"""CT-to-PET comparison models aligned with the SLMF-BBDM interface.

All models in this file expose the same minimal contract as the project
model:

    loss, logs = model(batch)
    result = model.sample(batch)

The batch interface is intentionally identical to SLMF-BBDM:
``batch["ct"]`` is the source CT tensor and ``batch["pet"]`` is the target
PET tensor, both shaped [B, 1, H, W] and normalised to [-1, 1].

Implementation sources used for reproduction:
  - Pix2Pix/CycleGAN: official junyanz PyTorch implementation conventions.
  - RegGAN: official/public Reg-GAN implementation and paper description.
  - CPDM: official CPDM/BBDM repository and paper description, adapted to
    direct 2D tensor training instead of the original VQGAN-heavy pipeline.
  - District-specific GAN: paper-level reproduction of the 2025 whole-body
    virtual scanner idea when no directly reusable official code is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


def _requires_grad(modules: Iterable[nn.Module], flag: bool) -> None:
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(flag)


def _image_gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * ((pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean())


def _default_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float = 1.0,
    l1_weight: float = 1.0,
    gradient_weight: float = 0.1,
) -> tuple[torch.Tensor, TensorDict]:
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    grad = _image_gradient_l1(pred, target)
    loss = mse_weight * mse + l1_weight * l1 + gradient_weight * grad
    return loss, {
        "loss/recon_mse": mse.detach(),
        "loss/recon_l1": l1.detach(),
        "loss/recon_gradient": grad.detach(),
    }


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: bool = True,
        activation: str = "leaky",
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "silu":
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """Pix2Pix-style U-Net generator for 2D CT-to-PET slices."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.down1 = ConvBlock(in_channels, c, norm=False)
        self.down2 = ConvBlock(c, c * 2)
        self.down3 = ConvBlock(c * 2, c * 4)
        self.down4 = ConvBlock(c * 4, c * 8)
        self.bottleneck = ConvBlock(
            c * 8,
            c * 8,
            norm=False,
            activation="relu",
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up1 = UpBlock(c * 8, c * 4, dropout=True)
        self.up2 = UpBlock(c * 8, c * 2)
        self.up3 = UpBlock(c * 4, c)
        self.final = nn.Sequential(nn.ConvTranspose2d(c * 2, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        b = self.bottleneck(d4)
        u1 = self.up1(b)
        if u1.shape[-2:] != d3.shape[-2:]:
            u1 = F.interpolate(u1, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        if u2.shape[-2:] != d2.shape[-2:]:
            u2 = F.interpolate(u2, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        if u3.shape[-2:] != d1.shape[-2:]:
            u3 = F.interpolate(u3, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        out = self.final(torch.cat([u3, d1], dim=1))
        if out.shape[-2:] != input_hw:
            out = F.interpolate(out, size=input_hw, mode="bilinear", align_corners=False)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """CycleGAN/RegGAN-style ResNet generator."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 6,
    ):
        super().__init__()
        c = base_channels
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, c, 7, bias=False),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True),
            ConvBlock(c, c * 2, activation="relu"),
            ConvBlock(c * 2, c * 4, activation="relu"),
        ]
        layers += [ResnetBlock(c * 4) for _ in range(num_blocks)]
        layers += [
            nn.ConvTranspose2d(c * 4, c * 2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, out_channels, 7),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        out = self.net(x)
        if out.shape[-2:] != input_hw:
            out = F.interpolate(out, size=input_hw, mode="bilinear", align_corners=False)
        return out


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator used by Pix2Pix/CycleGAN."""

    def __init__(self, in_channels: int = 2, base_channels: int = 64, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = base_channels
        for n in range(1, num_layers):
            nf_prev = nf
            nf = min(base_channels * 2**n, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(nf, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(nf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, 4, 1, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSGANLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)


class SimpleUNet(nn.Module):
    """Compact U-Net used for the CPDM-style diffusion baseline."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 64,
        final_activation: Optional[str] = "tanh",
    ):
        super().__init__()
        self.final_activation = final_activation
        c = base_channels
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.down1 = nn.Sequential(nn.Conv2d(c, c * 2, 4, 2, 1), nn.SiLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.SiLU(inplace=True))
        self.mid = nn.Sequential(ResnetBlock(c * 4), ResnetBlock(c * 4))
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),
            nn.SiLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c, 4, 2, 1),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Sequential(nn.Conv2d(c * 2, c, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(c, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        s1 = self.in_conv(x)
        s2 = self.down1(s1)
        z = self.down2(s2)
        z = self.mid(z)
        z = self.up1(z)
        if z.shape[-2:] != s2.shape[-2:]:
            z = F.interpolate(z, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        z = self.up2(torch.cat([z, s2], dim=1))
        if z.shape[-2:] != s1.shape[-2:]:
            z = F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        out = self.out(torch.cat([z, s1], dim=1))
        if self.final_activation == "tanh":
            out = out.tanh()
        if out.shape[-2:] != input_hw:
            out = F.interpolate(out, size=input_hw, mode="bilinear", align_corners=False)
        return out


class RegistrationNet(nn.Module):
    """Small deformation field estimator for RegGAN-style loss correction."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32, max_flow: float = 0.15):
        super().__init__()
        self.max_flow = max_flow
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 2, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        return self.max_flow * self.net(torch.cat([moving, fixed], dim=1))


def warp_image(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    B, _, H, W = image.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device, dtype=image.dtype),
        torch.linspace(-1, 1, W, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, H, W, 2)
    grid = base_grid + flow.permute(0, 2, 3, 1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


class VectorQuantizerEMA(nn.Module):
    """Small VQGAN-style codebook used by the CPDM first stage.

    Official CPDM trains a VQGAN first-stage model and then runs Brownian
    Bridge diffusion in that latent space.  This module keeps that mechanism
    available inside the unified CT/PET tensor interface.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 3,
        commitment_weight: float = 0.25,
        enabled: bool = True,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_weight = float(commitment_weight)
        self.enabled = bool(enabled)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return z, torch.zeros((), device=z.device, dtype=z.dtype)
        # [B, C, H, W] -> [BHW, C]
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(dist, dim=1)
        quant = self.embedding(indices).view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quant = quant.permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(quant.detach(), z) * self.commitment_weight + F.mse_loss(quant, z.detach())
        quant = z + (quant - z).detach()
        return quant, codebook_loss


class VQGANFirstStage(nn.Module):
    """VQGAN-like first stage for CPDM latent Brownian Bridge training."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 3,
        base_channels: int = 32,
        num_embeddings: int = 512,
        use_vq: bool = True,
    ):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 2, latent_channels, 3, padding=1),
        )
        self.quantizer = VectorQuantizerEMA(num_embeddings, latent_channels, enabled=use_vq)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, c * 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, in_channels, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor, *, quantize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        if quantize:
            return self.quantizer(z)
        return z, torch.zeros((), device=x.device, dtype=x.dtype)

    def decode(self, z: torch.Tensor, output_size: Optional[tuple[int, int]] = None) -> torch.Tensor:
        out = self.decoder(z)
        if output_size is not None and out.shape[-2:] != output_size:
            out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, qloss = self.encode(x, quantize=True)
        recon = self.decode(z, output_size=x.shape[-2:])
        return recon, z, qloss


class SpatialRescalerCondition(nn.Module):
    """CPDM-style spatial condition encoder.

    The official config uses a SpatialRescaler condition stage.  Here the
    source tensor is CT plus CPDM's task-specific attention/attenuation maps,
    rescaled to the latent resolution and projected to 3 channels.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_stages: int = 1):
        super().__init__()
        self.n_stages = int(n_stages)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        out = x
        for _ in range(self.n_stages):
            out = F.interpolate(out, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@dataclass
class LossWeights:
    lambda_l1: float = 100.0
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_registration: float = 10.0
    lambda_smooth: float = 0.1
    lambda_attention: float = 1.0
    lambda_district: float = 100.0
    lambda_vq: float = 1.0
    lambda_first_stage: float = 1.0
    lambda_latent: float = 1.0
    lambda_patch_overlap: float = 0.1
    mse_weight: float = 1.0
    l1_weight: float = 1.0
    gradient_weight: float = 0.1


class ComparisonBase(nn.Module):
    model_name = "comparison_base"

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_weights = LossWeights(**{k: v for k, v in kwargs.items() if k in LossWeights.__annotations__})
        # Keep these attributes so the existing Trainer can print safely.
        self.priors = nn.ModuleDict()
        self.loss_terms = nn.ModuleDict()

    @staticmethod
    def _ct(batch: TensorDict) -> torch.Tensor:
        return batch["ct"]

    @staticmethod
    def _pet(batch: TensorDict) -> torch.Tensor:
        return batch["pet"]

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def sample(self, batch: TensorDict, **_) -> TensorDict:
        self.eval()
        return {"synthetic_pet": self.generate(self._ct(batch))}

    def generate(self, ct: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1) Pix2Pix
# ---------------------------------------------------------------------------


class Pix2PixComparison(ComparisonBase):
    """Official-code-aligned Pix2Pix baseline: U-Net generator + PatchGAN."""

    model_name = "pix2pix"

    def __init__(self, base_channels: int = 64, lambda_l1: float = 100.0, **kwargs):
        super().__init__(lambda_l1=lambda_l1, **kwargs)
        self.generator = UNetGenerator(1, 1, base_channels)
        self.discriminator = PatchDiscriminator(2, base_channels)
        self.gan_loss = LSGANLoss()

    def generate(self, ct: torch.Tensor) -> torch.Tensor:
        return self.generator(ct)

    def forward(self, batch: TensorDict) -> tuple[torch.Tensor, TensorDict]:
        ct, pet = self._ct(batch), self._pet(batch)
        fake = self.generator(ct)

        _requires_grad([self.discriminator], False)
        pred_fake_for_g = self.discriminator(torch.cat([ct, fake], dim=1))
        loss_g_gan = self.gan_loss(pred_fake_for_g, True)
        loss_g_l1 = F.l1_loss(fake, pet) * self.loss_weights.lambda_l1
        loss_g = loss_g_gan + loss_g_l1

        _requires_grad([self.discriminator], True)
        pred_real = self.discriminator(torch.cat([ct, pet], dim=1))
        pred_fake = self.discriminator(torch.cat([ct, fake.detach()], dim=1))
        loss_d = 0.5 * (self.gan_loss(pred_real, True) + self.gan_loss(pred_fake, False))

        total = loss_g + loss_d
        logs = {
            "loss/total": total.detach(),
            "loss/g_gan": loss_g_gan.detach(),
            "loss/g_l1": loss_g_l1.detach(),
            "loss/d": loss_d.detach(),
        }
        return total, logs


# ---------------------------------------------------------------------------
# 2) CycleGAN
# ---------------------------------------------------------------------------


class CycleGANComparison(ComparisonBase):
    """Official-code-aligned CycleGAN baseline with CT<->PET generators."""

    model_name = "cyclegan"

    def __init__(
        self,
        base_channels: int = 64,
        num_res_blocks: int = 6,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        **kwargs,
    ):
        super().__init__(lambda_cycle=lambda_cycle, lambda_identity=lambda_identity, **kwargs)
        self.g_ct_to_pet = ResnetGenerator(1, 1, base_channels, num_res_blocks)
        self.g_pet_to_ct = ResnetGenerator(1, 1, base_channels, num_res_blocks)
        self.d_pet = PatchDiscriminator(1, base_channels)
        self.d_ct = PatchDiscriminator(1, base_channels)
        self.gan_loss = LSGANLoss()

    def generate(self, ct: torch.Tensor) -> torch.Tensor:
        return self.g_ct_to_pet(ct)

    def forward(self, batch: TensorDict) -> tuple[torch.Tensor, TensorDict]:
        ct, pet = self._ct(batch), self._pet(batch)
        fake_pet = self.g_ct_to_pet(ct)
        rec_ct = self.g_pet_to_ct(fake_pet)
        fake_ct = self.g_pet_to_ct(pet)
        rec_pet = self.g_ct_to_pet(fake_ct)

        _requires_grad([self.d_pet, self.d_ct], False)
        loss_g_adv = self.gan_loss(self.d_pet(fake_pet), True) + self.gan_loss(self.d_ct(fake_ct), True)
        loss_cycle = (F.l1_loss(rec_ct, ct) + F.l1_loss(rec_pet, pet)) * self.loss_weights.lambda_cycle
        loss_identity = (
            F.l1_loss(self.g_ct_to_pet(pet), pet) + F.l1_loss(self.g_pet_to_ct(ct), ct)
        ) * self.loss_weights.lambda_identity
        loss_g = loss_g_adv + loss_cycle + loss_identity

        _requires_grad([self.d_pet, self.d_ct], True)
        loss_d_pet = 0.5 * (self.gan_loss(self.d_pet(pet), True) + self.gan_loss(self.d_pet(fake_pet.detach()), False))
        loss_d_ct = 0.5 * (self.gan_loss(self.d_ct(ct), True) + self.gan_loss(self.d_ct(fake_ct.detach()), False))
        loss_d = loss_d_pet + loss_d_ct

        total = loss_g + loss_d
        logs = {
            "loss/total": total.detach(),
            "loss/g_adv": loss_g_adv.detach(),
            "loss/cycle": loss_cycle.detach(),
            "loss/identity": loss_identity.detach(),
            "loss/d_pet": loss_d_pet.detach(),
            "loss/d_ct": loss_d_ct.detach(),
        }
        return total, logs


# ---------------------------------------------------------------------------
# 3) RegGAN
# ---------------------------------------------------------------------------


class RegGANComparison(ComparisonBase):
    """RegGAN-style medical I2I baseline with registration loss correction."""

    model_name = "reggan"

    def __init__(
        self,
        base_channels: int = 64,
        num_res_blocks: int = 6,
        lambda_registration: float = 10.0,
        lambda_smooth: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            lambda_registration=lambda_registration,
            lambda_smooth=lambda_smooth,
            **kwargs,
        )
        self.generator = ResnetGenerator(1, 1, base_channels, num_res_blocks)
        self.registration = RegistrationNet(2, max(16, base_channels // 2))
        self.discriminator = PatchDiscriminator(2, base_channels)
        self.gan_loss = LSGANLoss()

    def generate(self, ct: torch.Tensor) -> torch.Tensor:
        return self.generator(ct)

    def forward(self, batch: TensorDict) -> tuple[torch.Tensor, TensorDict]:
        ct, pet = self._ct(batch), self._pet(batch)
        fake_pet = self.generator(ct)
        flow = self.registration(pet, fake_pet)
        corrected_pet = warp_image(pet, flow)

        _requires_grad([self.discriminator], False)
        loss_g_adv = self.gan_loss(self.discriminator(torch.cat([ct, fake_pet], dim=1)), True)
        loss_reg = F.l1_loss(fake_pet, corrected_pet) * self.loss_weights.lambda_registration
        loss_smooth = smoothness_loss(flow) * self.loss_weights.lambda_smooth
        loss_g = loss_g_adv + loss_reg + loss_smooth

        _requires_grad([self.discriminator], True)
        pred_real = self.discriminator(torch.cat([ct, pet], dim=1))
        pred_fake = self.discriminator(torch.cat([ct, fake_pet.detach()], dim=1))
        loss_d = 0.5 * (self.gan_loss(pred_real, True) + self.gan_loss(pred_fake, False))

        total = loss_g + loss_d
        logs = {
            "loss/total": total.detach(),
            "loss/g_adv": loss_g_adv.detach(),
            "loss/registration": loss_reg.detach(),
            "loss/flow_smooth": loss_smooth.detach(),
            "loss/d": loss_d.detach(),
            "reggan/flow_abs": flow.abs().mean().detach(),
        }
        return total, logs


# ---------------------------------------------------------------------------
# 4) CPDM
# ---------------------------------------------------------------------------


class CPDMComparison(ComparisonBase):
    """CPDM-style CT-to-PET conditional Brownian Bridge diffusion baseline.

    The official CPDM code combines Brownian Bridge diffusion, VQGAN, a CT
    object-attention map, and an attenuation map.  This adaptation keeps the
    paper's task-specific conditioning and BBDM training objective, while
    exposing the same direct 2D tensor interface as SLMF-BBDM.
    """

    model_name = "cpdm"

    def __init__(
        self,
        base_channels: int = 64,
        first_stage_channels: int = 32,
        latent_channels: int = 3,
        vq_num_embeddings: int = 512,
        use_vq: bool = True,
        num_train_timesteps: int = 1000,
        lambda_attention: float = 1.0,
        lambda_vq: float = 1.0,
        lambda_first_stage: float = 1.0,
        lambda_latent: float = 1.0,
        mse_weight: float = 1.0,
        l1_weight: float = 1.0,
        gradient_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            lambda_attention=lambda_attention,
            lambda_vq=lambda_vq,
            lambda_first_stage=lambda_first_stage,
            lambda_latent=lambda_latent,
            mse_weight=mse_weight,
            l1_weight=l1_weight,
            gradient_weight=gradient_weight,
            **kwargs,
        )
        self.latent_channels = int(latent_channels)
        self.num_train_timesteps = int(num_train_timesteps)
        # Official CPDM uses a VQGAN first stage and a 9->3 UNet in latent space.
        self.first_stage = VQGANFirstStage(
            in_channels=1,
            latent_channels=self.latent_channels,
            base_channels=first_stage_channels,
            num_embeddings=vq_num_embeddings,
            use_vq=use_vq,
        )
        self.condition_stage = SpatialRescalerCondition(in_channels=3, out_channels=self.latent_channels)
        self.denoiser = SimpleUNet(
            in_channels=self.latent_channels * 3,
            out_channels=self.latent_channels,
            base_channels=base_channels,
            final_activation=None,
        )
        m_t = torch.linspace(1.0, 0.0, self.num_train_timesteps)
        sigma_t = torch.sin(torch.linspace(0.0, torch.pi, self.num_train_timesteps)) * 0.25
        self.register_buffer("m_t", m_t)
        self.register_buffer("sigma_t", sigma_t)

    @staticmethod
    def _attention_map(batch: TensorDict) -> torch.Tensor:
        if "mask" in batch and torch.is_tensor(batch["mask"]) and batch["mask"].shape[1] == 1:
            return batch["mask"].float().clamp(0.0, 1.0)
        ct = batch["ct"]
        pooled = F.avg_pool2d(ct, kernel_size=9, stride=1, padding=4)
        edge = (ct - pooled).abs()
        denom = edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (edge / denom).clamp(0.0, 1.0)

    @staticmethod
    def _attenuation_map(batch: TensorDict) -> torch.Tensor:
        if "mu_map" in batch and torch.is_tensor(batch["mu_map"]):
            mu = batch["mu_map"]
            if mu.shape[1] == 1:
                denom = mu.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                return (mu / denom).clamp(-1.0, 1.0)
        if "ct_hu" in batch and torch.is_tensor(batch["ct_hu"]):
            return (batch["ct_hu"] / 1000.0).clamp(-1.0, 1.0)
        return batch["ct"]

    def _condition_latent(self, batch: TensorDict, latent_size: tuple[int, int]) -> torch.Tensor:
        attention = self._attention_map(batch)
        attenuation = self._attenuation_map(batch)
        cond = torch.cat([batch["ct"], attention, attenuation], dim=1)
        return self.condition_stage(cond, size=latent_size)

    def _model_input(self, x_t: torch.Tensor, ct_z: torch.Tensor, cond_z: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_t, ct_z, cond_z], dim=1)

    def _add_bridge_noise(self, ct_z: torch.Tensor, pet_z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        m = self.m_t[t].view(-1, 1, 1, 1)
        sigma = self.sigma_t[t].view(-1, 1, 1, 1)
        return m * ct_z + (1.0 - m) * pet_z + sigma * torch.randn_like(pet_z)

    def generate(self, ct: torch.Tensor) -> torch.Tensor:
        batch = {"ct": ct, "mask": torch.zeros_like(ct), "mu_map": torch.zeros_like(ct)}
        return self._sample_impl(batch, num_steps=20)

    def forward(self, batch: TensorDict) -> tuple[torch.Tensor, TensorDict]:
        ct, pet = self._ct(batch), self._pet(batch)
        B = ct.shape[0]
        t = torch.randint(0, self.num_train_timesteps, (B,), device=ct.device)

        # First-stage VQGAN path: reconstruct inputs and diffuse in latent space.
        ct_recon, ct_z, ct_vq = self.first_stage(ct)
        pet_recon, pet_z, pet_vq = self.first_stage(pet)
        cond_z = self._condition_latent(batch, latent_size=pet_z.shape[-2:])

        x_t = self._add_bridge_noise(ct_z, pet_z, t)
        target_grad = ct_z - pet_z
        pred_grad = self.denoiser(self._model_input(x_t, ct_z, cond_z))
        pred_pet_z = ct_z - pred_grad
        pred_pet = self.first_stage.decode(pred_pet_z, output_size=pet.shape[-2:])

        recon, logs = _default_reconstruction_loss(
            pred_pet,
            pet,
            mse_weight=self.loss_weights.mse_weight,
            l1_weight=self.loss_weights.l1_weight,
            gradient_weight=self.loss_weights.gradient_weight,
        )
        attention = self._attention_map(batch)
        att_loss = ((pred_pet - pet).abs() * (1.0 + attention)).mean() * self.loss_weights.lambda_attention
        grad_loss = F.mse_loss(pred_grad, target_grad) * self.loss_weights.lambda_latent
        latent_loss = F.l1_loss(pred_pet_z, pet_z) * self.loss_weights.lambda_latent
        first_stage_loss = (
            F.l1_loss(ct_recon, ct) + F.l1_loss(pet_recon, pet)
        ) * self.loss_weights.lambda_first_stage
        vq_loss = (ct_vq + pet_vq) * self.loss_weights.lambda_vq
        total = recon + att_loss + grad_loss + latent_loss + first_stage_loss + vq_loss
        logs.update({
            "loss/total": total.detach(),
            "loss/cpdm_attention": att_loss.detach(),
            "loss/cpdm_grad": grad_loss.detach(),
            "loss/cpdm_latent_l1": latent_loss.detach(),
            "loss/cpdm_first_stage": first_stage_loss.detach(),
            "loss/cpdm_vq": vq_loss.detach(),
            "diffusion/t_mean": t.float().mean().detach(),
        })
        return total, logs

    @torch.no_grad()
    def _sample_impl(self, batch: TensorDict, num_steps: int = 20) -> torch.Tensor:
        ct = batch["ct"]
        B = ct.shape[0]
        device = ct.device
        ct_z, _ = self.first_stage.encode(ct, quantize=True)
        cond_z = self._condition_latent(batch, latent_size=ct_z.shape[-2:])
        x_t = ct_z + 0.05 * torch.randn_like(ct_z)
        pred_pet_z = ct_z
        steps = torch.linspace(self.num_train_timesteps - 1, 0, num_steps, device=device, dtype=torch.long)
        for idx, t_value in enumerate(steps):
            t = torch.full((B,), t_value, device=device, dtype=torch.long)
            pred_grad = self.denoiser(self._model_input(x_t, ct_z, cond_z))
            pred_pet_z = ct_z - pred_grad
            if idx < len(steps) - 1:
                t_next = steps[idx + 1]
                m_next = self.m_t[t_next].view(1, 1, 1, 1)
                x_t = m_next * ct_z + (1.0 - m_next) * pred_pet_z
            else:
                x_t = pred_pet_z
        return self.first_stage.decode(x_t, output_size=ct.shape[-2:]).clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(self, batch: TensorDict, num_steps: Optional[int] = None, **_) -> TensorDict:
        self.eval()
        return {"synthetic_pet": self._sample_impl(batch, num_steps=num_steps or 20)}


# ---------------------------------------------------------------------------
# 5) District-specific GAN
# ---------------------------------------------------------------------------


class DistrictSpecificGANComparison(ComparisonBase):
    """Paper-level reproduction of 2025 district-specific whole-body GAN.

    If a batch provides ``district_mask`` with four channels, those masks are
    used directly.  Otherwise the 2D slice is split into four deterministic
    anatomical proxy districts so the model remains runnable on the current
    dataset interface.
    """

    model_name = "district_gan"
    districts = ("head", "trunk", "arms", "legs")

    def __init__(
        self,
        base_channels: int = 64,
        lambda_district: float = 100.0,
        lambda_patch_overlap: float = 0.1,
        patch_size: int = 96,
        patch_overlap: int = 24,
        sliding_window_inference: bool = True,
        **kwargs,
    ):
        super().__init__(
            lambda_district=lambda_district,
            lambda_patch_overlap=lambda_patch_overlap,
            **kwargs,
        )
        self.patch_size = int(patch_size)
        self.patch_overlap = int(patch_overlap)
        self.sliding_window_inference = bool(sliding_window_inference)
        self.generators = nn.ModuleDict({
            name: UNetGenerator(1, 1, base_channels) for name in self.districts
        })
        self.discriminators = nn.ModuleDict({
            name: PatchDiscriminator(2, base_channels) for name in self.districts
        })
        self.gan_loss = LSGANLoss()

    def _soften_masks(self, masks: torch.Tensor, blur: int = 9) -> torch.Tensor:
        if blur <= 1:
            return masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pad = blur // 2
        flat = masks.reshape(masks.shape[0] * masks.shape[1], 1, masks.shape[2], masks.shape[3])
        flat = F.avg_pool2d(flat, kernel_size=blur, stride=1, padding=pad)
        soft = flat.reshape_as(masks)
        return soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _district_masks(self, batch: TensorDict) -> torch.Tensor:
        ct = batch["ct"]
        B, _, H, W = ct.shape
        if "district_mask" in batch and torch.is_tensor(batch["district_mask"]):
            masks = batch["district_mask"].to(device=ct.device, dtype=ct.dtype)
            if masks.shape[1] == 4:
                if masks.shape[-2:] != (H, W):
                    masks = F.interpolate(masks, size=(H, W), mode="nearest")
                denom = masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
                return self._soften_masks(masks / denom)
        masks = torch.zeros(B, 4, H, W, device=ct.device, dtype=ct.dtype)
        # Whole-body paper districts approximated in 2D as ordered axial bands.
        # Softening below imitates overlap-averaging between neighbouring districts.
        cuts = [0, max(1, H // 5), max(2, H // 2), max(3, 4 * H // 5), H]
        for i in range(4):
            masks[:, i, cuts[i]:cuts[i + 1], :] = 1.0
        return self._soften_masks(masks)

    def random_patch_batch(self, batch: TensorDict) -> TensorDict:
        ct = batch["ct"]
        _, _, H, W = ct.shape
        p = self.patch_size
        if p <= 0 or (H <= p and W <= p):
            return batch
        top = torch.randint(0, max(H - p + 1, 1), (1,), device=ct.device).item()
        left = torch.randint(0, max(W - p + 1, 1), (1,), device=ct.device).item()
        bottom, right = min(top + p, H), min(left + p, W)
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() == 4 and value.shape[-2:] == (H, W):
                out[key] = value[..., top:bottom, left:right]
            else:
                out[key] = value
        return out

    def _generate_direct(self, ct: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        fake_parts = []
        for idx, name in enumerate(self.districts):
            fake_parts.append(self.generators[name](ct) * masks[:, idx:idx + 1])
        return torch.stack(fake_parts, dim=0).sum(dim=0).clamp(-1.0, 1.0)

    def _window_starts(self, length: int, window: int, stride: int) -> list[int]:
        if length <= window:
            return [0]
        starts = list(range(0, max(length - window + 1, 1), stride))
        if starts[-1] != length - window:
            starts.append(length - window)
        return starts

    def _sliding_window_generate(self, ct: torch.Tensor) -> torch.Tensor:
        B, C, H, W = ct.shape
        p = self.patch_size if self.patch_size > 0 else min(H, W)
        if not self.sliding_window_inference or (H <= p and W <= p):
            return self._generate_direct(ct, self._district_masks({"ct": ct}))
        stride = max(1, p - self.patch_overlap)
        output = torch.zeros_like(ct)
        weight = torch.zeros_like(ct)
        for top in self._window_starts(H, p, stride):
            for left in self._window_starts(W, p, stride):
                patch = ct[..., top:top + p, left:left + p]
                pred = self._generate_direct(patch, self._district_masks({"ct": patch}))
                output[..., top:top + pred.shape[-2], left:left + pred.shape[-1]] += pred
                weight[..., top:top + pred.shape[-2], left:left + pred.shape[-1]] += 1.0
        return (output / weight.clamp_min(1.0)).clamp(-1.0, 1.0)

    def generate(self, ct: torch.Tensor) -> torch.Tensor:
        return self._sliding_window_generate(ct)

    def forward(self, batch: TensorDict) -> tuple[torch.Tensor, TensorDict]:
        ct, pet = self._ct(batch), self._pet(batch)
        masks = self._district_masks(batch)

        fake_parts: list[torch.Tensor] = []
        loss_g_adv = torch.tensor(0.0, device=ct.device)
        loss_l1 = torch.tensor(0.0, device=ct.device)

        _requires_grad(self.discriminators.values(), False)
        for idx, name in enumerate(self.districts):
            mask = masks[:, idx:idx + 1]
            fake_i = self.generators[name](ct) * mask
            real_i = pet * mask
            ct_i = ct * mask
            fake_parts.append(fake_i)
            loss_g_adv = loss_g_adv + self.gan_loss(self.discriminators[name](torch.cat([ct_i, fake_i], dim=1)), True)
            loss_l1 = loss_l1 + F.l1_loss(fake_i, real_i) * self.loss_weights.lambda_district
        fake = torch.stack(fake_parts, dim=0).sum(dim=0).clamp(-1.0, 1.0)
        overlap_loss = torch.tensor(0.0, device=ct.device)
        mask_sum = masks.sum(dim=1, keepdim=True)
        if (mask_sum > 1.01).any():
            overlap_loss = (fake - pet).abs().mul((mask_sum - 1.0).clamp_min(0.0)).mean()

        _requires_grad(self.discriminators.values(), True)
        loss_d = torch.tensor(0.0, device=ct.device)
        for idx, name in enumerate(self.districts):
            mask = masks[:, idx:idx + 1]
            real_pair = torch.cat([ct * mask, pet * mask], dim=1)
            fake_pair = torch.cat([ct * mask, fake.detach() * mask], dim=1)
            loss_d = loss_d + 0.5 * (
                self.gan_loss(self.discriminators[name](real_pair), True)
                + self.gan_loss(self.discriminators[name](fake_pair), False)
            )

        total = loss_g_adv + loss_l1 + loss_d + overlap_loss * self.loss_weights.lambda_patch_overlap
        logs = {
            "loss/total": total.detach(),
            "loss/g_adv": loss_g_adv.detach(),
            "loss/district_l1": loss_l1.detach(),
            "loss/d": loss_d.detach(),
            "loss/patch_overlap": overlap_loss.detach(),
        }
        return total, logs

    @torch.no_grad()
    def sample(self, batch: TensorDict, **_) -> TensorDict:
        self.eval()
        masks = self._district_masks(batch)
        ct = batch["ct"]
        return {
            "synthetic_pet": self._sliding_window_generate(ct),
            "district_masks": masks,
        }


MODEL_REGISTRY = {
    "pix2pix": Pix2PixComparison,
    "cyclegan": CycleGANComparison,
    "reggan": RegGANComparison,
    "cpdm": CPDMComparison,
    "district_gan": DistrictSpecificGANComparison,
}


def build_comparison_model(config: Dict) -> ComparisonBase:
    model_cfg = dict(config.get("comparison_model", config.get("model", {})))
    name = model_cfg.pop("name", "pix2pix")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown comparison model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**model_cfg)
