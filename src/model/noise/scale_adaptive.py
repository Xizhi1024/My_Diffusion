"""Scale-adaptive noise schedule – protects small high-frequency lesions.

Decomposes the image via Laplacian pyramid, applies different noise rates
per frequency band, and reconstructs. Gabor energy further modulates the
high-frequency noise level: sharper regions receive less corruption.

The schedule keeps DDPM-style signal attenuation, i.e.

    x_t = sqrt(alpha_bar_t) * x_0 + sigma_t(scale, region) * eps

so training states remain compatible with the reverse process.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..interfaces import ConditionBundle, NoiseSchedule


class ScaleAdaptiveNoise(NoiseSchedule):
    """Scale-adaptive noise schedule with optional BBDM bridge mode.

    When bridge_mode=False (default): standard DDPM signal attenuation per band.
    When bridge_mode=True: BBDM bridge CT→PET formulation per band, i.e.
        x_t = m_t * CT + (1-m_t) * PET + sigma_t(scale) * eps
    where sigma_t varies per Laplacian pyramid band and is Gabor-modulated
    at high frequencies to protect small lesion edges.

    The m_t buffer is exposed so the sampling loop detects is_bbdm=True and
    uses the correct bridge reverse step.
    """

    name = "scale_adaptive_noise"

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_scales: int = 3,
        low_sigma_mult: float = 1.0,
        mid_sigma_mult: float = 0.75,
        high_sigma_mult: float = 0.45,
        use_gabor_energy: bool = True,
        gabor_weight: float = 1.0,
        bridge_mode: bool = False,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.num_train_timesteps = num_train_timesteps
        self.num_scales = num_scales
        self.low_sigma_mult = low_sigma_mult
        self.mid_sigma_mult = mid_sigma_mult
        self.high_sigma_mult = high_sigma_mult
        self.use_gabor_energy = use_gabor_energy
        self.gabor_weight = gabor_weight
        self.bridge_mode = bridge_mode

        # Base DDPM noise levels (reference) + alphas for DDIM sampling
        betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("base_sigma", (1.0 - alphas_cumprod).sqrt(), persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt(), persistent=False)

        # Pre-compute bridge coefficient m_t when in bridge mode.
        # Exposing m_t triggers the BBDM reverse-step path in the sampling loop.
        if bridge_mode:
            t = torch.arange(num_train_timesteps)
            m_t = t.float() / num_train_timesteps  # linear bridge: 0→1 as t: 0→T
            sigma_bridge = torch.sqrt(2 * m_t * (1 - m_t)).clamp_min(1e-8)
            self.register_buffer("m_t", m_t, persistent=False)
            self.register_buffer("sigma_t", sigma_bridge, persistent=False)

    def _laplacian_pyramid(self, x: torch.Tensor, levels: int = 3) -> List[torch.Tensor]:
        """Decompose into [low, mid1, mid2, ..., high_residual].

        Returns levels+1 tensors: the low-res base + N high-pass residuals.
        """
        pyramid = []
        current = x
        for _ in range(levels):
            low = F.avg_pool2d(current, kernel_size=2, stride=2)
            up = F.interpolate(low, size=current.shape[2:], mode="bilinear", align_corners=False)
            residual = current - up
            pyramid.append(residual)
            current = low
        pyramid.append(current)  # lowest resolution base
        return pyramid

    def _reconstruct_from_pyramid(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct image from Laplacian pyramid."""
        result = pyramid[-1]
        for residual in reversed(pyramid[:-1]):
            result = F.interpolate(result, size=residual.shape[2:], mode="bilinear", align_corners=False)
            result = result + residual
        return result

    def _broadcast_coeff(
        self,
        coeffs: torch.Tensor,
        timesteps: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Expand timestep coefficients to match a reference tensor."""
        out = coeffs[timesteps].to(device=ref.device, dtype=ref.dtype)
        while out.dim() < ref.dim():
            out = out.unsqueeze(-1)
        return out

    def _get_scale_multipliers(
        self,
        timesteps: torch.Tensor,
        gabor_energy: Optional[torch.Tensor] = None,
        shape: Tuple[int, ...] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-band noise multipliers [low, mid, high], each [B, 1, H, W]."""
        device = timesteps.device
        B = timesteps.shape[0]

        base = self.base_sigma[timesteps].to(device)
        while base.dim() < 4:
            base = base.unsqueeze(-1)

        if shape is not None:
            low = torch.ones(B, 1, *shape[2:], device=device) * self.low_sigma_mult * base
            mid = torch.ones(B, 1, *shape[2:], device=device) * self.mid_sigma_mult * base
            high = torch.ones(B, 1, *shape[2:], device=device) * self.high_sigma_mult * base
        else:
            low = self.low_sigma_mult * base
            mid = self.mid_sigma_mult * base
            high = self.high_sigma_mult * base

        # Gabor modulation: reduce high-frequency noise where Gabor energy is high
        if self.use_gabor_energy and gabor_energy is not None:
            gabor = F.avg_pool2d(gabor_energy, kernel_size=8)  # spatial smooth
            if gabor.shape[2:] != high.shape[2:]:
                gabor = F.interpolate(gabor, size=high.shape[2:], mode="bilinear", align_corners=False)
            high = high * (1.0 - self.gabor_weight * gabor.clamp(0.0, 1.0))

        return low, mid, high

    def _get_band_sigmas(
        self,
        timesteps: torch.Tensor,
        condition: ConditionBundle,
        pyramid: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Return the per-band sigma tensor for each Laplacian pyramid level."""
        gabor_energy = condition.get_map("gabor_energy")
        band_sigmas: List[torch.Tensor] = []
        for i, band in enumerate(pyramid):
            low, mid, high = self._get_scale_multipliers(timesteps, gabor_energy, band.shape)
            sigma = high if i == 0 else mid if i < len(pyramid) - 1 else low
            band_sigmas.append(sigma.to(device=band.device, dtype=band.dtype))
        return band_sigmas

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: ConditionBundle,
        x_source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decompose, noise each band differently, reconstruct.

        When bridge_mode=True and x_source is provided, uses BBDM bridge
        formulation per band:  x_t = m·CT + (1-m)·PET + σ·ε.
        """
        if not self.enabled:
            sqrt_alpha = self._broadcast_coeff(self.sqrt_alphas_cumprod, timesteps, x0)
            sigma = self._broadcast_coeff(self.sqrt_one_minus_alphas_cumprod, timesteps, x0)
            return sqrt_alpha * x0 + sigma * noise

        pyramid = self._laplacian_pyramid(noise, self.num_scales)
        x0_pyramid = self._laplacian_pyramid(x0, self.num_scales)
        band_sigmas = self._get_band_sigmas(timesteps, condition, x0_pyramid)

        if self.bridge_mode and x_source is not None:
            # BBDM bridge per band: x_t = m·CT + (1-m)·PET + σ·ε
            x_source_pyramid = self._laplacian_pyramid(x_source, self.num_scales)
            noisy_pyramid = []
            for x0_band, src_band, n_band, sigma_band in zip(
                x0_pyramid, x_source_pyramid, pyramid, band_sigmas
            ):
                m = self._broadcast_coeff(self.m_t, timesteps, x0_band)
                noisy_pyramid.append(m * src_band + (1 - m) * x0_band + sigma_band * n_band)
        else:
            # Standard DDPM per band: x_t = √ᾱ·x0 + σ·ε
            noisy_pyramid = []
            for x0_band, n_band, sigma_band in zip(x0_pyramid, pyramid, band_sigmas):
                signal = self._broadcast_coeff(self.sqrt_alphas_cumprod, timesteps, x0_band)
                noisy_pyramid.append(signal * x0_band + sigma_band * n_band)

        return self._reconstruct_from_pyramid(noisy_pyramid)

    def step_from_prediction(
        self,
        x_t: torch.Tensor,
        pred_x0: torch.Tensor,
        timesteps: torch.Tensor,
        next_timesteps: torch.Tensor,
        condition: ConditionBundle,
        x_source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform one DDIM-style reverse step consistent with this schedule."""
        if not self.enabled:
            alpha_t = self._broadcast_coeff(self.alphas_cumprod, timesteps, x_t)
            alpha_next = self._broadcast_coeff(self.alphas_cumprod, next_timesteps, x_t)
            eps_pred = (x_t - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt().clamp_min(1e-8)
            return alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * eps_pred

        x_t_pyramid = self._laplacian_pyramid(x_t, self.num_scales)
        x0_pyramid = self._laplacian_pyramid(pred_x0, self.num_scales)
        sigma_t_bands = self._get_band_sigmas(timesteps, condition, x_t_pyramid)
        sigma_next_bands = self._get_band_sigmas(next_timesteps, condition, x0_pyramid)

        if self.bridge_mode and x_source is not None:
            src_pyramid = self._laplacian_pyramid(x_source, self.num_scales)
            next_pyramid = []
            for x_t_band, x0_band, src_band, sigma_t, sigma_next in zip(
                x_t_pyramid,
                x0_pyramid,
                src_pyramid,
                sigma_t_bands,
                sigma_next_bands,
            ):
                m_t = self._broadcast_coeff(self.m_t, timesteps, x_t_band)
                m_next = self._broadcast_coeff(self.m_t, next_timesteps, x_t_band)
                eps_pred = (
                    x_t_band - m_t * src_band - (1.0 - m_t) * x0_band
                ) / sigma_t.clamp_min(1e-8)
                next_pyramid.append(m_next * src_band + (1.0 - m_next) * x0_band + sigma_next * eps_pred)

            return self._reconstruct_from_pyramid(next_pyramid)

        next_pyramid = []
        for x_t_band, x0_band, sigma_t, sigma_next in zip(
            x_t_pyramid,
            x0_pyramid,
            sigma_t_bands,
            sigma_next_bands,
        ):
            signal_t = self._broadcast_coeff(self.sqrt_alphas_cumprod, timesteps, x_t_band)
            signal_next = self._broadcast_coeff(self.sqrt_alphas_cumprod, next_timesteps, x0_band)
            eps_pred = (x_t_band - signal_t * x0_band) / sigma_t.clamp_min(1e-8)
            next_pyramid.append(signal_next * x0_band + sigma_next * eps_pred)

        return self._reconstruct_from_pyramid(next_pyramid)

    def get_tau(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / self.num_train_timesteps
