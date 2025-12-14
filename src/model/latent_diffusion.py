import torch
import torch.nn as nn
from typing import Optional
from diffusers import UNet2DModel, DDPMScheduler, AutoencoderKL
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        image_size: int = 128,
        latent_channels: int = 4,
        unet_config: Optional[dict] = None,
    ):
        super().__init__()

        self.image_size = image_size
        self.latent_channels = latent_channels

        # VAE for latent space
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae.eval()

        # U-Net for denoising
        unet_config = unet_config or {
            "in_channels": latent_channels * 2,  # target + condition
            "out_channels": latent_channels,
            "sample_size": image_size // 8,  # Add explicit sample size for latent dimensions
            "layers_per_block": 2,
            "block_out_channels": (64, 128, 256, 512),  # Simplified architecture
            "down_block_types": (
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            "up_block_types": (
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        }

        self.unet = UNet2DModel(**unet_config)

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )

        # DDIM scheduler for sampling
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )

    def encode_to_latent(self, x):
        # If input is single channel, repeat it to 3 channels for VAE
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        with torch.no_grad():
            posterior = self.vae.encode(x)
            latents = posterior.latent_dist.sample() * self.vae.config.scaling_factor
        return latents

    def decode_from_latent(self, latents):
        with torch.no_grad():
            decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            # Convert back to single channel by taking the mean across channels
            if decoded.shape[1] == 3:
                decoded = decoded.mean(dim=1, keepdim=True)
        return decoded

    def add_noise(self, latents, noise, timesteps):
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    def predict_noise(self, noisy_latents, timesteps, condition=None):
        if condition is not None:
            # Encode condition to latent space
            condition_latent = self.encode_to_latent(condition)
            # Ensure both tensors have the same spatial dimensions
            if noisy_latents.shape[2:] != condition_latent.shape[2:]:
                condition_latent = torch.nn.functional.interpolate(
                    condition_latent,
                    size=noisy_latents.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            # Simple concat conditioning for now
            # Will be replaced by dual-stream encoder later
            noisy_latents = torch.cat([noisy_latents, condition_latent], dim=1)

        return self.unet(noisy_latents, timesteps).sample

    @torch.no_grad()
    def sample(self, condition=None, num_inference_steps=50, generator=None):
        device = next(self.unet.parameters()).device

        # Determine batch size from condition
        batch_size = condition.shape[0] if condition is not None else 1

        # Start with pure noise
        latents = torch.randn(
            (batch_size, self.latent_channels, self.image_size // 8, self.image_size // 8),
            device=device,
            generator=generator,
        )

        self.ddim_scheduler.set_timesteps(num_inference_steps)

        for t in self.ddim_scheduler.timesteps:
            # Predict noise
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = self.predict_noise(latents, t_tensor, condition)

            # Compute previous sample
            latents = self.ddim_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode to image space
        image = self.decode_from_latent(latents)
        return image

    def forward(self, x, condition=None):
        # Encode to latent space
        latents = self.encode_to_latent(x)

        # Sample noise and timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x.shape[0],))
        timesteps = timesteps.to(x.device)

        # Add noise to latents
        noisy_latents = self.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = self.predict_noise(noisy_latents, timesteps, condition)

        return noise_pred, noise