"""
ControlNet Injection Module for CT-to-PET Diffusion Model.

This module implements the ControlNet mechanism to inject CT condition features
into a frozen Stable Diffusion U-Net.

Reference: 
    - ControlNet (Zhang et al., 2023) - "Adding Conditional Control to Text-to-Image Diffusion Models"
    - Uses Zero Convolutions for stable training
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from copy import deepcopy


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    Used for Zero Convolution initialization in ControlNet.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ZeroConv2d(nn.Module):
    """
    A Conv2d layer initialized with zeros.
    
    This is crucial for ControlNet's stable training - it ensures that
    the control signal starts at zero and gradually learns to contribute.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, padding: int = 0):
        super().__init__()
        self.conv = zero_module(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ControlNetInjection(nn.Module):
    """
    ControlNet-style injection module for CT-conditioned PET generation.
    
    This module:
    1. Encodes the CT image using DualStreamCTEncoder
    2. Processes the noisy latents through a trainable copy of the U-Net encoder
    3. Outputs control signals via Zero Convolutions to be added to the frozen U-Net decoder
    
    Args:
        condition_encoder: The DualStreamCTEncoder for extracting CT features
        control_model: A trainable copy of the SD U-Net encoder blocks
        block_out_channels: Channel sizes at each encoder block level
                           (default matches SD 1.5: [320, 640, 1280, 1280])
    """
    def __init__(
        self,
        condition_encoder: nn.Module,
        control_model: Optional[nn.Module] = None,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        latent_channels: int = 4,  # VAE latent channels
        condition_channels: int = 320,  # Output of DualStreamCTEncoder
    ):
        super().__init__()
        
        # CT Condition Encoder
        self.condition_encoder = condition_encoder
        
        # Trainable control model (copy of SD encoder)
        # If not provided, we'll use a simplified version
        self.control_model = control_model
        
        # Store channel configuration
        self.block_out_channels = block_out_channels
        
        # Zero Conv for initial condition injection
        # This aligns CT features (condition_channels) to latent space (latent_channels)
        self.condition_zero_conv = ZeroConv2d(
            in_channels=condition_channels,
            out_channels=latent_channels,
            kernel_size=1
        )
        
        # Zero Convs for each encoder block output
        # These process the control signals before adding to the main U-Net
        self.zero_convs = nn.ModuleList()
        
        # For SD-style architecture: 3 blocks per resolution level + 1 middle block
        # Total: 4 levels * 2 blocks + 1 middle = 12 control outputs typically
        # Simplified here to 1 per level + middle
        num_blocks = len(block_out_channels) + 1  # +1 for middle block
        for i, ch in enumerate(block_out_channels):
            self.zero_convs.append(ZeroConv2d(ch, ch))
        # Middle block
        self.zero_convs.append(ZeroConv2d(block_out_channels[-1], block_out_channels[-1]))
        
        # If no control_model provided, create simplified encoder blocks
        if self.control_model is None:
            self._build_simplified_control_model(latent_channels, block_out_channels)
    
    def _build_simplified_control_model(
        self, 
        latent_channels: int,
        block_out_channels: Tuple[int, ...]
    ):
        """
        Build a simplified control model for standalone use.
        
        This creates a basic encoder that mimics SD U-Net encoder structure:
        - Input projection
        - Downsampling blocks
        - Middle block
        """
        from .encoder import ResnetBlock2D  # Reuse our ResNet blocks
        
        layers = nn.ModuleList()
        
        # Input projection: 4 (latent) -> 320 (first block)
        layers.append(nn.Conv2d(latent_channels, block_out_channels[0], kernel_size=3, padding=1))
        
        # Encoder blocks with downsampling
        in_ch = block_out_channels[0]
        for out_ch in block_out_channels:
            # ResNet block
            layers.append(ResnetBlock2D(in_ch, out_ch))
            # Downsample (except for last block)
            if out_ch != block_out_channels[-1]:
                layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            in_ch = out_ch
        
        # Middle block
        layers.append(ResnetBlock2D(block_out_channels[-1], block_out_channels[-1]))
        
        self.control_blocks = layers
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        ct_image: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass for ControlNet injection.
        
        Args:
            noisy_latents: Noisy latent representation from VAE, shape (B, 4, H/8, W/8)
            timesteps: Diffusion timesteps, shape (B,)
            ct_image: Input CT image, shape (B, 1, H, W)
            encoder_hidden_states: Optional text embeddings (for compatibility), shape (B, seq_len, dim)
        
        Returns:
            control_outputs: List of feature maps to be added to the U-Net decoder
        """
        # Step 1: Extract condition features from CT
        # (B, 1, H, W) -> (B, 320, H, W)
        condition_features = self.condition_encoder(ct_image)
        
        # Step 2: Downsample condition features to match latent resolution
        # CT image is H×W, latent is H/8 × W/8
        # Use adaptive pooling to match
        latent_h, latent_w = noisy_latents.shape[2], noisy_latents.shape[3]
        condition_features = nn.functional.adaptive_avg_pool2d(
            condition_features, (latent_h, latent_w)
        )
        
        # Step 3: Project condition features and add to noisy latents
        condition_injection = self.condition_zero_conv(condition_features)
        controlled_latents = noisy_latents + condition_injection
        
        # Step 4: Pass through control model and collect outputs
        control_outputs = []
        
        if self.control_model is not None:
            # Use provided control model (full SD encoder copy)
            # This would need to match the exact diffusers UNet2DConditionModel structure
            h = controlled_latents
            for block in self.control_model.down_blocks:
                h, res_samples = block(h, timesteps, encoder_hidden_states)
                for res in res_samples:
                    control_outputs.append(res)
            # Middle
            h = self.control_model.mid_block(h, timesteps, encoder_hidden_states)
            control_outputs.append(h)
        else:
            # Use our simplified control blocks
            h = controlled_latents
            block_idx = 0
            for i, layer in enumerate(self.control_blocks):
                h = layer(h)
                # Save output after each ResNet block (not downsample)
                if isinstance(layer, ResnetBlock2D if hasattr(self, 'ResnetBlock2D') else nn.Module):
                    if hasattr(layer, 'skip'):  # It's a ResnetBlock2D
                        control_outputs.append(h)
                        block_idx += 1
        
        # Step 5: Apply Zero Convolutions to control outputs
        processed_outputs = []
        for i, (feat, zc) in enumerate(zip(control_outputs, self.zero_convs)):
            processed_outputs.append(zc(feat))
        
        return processed_outputs


class ControlledUNet(nn.Module):
    """
    Wrapper that combines a frozen U-Net with ControlNet injection.
    
    This is the main model to use during training/inference.
    
    Args:
        unet: The base (frozen) U-Net model
        controlnet: The ControlNetInjection module
        freeze_unet: Whether to freeze the base U-Net (default: True)
    """
    def __init__(
        self,
        unet: nn.Module,
        controlnet: ControlNetInjection,
        freeze_unet: bool = True,
    ):
        super().__init__()
        
        self.unet = unet
        self.controlnet = controlnet
        
        # Freeze U-Net if specified
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        ct_image: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with control injection.
        
        Args:
            noisy_latents: Noisy latent representation
            timesteps: Diffusion timesteps
            ct_image: Condition CT image
            encoder_hidden_states: Optional text embeddings
        
        Returns:
            Predicted noise (or velocity, depending on prediction type)
        """
        # Get control signals
        control_outputs = self.controlnet(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            ct_image=ct_image,
            encoder_hidden_states=encoder_hidden_states,
        )
        
        # Pass through U-Net with control injection
        # This requires modifying the U-Net forward to accept control signals
        # For diffusers, you would do:
        # return self.unet(noisy_latents, timesteps, encoder_hidden_states, down_block_additional_residuals=control_outputs)
        
        # Simplified version for now (just U-Net forward)
        return self.unet(noisy_latents, timesteps, encoder_hidden_states)


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    from encoder import DualStreamCTEncoder
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create condition encoder
    condition_encoder = DualStreamCTEncoder(
        in_channels=1,
        base_channels=64,
        out_channels=320,
    ).to(device)
    
    # Create ControlNet
    controlnet = ControlNetInjection(
        condition_encoder=condition_encoder,
        control_model=None,  # Use simplified version
        block_out_channels=(320, 640, 1280, 1280),
    ).to(device)
    
    # Test inputs
    batch_size = 2
    noisy_latents = torch.randn(batch_size, 4, 32, 32).to(device)  # Latent: 256/8 = 32
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    ct_image = torch.randn(batch_size, 1, 256, 256).to(device)
    
    # Forward pass
    with torch.no_grad():
        control_outputs = controlnet(noisy_latents, timesteps, ct_image)
    
    print(f"\nControlNet Injection Test:")
    print(f"  CT Image shape:     {ct_image.shape}")
    print(f"  Noisy Latents shape: {noisy_latents.shape}")
    print(f"  Number of control outputs: {len(control_outputs)}")
    for i, out in enumerate(control_outputs):
        print(f"    Control output {i}: {out.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in controlnet.parameters())
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
