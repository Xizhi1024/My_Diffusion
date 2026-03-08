import torch
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any


class MixedPrecisionTrainer:
    """
    Utility class for mixed precision training using PyTorch AMP.
    """

    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 2.**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """
        Args:
            enabled: Whether to enable mixed precision training
            init_scale: Initial loss scaling
            growth_factor: Factor by which the scale is multiplied during
                           good training steps
            backoff_factor: Factor by which the scale is multiplied during
                            bad training steps
            growth_interval: Number of good training steps before increasing
                            the loss scale
        """
        self.enabled = enabled and torch.cuda.is_available()

        if self.enabled:
            self.scaler = torch.amp.GradScaler('cuda',
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
            print(f"Mixed precision training enabled (init_scale={init_scale})")
        else:
            self.scaler = None
            print("Mixed precision training disabled")

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale the loss for mixed precision training.
        """
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    @torch.amp.autocast('cuda', enabled=True)
    def forward(self, model: torch.nn.Module, *args, **kwargs):
        """
        Forward pass with autocast.
        """
        return model(*args, **kwargs)

    def backward(self, loss: torch.Tensor):
        """
        Backward pass with gradient scaling.
        """
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, optimizer):
        """
        Optimizer step with gradient unscale.
        """
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def clip_grad_norm_(self, model_parameters, max_norm: float, optimizer=None):
        """
        Clip gradient norm with unscaled gradients.
        """
        if self.enabled:
            self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_parameters, max_norm)

    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dict for saving.
        """
        if self.enabled:
            return {
                'scaler': self.scaler.state_dict(),
                'enabled': self.enabled
            }
        return {'enabled': False}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state dict from checkpoint.
        """
        if self.enabled and 'scaler' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler'])