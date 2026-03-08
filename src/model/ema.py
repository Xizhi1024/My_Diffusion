import torch
import copy
from typing import Optional


class EMA:
    """
    EMA (Exponential Moving Average) - 兼容 ema_pytorch 接口

    这是参考 diffusion 项目 (denoising-diffusion-pytorch) 的 EMA 实现
    使用 ema_pytorch 库的接口风格
    """

    def __init__(
        self,
        model: torch.nn.Module,
        beta: float = 0.995,
        update_every: int = 10,
    ):
        """
        Args:
            model: 要应用 EMA 的模型
            beta: EMA 衰减率 (默认 0.995)
            update_every: 每隔多少步更新一次 EMA (默认 10)
        """
        self.beta = beta
        self.update_every = update_every
        self.step = 0

        # 创建 EMA 模型的深拷贝
        self.ema_model = copy.deepcopy(model)

        # EMA 模型的参数不需要梯度
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        # 存储参数引用
        self.model_params = list(model.parameters())
        self.ema_params = list(self.ema_model.parameters())

    @torch.no_grad()
    def update(self):
        """更新 EMA 参数"""
        self.step += 1

        # 只在指定的步数间隔更新
        if self.step % self.update_every != 0:
            return

        # EMA 更新公式: ema = beta * ema + (1 - beta) * model
        for ema_param, model_param in zip(self.ema_params, self.model_params):
            ema_param.data.mul_(self.beta).add_(model_param.data, alpha=1 - self.beta)

    def state_dict(self):
        """返回 EMA 的状态字典"""
        return {
            'step': self.step,
            'beta': self.beta,
            'update_every': self.update_every,
            'ema_model': self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """加载 EMA 的状态字典"""
        self.step = state_dict['step']
        self.beta = state_dict['beta']
        self.update_every = state_dict['update_every']
        self.ema_model.load_state_dict(state_dict['ema_model'])


class EMAModel:
    """
    Exponential Moving Average (EMA) for model weights.

    EMA helps stabilize training by maintaining a moving average of model parameters.
    The EMA model is typically used for evaluation and inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.995,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 0.75,
    ):
        """
        Args:
            model: The model to apply EMA to
            decay: EMA decay rate (higher means more averaging, slower to adapt)
            min_decay: Minimum decay rate
            update_after_step: Number of steps to wait before starting EMA updates
            use_ema_warmup: Whether to use warmup for EMA decay
            inv_gamma: Inverse gamma for EMA warmup
            power: Power for EMA warmup
        """
        self.model = model
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.current_step = 0

        # Create shadow model for EMA
        self.shadow = copy.deepcopy(model)
        for param in self.shadow.parameters():
            param.requires_grad_(False)

        # Store shadow parameters
        self.shadow_params = list(self.shadow.parameters())
        self.model_params = list(model.parameters())

        # Initialize with model parameters
        self.copy_to()

    def get_decay(self) -> float:
        """
        Compute the current decay rate, possibly with warmup.
        """
        if self.current_step < self.update_after_step:
            return 0.0

        if not self.use_ema_warmup:
            return self.decay

        # EMA warmup
        step = self.current_step - self.update_after_step
        decay = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        decay = min(decay, self.decay)
        decay = max(decay, self.min_decay)
        return decay

    @torch.no_grad()
    def step(self, current_step: Optional[int] = None):
        """
        Update the EMA parameters.

        Args:
            current_step: Current training step. If None, use internal counter.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        decay = self.get_decay()

        # Don't update if decay is 0
        if decay == 0.0:
            return

        # Update shadow parameters
        for shadow_param, model_param in zip(self.shadow_params, self.model_params):
            if model_param.requires_grad:
                # EMA update: shadow = decay * shadow + (1 - decay) * model
                shadow_param.sub_((1 - decay) * (shadow_param - model_param))

    def copy_to(self):
        """
        Copy current model parameters to shadow parameters.
        """
        for shadow_param, model_param in zip(self.shadow_params, self.model_params):
            shadow_param.data.copy_(model_param.data)

    def store(self, model_parameters: list):
        """
        Store model parameters to restore later.
        """
        self.temp_stored_params = [param.clone() for param in model_parameters]

    def restore(self, model_parameters: list):
        """
        Restore stored model parameters.
        """
        for param, stored_param in zip(model_parameters, self.temp_stored_params):
            param.data.copy_(stored_param.data)

    def apply_shadow(self) -> torch.nn.Module:
        """
        Apply the shadow parameters to the model and return the model.
        Useful for evaluation.
        """
        self.store(self.model_params)
        for model_param, shadow_param in zip(self.model_params, self.shadow_params):
            model_param.data.copy_(shadow_param.data)
        return self.model

    def restore_model(self):
        """
        Restore the original model parameters.
        Call this after evaluation when using apply_shadow.
        """
        if hasattr(self, 'temp_stored_params'):
            self.restore(self.model_params)

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'shadow': self.shadow.state_dict(),
            'current_step': self.current_step,
            'decay': self.decay,
            'min_decay': self.min_decay,
            'update_after_step': self.update_after_step,
            'use_ema_warmup': self.use_ema_warmup,
            'inv_gamma': self.inv_gamma,
            'power': self.power,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow.load_state_dict(state_dict['shadow'])
        self.current_step = state_dict.get('current_step', 0)
        self.decay = state_dict.get('decay', self.decay)
        self.min_decay = state_dict.get('min_decay', self.min_decay)
        self.update_after_step = state_dict.get('update_after_step', self.update_after_step)
        self.use_ema_warmup = state_dict.get('use_ema_warmup', self.use_ema_warmup)
        self.inv_gamma = state_dict.get('inv_gamma', self.inv_gamma)
        self.power = state_dict.get('power', self.power)

    def __call__(self, *args, **kwargs):
        """
        Forward pass through the EMA model.
        """
        return self.shadow(*args, **kwargs)
