import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from contextlib import nullcontext
import torch.nn.functional as F

from .latent_diffusion import LatentDiffusionModel, extract
from .ema import EMAModel
from .amp_utils import MixedPrecisionTrainer
from .losses import WeightedPETLoss, GradientLoss, QuantitativeConstraintLoss
from ..utils import (
    DiffusionMetrics,
    pet_to_zero_one,
    map_pet_to_output_domain,
    prepare_pet_eval_tensors,
)


class DiffusionTrainerV2:
    """
    Enhanced Diffusion Trainer with EMA and Mixed Precision support.
    """
    def __init__(
        self,
        model: LatentDiffusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.config.setdefault('experimental', {})
        self.config['experimental'].setdefault('enable_controlnet', False)
        self.config['experimental'].setdefault('enable_continuous_time', False)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get('learning_rate', 1e-4)),
            weight_decay=float(self.config.get('weight_decay', 0.01)),
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.config.get('num_epochs', 100)),
            eta_min=float(self.config.get('lr_min', 1e-6)),
        )

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # EMA (Exponential Moving Average)
        if self.config.get('use_ema', True):
            self.ema = EMAModel(
                model=self.model,
                decay=self.config.get('ema_decay', 0.995),
                min_decay=self.config.get('ema_min_decay', 0.0),
                update_after_step=self.config.get('ema_update_after_step', 0),
                use_ema_warmup=self.config.get('ema_use_warmup', False),
            )
            self.ema_update_every = max(1, int(self.config.get('ema_update_every', 1)))
            print(
                f"EMA enabled with decay={self.config.get('ema_decay', 0.995)}, "
                f"update_every={self.ema_update_every}"
            )
        else:
            self.ema = None
            self.ema_update_every = 1
            print("EMA disabled")

        # Mixed Precision Training
        self.mixed_precision = MixedPrecisionTrainer(
            enabled=self.config.get('use_amp', True),
            init_scale=self.config.get('amp_init_scale', 2.**16),
            growth_factor=self.config.get('amp_growth_factor', 2.0),
            backoff_factor=self.config.get('amp_backoff_factor', 0.5),
            growth_interval=self.config.get('amp_growth_interval', 2000),
        )

        # Metrics
        self.metrics = DiffusionMetrics(device=self.device)

        # Sampling settings for eval / visualization
        self.eval_num_inference_steps = int(self.config.get('num_inference_steps_eval', 1000))
        self.sample_num_inference_steps = int(
            self.config.get('num_inference_steps_sample', self.eval_num_inference_steps)
        )
        self.eval_mc_samples = max(1, int(self.config.get('num_mc_samples_eval', 1)))
        self.invert_pet = bool(self.config.get('invert_pet', False))
        self.invert_pet_on_output = bool(self.config.get('invert_pet_on_output', True))
        if self.invert_pet:
            print(
                f"PET inversion enabled (data domain). "
                f"output_inverse={self.invert_pet_on_output}"
            )
        if self.eval_mc_samples > 1:
            print(f"MC sampling for eval enabled: samples={self.eval_mc_samples}")

        self.enable_heteroscedastic = bool(getattr(self.model, 'enable_heteroscedastic', False))
        self.use_heteroscedastic_nll = bool(
            self.config.get('use_heteroscedastic_nll', self.enable_heteroscedastic)
        )
        self.heteroscedastic_nll_weight = float(self.config.get('heteroscedastic_nll_weight', 1.0))
        self.heteroscedastic_mse_weight = float(self.config.get('heteroscedastic_mse_weight', 0.0))
        if self.use_heteroscedastic_nll and not self.enable_heteroscedastic:
            print(
                "Heteroscedastic NLL requested but model has no heteroscedastic head. "
                "Falling back to standard MSE diffusion loss."
            )
            self.use_heteroscedastic_nll = False
        if self.enable_heteroscedastic:
            print(
                "Heteroscedastic head enabled: "
                f"use_nll={self.use_heteroscedastic_nll}, "
                f"nll_weight={self.heteroscedastic_nll_weight}, "
                f"mse_weight={self.heteroscedastic_mse_weight}"
            )

        experimental_cfg = self.config.get('experimental', {})
        if experimental_cfg.get('enable_controlnet', False) or experimental_cfg.get('enable_continuous_time', False):
            print(
                "Experimental flags detected in config "
                f"(controlnet={experimental_cfg.get('enable_controlnet', False)}, "
                f"continuous_time={experimental_cfg.get('enable_continuous_time', False)}). "
                "DiffusionTrainerV2 runs the default latent diffusion path only."
            )

        # Optional x0 auxiliary losses to sharpen internal structures.
        self.aux_x0_l1_weight = float(self.config.get('aux_x0_l1_weight', 0.0))
        self.aux_x0_grad_weight = float(self.config.get('aux_x0_grad_weight', 0.0))
        self.aux_x0_pet_focus_weight = float(self.config.get('aux_x0_pet_focus_weight', 0.0))
        self.aux_x0_pet_threshold = float(self.config.get('aux_x0_pet_threshold', 0.15))
        self.aux_x0_bg_suppress_weight = float(self.config.get('aux_x0_bg_suppress_weight', 0.0))
        self.tumor_weight_min = float(self.config.get('tumor_weight_min', 10.0))
        self.tumor_weight_max = float(self.config.get('tumor_weight_max', 50.0))
        if self.tumor_weight_min > self.tumor_weight_max:
            self.tumor_weight_min, self.tumor_weight_max = self.tumor_weight_max, self.tumor_weight_min

        self.enforce_tumor_weight_range = bool(self.config.get('enforce_tumor_weight_range', False))
        raw_tumor_weight = float(
            self.config.get(
                'tumor_region_weight',
                self.config.get('aux_x0_pet_high_weight', self.tumor_weight_min)
            )
        )
        if self.enforce_tumor_weight_range:
            clamped = min(max(raw_tumor_weight, self.tumor_weight_min), self.tumor_weight_max)
            if clamped != raw_tumor_weight:
                print(
                    f"Tumor region weight {raw_tumor_weight:.2f} out of range "
                    f"[{self.tumor_weight_min:.1f}, {self.tumor_weight_max:.1f}], clamped to {clamped:.2f}"
                )
            raw_tumor_weight = clamped
        self.aux_x0_pet_high_weight = raw_tumor_weight
        # Keep both keys aligned for backward compatibility.
        self.config['aux_x0_pet_high_weight'] = self.aux_x0_pet_high_weight
        self.config['tumor_region_weight'] = self.aux_x0_pet_high_weight

        # Structure-aware reweighting to reduce background dominance in training.
        self.use_structure_weighted_noise = bool(self.config.get('use_structure_weighted_noise', True))
        self.use_structure_weighted_x0_l1 = bool(self.config.get('use_structure_weighted_x0_l1', True))
        self.structure_focus_threshold = float(
            self.config.get('structure_focus_threshold', self.aux_x0_pet_threshold)
        )
        self.structure_focus_weight = float(
            self.config.get('structure_focus_weight', self.aux_x0_pet_high_weight)
        )
        self.structure_background_weight = float(
            self.config.get('structure_background_weight', 0.2)
        )
        self.structure_weight_warmup_epochs = max(
            0,
            int(self.config.get('structure_weight_warmup_epochs', 0)),
        )
        self.background_target_value = float(
            self.config.get('background_target_value', 0.0 if self.invert_pet else 1.0)
        )

        self.structure_focus_threshold = min(max(self.structure_focus_threshold, 0.0), 1.0)
        self.structure_focus_weight = max(1.0, self.structure_focus_weight)
        self.structure_background_weight = max(0.0, self.structure_background_weight)
        self.background_target_value = min(max(self.background_target_value, 0.0), 1.0)

        self.pet_focus_loss_fn = None
        self.grad_loss_fn = None
        self.quantitative_loss_fn = None
        if self.aux_x0_pet_focus_weight > 0:
            self.pet_focus_loss_fn = WeightedPETLoss(
                threshold=self.aux_x0_pet_threshold,
                high_weight=self.aux_x0_pet_high_weight,
                low_weight=1.0,
                loss_type='l1',
            ).to(self.device)
        if self.aux_x0_grad_weight > 0:
            self.grad_loss_fn = GradientLoss().to(self.device)

        self.enable_quantitative_constraints = bool(
            self.config.get('enable_quantitative_constraints', False)
        )
        self.quantitative_constraint_weight = float(
            self.config.get('quantitative_constraint_weight', 0.0)
        )
        if self.enable_quantitative_constraints and self.quantitative_constraint_weight > 0:
            self.quantitative_loss_fn = QuantitativeConstraintLoss(
                hotspot_threshold=float(
                    self.config.get('quant_hotspot_threshold', self.structure_focus_threshold)
                ),
                topk_percent=float(self.config.get('quant_topk_percent', 0.01)),
                mean_weight=float(self.config.get('quant_mean_weight', 1.0)),
                integral_weight=float(self.config.get('quant_integral_weight', 1.0)),
                hotspot_weight=float(self.config.get('quant_hotspot_weight', 1.0)),
                peak_weight=float(self.config.get('quant_peak_weight', 1.0)),
            ).to(self.device)
            print(
                "Quantitative constraints enabled: "
                f"weight={self.quantitative_constraint_weight}, "
                f"hotspot_th={self.config.get('quant_hotspot_threshold', self.structure_focus_threshold)}, "
                f"topk={self.config.get('quant_topk_percent', 0.01)}"
            )

        if (
            self.aux_x0_l1_weight > 0
            or self.aux_x0_grad_weight > 0
            or self.aux_x0_pet_focus_weight > 0
            or self.aux_x0_bg_suppress_weight > 0
        ):
            print(
                "x0 auxiliary losses enabled: "
                f"l1={self.aux_x0_l1_weight}, "
                f"grad={self.aux_x0_grad_weight}, "
                f"pet_focus={self.aux_x0_pet_focus_weight}, "
                f"bg_suppress={self.aux_x0_bg_suppress_weight}, "
                f"tumor_region_weight={self.aux_x0_pet_high_weight}"
            )

        if self.use_structure_weighted_noise or self.use_structure_weighted_x0_l1:
            print(
                "Structure-aware weighting enabled: "
                f"threshold={self.structure_focus_threshold}, "
                f"fg_weight={self.structure_focus_weight}, "
                f"bg_weight={self.structure_background_weight}, "
                f"warmup_epochs={self.structure_weight_warmup_epochs}, "
                f"noise={self.use_structure_weighted_noise}, "
                f"x0_l1={self.use_structure_weighted_x0_l1}"
            )

        # Initialize wandb
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('project_name', 'pet-ct-diffusion'),
                config=self.config,
            )

        # Training state
        self.global_step = 0
        self.start_epoch = 1
        self.current_epoch = 1
        self.best_val_loss = float('inf')

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _count_trigger_epochs(start_epoch: int, end_epoch: int, interval: int) -> int:
        if interval <= 0 or start_epoch > end_epoch:
            return 0

        first = ((start_epoch + interval - 1) // interval) * interval
        if first > end_epoch:
            return 0
        return ((end_epoch - first) // interval) + 1

    def _split_batch(self, batch):
        """Split batch into (target_pet, condition_ct)."""
        if isinstance(batch, dict):
            pet = batch['pet'].to(self.device)
            ct = batch['ct'].to(self.device)
        else:
            pet, ct = batch.chunk(2, dim=1)
            pet = pet.to(self.device)
            ct = ct.to(self.device)
        return pet, ct

    def _predict_x0_for_aux(
        self,
        model: LatentDiffusionModel,
        target: torch.Tensor,
        model_pred: torch.Tensor,
        loss_target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if model.objective == 'pred_noise':
            noisy_x = model.add_noise(target, loss_target, timesteps)
            return model.predict_start_from_noise(noisy_x, timesteps, model_pred)

        if model.objective == 'pred_x0':
            return model_pred

        if model.objective == 'pred_v':
            alpha_t = extract(model.sqrt_alphas_cumprod, timesteps, target.shape)
            sigma_t = extract(model.sqrt_one_minus_alphas_cumprod, timesteps, target.shape)
            noisy_x = (target + sigma_t * loss_target) / torch.clamp(alpha_t, min=1e-8)
            return model.predict_start_from_v(noisy_x, timesteps, model_pred)

        return None

    def _current_focus_scale(self) -> float:
        if self.structure_weight_warmup_epochs <= 0:
            return 1.0
        return min(1.0, max(0.0, self.current_epoch / float(self.structure_weight_warmup_epochs)))

    def _get_structure_maps(self, target_01: torch.Tensor):
        fg_mask = (target_01 >= self.structure_focus_threshold).float()
        bg_mask = 1.0 - fg_mask

        focus_scale = self._current_focus_scale()
        fg_weight = 1.0 + (self.structure_focus_weight - 1.0) * focus_scale
        bg_weight = 1.0 + (self.structure_background_weight - 1.0) * focus_scale
        weight_map = fg_mask * fg_weight + bg_mask * bg_weight
        return fg_mask, bg_mask, weight_map

    @staticmethod
    def _masked_mean(loss_map: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(1, loss_map.ndim))
        numerator = (loss_map * weight_map).sum(dim=reduce_dims)
        denominator = weight_map.sum(dim=reduce_dims).clamp(min=1e-6)
        return numerator / denominator

    def _compute_weighted_loss(self, model, target, condition):
        """Compute total loss (SNR-weighted noise loss + optional x0 auxiliary losses)."""
        model_out = model(target, condition)
        if isinstance(model_out, tuple) and len(model_out) == 4:
            model_pred, loss_target, timesteps, pred_logvar = model_out
        else:
            model_pred, loss_target, timesteps = model_out
            pred_logvar = None

        target_01 = self._pet_to_zero_one(target)
        fg_mask, bg_mask, structure_weight_map = self._get_structure_maps(target_01)

        mse_map = F.mse_loss(model_pred, loss_target, reduction='none')
        if self.use_structure_weighted_noise:
            mse_per_sample = self._masked_mean(mse_map, structure_weight_map)
        else:
            mse_per_sample = mse_map.mean(dim=list(range(1, len(mse_map.shape))))

        # Keep loss_weight as shape [B] to avoid unintended cross-batch broadcasting.
        loss_weight = extract(model.loss_weight, timesteps, mse_per_sample.shape)
        weighted_mse_loss = (mse_per_sample * loss_weight).mean()

        diffusion_loss = weighted_mse_loss
        losses = {
            'mse': mse_per_sample.mean(),
            'snr_weighted_mse': weighted_mse_loss,
        }

        if self.use_heteroscedastic_nll and pred_logvar is not None:
            inv_var = torch.exp(-pred_logvar)
            nll_map = 0.5 * (inv_var * mse_map + pred_logvar)
            if self.use_structure_weighted_noise:
                nll_per_sample = self._masked_mean(nll_map, structure_weight_map)
            else:
                nll_per_sample = nll_map.mean(dim=list(range(1, len(nll_map.shape))))
            weighted_nll_loss = (nll_per_sample * loss_weight).mean()
            diffusion_loss = (
                self.heteroscedastic_nll_weight * weighted_nll_loss
                + self.heteroscedastic_mse_weight * weighted_mse_loss
            )
            losses['nll'] = nll_per_sample.mean()
            losses['snr_weighted_nll'] = weighted_nll_loss
            losses['pred_logvar_mean'] = pred_logvar.mean()
            losses['pred_logvar_std'] = pred_logvar.std(unbiased=False)

        losses['snr_weighted'] = diffusion_loss
        total_loss = diffusion_loss

        use_aux = (
            self.aux_x0_l1_weight > 0
            or self.aux_x0_grad_weight > 0
            or self.aux_x0_pet_focus_weight > 0
            or self.aux_x0_bg_suppress_weight > 0
            or (self.quantitative_loss_fn is not None and self.quantitative_constraint_weight > 0)
        )

        if use_aux:
            pred_x0 = self._predict_x0_for_aux(model, target, model_pred, loss_target, timesteps)
            if pred_x0 is not None:
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                pred_01 = self._pet_to_zero_one(pred_x0)

                if self.aux_x0_l1_weight > 0:
                    x0_l1_map = torch.abs(pred_01 - target_01)
                    if self.use_structure_weighted_x0_l1:
                        x0_l1 = self._masked_mean(x0_l1_map, structure_weight_map).mean()
                    else:
                        x0_l1 = x0_l1_map.mean()
                    losses['x0_l1'] = x0_l1
                    total_loss = total_loss + self.aux_x0_l1_weight * x0_l1

                if self.aux_x0_grad_weight > 0 and self.grad_loss_fn is not None:
                    x0_grad = self.grad_loss_fn(pred_01, target_01)
                    losses['x0_grad'] = x0_grad
                    total_loss = total_loss + self.aux_x0_grad_weight * x0_grad

                if self.aux_x0_pet_focus_weight > 0 and self.pet_focus_loss_fn is not None:
                    x0_pet_focus = self.pet_focus_loss_fn(pred_01, target_01)
                    losses['x0_pet_focus'] = x0_pet_focus
                    total_loss = total_loss + self.aux_x0_pet_focus_weight * x0_pet_focus

                if self.aux_x0_bg_suppress_weight > 0:
                    bg_target = torch.full_like(pred_01, self.background_target_value)
                    bg_error_map = torch.abs(pred_01 - bg_target)
                    x0_bg_suppress = self._masked_mean(bg_error_map, bg_mask).mean()
                    losses['x0_bg_suppress'] = x0_bg_suppress
                    total_loss = total_loss + self.aux_x0_bg_suppress_weight * x0_bg_suppress

                if self.quantitative_loss_fn is not None and self.quantitative_constraint_weight > 0:
                    x0_quant, quant_items = self.quantitative_loss_fn(
                        pred_01,
                        target_01,
                        mask=fg_mask,
                    )
                    losses['x0_quant'] = x0_quant
                    for key, val in quant_items.items():
                        losses[f'x0_quant_{key}'] = val
                    total_loss = total_loss + self.quantitative_constraint_weight * x0_quant

        losses['total'] = total_loss
        return losses

    def _pet_to_zero_one(self, pet_tensor: torch.Tensor) -> torch.Tensor:
        return pet_to_zero_one(pet_tensor)

    def _map_pet_to_output_domain(self, pet_tensor_01: torch.Tensor) -> torch.Tensor:
        return map_pet_to_output_domain(
            pet_tensor_01,
            invert_pet=self.invert_pet,
            invert_pet_on_output=self.invert_pet_on_output,
        )

    def _autocast_context(self):
        if self.mixed_precision.enabled:
            return torch.amp.autocast('cuda', enabled=True)
        return nullcontext()

    def _maybe_update_ema(self):
        if self.ema and (self.global_step % self.ema_update_every == 0):
            self.ema.step(current_step=self.global_step)

    def compute_loss(self, batch):
        target, condition = self._split_batch(batch)
        losses = self._compute_weighted_loss(self.model, target, condition)
        return losses['total']

    def compute_loss_with_details(self, batch):
        """
        计算损失并返回各项损失的详细信息（用于记录和调试）
        """
        target, condition = self._split_batch(batch)
        return self._compute_weighted_loss(self.model, target, condition)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        # 用于记录各项损失的累计值（按出现的 key 动态聚合）
        loss_components = {}

        # 【新增】获取梯度累积配置
        gradient_accumulate_every = self.config.get('gradient_accumulate_every', 1)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}',
                   dynamic_ncols=False, miniters=1, unit='batch')
        total_batches = len(self.train_loader)

        # 【新增】在epoch开始时清零梯度（而不是每个batch都清零）
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # 获取损失
            with self._autocast_context():
                losses = self.compute_loss_with_details(batch)
            loss = losses['total']

            # For the last incomplete accumulation group, use its real size.
            group_start = (batch_idx // gradient_accumulate_every) * gradient_accumulate_every
            group_end = min(group_start + gradient_accumulate_every, total_batches)
            effective_accumulate = group_end - group_start
            loss = loss / effective_accumulate

            # Scale loss for AMP
            if self.mixed_precision.enabled:
                loss = self.mixed_precision.scale_loss(loss)

            # Backward pass（累积梯度）
            loss.backward()

            # 累积损失用于日志（使用原始loss，不是除以accumulation后的）
            total_loss += losses['total'].item()
            for key, value in losses.items():
                loss_components.setdefault(key, 0.0)
                loss_components[key] += value.item() if hasattr(value, 'item') else float(value)

            # Step at the end of each accumulation group.
            if (batch_idx + 1) == group_end:
                # Gradient clipping
                if self.config.get('grad_clip_norm', 0) > 0:
                    self.mixed_precision.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('grad_clip_norm', 1.0),
                        self.optimizer
                    )

                # Optimizer step
                self.mixed_precision.optimizer_step(self.optimizer)

                # 清零梯度，准备下一轮累积
                self.optimizer.zero_grad()

                # Update EMA
                self.global_step += 1
                self._maybe_update_ema()

            # 更新进度条显示
            postfix = {
                'loss': losses['total'].item(),
                'step': self.global_step
            }
            if batch_idx == 0:  # 第一个批次显示各项损失
                postfix.update({f'{k}_loss': v.item() if hasattr(v, 'item') else v
                               for k, v in losses.items() if k != 'total'})
            pbar.set_postfix(postfix)

        # 计算平均损失（基于实际batch数）
        num_batches = len(pbar)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            if num_batches > 0:
                loss_components[key] /= num_batches

        # 将损失组件作为属性返回，供日志记录使用
        self.last_epoch_losses = loss_components
        return avg_loss

    @torch.no_grad()
    def validate(self, use_ema=False):
        if use_ema and self.ema:
            # Use EMA model for validation
            model = self.ema.apply_shadow()
        else:
            model = self.model

        model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            target, condition = self._split_batch(batch)
            with self._autocast_context():
                losses = self._compute_weighted_loss(model, target, condition)

            total_loss += losses['total'].item()
            num_batches += 1

        # Restore original model if we used EMA
        if use_ema and self.ema:
            self.ema.restore_model()

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def evaluate(self, num_samples: Optional[int] = None, use_ema=True):
        # Use EMA model for evaluation by default
        if use_ema and self.ema:
            model = self.ema.shadow
        else:
            model = self.model

        model.eval()
        all_metrics = []
        num_processed = 0

        for batch in tqdm(self.val_loader, desc='Evaluating'):
            if isinstance(batch, dict):
                pet = batch['pet'].to(self.device)
                ct = batch['ct'].to(self.device)
            else:
                pet, ct = batch.chunk(2, dim=1)
                pet = pet.to(self.device)
                ct = ct.to(self.device)

            batch_uncertainty_mean = None
            if self.eval_mc_samples > 1:
                sampled = model.sample_multiple(
                    condition=ct,
                    num_inference_steps=self.eval_num_inference_steps,
                    num_samples=self.eval_mc_samples,
                    return_stats=True,
                )
                generated = sampled['mean']
                batch_uncertainty_mean = sampled['std'].mean().item()
            else:
                generated = model.sample(
                    condition=ct,
                    num_inference_steps=self.eval_num_inference_steps
                )
            generated_eval, pet_eval = prepare_pet_eval_tensors(
                pred_pet_01=generated,
                target_pet_raw=pet,
                invert_pet=self.invert_pet,
                invert_pet_on_output=self.invert_pet_on_output,
            )

            # Compute metrics（现在两者都在 [0, 1] 范围）
            batch_metrics = self.metrics.evaluate_batch(generated_eval, pet_eval)
            if batch_uncertainty_mean is not None:
                batch_metrics['uncertainty_mean'] = float(batch_uncertainty_mean)
            all_metrics.append(batch_metrics)

            num_processed += pet.shape[0]
            if num_samples and num_processed >= num_samples:
                break

        # Compute average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        return avg_metrics

    def sample_images(self, num_samples: int = 4, use_ema=True):
        # Use EMA model for sampling by default
        if use_ema and self.ema:
            model = self.ema.shadow
        else:
            model = self.model

        model.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))

        if isinstance(batch, dict):
            pet = batch['pet'][:num_samples].to(self.device)
            ct = batch['ct'][:num_samples].to(self.device)
        else:
            pet, ct = batch.chunk(2, dim=1)
            pet = pet[:num_samples].to(self.device)
            ct = ct[:num_samples].to(self.device)

        with torch.no_grad():
            uncertainty = None
            if self.eval_mc_samples > 1:
                sampled = model.sample_multiple(
                    condition=ct,
                    num_inference_steps=self.sample_num_inference_steps,
                    num_samples=self.eval_mc_samples,
                    return_stats=True,
                )
                generated = sampled['mean']
                uncertainty = sampled['std']
            else:
                generated = model.sample(
                    condition=ct,
                    num_inference_steps=self.sample_num_inference_steps
                )

        out = {
            'pet': pet.cpu(),  # [-1, 1]
            'ct': ct.cpu(),    # [-1, 1]
            'generated': generated.cpu(),  # [0, 1]
        }
        if uncertainty is not None:
            out['uncertainty'] = uncertainty.cpu()
        return out

    def train(self):
        num_epochs = self.config.get('num_epochs', 100)
        save_interval = self.config.get('save_interval', 10)
        eval_interval = self.config.get('eval_interval', save_interval)
        eval_num_samples = self.config.get('eval_num_samples', 50)

        run_start_time = time.perf_counter()
        total_epoch_count = max(0, num_epochs - self.start_epoch + 1)
        total_eval_events = self._count_trigger_epochs(self.start_epoch, num_epochs, eval_interval)
        total_save_events = self._count_trigger_epochs(self.start_epoch, num_epochs, save_interval)

        timing_stats = {
            'train_val_seconds': 0.0,
            'eval_seconds': 0.0,
            'save_seconds': 0.0,
            'epochs_done': 0,
            'eval_done': 0,
            'save_done': 0,
        }

        print(
            "Timing plan: "
            f"epochs={total_epoch_count}, eval_events={total_eval_events}, save_events={total_save_events}"
        )

        for epoch in range(self.start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.perf_counter()

            # Training
            train_start_time = time.perf_counter()
            train_loss = self.train_epoch(epoch)
            train_seconds = time.perf_counter() - train_start_time

            # Validation (use regular model during training for fair comparison)
            val_start_time = time.perf_counter()
            val_loss = self.validate(use_ema=False)
            val_seconds = time.perf_counter() - val_start_time
            train_val_seconds = train_seconds + val_seconds

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Logging
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
            }

            # 添加各项损失组件到日志
            if hasattr(self, 'last_epoch_losses'):
                for k, v in self.last_epoch_losses.items():
                    if v > 0:  # 只记录非零值
                        log_dict[f'train_{k}_loss'] = v

            # Evaluation
            eval_seconds = 0.0
            do_eval = (eval_interval > 0) and (epoch % eval_interval == 0)
            if do_eval:
                eval_start_time = time.perf_counter()
                eval_metrics = self.evaluate(num_samples=eval_num_samples, use_ema=True)
                eval_seconds = time.perf_counter() - eval_start_time
                log_dict.update({f'eval_{k}': v for k, v in eval_metrics.items()})
                print("\nEvaluation Metrics:")
                for k, v in eval_metrics.items():
                    print(f"  {k.upper()}: {v:.4f}")

            if self.config.get('use_wandb', False):
                wandb.log(log_dict)

            # 打印训练信息（包含各项损失）
            print_str = f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.2e}"
            if hasattr(self, 'last_epoch_losses') and self.last_epoch_losses.get('mse', 0) > 0:
                print_str += "\n  Loss Components: "
                components = [f"{k}={v:.4f}" for k, v in self.last_epoch_losses.items()
                            if v > 0 and k != 'total']
                print_str += ", ".join(components)
            print_str += f" (EMA: {'ON' if self.ema else 'OFF'}, AMP: {'ON' if self.mixed_precision.enabled else 'OFF'})"
            print(print_str)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f'best_model.pth', epoch)
                # Also save EMA version separately
                if self.ema:
                    self.save_checkpoint(f'best_model_ema.pth', epoch, is_ema=True)

            # Periodic saving
            save_seconds = 0.0
            do_save = (save_interval > 0) and (epoch % save_interval == 0)
            if do_save:
                save_start_time = time.perf_counter()
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch)
                if self.ema:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}_ema.pth', epoch, is_ema=True)

                # 强制保存采样图片到本地文件夹
                import torchvision.utils as vutils
                os.makedirs('samples', exist_ok=True)

                # 采样
                samples = self.sample_images(use_ema=True)

                # 拼接图片：左边是CT(条件)，中间是PET(真值)，右边是Generated(生成)
                # CT 和 PET 是 [-1, 1]，需要转换到 [0, 1]
                # generated 已经是 [0, 1]，不需要再转换
                ct_show = torch.clamp((samples['ct'] + 1) / 2, 0, 1)
                pet_show = self._pet_to_zero_one(samples['pet'])
                gen_show = torch.clamp(samples['generated'], 0.0, 1.0)
                pet_show = self._map_pet_to_output_domain(pet_show)
                gen_show = self._map_pet_to_output_domain(gen_show)

                # 拼接成一个网格
                grid = torch.cat([ct_show, pet_show, gen_show], dim=3) # 在宽度方向拼接
                vutils.save_image(grid, f'samples/epoch_{epoch}.png', normalize=False)
                print(f"Saved sample images to samples/epoch_{epoch}.png")

                # Sample and log images
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'sample_images': [
                            wandb.Image(img, caption=f'Gen {i}')
                            for i, img in enumerate(gen_show)
                        ]
                    })
                save_seconds = time.perf_counter() - save_start_time

            epoch_seconds = time.perf_counter() - epoch_start_time

            timing_stats['train_val_seconds'] += train_val_seconds
            timing_stats['epochs_done'] += 1
            if do_eval:
                timing_stats['eval_seconds'] += eval_seconds
                timing_stats['eval_done'] += 1
            if do_save:
                timing_stats['save_seconds'] += save_seconds
                timing_stats['save_done'] += 1

            elapsed_seconds = time.perf_counter() - run_start_time
            avg_train_val = timing_stats['train_val_seconds'] / max(1, timing_stats['epochs_done'])
            avg_eval = (
                timing_stats['eval_seconds'] / timing_stats['eval_done']
                if timing_stats['eval_done'] > 0 else 0.0
            )
            avg_save = (
                timing_stats['save_seconds'] / timing_stats['save_done']
                if timing_stats['save_done'] > 0 else 0.0
            )

            remaining_epochs = max(0, num_epochs - epoch)
            remaining_eval_events = self._count_trigger_epochs(epoch + 1, num_epochs, eval_interval)
            remaining_save_events = self._count_trigger_epochs(epoch + 1, num_epochs, save_interval)

            remaining_seconds = (
                avg_train_val * remaining_epochs
                + avg_eval * remaining_eval_events
                + avg_save * remaining_save_events
            )
            estimated_total_seconds = elapsed_seconds + remaining_seconds
            eta_datetime = datetime.now() + timedelta(seconds=remaining_seconds)

            print(
                "Timing: "
                f"train={train_seconds:.1f}s, val={val_seconds:.1f}s, "
                f"eval={eval_seconds:.1f}s, save={save_seconds:.1f}s, epoch={epoch_seconds:.1f}s"
            )
            print(
                "Timing Summary: "
                f"elapsed={self._format_duration(elapsed_seconds)}, "
                f"remaining={self._format_duration(remaining_seconds)}, "
                f"estimated_total={self._format_duration(estimated_total_seconds)}, "
                f"ETA={eta_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if self.config.get('use_wandb', False):
                wandb.log({
                    'time/train_seconds': train_seconds,
                    'time/val_seconds': val_seconds,
                    'time/eval_seconds': eval_seconds,
                    'time/save_seconds': save_seconds,
                    'time/epoch_seconds': epoch_seconds,
                    'time/elapsed_seconds': elapsed_seconds,
                    'time/remaining_seconds_est': remaining_seconds,
                    'time/estimated_total_seconds': estimated_total_seconds,
                    'time/eval_num_samples': eval_num_samples,
                })

        total_elapsed_seconds = time.perf_counter() - run_start_time
        print(
            "Final Timing: "
            f"total={self._format_duration(total_elapsed_seconds)}, "
            f"train_val={self._format_duration(timing_stats['train_val_seconds'])}, "
            f"eval={self._format_duration(timing_stats['eval_seconds'])}, "
            f"save={self._format_duration(timing_stats['save_seconds'])}"
        )

        print("\nTraining completed!")
        return self.ema.shadow if self.ema else self.model

    def save_checkpoint(self, filename: str, epoch: int = None, is_ema=False):
        os.makedirs('checkpoints', exist_ok=True)

        # Save EMA model if requested
        if is_ema and self.ema:
            model_state_dict = self.ema.shadow.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'is_ema': is_ema,
        }

        if self.ema is not None:
            checkpoint['ema_state'] = self.ema.state_dict()
            checkpoint['ema_shadow_state_dict'] = self.ema.shadow.state_dict()

        # Save mixed precision scaler state
        if self.mixed_precision.enabled:
            checkpoint['amp_scaler'] = self.mixed_precision.state_dict()

        # Save epoch information
        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, os.path.join('checkpoints', filename))

    def _resolve_checkpoint_path(self, filename: str) -> str:
        """
        Resolve checkpoint path robustly.

        Supports:
        - absolute paths
        - relative paths already including 'checkpoints/'
        - bare checkpoint filenames saved under 'checkpoints/'
        """
        candidate_paths = []

        if os.path.isabs(filename):
            candidate_paths.append(filename)
        else:
            candidate_paths.append(filename)
            candidate_paths.append(os.path.join('checkpoints', filename))

        for path in candidate_paths:
            if os.path.exists(path):
                return path

        searched = ", ".join(candidate_paths)
        raise FileNotFoundError(
            f"Checkpoint not found. Tried: {searched}"
        )

    def _validate_checkpoint_compatibility(self, checkpoint: Dict[str, Any], checkpoint_path: str):
        """
        Ensure critical model settings match before loading state dicts.
        """
        checkpoint_config = checkpoint.get('config', None)
        if not isinstance(checkpoint_config, dict):
            return

        expected = {
            'unet_type': getattr(self.model, 'unet_type', None),
            'model_variant': getattr(self.model, 'model_variant', None),
            'objective': getattr(self.model, 'objective', None),
            'enable_heteroscedastic': getattr(self.model, 'enable_heteroscedastic', None),
        }
        found = {
            'unet_type': checkpoint_config.get('unet_type', None),
            'model_variant': checkpoint_config.get(
                'model_variant',
                checkpoint_config.get('pipeline_profile', None),
            ),
            'objective': checkpoint_config.get('objective', None),
            'enable_heteroscedastic': checkpoint_config.get('enable_heteroscedastic', None),
        }

        mismatches = []
        for key, expected_value in expected.items():
            found_value = found.get(key, None)
            if expected_value is None or found_value is None:
                continue
            if str(expected_value) != str(found_value):
                mismatches.append((key, found_value, expected_value))

        if not mismatches:
            return

        mismatch_text = "; ".join(
            f"{k}: checkpoint={v_ckpt}, current={v_current}"
            for k, v_ckpt, v_current in mismatches
        )
        raise ValueError(
            "Checkpoint/model compatibility check failed. "
            f"{mismatch_text}. "
            f"Checkpoint: {checkpoint_path}. "
            "Please use a matching config/checkpoint pair or migrate the checkpoint."
        )

    def load_checkpoint(self, filename: str):
        checkpoint_path = self._resolve_checkpoint_path(filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._validate_checkpoint_compatibility(checkpoint, checkpoint_path)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load EMA state if available
        if self.ema:
            if checkpoint.get('is_ema', False):
                self.ema.shadow.load_state_dict(checkpoint['model_state_dict'])
            elif 'ema_state' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state'])
            elif 'ema_shadow_state_dict' in checkpoint:
                self.ema.shadow.load_state_dict(checkpoint['ema_shadow_state_dict'])
            else:
                # Fallback: keep EMA shadow aligned with the loaded model
                self.ema.copy_to()

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load mixed precision scaler state
        if self.mixed_precision.enabled and 'amp_scaler' in checkpoint:
            self.mixed_precision.load_state_dict(checkpoint['amp_scaler'])

        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}, global_step {self.global_step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        return self.start_epoch
