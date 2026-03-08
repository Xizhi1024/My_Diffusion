import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
from typing import Optional, Dict, Any
import torch.nn.functional as F

from .latent_diffusion import LatentDiffusionModel, extract
from .ema import EMAModel
from .amp_utils import MixedPrecisionTrainer
from ..utils import DiffusionMetrics


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

        # Initialize wandb
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('project_name', 'pet-ct-diffusion'),
                config=self.config,
            )

        # Training state
        self.global_step = 0
        self.start_epoch = 1
        self.best_val_loss = float('inf')

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

    def _compute_weighted_loss(self, model, target, condition):
        """Compute SNR-weighted MSE used by training and validation."""
        model_pred, loss_target, timesteps = model(target, condition)
        loss_weight = extract(model.loss_weight, timesteps, model_pred.shape)
        mse_loss = F.mse_loss(model_pred, loss_target, reduction='none')
        mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape))))
        weighted_loss = (mse_loss * loss_weight).mean()
        return weighted_loss, mse_loss.mean()

    def _maybe_update_ema(self):
        if self.ema and (self.global_step % self.ema_update_every == 0):
            self.ema.step(current_step=self.global_step)

    def compute_loss(self, batch):
        target, condition = self._split_batch(batch)
        weighted_loss, _ = self._compute_weighted_loss(self.model, target, condition)
        return weighted_loss

    def compute_loss_with_details(self, batch):
        """
        计算损失并返回各项损失的详细信息（用于记录和调试）
        """
        target, condition = self._split_batch(batch)
        weighted_loss, mse_mean = self._compute_weighted_loss(self.model, target, condition)

        # 返回各项损失详情
        losses = {
            'total': weighted_loss,
            'mse': mse_mean,
            'snr_weighted': weighted_loss,
        }

        return losses

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        # 用于记录各项损失的累计值
        loss_components = {
            'mse': 0.0,
            'snr_weighted': 0.0,
            'total': 0.0
        }

        # 【新增】获取梯度累积配置
        gradient_accumulate_every = self.config.get('gradient_accumulate_every', 1)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}',
                   dynamic_ncols=False, miniters=1, unit='batch')
        total_batches = len(self.train_loader)

        # 【新增】在epoch开始时清零梯度（而不是每个batch都清零）
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # 获取损失
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
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()

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
            if num_batches > 0 and loss_components[key] > 0:
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

            if self.mixed_precision.enabled:
                with torch.amp.autocast('cuda'):
                    weighted_loss, _ = self._compute_weighted_loss(model, target, condition)
            else:
                weighted_loss, _ = self._compute_weighted_loss(model, target, condition)

            total_loss += weighted_loss.item()
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

            generated = model.sample(
                condition=ct,
                num_inference_steps=self.eval_num_inference_steps
            )

            # 【修复】将真实图像从 [-1, 1] 转换到 [0, 1]
            # model.sample() 返回的是 [0, 1] 范围的图像
            # 而 Dataset 输出的是 [-1, 1] 范围，需要转换才能正确比较
            pet_denorm = (pet + 1.0) * 0.5

            # Compute metrics（现在两者都在 [0, 1] 范围）
            batch_metrics = self.metrics.evaluate_batch(generated, pet_denorm)
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
            generated = model.sample(
                condition=ct,
                num_inference_steps=self.sample_num_inference_steps
            )

        return {
            'pet': pet.cpu(),  # [-1, 1]
            'ct': ct.cpu(),    # [-1, 1]
            'generated': generated.cpu(),  # [0, 1]
        }

    def train(self):
        num_epochs = self.config.get('num_epochs', 100)
        save_interval = self.config.get('save_interval', 10)
        eval_interval = self.config.get('eval_interval', save_interval)

        for epoch in range(self.start_epoch, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)

            # Validation (use regular model during training for fair comparison)
            val_loss = self.validate(use_ema=False)

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
            if epoch % eval_interval == 0:
                eval_metrics = self.evaluate(num_samples=50, use_ema=True)
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
            if epoch % save_interval == 0:
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
                pet_show = torch.clamp((samples['pet'] + 1) / 2, 0, 1)
                gen_show = samples['generated']  # 【修复】已经是 [0, 1]，不需要再转换

                # 拼接成一个网格
                grid = torch.cat([ct_show, pet_show, gen_show], dim=3) # 在宽度方向拼接
                vutils.save_image(grid, f'samples/epoch_{epoch}.png', normalize=False)
                print(f"Saved sample images to samples/epoch_{epoch}.png")

                # Sample and log images
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'sample_images': [
                            wandb.Image(img, caption=f'Gen {i}')
                            for i, img in enumerate(samples['generated'])
                        ]
                    })

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

    def load_checkpoint(self, filename: str):
        checkpoint_path = os.path.join('checkpoints', filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

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

        print(f"Loaded checkpoint from {filename}")
        print(f"Resuming from epoch {self.start_epoch}, global_step {self.global_step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        return self.start_epoch
