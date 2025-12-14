import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
from typing import Optional, Dict, Any

from .latent_diffusion import LatentDiffusionModel
from ..utils import DiffusionMetrics


class DiffusionTrainer:
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

        # Metrics
        self.metrics = DiffusionMetrics(device=self.device)

        # Initialize wandb
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('project_name', 'pet-ct-diffusion'),
                config=self.config,
            )

    def compute_loss(self, batch):
        # Extract PET and CT from batch
        if isinstance(batch, dict):
            pet = batch['pet'].to(self.device)
            ct = batch['ct'].to(self.device)
        else:
            # PairedDataset returns concatenated tensor
            pet, ct = batch.chunk(2, dim=1)
            pet = pet.to(self.device)
            ct = ct.to(self.device)

        # Use CT as target, PET as condition (for now)
        condition = pet
        target = ct

        # Forward pass
        noise_pred, noise = self.model(target, condition)

        # MSE loss on noise prediction
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            self.optimizer.zero_grad()

            loss = self.compute_loss(batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip_norm', 1.0)
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            loss = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def evaluate(self, num_samples: Optional[int] = None):
        return self.metrics.evaluate_model(self.model, self.val_loader, num_samples)

    def sample_images(self, num_samples: int = 4):
        self.model.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))

        if isinstance(batch, dict):
            pet = batch['pet'][:num_samples].to(self.device)
            ct = batch['ct'][:num_samples].to(self.device)
        else:
            pet, ct = batch.chunk(2, dim=1)
            pet = pet[:num_samples].to(self.device)
            ct = ct[:num_samples].to(self.device)

        # Generate samples
        with torch.no_grad():
            generated = self.model.sample(condition=pet, num_inference_steps=50)

        return {
            'pet': pet.cpu(),
            'ct': ct.cpu(),
            'generated': generated.cpu(),
        }

    def train(self):
        num_epochs = self.config.get('num_epochs', 100)
        save_interval = self.config.get('save_interval', 10)
        eval_interval = self.config.get('eval_interval', save_interval)

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.validate()

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

            # Evaluation
            if epoch % eval_interval == 0:
                eval_metrics = self.evaluate(num_samples=50)
                log_dict.update({f'eval_{k}': v for k, v in eval_metrics.items()})
                print("\nEvaluation Metrics:")
                for k, v in eval_metrics.items():
                    print(f"  {k.upper()}: {v:.4f}")

            if self.config.get('use_wandb', False):
                wandb.log(log_dict)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.2e}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f'best_model.pth')

            # Periodic saving
            if epoch % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

                # Sample and log images
                if self.config.get('use_wandb', False):
                    samples = self.sample_images()
                    wandb.log({
                        'sample_images': [
                            wandb.Image(img, caption=f'Gen {i}')
                            for i, img in enumerate(samples['generated'])
                        ]
                    })

    def save_checkpoint(self, filename: str):
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, os.path.join('checkpoints', filename))

    def load_checkpoint(self, filename: str):
        checkpoint_path = os.path.join('checkpoints', filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])