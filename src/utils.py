import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm


class DiffusionMetrics:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if pred_np.ndim == 4:
            pred_np = np.transpose(pred_np, (0, 2, 3, 1))
            target_np = np.transpose(target_np, (0, 2, 3, 1))

        scores = []
        for i in range(pred_np.shape[0]):
            if pred_np.shape[-1] == 1:
                scores.append(psnr(target_np[i, :, :, 0], pred_np[i, :, :, 0], data_range=data_range))
            else:
                try:
                    scores.append(ssim(target_np[i], pred_np[i], data_range=data_range, channel_axis=-1, win_size=7))
                except:
                    scores.append(ssim(target_np[i], pred_np[i], data_range=data_range, multichannel=True, win_size=7))
        return np.mean(scores)

    @staticmethod
    def compute_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if pred_np.ndim == 4:
            pred_np = np.transpose(pred_np, (0, 2, 3, 1))
            target_np = np.transpose(target_np, (0, 2, 3, 1))

        scores = []
        for i in range(pred_np.shape[0]):
            if pred_np.shape[-1] == 1:
                scores.append(ssim(target_np[i, :, :, 0], pred_np[i, :, :, 0], data_range=data_range))
            else:
                try:
                    scores.append(ssim(target_np[i], pred_np[i], data_range=data_range, channel_axis=-1, win_size=7))
                except:
                    scores.append(ssim(target_np[i], pred_np[i], data_range=data_range, multichannel=True, win_size=7))
        return np.mean(scores)

    @staticmethod
    def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        return F.l1_loss(pred, target).item()

    @staticmethod
    def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        return F.mse_loss(pred, target).item()

    @staticmethod
    def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        return torch.sqrt(F.mse_loss(pred, target)).item()

    @staticmethod
    def compute_nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = F.mse_loss(pred, target)
        rmse = torch.sqrt(mse)
        target_range = target.max() - target.min()
        if target_range == 0:
            return float('inf')
        return (rmse / target_range).item()

    def evaluate_batch(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        metrics = {
            'mae': self.compute_mae(pred, target),
            'mse': self.compute_mse(pred, target),
            'rmse': self.compute_rmse(pred, target),
            'nrmse': self.compute_nrmse(pred, target),
            'psnr': self.compute_psnr(pred, target),
            'ssim': self.compute_ssim(pred, target)
        }
        return metrics

    def evaluate_model(self, model, dataloader, num_samples: Optional[int] = None) -> Dict[str, float]:
        model.eval()
        all_metrics = []

        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                if isinstance(batch, dict):
                    pet = batch['pet'].to(self.device)
                    ct = batch['ct'].to(self.device)
                else:
                    pet, ct = batch.chunk(2, dim=1)
                    pet = pet.to(self.device)
                    ct = ct.to(self.device)

                generated = model.sample(condition=pet, num_inference_steps=50)
                batch_metrics = self.evaluate_batch(generated, ct)
                all_metrics.append(batch_metrics)

                total_samples += pet.shape[0]
                if num_samples and total_samples >= num_samples:
                    break

        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics