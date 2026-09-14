"""Original-protocol trainers for SLMF-aligned comparison models.

The main project Trainer is excellent for SLMF-BBDM, but official GAN
implementations usually use separate generator/discriminator optimisers and
carefully staged gradient updates.  This file keeps the public CT/PET interface
unified while making the internal optimisation procedure closer to the source
papers and official code.
"""

from __future__ import annotations

import os
import random
import time
from collections import OrderedDict
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from comparison_experiments.models import (
    CPDMComparison,
    CycleGANComparison,
    DistrictSpecificGANComparison,
    Pix2PixComparison,
    RegGANComparison,
    _requires_grad,
    smoothness_loss,
    warp_image,
)


class ImagePool:
    """CycleGAN official-style fake image buffer.

    The original implementation stores previously generated images and
    randomly replays them to stabilise discriminator training.
    """

    def __init__(self, pool_size: int = 50):
        self.pool_size = int(pool_size)
        self.images: list[torch.Tensor] = []

    def query(self, image: torch.Tensor) -> torch.Tensor:
        if self.pool_size <= 0:
            return image.detach()
        returned = []
        for item in image.detach():
            item = item.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(item.clone())
                returned.append(item)
            elif random.random() > 0.5:
                idx = random.randrange(len(self.images))
                old = self.images[idx].clone()
                self.images[idx] = item.clone()
                returned.append(old)
            else:
                returned.append(item)
        return torch.cat(returned, dim=0)


def _detect_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _move_batch(batch: Dict, device: str) -> Dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def _adam(params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float, betas=(0.5, 0.999)):
    return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)


class OriginalProtocolTrainer:
    """Dispatch to the closest available original training protocol."""

    def __init__(
        self,
        model,
        config: Dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        run_cfg = config.get("runtime", {})
        train_cfg = config.get("training", {})
        self.amp = bool(run_cfg.get("amp", True)) and self.device == "cuda"
        self.amp_dtype = _detect_dtype()
        self.log_interval = int(run_cfg.get("log_interval", 10))
        self.eval_interval = int(run_cfg.get("eval_interval", 50))
        self.sample_interval = int(run_cfg.get("sample_interval", 50))
        self.save_interval = int(run_cfg.get("save_interval", 50))
        self.num_epochs = int(train_cfg.get("num_epochs", 1000))
        self.grad_clip_norm = float(run_cfg.get("grad_clip_norm", 0.0))
        self.lr = float(train_cfg.get("learning_rate", 2e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0))
        self.lr_min = float(train_cfg.get("lr_min", 1e-6))
        self.epoch_count = 0
        self.step_count = 0
        self.best_val_loss = float("inf")
        early_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = bool(early_cfg.get("enabled", True))
        self.early_stopping_patience = int(early_cfg.get("patience", 80))
        self.early_stopping_min_delta = float(early_cfg.get("min_delta", 1e-4))
        self.early_stopping_warmup_epochs = int(early_cfg.get("warmup_epochs", 50))
        self.no_improve_epochs = 0

        if isinstance(model, Pix2PixComparison):
            self.protocol = "pix2pix_official"
            self.opt_g = _adam(model.generator.parameters(), self.lr, self.weight_decay)
            self.opt_d = _adam(model.discriminator.parameters(), self.lr, self.weight_decay)
            sched_params = [self.opt_g, self.opt_d]
        elif isinstance(model, CycleGANComparison):
            self.protocol = "cyclegan_official"
            self.opt_g = _adam(
                list(model.g_ct_to_pet.parameters()) + list(model.g_pet_to_ct.parameters()),
                self.lr,
                self.weight_decay,
            )
            self.opt_d = _adam(
                list(model.d_pet.parameters()) + list(model.d_ct.parameters()),
                self.lr,
                self.weight_decay,
            )
            pool_size = int(config.get("comparison_model", {}).get("pool_size", 50))
            self.fake_pet_pool = ImagePool(pool_size)
            self.fake_ct_pool = ImagePool(pool_size)
            sched_params = [self.opt_g, self.opt_d]
        elif isinstance(model, RegGANComparison):
            self.protocol = "reggan_public_code_aligned"
            self.opt_g = _adam(
                list(model.generator.parameters()) + list(model.registration.parameters()),
                self.lr,
                self.weight_decay,
            )
            self.opt_d = _adam(model.discriminator.parameters(), self.lr, self.weight_decay)
            sched_params = [self.opt_g, self.opt_d]
        elif isinstance(model, DistrictSpecificGANComparison):
            self.protocol = "district_gan_paper_reproduction"
            self.opt_g = _adam(model.generators.parameters(), self.lr, self.weight_decay)
            self.opt_d = _adam(model.discriminators.parameters(), self.lr, self.weight_decay)
            sched_params = [self.opt_g, self.opt_d]
        elif isinstance(model, CPDMComparison):
            self.protocol = "cpdm_bbdm_aligned"
            self.opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sched_params = [self.opt]
        else:
            self.protocol = "generic"
            self.opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sched_params = [self.opt]

        self.schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.num_epochs, eta_min=self.lr_min)
            for opt in sched_params
        ]

    def _clip(self, modules: Iterable[torch.nn.Module]) -> None:
        if self.grad_clip_norm <= 0:
            return
        params = []
        for module in modules:
            params.extend([p for p in module.parameters() if p.requires_grad and p.grad is not None])
        if params:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)

    # ------------------------------------------------------------------
    # Official-style train steps
    # ------------------------------------------------------------------

    def _step_pix2pix(self, batch: Dict) -> Dict[str, float]:
        model: Pix2PixComparison = self.model
        ct, pet = batch["ct"], batch["pet"]

        fake = model.generator(ct)

        # D update: official Pix2Pix uses detached fake pairs.
        _requires_grad([model.discriminator], True)
        self.opt_d.zero_grad(set_to_none=True)
        pred_real = model.discriminator(torch.cat([ct, pet], dim=1))
        pred_fake = model.discriminator(torch.cat([ct, fake.detach()], dim=1))
        loss_d = 0.5 * (model.gan_loss(pred_real, True) + model.gan_loss(pred_fake, False))
        loss_d.backward()
        self._clip([model.discriminator])
        self.opt_d.step()

        # G update.
        _requires_grad([model.discriminator], False)
        self.opt_g.zero_grad(set_to_none=True)
        fake = model.generator(ct)
        loss_g_gan = model.gan_loss(model.discriminator(torch.cat([ct, fake], dim=1)), True)
        loss_g_l1 = F.l1_loss(fake, pet) * model.loss_weights.lambda_l1
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        self._clip([model.generator])
        self.opt_g.step()

        return {
            "loss/total": float((loss_g + loss_d).detach()),
            "loss/g_gan": float(loss_g_gan.detach()),
            "loss/g_l1": float(loss_g_l1.detach()),
            "loss/d": float(loss_d.detach()),
        }

    def _step_cyclegan(self, batch: Dict) -> Dict[str, float]:
        model: CycleGANComparison = self.model
        ct, pet = batch["ct"], batch["pet"]

        # G update first, following the official PyTorch CycleGAN flow.
        _requires_grad([model.d_pet, model.d_ct], False)
        self.opt_g.zero_grad(set_to_none=True)
        fake_pet = model.g_ct_to_pet(ct)
        rec_ct = model.g_pet_to_ct(fake_pet)
        fake_ct = model.g_pet_to_ct(pet)
        rec_pet = model.g_ct_to_pet(fake_ct)

        loss_g_adv = model.gan_loss(model.d_pet(fake_pet), True) + model.gan_loss(model.d_ct(fake_ct), True)
        loss_cycle = (F.l1_loss(rec_ct, ct) + F.l1_loss(rec_pet, pet)) * model.loss_weights.lambda_cycle
        loss_identity = (
            F.l1_loss(model.g_ct_to_pet(pet), pet) + F.l1_loss(model.g_pet_to_ct(ct), ct)
        ) * model.loss_weights.lambda_identity
        loss_g = loss_g_adv + loss_cycle + loss_identity
        loss_g.backward()
        self._clip([model.g_ct_to_pet, model.g_pet_to_ct])
        self.opt_g.step()

        # D update with image pools.
        _requires_grad([model.d_pet, model.d_ct], True)
        self.opt_d.zero_grad(set_to_none=True)
        pooled_pet = self.fake_pet_pool.query(fake_pet)
        pooled_ct = self.fake_ct_pool.query(fake_ct)
        loss_d_pet = 0.5 * (
            model.gan_loss(model.d_pet(pet), True)
            + model.gan_loss(model.d_pet(pooled_pet), False)
        )
        loss_d_ct = 0.5 * (
            model.gan_loss(model.d_ct(ct), True)
            + model.gan_loss(model.d_ct(pooled_ct), False)
        )
        loss_d = loss_d_pet + loss_d_ct
        loss_d.backward()
        self._clip([model.d_pet, model.d_ct])
        self.opt_d.step()

        return {
            "loss/total": float((loss_g + loss_d).detach()),
            "loss/g_adv": float(loss_g_adv.detach()),
            "loss/cycle": float(loss_cycle.detach()),
            "loss/identity": float(loss_identity.detach()),
            "loss/d_pet": float(loss_d_pet.detach()),
            "loss/d_ct": float(loss_d_ct.detach()),
        }

    def _step_reggan(self, batch: Dict) -> Dict[str, float]:
        model: RegGANComparison = self.model
        ct, pet = batch["ct"], batch["pet"]

        fake_pet = model.generator(ct)

        _requires_grad([model.discriminator], True)
        self.opt_d.zero_grad(set_to_none=True)
        pred_real = model.discriminator(torch.cat([ct, pet], dim=1))
        pred_fake = model.discriminator(torch.cat([ct, fake_pet.detach()], dim=1))
        loss_d = 0.5 * (model.gan_loss(pred_real, True) + model.gan_loss(pred_fake, False))
        loss_d.backward()
        self._clip([model.discriminator])
        self.opt_d.step()

        _requires_grad([model.discriminator], False)
        self.opt_g.zero_grad(set_to_none=True)
        fake_pet = model.generator(ct)
        flow = model.registration(pet, fake_pet)
        corrected_pet = warp_image(pet, flow)
        loss_g_adv = model.gan_loss(model.discriminator(torch.cat([ct, fake_pet], dim=1)), True)
        loss_reg = F.l1_loss(fake_pet, corrected_pet) * model.loss_weights.lambda_registration
        loss_smooth = smoothness_loss(flow) * model.loss_weights.lambda_smooth
        loss_g = loss_g_adv + loss_reg + loss_smooth
        loss_g.backward()
        self._clip([model.generator, model.registration])
        self.opt_g.step()

        return {
            "loss/total": float((loss_g + loss_d).detach()),
            "loss/g_adv": float(loss_g_adv.detach()),
            "loss/registration": float(loss_reg.detach()),
            "loss/flow_smooth": float(loss_smooth.detach()),
            "loss/d": float(loss_d.detach()),
            "reggan/flow_abs": float(flow.abs().mean().detach()),
        }

    def _step_district_gan(self, batch: Dict) -> Dict[str, float]:
        model: DistrictSpecificGANComparison = self.model
        batch = model.random_patch_batch(batch)
        ct, pet = batch["ct"], batch["pet"]
        masks = model._district_masks(batch)

        _requires_grad(model.discriminators.values(), True)
        self.opt_d.zero_grad(set_to_none=True)
        loss_d = torch.tensor(0.0, device=ct.device)
        with torch.no_grad():
            fake_for_d = torch.stack(
                [
                    model.generators[name](ct) * masks[:, idx:idx + 1]
                    for idx, name in enumerate(model.districts)
                ],
                dim=0,
            ).sum(dim=0).clamp(-1.0, 1.0)
        for idx, name in enumerate(model.districts):
            mask = masks[:, idx:idx + 1]
            real_pair = torch.cat([ct * mask, pet * mask], dim=1)
            fake_pair = torch.cat([ct * mask, fake_for_d * mask], dim=1)
            loss_d = loss_d + 0.5 * (
                model.gan_loss(model.discriminators[name](real_pair), True)
                + model.gan_loss(model.discriminators[name](fake_pair), False)
            )
        loss_d.backward()
        self._clip(model.discriminators.values())
        self.opt_d.step()

        _requires_grad(model.discriminators.values(), False)
        self.opt_g.zero_grad(set_to_none=True)
        loss_g_adv = torch.tensor(0.0, device=ct.device)
        loss_l1 = torch.tensor(0.0, device=ct.device)
        fake_parts = []
        for idx, name in enumerate(model.districts):
            mask = masks[:, idx:idx + 1]
            fake_i = model.generators[name](ct) * mask
            fake_parts.append(fake_i)
            loss_g_adv = loss_g_adv + model.gan_loss(
                model.discriminators[name](torch.cat([ct * mask, fake_i], dim=1)), True
            )
            loss_l1 = loss_l1 + F.l1_loss(fake_i, pet * mask) * model.loss_weights.lambda_district
        loss_g = loss_g_adv + loss_l1
        loss_g.backward()
        self._clip(model.generators.values())
        self.opt_g.step()

        return {
            "loss/total": float((loss_g + loss_d).detach()),
            "loss/g_adv": float(loss_g_adv.detach()),
            "loss/district_l1": float(loss_l1.detach()),
            "loss/d": float(loss_d.detach()),
        }

    def _step_diffusion_or_generic(self, batch: Dict) -> Dict[str, float]:
        self.opt.zero_grad(set_to_none=True)
        loss, logs = self.model(batch)
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.opt.step()
        return {
            key: (float(value.detach()) if torch.is_tensor(value) else float(value))
            for key, value in logs.items()
        }

    def train_step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        batch = _move_batch(batch, self.device)
        use_autocast = self.amp and self.protocol in {"cpdm_bbdm_aligned", "generic"}
        with torch.amp.autocast("cuda", enabled=use_autocast, dtype=self.amp_dtype):
            if isinstance(self.model, Pix2PixComparison):
                logs = self._step_pix2pix(batch)
            elif isinstance(self.model, CycleGANComparison):
                logs = self._step_cyclegan(batch)
            elif isinstance(self.model, RegGANComparison):
                logs = self._step_reggan(batch)
            elif isinstance(self.model, DistrictSpecificGANComparison):
                logs = self._step_district_gan(batch)
            else:
                logs = self._step_diffusion_or_generic(batch)
        self.step_count += 1
        return logs

    @torch.no_grad()
    def eval_step(self, batch: Dict) -> Dict[str, float]:
        self.model.eval()
        batch = _move_batch(batch, self.device)
        loss, logs = self.model(batch)
        return {
            key: (float(value.detach()) if torch.is_tensor(value) else float(value))
            for key, value in logs.items()
        }

    @torch.no_grad()
    def evaluate_loader(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        eval_logs = [self.eval_step(batch) for batch in self.val_loader]
        if not eval_logs:
            return {}
        keys = eval_logs[0].keys()
        return {key: sum(item.get(key, 0.0) for item in eval_logs) / len(eval_logs) for key in keys}

    def train_epoch(self) -> Dict[str, float]:
        start = time.time()
        collected: OrderedDict[str, list[float]] = OrderedDict()
        for batch in self.train_loader:
            logs = self.train_step(batch)
            if self.step_count % self.log_interval == 0:
                for key, value in logs.items():
                    collected.setdefault(key, []).append(value)
        self.epoch_count += 1
        for scheduler in self.schedulers:
            scheduler.step()
        if not collected:
            collected.setdefault("loss/total", [logs.get("loss/total", 0.0)])
        avg = {key: sum(values) / len(values) for key, values in collected.items()}
        avg["perf/epoch_seconds"] = time.time() - start
        avg["perf/lr"] = self.schedulers[0].get_last_lr()[0]
        return avg

    def run(self, num_epochs: Optional[int] = None) -> None:
        num_epochs = int(num_epochs or self.num_epochs)
        print("\n" + "=" * 60)
        print("Comparison Training")
        print(f"Model: {self.model.model_name} | Protocol: {self.protocol}")
        print(f"Device: {self.device} | AMP(diffusion only): {self.amp_dtype if self.amp else 'off'}")
        print(f"Trainable: {self.model.get_trainable_params():,} | Total: {self.model.get_total_params():,}")
        if self.val_loader is not None and self.early_stopping_enabled:
            print(
                "Early stopping: "
                f"patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta}, "
                f"warmup={self.early_stopping_warmup_epochs}"
            )
        print("=" * 60 + "\n")

        start = time.time()
        for _ in range(num_epochs):
            should_stop = False
            logs = self.train_epoch()
            print(
                f"Epoch {self.epoch_count:4d}/{num_epochs} | "
                f"Loss: {logs.get('loss/total', 0.0):.4f} | "
                f"Time: {time.time() - start:.0f}s ({logs.get('perf/epoch_seconds', 0.0):.1f}s/ep) | "
                f"LR: {logs.get('perf/lr', 0.0):.2e}"
            )
            if self.val_loader is not None and self.epoch_count % self.eval_interval == 0:
                avg_eval = self.evaluate_loader()
                print(f"  Eval  Loss: {avg_eval.get('loss/total', 0.0):.4f}")
                val_loss = float(avg_eval.get("loss/total", float("inf")))
                improved = val_loss < (self.best_val_loss - self.early_stopping_min_delta)
                if improved:
                    self.best_val_loss = val_loss
                    self.no_improve_epochs = 0
                    self.save_checkpoint(tag="best")
                    print(f"  Best  Loss: {self.best_val_loss:.4f}")
                else:
                    self.no_improve_epochs += self.eval_interval
                    print(
                        f"  No improvement: {self.no_improve_epochs}/"
                        f"{self.early_stopping_patience} epochs"
                    )
                    if (
                        self.early_stopping_enabled
                        and self.epoch_count >= self.early_stopping_warmup_epochs
                        and self.no_improve_epochs >= self.early_stopping_patience
                    ):
                        should_stop = True
                        print(
                            "  Early stopping triggered: "
                            f"best_val_loss={self.best_val_loss:.4f}, "
                            f"epoch={self.epoch_count}"
                        )
            if self.epoch_count % self.save_interval == 0:
                self.save_checkpoint()
            if self.val_loader is not None and self.epoch_count % self.sample_interval == 0:
                batch = _move_batch(next(iter(self.val_loader)), self.device)
                result = self.model.sample(batch)
                synth = result["synthetic_pet"]
                print(f"  Sample PET range: [{synth.min().item():.4f}, {synth.max().item():.4f}]")
            if should_stop:
                break

        if self.val_loader is not None and self.best_val_loss == float("inf"):
            avg_eval = self.evaluate_loader()
            self.best_val_loss = float(avg_eval.get("loss/total", float("inf")))
            self.save_checkpoint(tag="best")
            print(f"Final best checkpoint saved with val loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, tag: Optional[str] = None) -> None:
        exp_name = self.config.get("experiment", {}).get("name", "comparison")
        save_dir = os.path.join("checkpoints", exp_name)
        os.makedirs(save_dir, exist_ok=True)
        fname = f"ckpt_epoch{self.epoch_count:04d}.pt" if tag is None else f"ckpt_{tag}.pt"
        path = os.path.join(save_dir, fname)
        checkpoint = {
            "model": self.model.state_dict(),
            "epoch": self.epoch_count,
            "step": self.step_count,
            "config": self.config,
            "protocol": self.protocol,
            "best_val_loss": self.best_val_loss,
            "early_stopping": {
                "enabled": self.early_stopping_enabled,
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_min_delta,
                "warmup_epochs": self.early_stopping_warmup_epochs,
                "no_improve_epochs": self.no_improve_epochs,
            },
        }
        for name in ("opt", "opt_g", "opt_d"):
            if hasattr(self, name):
                checkpoint[name] = getattr(self, name).state_dict()
        torch.save(checkpoint, path)
        print(f"  Saved -> {path}")
