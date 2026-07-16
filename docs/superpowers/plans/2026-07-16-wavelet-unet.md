# Complete Wavelet U-Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configuration-selectable BBDM U-Net whose three encoder and decoder scale transitions use Haar DWT/IWT while preserving the original U-Net API and conditioning routes.

**Architecture:** New focused wavelet down/up modules mix all four Haar subbands with learned convolutions. A separate `WaveletBBDMUNet` mirrors the existing residual blocks, cross-attention, skips, and output head, so legacy construction remains byte-for-byte untouched when disabled.

**Tech Stack:** Python 3.12, PyTorch, pytest, existing dependency-free Haar transform

---

### Task 1: Wavelet scale-transition contracts

**Files:**
- Create: `tests/test_wavelet_unet.py`
- Create: `src/model/wavelet_unet.py`

- [ ] **Step 1: Write failing shape and gradient tests**

```python
import torch


def test_wavelet_downsample_mixes_all_subbands_and_backpropagates():
    from src.model.wavelet_unet import WaveletDownsample

    x = torch.randn(2, 8, 32, 32, requires_grad=True)
    layer = WaveletDownsample(8, 16, mix_kernel_size=3)
    y = layer(x)
    assert y.shape == (2, 16, 16, 16)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_wavelet_upsample_expands_to_four_subbands_and_backpropagates():
    from src.model.wavelet_unet import WaveletUpsample

    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    layer = WaveletUpsample(16, 8, mix_kernel_size=3)
    y = layer(x)
    assert y.shape == (2, 8, 32, 32)
    y.abs().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m pytest tests/test_wavelet_unet.py -q`

Expected: FAIL with `ModuleNotFoundError: src.model.wavelet_unet`.

- [ ] **Step 3: Implement minimal wavelet transitions**

```python
class WaveletDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mix_kernel_size: int = 3):
        super().__init__()
        padding = mix_kernel_size // 2
        self.mix = nn.Sequential(
            nn.Conv2d(4 * in_channels, out_channels, 1),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, mix_kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, details = haar_dwt2(x)
        return self.mix(torch.cat((ll, *details), dim=1))


class WaveletUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mix_kernel_size: int = 3):
        super().__init__()
        padding = mix_kernel_size // 2
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, 4 * out_channels, mix_kernel_size, padding=padding),
            nn.GroupNorm(min(32, 4 * out_channels), 4 * out_channels),
            nn.SiLU(),
            nn.Conv2d(4 * out_channels, 4 * out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.expand(x).chunk(4, dim=1)
        return haar_idwt2(ll, (lh, hl, hh))
```

Validate odd `mix_kernel_size` and positive channel counts with explicit `ValueError`s.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_wavelet_unet.py -q`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/model/wavelet_unet.py tests/test_wavelet_unet.py
git commit -m "feat: add Haar wavelet scale transitions"
```

### Task 2: Complete wavelet BBDM U-Net

**Files:**
- Modify: `src/model/wavelet_unet.py`
- Modify: `tests/test_wavelet_unet.py`

- [ ] **Step 1: Write failing full-backbone tests**

```python
def test_complete_wavelet_unet_preserves_public_shapes_and_skip_contract():
    from src.model.wavelet_unet import WaveletBBDMUNet

    model = WaveletBBDMUNet(
        in_channels=2,
        base_channels=8,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=1,
        time_dim=32,
        enable_heteroscedastic=False,
        ca_kv_dim=8,
        ca_num_heads=1,
    )
    x = torch.randn(2, 2, 32, 32)
    t = torch.tensor([10, 20])
    injections = [
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 8, 32, 32),
    ]
    y = model(x, t, skip_injections=injections)
    assert y.shape == (2, 1, 32, 32)
    assert sum(isinstance(m, WaveletDownsample) for m in model.modules()) == 3
    assert sum(isinstance(m, WaveletUpsample) for m in model.modules()) == 3


def test_complete_wavelet_unet_scale_changes_do_not_call_interpolate(monkeypatch):
    from src.model import wavelet_unet as module

    monkeypatch.setattr(module.F, "interpolate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("interpolate")))
    model = module.WaveletBBDMUNet(
        in_channels=2, base_channels=8, channel_mult=(1, 2, 4, 4),
        num_res_blocks=1, time_dim=32, enable_heteroscedastic=False,
        ca_kv_dim=8, ca_num_heads=1,
    )
    assert model(torch.randn(1, 2, 32, 32), torch.tensor([1])).shape == (1, 1, 32, 32)
```

- [ ] **Step 2: Run tests and verify missing-class failure**

Run: `python -m pytest tests/test_wavelet_unet.py -q`

Expected: FAIL because `WaveletBBDMUNet` is not defined.

- [ ] **Step 3: Implement the full backbone**

Reuse `_TimeEmbedding`, `_ResBlock`, and `_CrossAttention` from `src/model/bbdm_unet.py`. Mirror `BBDMUNet` exactly except:

```python
self.downsamples.append(
    WaveletDownsample(ch, chs[i + 1], mix_kernel_size=mix_kernel_size)
)
...
self.upsamples.append(
    WaveletUpsample(ch, rev_chs[i + 1], mix_kernel_size=mix_kernel_size)
)
```

In `forward`, replace bilinear decoder scaling with:

```python
if i > 0:
    h = self.upsamples[i - 1](h)
```

Require spatial dimensions divisible by `2 ** (len(channel_mult) - 1)` and require exact skip-injection spatial shapes. A mismatched injection raises `ValueError`; it is not silently interpolated.

- [ ] **Step 4: Run all wavelet tests**

Run: `python -m pytest tests/test_wavelet_unet.py -q`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/model/wavelet_unet.py tests/test_wavelet_unet.py
git commit -m "feat: add complete wavelet BBDM U-Net"
```

### Task 3: Configuration and legacy compatibility

**Files:**
- Modify: `src/model/slmf_bbdm.py`
- Modify: `tests/test_wavelet_unet.py`
- Modify: `tests/test_residual_frequency.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_slmf_selects_wavelet_unet_only_when_enabled():
    import copy

    from src.model.bbdm_unet import BBDMUNet
    from src.model.slmf_bbdm import SLMFBBDM
    from src.model.wavelet_unet import WaveletBBDMUNet

    legacy_cfg = _residual_config(frequency=False, gabor=False)
    wavelet_cfg = copy.deepcopy(legacy_cfg)
    wavelet_cfg["modules"]["wavelet_unet"] = {"enabled": True, "mix_kernel_size": 3}
    assert isinstance(SLMFBBDM.from_config(legacy_cfg).unet, BBDMUNet)
    assert isinstance(SLMFBBDM.from_config(wavelet_cfg).unet, WaveletBBDMUNet)


def test_wavelet_unet_forward_keeps_optional_organ_and_suv_interfaces():
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = _residual_config(frequency=False, gabor=False)
    cfg["modules"]["wavelet_unet"] = {"enabled": True}
    cfg["modules"]["organ_prior"] = {"enabled": True, "organ_channels": 6}
    cfg["modules"]["zero_adapter"] = {"enabled": True}
    cfg["losses"]["roi_suv"] = {"enabled": True, "weight": 0.1}
    model = SLMFBBDM.from_config(cfg)
    batch = _model_batch()
    batch["organ_mask"][:, 1, 8:16, 8:16] = 1.0
    loss, logs = model(batch, timesteps=torch.tensor([10]))
    assert torch.isfinite(loss)
    assert "loss/roi_suv/loss" in logs
```

Place these two tests in `tests/test_residual_frequency.py`, where `_residual_config` and `_model_batch` already exist.

- [ ] **Step 2: Run tests and verify configuration failure**

Run: `python -m pytest tests/test_wavelet_unet.py tests/test_residual_frequency.py -q`

Expected: FAIL because `modules.wavelet_unet` is ignored.

- [ ] **Step 3: Route the configuration**

Add `wavelet_unet_config: Optional[Dict[str, Any]] = None` to `SLMFBBDM.__init__`. Build either class inside the existing seeded RNG scope:

```python
wavelet_cfg = wavelet_unet_config or {}
unet_cls = WaveletBBDMUNet if wavelet_cfg.get("enabled", False) else BBDMUNet
if unet_cls is WaveletBBDMUNet:
    unet_kwargs["mix_kernel_size"] = int(wavelet_cfg.get("mix_kernel_size", 3))
```

Pass `modules_cfg.get("wavelet_unet", {})` from `from_config`. Add `module/wavelet_unet` to model logs without changing old state-dict keys when disabled.

- [ ] **Step 4: Run compatibility tests**

Run: `python -m pytest tests/test_wavelet_unet.py tests/test_residual_frequency.py tests/test_smoke.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/model/slmf_bbdm.py tests/test_wavelet_unet.py tests/test_residual_frequency.py
git commit -m "feat: configure wavelet U-Net without breaking legacy routes"
```
