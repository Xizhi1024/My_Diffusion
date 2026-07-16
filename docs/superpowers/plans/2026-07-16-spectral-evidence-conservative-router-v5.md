# Spectral-Evidence Conservative Frequency Router V5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and ablate a safe Gabor-DCT evidence module that conservatively routes Haar detail packets to adjacent decoder levels or an exact null destination.

**Architecture:** Haar remains the only injected frequency content. A selected-DCT descriptor and the existing bounded quadrature Gabor prior produce inference-available evidence for a bounded trust-amplitude head and a null-aware adjacent-level router. Zero-initialized projections make the first forward pass an exact no-op; hard all-null mode short-circuits before projections.

**Tech Stack:** Python 3.13, PyTorch, PyYAML, pytest, Pixi, existing residual Brownian bridge/Haar/Gabor modules

---

## File map

**Create**

- `src/model/frequency/dct_descriptor.py` — fixed selected DCT basis, constrained learnable frequency weights, descriptor diagnostics.
- `src/model/frequency/spectral_router.py` — bounded amplitude heads, conservative route heads, Haar packet routing, exact fallback, diagnostics.
- `src/model/loss_terms/spectral_router.py` — non-image router/DCT/Gabor offset regularization.
- `tests/test_spectral_router.py` — unit contracts for DCT, routing, no-op behavior, gradients, and diagnostic shapes.
- `configs/experiments/slmf_png_spectral_router_v5.yaml` — V5 base configuration.
- `configs/experiments/spectral_router_ablation_plan_v5.yaml` — fixed Stage A/Stage B/300-epoch promotion plan.
- `scripts/run_spectral_router_v5.py` — dynamic two-stage V5 runner.

**Modify**

- `src/model/frequency/__init__.py` — export the V5 router and DCT descriptor.
- `src/model/priors/gabor.py` — expose differentiable raw-parameter offset energy for regularization and logging.
- `src/model/loss_terms/__init__.py` — export router regularization.
- `src/model/slmf_bbdm.py` — construct the new mode, route inference-only descriptors, publish regularizers/diagnostics, preserve legacy modes.
- `configs/experiments/ablations.yaml` — add S0/S1/S2/S3 V5 presets.
- `tests/test_residual_frequency.py` — integration and configuration contracts.
- `tests/test_frequency_ablation_runner.py` — V5 plan, dynamic inheritance, dry-run, and promotion contracts.
- `pixi.toml` — add dry-run and full V5 ablation tasks.

## Task 1: Selected-DCT descriptor

**Files:**

- Create: `src/model/frequency/dct_descriptor.py`
- Create: `tests/test_spectral_router.py`
- Modify: `src/model/frequency/__init__.py`

- [ ] **Step 1: Write failing descriptor tests**

Add these tests to `tests/test_spectral_router.py`:

```python
import torch


def test_selected_dct_descriptor_is_finite_normalized_and_trainable():
    from src.model.frequency.dct_descriptor import SelectedDCTDescriptor

    module = SelectedDCTDescriptor(pooled_size=8, selected_frequencies=12)
    details = torch.randn(2, 3, 16, 16, requires_grad=True)
    descriptor, diagnostics = module(details)

    assert descriptor.shape == (2, 3, 12)
    assert torch.isfinite(descriptor).all()
    assert torch.allclose(
        diagnostics["frequency_weights"].sum(), torch.tensor(1.0), atol=1e-6
    )
    assert diagnostics["weight_offset_energy"].ndim == 0

    descriptor.sum().backward()
    assert details.grad is not None
    assert torch.isfinite(details.grad).all()


def test_selected_dct_descriptor_starts_at_declared_prior():
    from src.model.frequency.dct_descriptor import SelectedDCTDescriptor

    module = SelectedDCTDescriptor(pooled_size=8, selected_frequencies=12)
    _, diagnostics = module(torch.ones(1, 3, 8, 8))
    assert torch.allclose(
        diagnostics["frequency_weights"], module.prior_weights, atol=1e-6
    )
    assert diagnostics["weight_offset_energy"].item() == 0.0


def test_selected_dct_descriptor_rejects_invalid_configuration():
    from src.model.frequency.dct_descriptor import SelectedDCTDescriptor

    for kwargs in (
        {"pooled_size": 7, "selected_frequencies": 12},
        {"pooled_size": 8, "selected_frequencies": 0},
        {"pooled_size": 8, "selected_frequencies": 13},
    ):
        try:
            SelectedDCTDescriptor(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"configuration should fail: {kwargs}")
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```powershell
pixi run python -m pytest tests/test_spectral_router.py -q
```

Expected: collection or import fails with `ModuleNotFoundError: src.model.frequency.dct_descriptor`.

- [ ] **Step 3: Implement the dependency-free selected-DCT descriptor**

Create `src/model/frequency/dct_descriptor.py` with these concrete rules:

```python
"""Small selected-frequency DCT descriptor for Haar detail energy maps."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


_SELECTED_8X8 = (
    (0, 1), (1, 0), (1, 1), (0, 2),
    (2, 0), (1, 2), (2, 1), (2, 2),
    (0, 3), (3, 0), (1, 3), (3, 1),
)


def _dct_basis(size: int, frequencies: tuple[tuple[int, int], ...]) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    rows = []
    for u, v in frequencies:
        cu = math.sqrt(1 / size) if u == 0 else math.sqrt(2 / size)
        cv = math.sqrt(1 / size) if v == 0 else math.sqrt(2 / size)
        basis_u = cu * torch.cos(math.pi * (2 * coords + 1) * u / (2 * size))
        basis_v = cv * torch.cos(math.pi * (2 * coords + 1) * v / (2 * size))
        rows.append(torch.outer(basis_u, basis_v))
    return torch.stack(rows, dim=0)


class SelectedDCTDescriptor(nn.Module):
    def __init__(self, pooled_size: int = 8, selected_frequencies: int = 12) -> None:
        super().__init__()
        if pooled_size != 8:
            raise ValueError("V5 selected DCT requires pooled_size=8")
        if not 1 <= selected_frequencies <= len(_SELECTED_8X8):
            raise ValueError("selected_frequencies must be in [1, 12]")
        frequencies = _SELECTED_8X8[:selected_frequencies]
        self.pooled_size = pooled_size
        self.selected_frequencies = selected_frequencies
        self.register_buffer("basis", _dct_basis(pooled_size, frequencies))

        prior = torch.tensor(
            [1.0, 1.0, 1.2, 1.2, 1.2, 1.4, 1.4, 1.4, 1.1, 1.1, 1.0, 1.0],
            dtype=torch.float32,
        )[:selected_frequencies]
        prior = prior / prior.sum()
        self.register_buffer("prior_weights", prior)
        self.weight_offsets = nn.Parameter(torch.zeros(selected_frequencies))

    def frequency_weights(self) -> torch.Tensor:
        logits = self.prior_weights.clamp_min(1e-8).log() + self.weight_offsets
        return torch.softmax(logits, dim=0)

    def forward(
        self, details: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if details.ndim != 4 or details.shape[1] != 3:
            raise ValueError("DCT descriptor expects [B,3,H,W] Haar details")
        energy = torch.log1p(details.abs())
        pooled = F.adaptive_avg_pool2d(
            energy, output_size=(self.pooled_size, self.pooled_size)
        )
        basis = self.basis.to(device=details.device, dtype=details.dtype)
        coefficients = torch.einsum("bchw,khw->bck", pooled, basis).abs()
        coefficients = coefficients / coefficients.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.frequency_weights().to(details)
        descriptor = coefficients * weights.view(1, 1, -1)
        return descriptor, {
            "frequency_weights": weights,
            "weight_offset_energy": self.weight_offsets.square().mean(),
        }
```

Export it from `src/model/frequency/__init__.py` by adding:

```python
from .dct_descriptor import SelectedDCTDescriptor
```

and adding `"SelectedDCTDescriptor"` to `__all__`.

- [ ] **Step 4: Run descriptor tests**

Run:

```powershell
pixi run python -m pytest tests/test_spectral_router.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit the descriptor**

```powershell
git add src/model/frequency/dct_descriptor.py src/model/frequency/__init__.py tests/test_spectral_router.py
git commit -m "feat: add selected DCT frequency descriptor"
```

## Task 2: Bounded trust and conservative route heads

**Files:**

- Create: `src/model/frequency/spectral_router.py`
- Modify: `tests/test_spectral_router.py`

- [ ] **Step 1: Write failing head tests**

Append:

```python
def test_bounded_amplitude_head_starts_neutral_and_respects_limits():
    from src.model.frequency.spectral_router import BoundedAmplitudeHead

    head = BoundedAmplitudeHead(16, hidden_channels=8, minimum=-0.05, maximum=0.10)
    evidence = torch.randn(4, 16)
    initial = head(evidence)
    assert torch.allclose(initial, torch.zeros_like(initial), atol=1e-7)

    with torch.no_grad():
        head.final.weight.fill_(100.0)
    bounded = head(evidence)
    assert bounded.min() >= -0.05
    assert bounded.max() <= 0.10


def test_conservative_route_head_is_null_biased_and_sums_to_one():
    from src.model.frequency.spectral_router import ConservativeRouteHead

    head = ConservativeRouteHead(16, hidden_channels=8, initial_null_probability=0.90)
    probabilities = head(torch.randn(6, 16))
    assert probabilities.shape == (6, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(6), atol=1e-6)
    assert torch.allclose(probabilities[:, 2], torch.full((6,), 0.90), atol=1e-6)
```

- [ ] **Step 2: Run the focused tests**

Run:

```powershell
pixi run python -m pytest tests/test_spectral_router.py -q
```

Expected: imports of `BoundedAmplitudeHead` and `ConservativeRouteHead` fail.

- [ ] **Step 3: Implement both heads**

Create `src/model/frequency/spectral_router.py` and add:

```python
"""Spectral-evidence trust and conservative cross-level routing."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_reliable import (
    BoundaryReliableFrequencyInjector,
    _ZeroProjection,
    _split_details,
    _stack_details,
    _total_variation,
)
from .dct_descriptor import SelectedDCTDescriptor


class BoundedAmplitudeHead(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        minimum: float = -0.05,
        maximum: float = 0.10,
    ) -> None:
        super().__init__()
        if not minimum < 0 < maximum:
            raise ValueError("amplitude bounds must straddle zero")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.features = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.final.weight)
        neutral = (0.0 - minimum) / (maximum - minimum)
        nn.init.constant_(self.final.bias, math.log(neutral / (1.0 - neutral)))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        unit = torch.sigmoid(self.final(self.features(evidence))).squeeze(-1)
        return self.minimum + (self.maximum - self.minimum) * unit


class ConservativeRouteHead(nn.Module):
    """Route one packet to native, adjacent-shallow, or null."""

    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        initial_null_probability: float = 0.90,
    ) -> None:
        super().__init__()
        if not 0.5 < initial_null_probability < 1.0:
            raise ValueError("initial_null_probability must be in (0.5, 1)")
        self.features = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_channels, 3)
        nn.init.zeros_(self.final.weight)
        active = (1.0 - initial_null_probability) / 2.0
        prior = torch.tensor([active, active, initial_null_probability])
        with torch.no_grad():
            self.final.bias.copy_(prior.log())

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.final(self.features(evidence)), dim=-1)
```

- [ ] **Step 4: Run head tests**

Run:

```powershell
pixi run python -m pytest tests/test_spectral_router.py -q
```

Expected: 5 passed.

- [ ] **Step 5: Commit the heads**

```powershell
git add src/model/frequency/spectral_router.py tests/test_spectral_router.py
git commit -m "feat: add bounded trust and conservative route heads"
```

## Task 3: Full Haar-packet router and exact fallback

**Files:**

- Modify: `src/model/frequency/spectral_router.py`
- Modify: `src/model/frequency/__init__.py`
- Modify: `tests/test_spectral_router.py`

- [ ] **Step 1: Write failing router behavior tests**

Add a local schedule fixture and contracts:

```python
class _BridgeSchedule:
    def __init__(self, steps: int = 100):
        self.num_train_timesteps = steps
        self.m_t = torch.linspace(0.0, 1.0, steps)
        self.sigma_t = (2 * self.m_t * (1 - self.m_t)).sqrt().clamp_min(1e-4)


def _router(**overrides):
    from src.model.frequency.spectral_router import SpectralEvidenceFrequencyRouter

    kwargs = {
        "output_channels": (16, 16, 8, 4),
        "band_scales": (0.5, 0.25),
        "ct_reliability_floors": (0.25, 0.50),
        "gabor_orientations": 8,
        "hidden_channels": 16,
        "initial_null_probability": 0.90,
    }
    kwargs.update(overrides)
    return SpectralEvidenceFrequencyRouter(**kwargs)


def test_router_first_forward_is_exact_noop_and_routes_are_conservative():
    module = _router()
    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    orientation = torch.rand(2, 8, 32, 32)
    anisotropy = torch.rand(2, 1, 32, 32)

    injections, diagnostics = module(
        residual,
        torch.tensor([20, 60]),
        _BridgeSchedule(),
        ct,
        gabor_orientation=orientation,
        gabor_anisotropy=anisotropy,
    )

    assert [tuple(x.shape) for x in injections] == [
        (2, 16, 4, 4), (2, 16, 8, 8), (2, 8, 16, 16), (2, 4, 32, 32)
    ]
    assert all(torch.count_nonzero(x).item() == 0 for x in injections)
    assert torch.allclose(diagnostics["routes_l2"].sum(-1), torch.ones(2, 3))
    assert torch.allclose(diagnostics["routes_l1"].sum(-1), torch.ones(2, 3))
    assert diagnostics["routes_l2"][..., 2].mean() > 0.89
    assert diagnostics["routes_l1"][..., 2].mean() > 0.89


def test_hard_all_null_short_circuits_before_projection():
    module = _router(hard_all_null=True)

    class _RejectProjection(torch.nn.Module):
        def forward(self, value):
            raise AssertionError("hard all-null must not call a projection")

    module.projection_heads = torch.nn.ModuleList(
        [_RejectProjection(), _RejectProjection(), _RejectProjection()]
    )
    module.l2_to_l1_projection = _RejectProjection()
    residual = torch.randn(1, 1, 32, 32)
    injections, diagnostics = module(
        residual, torch.tensor([50]), _BridgeSchedule(), torch.randn_like(residual)
    )
    assert all(torch.count_nonzero(x).item() == 0 for x in injections)
    assert torch.all(diagnostics["routes_l2"][..., 2] == 1)
    assert torch.all(diagnostics["routes_l1"][..., 2] == 1)


def test_router_receives_gradients_after_zero_projection_learns():
    module = _router()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    args = (residual, torch.tensor([20, 60]), _BridgeSchedule(), ct)

    first, _ = module(*args)
    sum(x.sum() for x in first).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second, _ = module(*args)
    sum(x.square().mean() for x in second).backward()
    route_grads = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if "route_heads" in name and parameter.grad is not None
    ]
    assert route_grads
    assert all(torch.isfinite(grad).all() for grad in route_grads)
```

- [ ] **Step 2: Run tests to verify the router class is missing**

```powershell
pixi run python -m pytest tests/test_spectral_router.py -q
```

Expected: failure importing `SpectralEvidenceFrequencyRouter`.

- [ ] **Step 3: Implement `SpectralEvidenceFrequencyRouter`**

Add a subclass of `BoundaryReliableFrequencyInjector` with these exact public constructor fields:

```python
class SpectralEvidenceFrequencyRouter(BoundaryReliableFrequencyInjector):
    def __init__(
        self,
        output_channels: Sequence[int] = (256, 256, 128, 64),
        band_scales: Sequence[float] = (0.5, 0.25),
        ct_reliability_floors: Sequence[float] = (0.25, 0.50),
        gabor_orientations: int = 8,
        hidden_channels: int = 32,
        dct_enabled: bool = True,
        gabor_enabled: bool = True,
        cross_level_enabled: bool = True,
        hard_all_null: bool = False,
        initial_null_probability: float = 0.90,
        amplitude_delta_min: float = -0.05,
        amplitude_delta_max: float = 0.10,
        **base_kwargs,
    ) -> None:
```

Use this exact public forward contract so the model integration and direct module tests agree:

```python
def forward(
    self,
    current_residual: torch.Tensor,
    ct: torch.Tensor,
    timestep: torch.Tensor,
    lesion_score: torch.Tensor | None = None,
    topq_mask: torch.Tensor | None = None,
    gabor_feat: torch.Tensor | None = None,
    gabor_orientation: torch.Tensor | None = None,
    gabor_anisotropy: torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
```

Implementation rules:

1. Call `super().__init__` with `use_directional_reliability=False` and `use_gabor_agreement=False`; V5 Gabor evidence must not be multiplied through the legacy fixed rules.
2. Instantiate one `SelectedDCTDescriptor` when `dct_enabled=true`.
3. Use a fixed evidence width computed from 36 DCT values plus 12 scalar statistics. When DCT or Gabor is disabled, or an optional Gabor tensor is absent in direct module use, insert zero tensors of the same declared width so checkpoints and heads remain shape-compatible across ablations.
4. Create two amplitude heads and two route heads, one for each native source level.
5. Reuse native `projection_heads` for `L2->L2`, `L1->L1`, and `L1->L0`; add `l2_to_l1_projection = _ZeroProjection(1, output_channels[2])`.
6. `H_L2` may route only to native L2, one-step IDWT L1, or null. `H_L1` may route only to native L1, one-step IDWT L0, or null.
7. When `cross_level_enabled=false`, retain the same trust-amplitude computation but use native/shallow weights matching current independent behavior: L2 native=1; L1 native=1 and shallow=1. Do not use a softmax in this control path; expose its diagnostic row as `[1, 0, 0]` for L2 and `[1, 1, 0]` for L1 under the key `independent_route_weights_*`.
8. When `hard_all_null=true`, return exact correctly shaped zeros before DCT, Gabor, or projection calls.

Use these output equations in `forward`:

```python
native_l2 = gated_l2 * routes_l2[..., 0, None, None]
shallow_l2 = gated_l2 * routes_l2[..., 1, None, None]
native_l1 = gated_l1 * routes_l1[..., 0, None, None]
shallow_l1 = gated_l1 * routes_l1[..., 1, None, None]

l2 = self.projection_heads[0](native_l2)
l1 = self.projection_heads[1](native_l1)
l1 = l1 + self.l2_to_l1_projection(
    self.reconstruct_l0(_split_details(shallow_l2))
)
l0 = self.projection_heads[2](
    self.reconstruct_l0(_split_details(shallow_l1))
)
```

Return diagnostics with these stable keys:

```python
{
    "gates_l2": amplitude_l2,
    "gates_l1": amplitude_l1,
    "routes_l2": routes_l2,
    "routes_l1": routes_l1,
    "noise_reliability": noise,
    "gate_tv": 0.5 * (_total_variation(amplitude_l2) + _total_variation(amplitude_l1)),
    "route_temporal_smoothness": temporal_smoothness,
    "dct_weight_offset": dct_weight_offset,
    "dct_frequency_weights": dct_frequency_weights,
    "gabor_haar_agreement": gabor_haar_agreement,
    "gabor_dct_agreement": gabor_dct_agreement,
}
```

Compute `route_temporal_smoothness` by holding content descriptors fixed, replacing the normalized timestep/log-SNR scalars with those for `clamp(t+1, 0, T-1)`, re-evaluating only the small route heads, and averaging absolute probability differences. This isolates temporal smoothness from patient differences.

Export the class from `src/model/frequency/__init__.py`.

- [ ] **Step 4: Run all spectral-router tests**

```powershell
pixi run python -m pytest tests/test_spectral_router.py -q
```

Expected: 8 passed.

- [ ] **Step 5: Run legacy frequency tests**

```powershell
pixi run python -m pytest tests/test_reliable_frequency.py tests/test_residual_frequency.py -q
```

Expected: all existing tests pass unchanged.

- [ ] **Step 6: Commit the full router**

```powershell
git add src/model/frequency/spectral_router.py src/model/frequency/__init__.py tests/test_spectral_router.py
git commit -m "feat: route Haar packets with spectral evidence"
```

## Task 4: Model integration, diagnostics, and non-image regularization

**Files:**

- Modify: `src/model/priors/gabor.py`
- Create: `src/model/loss_terms/spectral_router.py`
- Modify: `src/model/loss_terms/__init__.py`
- Modify: `src/model/slmf_bbdm.py`
- Modify: `tests/test_residual_frequency.py`
- Modify: `tests/test_spectral_router.py`

- [ ] **Step 1: Write failing model-integration tests**

Add tests that start from `_residual_config(frequency=True, gabor=True)` and override:

```python
def _enable_v5(cfg):
    frequency = cfg["modules"]["residual_frequency"]
    frequency.update({
        "mode": "spectral_evidence_router",
        "dct_descriptor": {"enabled": True, "pooled_size": 8, "selected_frequencies": 12},
        "gabor_descriptor": {"enabled": True},
        "cross_level_router": {
            "enabled": True,
            "hard_all_null": False,
            "initial_null_probability": 0.90,
        },
    })
    cfg["losses"]["spectral_router_regularization"] = {
        "enabled": True,
        "weight": 1.0,
        "temporal_weight": 1e-4,
        "dct_weight": 1e-4,
        "gabor_weight": 1e-4,
    }
    return cfg


def test_model_constructs_v5_and_routes_only_inference_available_maps():
    from src.model.slmf_bbdm import SLMFBBDM
    from src.model.frequency.spectral_router import SpectralEvidenceFrequencyRouter

    model = SLMFBBDM.from_config(_enable_v5(_residual_config()))
    assert isinstance(model.residual_preconditioner, SpectralEvidenceFrequencyRouter)
    loss, logs = model(_fake_batch(), timesteps=torch.tensor([50]))
    assert torch.isfinite(loss)
    assert "frequency/route_l2_null" in logs
    assert "frequency/route_l1_null" in logs
    assert "loss/spectral_router_regularization/loss" in logs


def test_v5_requires_gabor_only_when_gabor_evidence_is_enabled():
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = _enable_v5(_residual_config(gabor=False))
    with pytest.raises(ValueError, match="Gabor evidence requires"):
        SLMFBBDM.from_config(cfg)
    cfg["modules"]["residual_frequency"]["gabor_descriptor"]["enabled"] = False
    SLMFBBDM.from_config(cfg)
```

Use the existing `_fake_batch` and `pytest` imports already present in `tests/test_residual_frequency.py`.

- [ ] **Step 2: Run integration tests and verify failure**

```powershell
pixi run python -m pytest tests/test_residual_frequency.py -k "v5 or spectral" -q
```

Expected: construction rejects the unknown mode or the router regularization loss.

- [ ] **Step 3: Add Gabor offset energy**

Add to `GaborPrior`:

```python
def parameter_offset_energy(self) -> torch.Tensor:
    trainable = (
        self.log_frequency,
        self.theta_raw,
        self.log_sigma,
        self.gamma_raw,
    )
    return torch.stack([parameter.square().mean() for parameter in trainable]).mean()
```

Add a focused test asserting it starts at zero, becomes positive after changing one raw parameter, and backpropagates.

- [ ] **Step 4: Implement router regularization loss**

Create `src/model/loss_terms/spectral_router.py`:

```python
"""Non-image regularization for spectral evidence routing."""

from __future__ import annotations

from typing import Dict

import torch

from ..interfaces import LossContext, LossTerm


class SpectralRouterRegularizationLoss(LossTerm):
    name = "spectral_router_regularization"

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        temporal_weight: float = 1e-4,
        dct_weight: float = 1e-4,
        gabor_weight: float = 1e-4,
    ) -> None:
        super().__init__(enabled=enabled, weight=weight)
        self.temporal_weight = float(temporal_weight)
        self.dct_weight = float(dct_weight)
        self.gabor_weight = float(gabor_weight)

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = ctx.target_pet.new_zeros(())
        if not self.enabled:
            return zero, {f"{self.name}/enabled": zero, f"{self.name}/loss": zero}
        temporal = ctx.condition.scalars.get("spectral_route_temporal_smoothness", zero).mean()
        dct = ctx.condition.scalars.get("spectral_dct_weight_offset", zero).mean()
        gabor = ctx.condition.scalars.get("spectral_gabor_parameter_offset", zero).mean()
        raw = self.temporal_weight * temporal + self.dct_weight * dct + self.gabor_weight * gabor
        return self.weight * raw, {
            f"{self.name}/enabled": torch.ones_like(zero),
            f"{self.name}/temporal": temporal.detach(),
            f"{self.name}/dct": dct.detach(),
            f"{self.name}/gabor": gabor.detach(),
            f"{self.name}/loss": raw.detach(),
        }
```

Export it from `src/model/loss_terms/__init__.py` and add a `_build_loss` branch in `SLMFBBDM` with all four configured weights.

- [ ] **Step 5: Integrate the new frequency mode**

In `src/model/slmf_bbdm.py`:

1. Extend allowed modes to `{"legacy", "boundary_reliable", "spectral_evidence_router"}`.
2. Read nested `dct_descriptor`, `gabor_descriptor`, and `cross_level_router` dictionaries.
3. Require `modules.gabor.enabled=true` only when `gabor_descriptor.enabled=true`.
4. Construct `SpectralEvidenceFrequencyRouter` in its own explicit branch; do not fold it into the `boundary_reliable` constructor.
5. Pass `gabor_feat`, `gabor_orientation`, and `gabor_anisotropy` only in this mode.
6. Copy differentiable router diagnostics into `condition.scalars` before `LossContext` is created:

```python
condition.scalars["spectral_route_temporal_smoothness"] = diagnostics[
    "route_temporal_smoothness"
]
condition.scalars["spectral_dct_weight_offset"] = diagnostics["dct_weight_offset"]
gabor_prior = self.priors.get("gabor")
condition.scalars["spectral_gabor_parameter_offset"] = (
    gabor_prior.parameter_offset_energy()
    if gabor_prior is not None and hasattr(gabor_prior, "parameter_offset_energy")
    else noisy_residual.new_zeros(())
)
```

7. Add detached scalar logs for all three destinations per level and band-averaged trust amplitudes. Stable keys must include `frequency/route_l2_native`, `frequency/route_l2_shallow`, `frequency/route_l2_null`, and the corresponding L1 keys.
8. Preserve the existing boundary-reliable logging branch exactly.

- [ ] **Step 6: Run focused model and loss tests**

```powershell
pixi run python -m pytest tests/test_spectral_router.py tests/test_residual_frequency.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit integration**

```powershell
git add src/model/priors/gabor.py src/model/loss_terms/spectral_router.py src/model/loss_terms/__init__.py src/model/slmf_bbdm.py tests/test_spectral_router.py tests/test_residual_frequency.py
git commit -m "feat: integrate spectral evidence router"
```

## Task 5: V5 base configuration and identifiable presets

**Files:**

- Create: `configs/experiments/slmf_png_spectral_router_v5.yaml`
- Modify: `configs/experiments/ablations.yaml`
- Modify: `tests/test_frequency_ablation_runner.py`

- [ ] **Step 1: Write failing config construction tests**

Parameterize S0-S3 and assert exact factors:

```python
@pytest.mark.parametrize(
    ("preset", "mode", "gabor", "dct"),
    [
        ("sr_v5_s0", "boundary_reliable", False, False),
        ("sr_v5_s1", "spectral_evidence_router", True, False),
        ("sr_v5_s2", "spectral_evidence_router", False, True),
        ("sr_v5_s3", "spectral_evidence_router", True, True),
    ],
)
def test_v5_stage_a_presets_construct(preset, mode, gabor, dct):
    from src.model.config_utils import load_config
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = load_config(
        "configs/experiments/slmf_png_spectral_router_v5.yaml",
        ablation=preset,
        ablation_config_path="configs/experiments/ablations.yaml",
        overrides=["data.image_size=32", "runtime.eval_sampling_steps=2"],
    )
    model = SLMFBBDM.from_config(cfg)
    assert model.residual_frequency_mode == mode
    if mode == "spectral_evidence_router":
        assert model.residual_preconditioner.gabor_enabled is gabor
        assert model.residual_preconditioner.dct_enabled is dct
        assert model.residual_preconditioner.cross_level_enabled is False
```

- [ ] **Step 2: Verify presets are absent**

```powershell
pixi run python -m pytest tests/test_frequency_ablation_runner.py -k v5 -q
```

Expected: config or preset not found.

- [ ] **Step 3: Create the V5 base config**

Copy `configs/experiments/slmf_png_boundary_reliable_v4.yaml` and change only:

- experiment name to `slmf_png_spectral_router_v5`;
- residual-frequency mode to `spectral_evidence_router`;
- DCT/Gabor/router nested settings exactly as the approved design;
- `losses.frequency_gate_tv.enabled=true`, weight `0.001`;
- `losses.spectral_router_regularization.enabled=true`, top-level weight `1.0`, component weights `1e-4`;
- retain frozen mean, Top-Q configuration, CT floors `(0.25, 0.50)`, all masks as loss-only, and all legacy Gabor routes false.

- [ ] **Step 4: Add four Stage-A presets**

Append `sr_v5_s0` through `sr_v5_s3` to `configs/experiments/ablations.yaml`:

```yaml
  sr_v5_s0:
    description: V5 reference reproducing G0 without spectral evidence routing
    overrides:
      modules.residual_frequency.mode: boundary_reliable
      modules.residual_frequency.use_gabor_agreement: false
      losses.spectral_router_regularization.enabled: false

  sr_v5_s1:
    description: Bounded learnable Gabor evidence with independent per-level injection
    overrides:
      modules.residual_frequency.mode: spectral_evidence_router
      modules.residual_frequency.dct_descriptor.enabled: false
      modules.residual_frequency.gabor_descriptor.enabled: true
      modules.residual_frequency.cross_level_router.enabled: false

  sr_v5_s2:
    description: Selected-DCT evidence with independent per-level injection
    overrides:
      modules.residual_frequency.mode: spectral_evidence_router
      modules.residual_frequency.dct_descriptor.enabled: true
      modules.residual_frequency.gabor_descriptor.enabled: false
      modules.residual_frequency.cross_level_router.enabled: false

  sr_v5_s3:
    description: Complementary Gabor-DCT evidence with independent per-level injection
    overrides:
      modules.residual_frequency.mode: spectral_evidence_router
      modules.residual_frequency.dct_descriptor.enabled: true
      modules.residual_frequency.gabor_descriptor.enabled: true
      modules.residual_frequency.cross_level_router.enabled: false
```

Every preset must also explicitly keep direct Gabor adapter/noise/hotspot/loss routes false and keep new boundary/image losses unchanged.

- [ ] **Step 5: Run construction and 32x32 real-forward tests**

```powershell
pixi run python -m pytest tests/test_frequency_ablation_runner.py -k v5 -q
pixi run python -m pytest tests/test_residual_frequency.py -q
```

Expected: all selected tests pass; every preset completes a finite forward.

- [ ] **Step 6: Commit configuration**

```powershell
git add configs/experiments/slmf_png_spectral_router_v5.yaml configs/experiments/ablations.yaml tests/test_frequency_ablation_runner.py
git commit -m "feat: add spectral router v5 presets"
```

## Task 6: Two-stage V5 ablation runner

**Files:**

- Create: `configs/experiments/spectral_router_ablation_plan_v5.yaml`
- Create: `scripts/run_spectral_router_v5.py`
- Modify: `tests/test_frequency_ablation_runner.py`
- Modify: `pixi.toml`

- [ ] **Step 1: Write failing runner tests**

Add tests for these contracts:

```python
def test_v5_plan_has_fixed_stages_and_pareto_gates():
    import yaml
    from pathlib import Path

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert plan["output_dir"] == "results/spectral_router_ablations_v5"
    assert [row["id"] for row in plan["stage_a"]["variants"]] == ["S0", "S1", "S2", "S3"]
    assert [row["id"] for row in plan["stage_b"]["variants"]] == ["N0", "T0", "C0", "C1"]
    assert plan["stage_a"]["top_k"] == 1
    assert plan["stage_b"]["top_k"] == 2
    assert plan["promote"]["epochs"] == 300
    assert plan["stage_b"]["hard_gates"]["ssim_mean"]["min_value"] == 0.947
    assert plan["stage_b"]["hard_gates"]["lesion_topq_peak_error_norm_mean"]["max_value"] == 0.080


def test_v5_stage_b_inherits_selected_evidence_and_keeps_routes_distinct():
    import yaml
    from pathlib import Path
    from scripts.run_spectral_router_v5 import build_stage_b_variants

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml").read_text(
            encoding="utf-8"
        )
    )
    rows = build_stage_b_variants(plan, selected_evidence_id="S3")
    by_id = {row["id"]: row for row in rows}
    assert by_id["N0"]["overrides"]["modules.residual_frequency.cross_level_router.hard_all_null"] is True
    assert by_id["T0"]["overrides"]["modules.residual_frequency.cross_level_router.enabled"] is False
    assert by_id["C0"]["overrides"]["modules.residual_frequency.dct_descriptor.enabled"] is False
    assert by_id["C0"]["overrides"]["modules.residual_frequency.gabor_descriptor.enabled"] is False
    assert by_id["C1"]["preset"] == "sr_v5_s3"
    assert by_id["C1"]["overrides"]["modules.residual_frequency.cross_level_router.enabled"] is True
    assert all(
        row["overrides"]["modules.residual_frequency.mode"] == "spectral_evidence_router"
        for row in rows
    )
```

Also assert the dry-run manifest contains 4 Stage-A runs, 4 Stage-B runs, at most 2 promotion runs, V5-only output paths, fixed initialization seed 4242, frozen mean checkpoint, 64 evaluation samples, seed 42, 20 MC steps, and exact 300-epoch final checkpoint requirements.

- [ ] **Step 2: Run runner tests and verify missing files**

```powershell
pixi run python -m pytest tests/test_frequency_ablation_runner.py -k v5 -q
```

Expected: missing plan/runner failures.

- [ ] **Step 3: Create the V5 plan YAML**

Use:

```yaml
base_config: configs/experiments/slmf_png_spectral_router_v5.yaml
ablation_config: configs/experiments/ablations.yaml
output_dir: results/spectral_router_ablations_v5

mean_pretrain:
  enabled: false
  checkpoint: checkpoints/freq_mean_pretrain_v2/mean_best.pt

common_train_overrides:
  model.initialization_seed: 4242
  modules.conditional_mean.checkpoint: checkpoints/freq_mean_pretrain_v2/mean_best.pt
  modules.conditional_mean.freeze: true
  modules.conditional_mean.loss_weight: 0.0

stage_a:
  reference_id: S0
  top_k: 1
  dry_run_selected_evidence: S3
  experiment_prefix: sr_v5_stage_a
  epochs: 50
  eval_interval: 10
  split: val
  max_samples: 64
  seed: 42
  mc_steps: 20
  early_stopping: false
  variants:
    - {id: S0, preset: sr_v5_s0}
    - {id: S1, preset: sr_v5_s1}
    - {id: S2, preset: sr_v5_s2}
    - {id: S3, preset: sr_v5_s3}
  hard_gates:
    lesion_topq_peak_error_norm_mean:
      direction: lower
      max_value: 0.15
    lesion_peak_error_norm_mean:
      direction: lower
      max_value: 0.20
    failure_any_mean:
      direction: lower
      max_value: 0.50
      max_delta: 0.0625
    false_hotspot_density_mean:
      direction: lower
      max_ratio: 1.25
      epsilon: 1.0e-6
    stripe_excess_mean:
      direction: lower
      max_value: 0.20
      max_delta: 0.05
    ssim_mean:
      direction: higher
      min_value: 0.85
      max_delta: 0.03

stage_b:
  reference_id: T0
  top_k: 2
  experiment_prefix: sr_v5_stage_b
  epochs: 50
  eval_interval: 10
  split: val
  max_samples: 64
  seed: 42
  mc_steps: 20
  early_stopping: false
  variants:
    - id: N0
      overrides:
        modules.residual_frequency.mode: spectral_evidence_router
        modules.residual_frequency.cross_level_router.enabled: true
        modules.residual_frequency.cross_level_router.hard_all_null: true
    - id: T0
      overrides:
        modules.residual_frequency.mode: spectral_evidence_router
        modules.residual_frequency.cross_level_router.enabled: false
        modules.residual_frequency.cross_level_router.hard_all_null: false
    - id: C0
      overrides:
        modules.residual_frequency.mode: spectral_evidence_router
        modules.residual_frequency.dct_descriptor.enabled: false
        modules.residual_frequency.gabor_descriptor.enabled: false
        modules.residual_frequency.cross_level_router.enabled: true
        modules.residual_frequency.cross_level_router.hard_all_null: false
    - id: C1
      overrides:
        modules.residual_frequency.mode: spectral_evidence_router
        modules.residual_frequency.cross_level_router.enabled: true
        modules.residual_frequency.cross_level_router.hard_all_null: false
  hard_gates:
    lesion_topq_peak_error_norm_mean:
      direction: lower
      max_value: 0.080
    lesion_peak_error_norm_mean:
      direction: lower
      max_value: 0.20
    lesion_centroid_distance_mean:
      direction: lower
      max_value: 3.15
    directional_spectrum_error_norm_mean:
      direction: lower
      max_value: 0.0095
    failure_any_mean:
      direction: lower
      max_value: 0.0625
      max_delta: 0.0625
    false_hotspot_density_mean:
      direction: lower
      max_value: 0.00014
      max_ratio: 1.25
      epsilon: 1.0e-6
    stripe_excess_mean:
      direction: lower
      max_value: 0.20
      max_delta: 0.05
    mae_mean:
      direction: lower
      max_value: 0.0365
    ssim_mean:
      direction: higher
      min_value: 0.947
      max_delta: 0.03

promote:
  experiment_prefix: sr_v5_full
  epochs: 300
  eval_interval: 20
  split: val
  max_samples: 64
  seed: 42
  mc_steps: 20
  early_stopping: false
  require_final_checkpoint: true

paired_comparison:
  seed: 42
  resamples: 10000
  metrics:
    lesion_topq_peak_error_norm_mean: lower
    lesion_peak_error_norm_mean: lower
    lesion_mean_error_norm_mean: lower
    lesion_centroid_distance_mean: lower
    lesion_boundary_gradient_mae_norm_mean: lower
    anatomy_edge_gradient_mae_norm_mean: lower
    directional_spectrum_error_norm_mean: lower
    failure_any_mean: lower
    false_hotspot_density_mean: lower
    stripe_excess_mean: lower
    mae_mean: lower
    ssim_mean: higher
```

- [ ] **Step 4: Implement the V5 runner**

Copy the proven orchestration structure from `scripts/run_boundary_reliable_v4.py`, rename public helpers to `build_stage_b_variants` and `build_v5_dry_run_manifest`, and apply these exact differences:

1. Stage A winner is called `selected_evidence_id`.
2. Stage B inherits the selected Stage-A preset for N0, T0, and C1, then explicitly overrides `modules.residual_frequency.mode=spectral_evidence_router` for every N0/T0/C0/C1 variant. This prevents S0's legacy control mode from silently bypassing the Stage-B route comparison.
3. C0 still inherits the selected preset but explicitly zeros both evidence switches, so all non-routing settings remain paired.
4. N0 hard-null short-circuit remains trainable only as a frequency-off safety control and is never automatically promoted unless it passes the same gates and score ranking.
5. Promotion is capped at two variants and requires the epoch-300 checkpoint.
6. Final paired comparisons include every promoted pair and comparisons against the Stage-B T0 reference result when it exists.

Use existing `_run_manifest_entry`, `_run_entries`, `_write_rankings`, `passes_hard_gates`, `composite_score`, and `compare_all_results`; do not duplicate those implementations.

- [ ] **Step 5: Add Pixi commands**

Append under `[tasks]`:

```toml
train-spectral-router-v5 = "python scripts/run_spectral_router_v5.py --plan configs/experiments/spectral_router_ablation_plan_v5.yaml --stage all"
dry-run-spectral-router-v5 = "python scripts/run_spectral_router_v5.py --plan configs/experiments/spectral_router_ablation_plan_v5.yaml --stage all --dry-run"
```

- [ ] **Step 6: Run runner tests and inspect the manifest**

```powershell
pixi run python -m pytest tests/test_frequency_ablation_runner.py -k v5 -q
pixi run dry-run-spectral-router-v5
Get-Content -Raw 'results\spectral_router_ablations_v5\dry_run_manifest.json'
```

Expected: tests pass; manifest lists 4+4+2 runs, never references V3/V4 output directories, uses the frozen mean, and requires epoch 300 for promotion completion.

- [ ] **Step 7: Commit orchestration**

```powershell
git add configs/experiments/spectral_router_ablation_plan_v5.yaml scripts/run_spectral_router_v5.py tests/test_frequency_ablation_runner.py pixi.toml
git commit -m "feat: orchestrate spectral router v5 ablations"
```

## Task 7: Integrated verification and cloud handoff

**Files:**

- Modify only if verification exposes a V5 defect in files already listed above.

- [ ] **Step 1: Run focused unit and integration tests**

```powershell
pixi run python -m pytest tests/test_spectral_router.py tests/test_reliable_frequency.py tests/test_residual_frequency.py tests/test_frequency_ablation_runner.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete suite**

```powershell
pixi run python -m pytest tests -q
```

Expected: full suite passes with no new warnings attributable to V5.

- [ ] **Step 3: Verify exact fallback numerically**

Run a deterministic 32x32 fake batch twice with identical model initialization: once with residual frequency disabled and once with V5 hard all-null. Compare model outputs and total loss before an optimizer step with `torch.equal`, not a tolerance. Expected: exact equality.

- [ ] **Step 4: Verify descriptor and routing safety**

Check from one finite forward/backward:

- no target PET or masks appear in router call arguments;
- every route row sums to one;
- all trust amplitudes lie in `[0, gate_max]`;
- L3 is exact zero;
- all-null bypass skips projections;
- DCT weights sum to one;
- Gabor parameter offsets remain within existing bounded parameterization;
- no signed Gabor or DCT reconstruction enters decoder injections;
- projection, route, DCT-weight, and Gabor parameters receive finite gradients after the zero-head warm-up behavior.

- [ ] **Step 5: Audit repository state**

```powershell
git diff --check
git status --short
git log -8 --oneline
```

Expected: no whitespace errors; only intentional user-owned pre-existing changes remain uncommitted; V5 work is split into the planned focused commits.

- [ ] **Step 6: Produce the cloud command and sync list**

Cloud command:

```powershell
pixi run train-spectral-router-v5
```

Sync the new/modified source, test, config, runner, Pixi, lockfile only if it changed, and the frozen mean checkpoint. Do not sync local V3/V4 result directories as training inputs.

- [ ] **Step 7: Commit any verification-only correction**

If Step 1-5 required a code correction, stage only the corrected V5 files and commit:

```powershell
git commit -m "fix: close spectral router verification gaps"
```

If no correction was required, do not create an empty commit.

## Completion criteria

- The first enabled V5 forward is an exact zero residual through zero-initialized projections.
- Hard all-null mode short-circuits and exactly matches frequency-disabled behavior.
- Training initialization remains null-biased but has nonzero active-route probability.
- Haar packets route only to native, one adjacent shallower, or null destinations.
- Gabor and DCT are evidence-only and cannot inject frequency carriers.
- S0-S3 and N0/T0/C0/C1 are identifiable, paired, fixed-budget experiments.
- At most two candidates train for exactly 300 epochs.
- Final output includes hard-gate decisions, route diagnostics, and paired 10,000-resample comparisons.
