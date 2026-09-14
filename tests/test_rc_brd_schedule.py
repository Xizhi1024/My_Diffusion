"""RC-BRD bandwise schedule tests (DESIGN §11 / PRD FR-3; C1, C7).

Covers: κ=0 identity, κ≠0 contract requirement, monotonicity/endpoints,
both forward modes smoke, clock boundedness ρ∈[e^{−|κ|η_max}, e^{+|κ|η_max}]
for both κ signs (AUDIT 5 §3.1 corrected form / DESIGN §4 v1.0g), vp
non-equivalence at κ=0 (PRD v1.0.2: only m≡u, marginal ≠ bridge),
forward_transition closed form (v1.0g), vp posterior/transition guards,
finite-domain config validation (v1.0g), endpoint fail-closed validation.
CPU-only, deterministic (explicit torch.Generator seeds).
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from dataclasses import dataclass, field

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.rc_brd.schedule import (
    ENDPOINT_MODES,
    FORWARD_MODES,
    BandwiseBridgeSchedule,
    BandwiseScheduleConfig,
)
from src.model.rc_brd.wavelet import band_groups, haar_forward2


@dataclass(frozen=True)
class _StubContract:
    """Duck-typed stand-in for RecoverabilityContract (batch 1B owns contract.py).

    Provides exactly the runtime surface schedule.py relies on:
    effective_c(group, log_snr)->Tensor, eta_max, band_groups, log_snr_grid,
    b_active.  c(λ)=0.5+0.2·tanh(λ/10)∈(0.3,0.7), clipped to [0,1], then
    shrunk toward 0.5 ([审计] §3.3): c̃=(1−η)·0.5+η·c.
    """

    eta_max: float = 0.8
    band_groups: dict = field(default_factory=lambda: band_groups(3))
    log_snr_grid: tuple = (-10.0, 0.0, 10.0)
    b_active: tuple = ("low", "mid", "high")

    def effective_c(self, group: str, log_snr: torch.Tensor) -> torch.Tensor:
        if group not in self.band_groups:
            raise KeyError(group)
        lam = log_snr.to(dtype=torch.float64)
        c = (0.5 + 0.2 * torch.tanh(lam / 10.0)).clamp(0.0, 1.0)
        return (1.0 - self.eta_max) * 0.5 + self.eta_max * c


def _identity_residual(seq: torch.Tensor) -> float:
    num_t = seq.numel() - 1
    u = torch.arange(num_t + 1, dtype=torch.float64) / num_t
    return (seq - u).abs().max().item()


def _random_bands(batch: int = 2, size: int = 32, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    return haar_forward2(torch.randn(batch, 1, size, size, generator=gen))


def _random_eps(bands, seed: int):
    gen = torch.Generator().manual_seed(seed)
    return {b: torch.randn(v.shape, generator=gen) for b, v in bands.items()}


def test_mode_constants():
    assert FORWARD_MODES == ("bridge_time_changed", "vp_bandwise")
    assert ENDPOINT_MODES == ("zeros", "ct_minus_mean")


def test_kappa0_identity_without_contract():
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=100))
    for group in ("low", "mid", "high"):
        seq = sched.m_sequence(group)
        assert seq.dtype == torch.float64
        assert seq.device.type == "cpu"
        assert seq.numel() == 101
        assert _identity_residual(seq) < 1e-12   # m[t] == t/T exactly
        assert seq[0].item() == 0.0
        assert seq[-1].item() == 1.0


def test_kappa0_identity_ignores_contract():
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=50), contract=_StubContract()
    )
    for group in ("low", "mid", "high"):
        assert _identity_residual(sched.m_sequence(group)) < 1e-12


def test_kappa_nonzero_without_contract_raises():
    with pytest.raises(ValueError):
        BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=50, kappa=0.5))


def test_m_monotone_and_endpoints_with_contract():
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=1.0), contract=_StubContract()
    )
    u = torch.arange(101, dtype=torch.float64) / 100
    for group in ("low", "mid", "high"):
        seq = sched.m_sequence(group)
        assert bool((seq[1:] - seq[:-1] > 0).all())   # strictly monotone
        assert seq[0].item() == 0.0                   # m[0] = 0
        assert seq[-1].item() == 1.0                  # m[T] = 1
        assert (seq - u).abs().max().item() > 1e-3    # clock genuinely warped


@pytest.mark.parametrize("kappa", [1.0, -1.0])
def test_clock_bounded_by_exp_abs_kappa_eta(kappa):
    # [AUDIT 5 §3.1 corrected form / DESIGN §4 v1.0g]: after the [审计] §3.3
    # shrink c̃=(1−η)·0.5+η·c with c∈[0,1], one has 2c̃−1∈[−η,η] pointwise, so
    # κ(2c̃−1)∈[−|κ|·η, |κ|·η] REGARDLESS of the sign of κ, i.e.
    # ρ(u)=exp{κ(2c̃(u)−1)} ∈ [ρ_min, ρ_max]=[e^{−|κ|η_max}, e^{+|κ|η_max}].
    # Observable proxy — the normalized clock m(u)=A/(A+C) with A=∫₀ᵘρ,
    # C=∫ᵤ¹ρ: segment-wise ρ bounds give
    #   A ≥ u·ρ_min, C ≤ (1−u)·ρ_max ⟹ m(u) ≥ u/(u+r(1−u)),
    #   A ≤ u·ρ_max, C ≥ (1−u)·ρ_min ⟹ m(u) ≤ r·u/(r·u+(1−u)),
    # with r=ρ_max/ρ_min=e^{2|κ|η_max}.  The bounds hold for the trapezoid
    # rule too: every increment 0.5(ρ_i+ρ_{i+1})/N lies in [ρ_min/N, ρ_max/N].
    eta = 0.8
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=kappa),
        contract=_StubContract(eta_max=eta),
    )
    r = math.exp(2.0 * abs(kappa) * eta)
    u = torch.arange(101, dtype=torch.float64) / 100
    lower = u / (u + r * (1.0 - u))
    upper = (r * u) / (r * u + (1.0 - u))
    for group in ("low", "mid", "high"):
        seq = sched.m_sequence(group)
        assert bool((seq >= lower - 1e-12).all())   # ρ ≥ e^{−|κ|η_max}
        assert bool((seq <= upper + 1e-12).all())   # ρ ≤ e^{+|κ|η_max}
        assert (seq - u).abs().max().item() > 1e-3  # clock genuinely warped


@dataclass(frozen=True)
class _GridStubContract:
    """3-knot grid contract with a true piecewise-linear effective_c.

    Mirrors the real contract semantics ([计划] §3.5): interpolate on the λ
    grid → endpoint clamp → [审计] §3.3 shrink c̃=(1−η)·0.5+η·c.  Non-uniform
    c values keep a genuine interior kink at λ=0.
    """

    eta_max: float = 0.8
    band_groups: dict = field(default_factory=lambda: band_groups(3))
    log_snr_grid: tuple = (-10.0, 0.0, 10.0)
    b_active: tuple = ("low", "mid", "high")
    c_grid: dict = field(default_factory=lambda: {
        "low": (0.1, 0.25, 0.45),
        "mid": (0.2, 0.55, 0.6),
        "high": (0.55, 0.7, 0.9),
    })

    def effective_c(self, group: str, log_snr: torch.Tensor) -> torch.Tensor:
        if group not in self.band_groups:
            raise KeyError(group)
        lam = log_snr.to(dtype=torch.float64)
        knots = torch.tensor(self.log_snr_grid, dtype=torch.float64)
        values = torch.tensor(self.c_grid[group], dtype=torch.float64)
        idx = torch.searchsorted(knots, lam).clamp(1, knots.numel() - 1)
        lo, hi = knots[idx - 1], knots[idx]
        w = ((lam - lo) / (hi - lo)).clamp(0.0, 1.0)  # endpoint clamp
        c = values[idx - 1] * (1.0 - w) + values[idx] * w
        return (1.0 - self.eta_max) * 0.5 + self.eta_max * c


def _hand_clock(contract, group, subdivisions, kappa, sigma_bridge):
    """Independent reference clock under the [计划] v1.5 §3.6 lookup.

    λ₀(u)=log((1−u)/(ν²·u)) with ν²=2·sigma_bridge², clipped to the config
    λ endpoints; trapezoid integration on a subdivisions-per-step grid of
    u=t/100, t=0..100 (num_timesteps fixed at 100 for these checks).
    """
    num_t = 100
    nodes = num_t * subdivisions
    u = torch.arange(nodes + 1, dtype=torch.float64) / nodes
    nu2 = 2.0 * sigma_bridge ** 2
    lam = torch.log((1.0 - u) / (nu2 * u)).clamp(-10.0, 10.0)
    c_tilde = contract.effective_c(group, lam)
    rho = torch.exp(kappa * (2.0 * c_tilde - 1.0))
    inc = 0.5 * (rho[:-1] + rho[1:]) / nodes
    cum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(inc, dim=0)])
    return (cum / cum[-1])[::subdivisions]


def test_m_sequence_matches_hand_integration_v1_5_lookup():
    # [计划] v1.5 §3.6 / DESIGN v1.0f: c̃ is queried at the bridge log-SNR
    # λ₀(u)=log((1−u)/(ν²·u)), not at the deprecated v1.4 linear λ warp.
    contract = _GridStubContract()
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=0.5), contract=contract
    )
    m = sched.m_sequence("mid")
    ref_same_grid = _hand_clock(contract, "mid", 4, 0.5, 1.0)
    assert torch.allclose(m, ref_same_grid, rtol=0.0, atol=1e-14)
    # Independent denser grid (40 evals/step) with the smooth tanh contract.
    # The 4-subdiv grid carries a measured ~3.3e-5 bias vs the continuum
    # (accumulated in the clipped-λ boundary layer at u→0 and spread by the
    # normalization), so the cross-grid tolerance is 1e-4; a wrong lookup
    # convention would deviate by ~3e-2, 300x this budget.
    smooth = _StubContract()
    sched_smooth = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=0.5), contract=smooth
    )
    ref_dense = _hand_clock(smooth, "mid", 40, 0.5, 1.0)
    assert torch.allclose(sched_smooth.m_sequence("mid"), ref_dense, rtol=1e-5, atol=1e-4)
    # σ enters the lookup through ν²: a different sigma_bridge shifts the clock
    sched_s3 = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=0.5, sigma_bridge=3.0),
        contract=contract,
    )
    ref_s3 = _hand_clock(contract, "mid", 4, 0.5, 3.0)
    assert torch.allclose(sched_s3.m_sequence("mid"), ref_s3, rtol=0.0, atol=1e-14)
    assert (m - ref_s3).abs().max().item() > 1e-6  # ν² dependence is real


def test_clock_lookup_departs_from_deprecated_linear_warp():
    # Behavioral guard: the v1.4 warp λ(u)=λ_max+(λ_min−λ_max)·u would give a
    # materially different clock; the new convention must not match it.
    contract = _GridStubContract()
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=0.5), contract=contract
    )
    m_new = sched.m_sequence("mid")
    nodes = 400
    u = torch.arange(nodes + 1, dtype=torch.float64) / nodes
    lam_old = 10.0 + (-10.0 - 10.0) * u  # deprecated v1.4 warp
    c_tilde = contract.effective_c("mid", lam_old)
    rho = torch.exp(0.5 * (2.0 * c_tilde - 1.0))
    inc = 0.5 * (rho[:-1] + rho[1:]) / nodes
    cum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(inc, dim=0)])
    m_old = (cum / cum[-1])[::4]
    assert (m_new - m_old).abs().max().item() > 1e-3


def test_deprecated_linear_warp_absent_from_clock_path():
    # No dead path: _compute_m_sequence must contain the v1.5 log-SNR lookup
    # and no residual v1.4 linear warp (which legitimately survives only in
    # _alpha_hat_values for the vp ᾱ endpoint interpolation).
    src = inspect.getsource(BandwiseBridgeSchedule._compute_m_sequence)
    assert "torch.log" in src
    assert "(self.config.lambda_min - self.config.lambda_max)" not in src


def test_unknown_group_raises_keyerror():
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=10, kappa=1.0), contract=_StubContract()
    )
    with pytest.raises(KeyError):
        sched.m_sequence("nonexistent")


def test_numerical_monotonicity_violation_raises():
    # κ=5000 overflows exp on the fine grid → non-finite clock → ValueError.
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=5000.0),
        contract=_StubContract(),
    )
    with pytest.raises(ValueError):
        sched.m_sequence("mid")


def test_bridge_forward_smoke_no_nan():
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=1.0), contract=_StubContract()
    )
    bands = _random_bands(batch=4, seed=11)
    eps = _random_eps(bands, seed=12)
    t = torch.tensor([0, 33, 77, 100], dtype=torch.long)
    z_t, target = sched.q_marginal(bands, t, eps)
    for band, z0 in bands.items():
        assert torch.isfinite(z_t[band]).all()
        assert z_t[band].shape == z0.shape
        assert target[band] is z0          # pred_x0 target aliases z0
    # bridge endpoint t=T is pinned: z_T == e == 0 exactly
    assert torch.equal(z_t["LL2"][3], torch.zeros_like(z_t["LL2"][3]))


def test_vp_forward_smoke_no_nan_and_alpha_hat():
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, forward_mode="vp_bandwise", kappa=1.0),
        contract=_StubContract(),
    )
    bands = _random_bands(batch=3, seed=13)
    eps = _random_eps(bands, seed=14)
    t = torch.tensor([0, 50, 100], dtype=torch.long)
    z_t, _ = sched.q_marginal(bands, t, eps)
    for band in bands:
        assert torch.isfinite(z_t[band]).all()
    alpha = sched.alpha_hat("mid", torch.arange(101, dtype=torch.long))
    assert bool((alpha > 0).all()) and bool((alpha < 1).all())
    assert bool((alpha[1:] <= alpha[:-1]).all())     # ᾱ non-increasing in t
    # κ=0 vp mode also degenerates to the identity clock (m≡u only — the
    # VP marginal itself never equals the bridge marginal; see the reverse
    # assertion in test_vp_kappa0_clock_identity_but_marginal_differs).
    sched0 = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, forward_mode="vp_bandwise")
    )
    assert _identity_residual(sched0.m_sequence("mid")) < 1e-12


def test_vp_kappa0_clock_identity_but_marginal_differs_from_bridge():
    # [PRD v1.0.2-1 / AUDIT 5 triage X item] κ=0 forces only m≡u: the VP
    # marginal √ᾱ·z0+√(1−ᾱ)·ε is NOT the bridge marginal even under the
    # same identity clock.  The vp arm is a non-equivalent exploration
    # switch — reverse assertion (any test claiming scalar/bridge equivalence
    # of the vp arm at κ=0 would be wrong and must not exist).
    bridge = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=100))
    vp = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, forward_mode="vp_bandwise")
    )
    gen = torch.Generator().manual_seed(30)
    bands = {"LH1": torch.randn(2, 1, 4, 4, generator=gen)}
    eps = {b: torch.randn(v.shape, generator=gen) for b, v in bands.items()}
    t = torch.tensor([30, 70], dtype=torch.long)
    assert _identity_residual(vp.m_sequence("LH1")) < 1e-12  # clock degenerates
    z_bridge, _ = bridge.q_marginal(bands, t, eps)
    z_vp, _ = vp.q_marginal(bands, t, eps)
    delta = (z_bridge["LH1"] - z_vp["LH1"]).abs().max().item()
    assert delta > 1e-2  # genuinely different laws (coefficients differ at O(0.1))


def test_vp_posterior_raises_not_implemented():
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=20, forward_mode="vp_bandwise"),
    )
    bands = _random_bands(size=8, seed=15)
    with pytest.raises(NotImplementedError):
        sched.posterior(bands, bands, torch.tensor([1]), torch.tensor([2]))


# ---- forward_transition ([审计] §3.2 / DESIGN §4 v1.0g) --------------------


def test_forward_transition_hand_formula_zeros_endpoint():
    # q(z_t|z_s,z0,e) with e=0 ([审计] §3.2): mean=g·z_s with
    # g=(1−m_t)/(1−m_s) — z0 cancels analytically (bridge Markov property
    # in m-time) — and var=2σ²(m_t−m_s)(1−m_t)/(1−m_s).
    sigma = 1.5
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, sigma_bridge=sigma)
    )
    gen = torch.Generator().manual_seed(31)
    z_s = {"LH1": torch.randn(2, 1, 4, 4, generator=gen, dtype=torch.float64)}
    t_s = torch.tensor([20, 40], dtype=torch.long)
    t = torch.tensor([60, 90], dtype=torch.long)
    mean_b, var_b = sched.forward_transition(z_s, t_s, t)
    m = sched.m_sequence("LH1")
    m_s, m_t = m[t_s], m[t]  # [B] float64; var is per-sample like posterior()'s
    gain = (1.0 - m_t) / (1.0 - m_s)
    expected_mean = gain.view(-1, 1, 1, 1) * z_s["LH1"]
    expected_var = 2.0 * sigma ** 2 * (m_t - m_s) * (1.0 - m_t) / (1.0 - m_s)
    assert torch.allclose(mean_b["LH1"], expected_mean, rtol=1e-8, atol=1e-10)
    assert torch.allclose(var_b["LH1"], expected_var, rtol=1e-8, atol=1e-10)


def test_forward_transition_ct_minus_mean_shifts_toward_endpoint():
    # Endpoint semantics identical to q_marginal: mean=g·z_s+(1−g)·e, an
    # affine interpolation of z_s toward the fixed endpoint as t→T.
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, endpoint_mode="ct_minus_mean")
    )
    gen = torch.Generator().manual_seed(32)
    z_s = {"HL2": torch.randn(1, 1, 4, 4, generator=gen)}
    endpoints = {"HL2": torch.randn(1, 1, 4, 4, generator=gen)}
    t_s = torch.tensor([30], dtype=torch.long)
    t = torch.tensor([80], dtype=torch.long)
    mean_b, _ = sched.forward_transition(z_s, t_s, t, endpoint_bands=endpoints)
    m = sched.m_sequence("HL2")
    gain = (1.0 - m[t].item()) / (1.0 - m[t_s].item())
    expected = gain * z_s["HL2"] + (1.0 - gain) * endpoints["HL2"]
    assert torch.allclose(mean_b["HL2"], expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):     # missing endpoint_bands → fail-closed
        sched.forward_transition(z_s, t_s, t)


def test_forward_transition_t_s_equals_t_is_identity():
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=50))
    gen = torch.Generator().manual_seed(33)
    z_s = {"HH1": torch.randn(3, 1, 4, 4, generator=gen)}
    tv = torch.tensor([5, 25, 49], dtype=torch.long)
    mean_b, var_b = sched.forward_transition(z_s, tv, tv)
    assert torch.equal(mean_b["HH1"], z_s["HH1"].to(mean_b["HH1"].dtype))
    assert bool((var_b["HH1"] == 0).all())


def test_forward_transition_vp_and_order_validation():
    vp = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=20, forward_mode="vp_bandwise")
    )
    bands = {"LH1": torch.randn(1, 1, 4, 4)}
    with pytest.raises(NotImplementedError):     # C1 / PRD v1.0.2
        vp.forward_transition(bands, torch.tensor([1]), torch.tensor([2]))
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=20))
    with pytest.raises(ValueError):              # t_s > t
        sched.forward_transition(bands, torch.tensor([5]), torch.tensor([2]))
    bad = {"XX": torch.randn(1, 1, 4, 4)}
    with pytest.raises(ValueError):              # unknown band
        sched.forward_transition(bad, torch.tensor([1]), torch.tensor([2]))


def test_forward_transition_from_t_zero_equals_q_marginal_moments():
    # s=0: z_s≡z0 is pinned by the marginal itself, so q(z_t|z_0) must
    # reproduce the q_marginal moments exactly (Markov consistency anchor).
    for endpoint_mode in ("zeros", "ct_minus_mean"):
        sched = BandwiseBridgeSchedule(
            BandwiseScheduleConfig(num_timesteps=100, endpoint_mode=endpoint_mode)
        )
        gen = torch.Generator().manual_seed(34)
        z0 = {"LL2": torch.randn(2, 1, 4, 4, generator=gen)}
        e_bands = {"LL2": torch.randn(2, 1, 4, 4, generator=gen)} if endpoint_mode == "ct_minus_mean" else None
        zero_eps = {b: torch.zeros_like(v) for b, v in z0.items()}
        t = torch.tensor([35, 85], dtype=torch.long)
        t0 = torch.zeros_like(t)
        mean_b, var_b = sched.forward_transition(z0, t0, t, endpoint_bands=e_bands)
        z_ref, _ = sched.q_marginal(z0, t, zero_eps, endpoint_bands=e_bands)
        # zero-eps q_marginal == its mean; var from the marginal formula.
        assert torch.allclose(mean_b["LL2"], z_ref["LL2"], rtol=1e-6, atol=1e-7)
        m = sched.m_sequence("LL2")[t].to(z0["LL2"].dtype)   # [B], like var
        assert torch.allclose(var_b["LL2"], 2.0 * m * (1.0 - m), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward_mode": "nope"},
        {"endpoint_mode": "nope"},
        {"num_timesteps": 0},
        {"num_timesteps": -5},
        {"sigma_bridge": 0.0},
        {"lambda_min": 10.0, "lambda_max": -10.0},
        {"lambda_min": 0.0, "lambda_max": 0.0},
        # [DESIGN §4 v1.0g / AUDIT 5 §3.3] finite-domain violations: NaN/±inf
        # must fail closed at construction, not poison the clock integral.
        {"kappa": float("nan")},
        {"kappa": float("inf")},
        {"kappa": float("-inf")},
        {"sigma_bridge": float("inf")},
        {"sigma_bridge": float("nan")},
        {"lambda_min": float("-inf")},
        {"lambda_min": float("nan")},
        {"lambda_max": float("inf")},
        {"lambda_max": float("nan")},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        BandwiseScheduleConfig(**kwargs)


def test_finite_domain_config_accepts_negative_kappa():
    # The finite-domain gate must reject only NaN/±inf: a finite negative κ
    # (with a contract) stays constructible and yields a valid clock.
    cfg = BandwiseScheduleConfig(num_timesteps=50, kappa=-0.75)
    assert cfg.kappa == -0.75
    sched = BandwiseBridgeSchedule(cfg, contract=_StubContract())
    seq = sched.m_sequence("mid")
    assert seq[0].item() == 0.0
    assert seq[-1].item() == 1.0
    assert bool((seq[1:] - seq[:-1] > 0).all())


def test_q_marginal_unknown_band_raises():
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=10))
    bad_bands = {"XX": torch.randn(1, 1, 4, 4)}
    with pytest.raises(ValueError):
        sched.q_marginal(bad_bands, torch.tensor([0], dtype=torch.long))


def test_timestep_validation():
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=10))
    bands = {"LL2": torch.randn(1, 1, 4, 4)}
    with pytest.raises(ValueError):
        sched.q_marginal(bands, torch.tensor([11], dtype=torch.long))
    with pytest.raises(ValueError):
        sched.q_marginal(bands, torch.tensor([-1], dtype=torch.long))
    with pytest.raises(ValueError):
        sched.q_marginal(bands, torch.tensor([1.0]))          # wrong dtype
    with pytest.raises(ValueError):
        sched.q_marginal(bands, torch.tensor([[1]], dtype=torch.long))  # not 1-D


def test_endpoint_ct_minus_mean_validation():
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=50, endpoint_mode="ct_minus_mean")
    )
    bands = _random_bands(seed=16)
    eps = _random_eps(bands, seed=17)
    t = torch.tensor([10, 40], dtype=torch.long)
    with pytest.raises(ValueError):     # endpoint_bands missing entirely
        sched.q_marginal(bands, t, eps)
    partial = {k: v for k, v in bands.items() if k != "HH1"}
    with pytest.raises(ValueError):     # missing band key
        sched.q_marginal(bands, t, eps, endpoint_bands=partial)
    gen = torch.Generator().manual_seed(18)
    image_like = {b: torch.randn(2, 1, 32, 32, generator=gen) for b in bands}
    with pytest.raises(ValueError, match="image-domain"):  # μ/CT-style tensor
        sched.q_marginal(bands, t, eps, endpoint_bands=image_like)
    wrong_shape = {b: torch.randn(2, 1, 3, 3, generator=gen) for b in bands}
    with pytest.raises(ValueError):     # generic shape mismatch
        sched.q_marginal(bands, t, eps, endpoint_bands=wrong_shape)


def test_endpoint_ct_minus_mean_mean_shift():
    # With eps=0 the bridge marginal is exactly (1−m_t)·z0 + m_t·e.
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=50, endpoint_mode="ct_minus_mean")
    )
    bands = _random_bands(seed=19)
    endpoints = _random_bands(seed=20)
    t = torch.tensor([10, 40], dtype=torch.long)
    zero_eps = {b: torch.zeros_like(v) for b, v in bands.items()}
    z_t, _ = sched.q_marginal(bands, t, zero_eps, endpoint_bands=endpoints)
    m = sched.m_sequence("mid")  # κ=0 → identity clock shared by all groups
    for band, z0 in bands.items():
        m_view = m[t].to(z0.dtype).view(-1, 1, 1, 1)
        expected = (1.0 - m_view) * z0 + m_view * endpoints[band]
        assert torch.allclose(z_t[band], expected, rtol=1e-5, atol=1e-6)


def test_bridge_step_endpoint_coupling():
    # [DESIGN v1.0a] 'ct_minus_mean' requires endpoint_bands; 'zeros' forbids it.
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=50, endpoint_mode="ct_minus_mean")
    )
    bands = _random_bands(size=8, seed=21)
    with pytest.raises(ValueError):     # endpoint_bands missing
        sched.step_from_prediction(
            bands, bands, torch.tensor([1], dtype=torch.long), torch.tensor([2], dtype=torch.long)
        )
    sched_zeros = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=50))
    endpoints = _random_bands(size=8, seed=22)
    with pytest.raises(ValueError):     # endpoint_bands given under 'zeros'
        sched_zeros.step_from_prediction(
            bands,
            bands,
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            endpoint_bands=endpoints,
        )
