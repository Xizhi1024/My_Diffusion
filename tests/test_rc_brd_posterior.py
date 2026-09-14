"""RC-BRD bridge posterior tests (DESIGN §11 / PRD FR-3.5; [审计] §3.2).

Covers: closed-form posterior vs torch.distributions.MultivariateNormal
conditional, variance non-negativity, κ=0 numerical equivalence with the
scalar BBDMBridgeSchedule (marginal and _bbdm_ddim_step sampler step),
endpoint pinning at t_prev=0, vp-mode posterior guard.  v1.0g additions
(AUDIT 5 §3.2/§3.3): ancestral noise semantics of step_from_prediction
(posterior mean+√var·ξ), explicit degenerate-endpoint branches (m_t∈{0,1}),
forward_transition Monte-Carlo marginal consistency and its Bayes mutual
inverse against posterior().
CPU-only, deterministic (explicit torch.Generator seeds).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.noise.base import BBDMBridgeSchedule
from src.model.rc_brd.schedule import (
    BandwiseBridgeSchedule,
    BandwiseScheduleConfig,
)
from src.model.rc_brd.wavelet import BAND_NAMES, band_groups, haar_forward2, haar_inverse2
from src.model.slmf_bbdm import _bbdm_ddim_step

# band → 3-group name (for warped-clock hand formulas under a bound contract).
_GROUP_OF = {b: g for g, bs in band_groups(3).items() for b in bs}

# [审计] §3.2 ↔ DESIGN §4 σ alignment: audit σ_b² = 2·sigma_bridge² (σ=1 here).
_NU_SQUARED = 2.0


@dataclass(frozen=True)
class _StubContract:
    """Duck-typed stand-in for RecoverabilityContract (batch 1B owns contract.py)."""

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


def _mvn_conditional_reference(sched, group, z0, t_prev_idx, t_idx, seed):
    """Analytic bridge conditional via a torch MVN over (z_s, z_t).

    Per coefficient the joint is ([审计] §3.2 marginal + Brownian-bridge
    covariance Cov(B_s,B_t)=min(s,t)−st evaluated at s=m_s, t=m_t):
    mean (μ_s, μ_t) with μ_j=(1−m_j)z0 (e=0), covariance
    ν²·[[m_s(1−m_s), m_s(1−m_t)],[m_s(1−m_t), m_t(1−m_t)]].  Coefficients are
    independent, so the flat ordering (s-block, t-block) gives block-diagonal
    quadrants.  Returns (cond_mean, cond_var_diag, z_t_observed).
    """
    m_seq = sched.m_sequence(group)
    m_s = m_seq[t_prev_idx].item()
    m_t = m_seq[t_idx].item()
    n_coef = z0.numel()
    eye = torch.eye(n_coef, dtype=torch.float64)
    cov = torch.zeros(2 * n_coef, 2 * n_coef, dtype=torch.float64)
    cov[:n_coef, :n_coef] = _NU_SQUARED * m_s * (1 - m_s) * eye
    cov[n_coef:, n_coef:] = _NU_SQUARED * m_t * (1 - m_t) * eye
    cov[:n_coef, n_coef:] = _NU_SQUARED * m_s * (1 - m_t) * eye
    cov[n_coef:, :n_coef] = _NU_SQUARED * m_s * (1 - m_t) * eye
    z0_flat = z0.to(torch.float64).flatten()
    loc = torch.cat([(1 - m_s) * z0_flat, (1 - m_t) * z0_flat])
    joint = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    gen = torch.Generator().manual_seed(seed)
    eps = torch.randn(n_coef, generator=gen, dtype=torch.float64)
    s_tt = joint.covariance_matrix[n_coef:, n_coef:]
    chol_tt = torch.linalg.cholesky(s_tt)
    z_t_obs = (joint.loc[n_coef:] + chol_tt @ eps).view_as(z0)
    residual = (z_t_obs.to(torch.float64).flatten() - joint.loc[n_coef:]).unsqueeze(1)
    sol = torch.linalg.solve(s_tt, residual).squeeze(1)
    cond_mean = joint.loc[:n_coef] + joint.covariance_matrix[:n_coef, n_coef:] @ sol
    cond_cov = (
        joint.covariance_matrix[:n_coef, :n_coef]
        - joint.covariance_matrix[:n_coef, n_coef:]
        @ torch.linalg.solve(s_tt, joint.covariance_matrix[n_coef:, :n_coef])
    )
    return cond_mean, torch.diagonal(cond_cov), z_t_obs


def _assert_posterior_matches_mvn(sched, group, band, z0, t_prev_idx, t_idx, seed):
    cond_mean, cond_var, z_t_obs = _mvn_conditional_reference(
        sched, group, z0, t_prev_idx, t_idx, seed
    )
    mean_bands, var_bands = sched.posterior(
        {band: z0},
        {band: z_t_obs},
        torch.tensor([t_prev_idx], dtype=torch.long),
        torch.tensor([t_idx], dtype=torch.long),
    )
    assert torch.allclose(
        mean_bands[band].to(torch.float64).flatten(), cond_mean, rtol=1e-5, atol=1e-6
    )
    assert torch.allclose(
        var_bands[band].to(torch.float64).flatten(), cond_var, rtol=1e-5, atol=1e-6
    )


def test_posterior_matches_mvn_conditional_kappa0():
    # 2-coefficient scalar band, hand-checked against the MVN conditional.
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=100))
    gen = torch.Generator().manual_seed(0)
    z0 = torch.randn(1, 1, 1, 2, generator=gen)
    _assert_posterior_matches_mvn(sched, "mid", "LH1", z0, 30, 70, seed=1)


def test_posterior_matches_mvn_conditional_kappa_nonzero():
    # Warped clock (κ=1 + stub contract): same Gaussian conditioning holds.
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=1.0), contract=_StubContract()
    )
    gen = torch.Generator().manual_seed(2)
    z0 = torch.randn(1, 1, 1, 2, generator=gen)
    _assert_posterior_matches_mvn(sched, "mid", "HH2", z0, 20, 80, seed=3)


def test_posterior_variance_nonnegative():
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=50))
    z0 = torch.randn(1, 1, 2, 2, generator=torch.Generator().manual_seed(4))
    for t_prev in range(0, 25):
        for t in range(t_prev, 50):
            _, var = sched.posterior(
                {"HL2": z0},
                {"HL2": z0},
                torch.tensor([t_prev], dtype=torch.long),
                torch.tensor([t], dtype=torch.long),
            )
            assert bool((var["HL2"] >= 0).all())
            if t == t_prev:
                assert bool((var["HL2"] == 0).all())


def test_posterior_t_prev_equals_t_returns_z_t():
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=50))
    gen = torch.Generator().manual_seed(5)
    z0 = torch.randn(1, 1, 2, 2, generator=gen)
    z_t = torch.randn(1, 1, 2, 2, generator=gen)
    mean, var = sched.posterior(
        {"LL2": z0},
        {"LL2": z_t},
        torch.tensor([17], dtype=torch.long),
        torch.tensor([17], dtype=torch.long),
    )
    assert torch.equal(mean["LL2"], z_t)
    assert bool((var["LL2"] == 0).all())


def test_kappa0_q_marginal_matches_scalar_bbdm():
    # DESIGN §4 equivalence: same inputs/seeds → elementwise match (fp32 tol).
    num_t = 100
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=num_t))
    scalar = BBDMBridgeSchedule(num_train_timesteps=num_t, sigma_scale=1.0)
    gen = torch.Generator().manual_seed(6)
    bands = haar_forward2(torch.randn(4, 1, 32, 32, generator=gen))
    t = torch.tensor([0, 17, 55, 99], dtype=torch.long)
    for band, z0 in bands.items():
        eps = torch.randn(z0.shape, generator=gen)
        z_t, target = sched.q_marginal({band: z0}, t, {band: eps})
        ref = scalar.add_noise(
            z0, eps, t, condition=None, x_source=torch.zeros_like(z0)
        )
        assert torch.allclose(z_t[band], ref, rtol=1e-5, atol=1e-6)
        assert target[band] is z0


def test_step_from_prediction_pins_endpoint_at_t_prev_zero():
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=50))
    gen = torch.Generator().manual_seed(7)
    bands_pred = haar_forward2(torch.randn(1, 1, 16, 16, generator=gen))
    bands_t = haar_forward2(torch.randn(1, 1, 16, 16, generator=gen))
    out = sched.step_from_prediction(
        bands_pred,
        bands_t,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([40], dtype=torch.long),
    )
    for band in bands_pred:
        assert torch.equal(out[band], bands_pred[band])


def test_kappa0_step_matches_scalar_bbdm_ddim_step():
    # [设计勘误] deterministic ε̂-reuse step == _bbdm_ddim_step at κ=0 (σ=1, e=0).
    num_t = 100
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=num_t))
    scalar = BBDMBridgeSchedule(num_train_timesteps=num_t, sigma_scale=1.0)
    gen = torch.Generator().manual_seed(8)
    x_t = torch.randn(2, 1, 16, 16, generator=gen)
    z0_pred = torch.randn(2, 1, 16, 16, generator=gen)
    zeros = torch.zeros_like(x_t)
    bands_t = haar_forward2(x_t)
    bands_pred = haar_forward2(z0_pred)
    for t_prev_vals in ([70, 20], [0, 20]):  # include the pinned t_prev=0 case
        t = torch.tensor([90, 60], dtype=torch.long)
        t_prev = torch.tensor(t_prev_vals, dtype=torch.long)
        ref = _bbdm_ddim_step(scalar, x_t, z0_pred, zeros, t, t_prev)
        out = sched.step_from_prediction(bands_pred, bands_t, t_prev, t)
        # band-domain comparison (no Haar roundtrip in the reference path)
        for band in BAND_NAMES:
            ref_band = _bbdm_ddim_step(
                scalar,
                bands_t[band],
                bands_pred[band],
                torch.zeros_like(bands_t[band]),
                t,
                t_prev,
            )
            assert torch.allclose(out[band], ref_band, rtol=1e-5, atol=1e-6)
        # image-domain comparison (adds one Haar roundtrip, still fp32-tolerant)
        rec = haar_inverse2(out)
        assert torch.allclose(rec, ref, rtol=1e-5, atol=1e-6)


def test_step_with_ct_minus_mean_endpoints_matches_hand_formula():
    # [DESIGN v1.0a] ε̂-reuse step with e≠0: z_s = μ_s + (σ_s/σ_t)·(z_t − μ_t),
    # μ_j = m_j·e + (1−m_j)·ẑ0 (bandwise _bbdm_ddim_step with nonzero source).
    num_t = 100
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=num_t, endpoint_mode="ct_minus_mean")
    )
    gen = torch.Generator().manual_seed(10)
    bands_pred = haar_forward2(torch.randn(2, 1, 16, 16, generator=gen))
    bands_t = haar_forward2(torch.randn(2, 1, 16, 16, generator=gen))
    endpoints = haar_forward2(torch.randn(2, 1, 16, 16, generator=gen))
    t = torch.tensor([80, 60], dtype=torch.long)
    t_prev = torch.tensor([45, 25], dtype=torch.long)
    out = sched.step_from_prediction(
        bands_pred, bands_t, t_prev, t, endpoint_bands=endpoints
    )
    m = sched.m_sequence("mid")  # κ=0 → identity clock shared by all groups
    sigma = sched.config.sigma_bridge
    for band in bands_pred:
        z0p = bands_pred[band].to(torch.float64)
        zt = bands_t[band].to(torch.float64)
        e_64 = endpoints[band].to(torch.float64)
        m_t = m[t].view(-1, 1, 1, 1)
        m_p = m[t_prev].view(-1, 1, 1, 1)
        s_t = (sigma * torch.sqrt(2.0 * m_t * (1.0 - m_t))).clamp_min(1e-6)
        s_p = sigma * torch.sqrt(2.0 * m_p * (1.0 - m_p))
        mu_t = m_t * e_64 + (1.0 - m_t) * z0p
        mu_p = m_p * e_64 + (1.0 - m_p) * z0p
        expected = mu_p + (s_p / s_t) * (zt - mu_t)
        assert torch.allclose(out[band].to(torch.float64), expected, rtol=1e-5, atol=1e-6)
    # mixed batch: t_prev==0 sample pinned to ẑ0 regardless of e
    out0 = sched.step_from_prediction(
        bands_pred,
        bands_t,
        torch.tensor([0, 25], dtype=torch.long),
        t,
        endpoint_bands=endpoints,
    )
    for band in bands_pred:
        assert torch.equal(out0[band][0], bands_pred[band][0])
    # fail-closed coupling (DESIGN v1.0a)
    with pytest.raises(ValueError):     # 'ct_minus_mean' without endpoint_bands
        sched.step_from_prediction(bands_pred, bands_t, t_prev, t)
    sched_zeros = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=num_t))
    with pytest.raises(ValueError):     # 'zeros' with endpoint_bands
        sched_zeros.step_from_prediction(
            bands_pred, bands_t, t_prev, t, endpoint_bands=endpoints
        )


# ---- v1.0g: ancestral noise semantics of step_from_prediction ---------


def test_step_noise_is_ancestral_posterior_sampling_kappa0():
    # [AUDIT 5 §3.3 X item / DESIGN §4 v1.0g] noise≠None ⟹ exact ancestral
    # draw z_s=posterior_mean+√posterior_var·ξ with the [审计] §3.2 closed
    # form (σ=1, e=0), hand-checked; the retired ε̂-reuse+σ_p·ξ stacking must
    # NOT match anymore.
    num_t = 100
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=num_t))
    gen = torch.Generator().manual_seed(40)
    bands_pred = haar_forward2(torch.randn(2, 1, 16, 16, generator=gen))
    bands_t = haar_forward2(torch.randn(2, 1, 16, 16, generator=gen))
    noise = {b: torch.randn(v.shape, generator=gen) for b, v in bands_pred.items()}
    t = torch.tensor([90, 60], dtype=torch.long)
    t_prev = torch.tensor([70, 20], dtype=torch.long)
    out = sched.step_from_prediction(bands_pred, bands_t, t_prev, t, noise=noise)
    mean_b, var_b = sched.posterior(bands_pred, bands_t, t_prev, t)
    m = sched.m_sequence("mid")  # κ=0 identity clock shared by all groups
    for band in bands_pred:
        z0p = bands_pred[band].to(torch.float64)
        zt = bands_t[band].to(torch.float64)
        m_t = m[t].view(-1, 1, 1, 1)
        m_p = m[t_prev].view(-1, 1, 1, 1)
        ratio = m_p / m_t                                   # m_s/m_t
        mean_hand = (1.0 - ratio) * z0p + ratio * zt        # e cancels
        var_hand = 2.0 * m_p * (m_t - m_p) / m_t            # 2σ²m_s(m_t−m_s)/m_t
        expected = mean_hand + torch.sqrt(var_hand) * noise[band].to(torch.float64)
        assert torch.allclose(out[band].to(torch.float64), expected, rtol=1e-5, atol=1e-6)
        assert torch.allclose(mean_b[band].to(torch.float64), mean_hand, rtol=1e-5, atol=1e-6)
        # posterior() variance is per-sample [B]; compare/expand accordingly.
        assert torch.allclose(var_b[band].to(torch.float64), var_hand.view(-1), rtol=1e-5, atol=1e-6)
        # retired buggy composite: μ_p+σ_p·(ε̂+ξ) — must differ
        sigma_t = torch.sqrt(2.0 * m_t * (1.0 - m_t)).clamp_min(1e-6)
        sigma_p = torch.sqrt(2.0 * m_p * (1.0 - m_p))
        mu_t, mu_p = (1.0 - m_t) * z0p, (1.0 - m_p) * z0p
        eps_hat = (zt - mu_t) / sigma_t
        retired = mu_p + sigma_p * eps_hat + sigma_p * noise[band].to(torch.float64)
        assert not torch.allclose(out[band], retired.to(out[band].dtype), rtol=1e-3, atol=1e-4)


def test_step_noise_is_ancestral_posterior_sampling_warped_clock():
    # Same ancestral semantics under the contracted warped clock (κ=1):
    # closed form evaluated at the warped m values.
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=100, kappa=1.0), contract=_StubContract()
    )
    gen = torch.Generator().manual_seed(42)
    bands_pred = haar_forward2(torch.randn(2, 1, 16, 16, generator=gen))
    bands_t = haar_forward2(torch.randn(2, 1, 16, 16, generator=gen))
    noise = {b: torch.randn(v.shape, generator=gen) for b, v in bands_pred.items()}
    t = torch.tensor([80, 60], dtype=torch.long)
    t_prev = torch.tensor([30, 20], dtype=torch.long)
    out = sched.step_from_prediction(bands_pred, bands_t, t_prev, t, noise=noise)
    for band in bands_pred:
        m_seq = sched.m_sequence(_GROUP_OF[band])
        m_t = m_seq[t].view(-1, 1, 1, 1)
        m_p = m_seq[t_prev].view(-1, 1, 1, 1)
        z0p = bands_pred[band].to(torch.float64)
        zt = bands_t[band].to(torch.float64)
        ratio = m_p / m_t
        expected = (
            (1.0 - ratio) * z0p + ratio * zt
            + torch.sqrt(2.0 * m_p * (m_t - m_p) / m_t) * noise[band].to(torch.float64)
        )
        assert torch.allclose(out[band].to(torch.float64), expected, rtol=1e-5, atol=1e-6)


def test_step_noise_ancestral_ct_minus_mean_endpoint_coupling():
    # The posterior mean is endpoint-independent (e cancels analytically),
    # but the v1.0a coupling validation still applies in ancestral mode:
    # 'ct_minus_mean' demands endpoint_bands; 'zeros' forbids them.
    num_t = 100
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=num_t, endpoint_mode="ct_minus_mean")
    )
    gen = torch.Generator().manual_seed(43)
    bands_pred = haar_forward2(torch.randn(1, 1, 8, 8, generator=gen))
    bands_t = haar_forward2(torch.randn(1, 1, 8, 8, generator=gen))
    endpoints = haar_forward2(torch.randn(1, 1, 8, 8, generator=gen))
    noise = {b: torch.randn(v.shape, generator=gen) for b, v in bands_pred.items()}
    t = torch.tensor([70], dtype=torch.long)
    t_prev = torch.tensor([40], dtype=torch.long)
    with pytest.raises(ValueError):     # endpoint_bands missing
        sched.step_from_prediction(bands_pred, bands_t, t_prev, t, noise=noise)
    out = sched.step_from_prediction(
        bands_pred, bands_t, t_prev, t, noise=noise, endpoint_bands=endpoints
    )
    mean_b, var_b = sched.posterior(bands_pred, bands_t, t_prev, t)
    for band in bands_pred:
        expected = mean_b[band] + torch.sqrt(var_b[band]).view(-1, 1, 1, 1) * noise[band]
        assert torch.allclose(out[band], expected, rtol=1e-5, atol=1e-6)
    sched_zeros = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=num_t))
    with pytest.raises(ValueError):     # 'zeros' + endpoint_bands
        sched_zeros.step_from_prediction(
            bands_pred, bands_t, t_prev, t, noise=noise, endpoint_bands=endpoints
        )


def test_step_vp_mode_noise_raises_not_implemented():
    # [PRD v1.0.2-2] the vp arm is deterministic-DDIM only — no ancestral
    # sampler.
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=20, forward_mode="vp_bandwise")
    )
    bands = haar_forward2(torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(45)))
    noise = {b: torch.randn_like(v) for b, v in bands.items()}
    with pytest.raises(NotImplementedError):
        sched.step_from_prediction(
            bands, bands, torch.tensor([1]), torch.tensor([2]), noise=noise
        )


def test_step_noise_validation():
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=50))
    bands = haar_forward2(torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(48)))
    partial = {k: v for k, v in bands.items() if k != "HH1"}
    with pytest.raises(ValueError):
        sched.step_from_prediction(
            bands, bands, torch.tensor([1]), torch.tensor([2]), noise=partial
        )


# ---- v1.0g: explicit degenerate-endpoint branches (m_t∈{0,1}) -------------


def test_step_deterministic_degenerate_at_t_equals_T_returns_posterior_mean():
    # [DESIGN §4 v1.0g / AUDIT 5 §3.3] m_t=1 (t=T): ε̂=(z_t−μ_t)/σ_t is not
    # invertible (σ_t=0) — the explicit branch returns the posterior mean
    # ((m_t−m_p)/m_t)·ẑ0+(m_p/m_t)·z_t, hand-checked; no clamp-masked
    # division happens at the endpoint.
    num_t = 100
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=num_t))
    gen = torch.Generator().manual_seed(46)
    bands_pred = haar_forward2(torch.randn(2, 1, 8, 8, generator=gen))
    bands_t = haar_forward2(torch.randn(2, 1, 8, 8, generator=gen))
    t = torch.full((2,), num_t, dtype=torch.long)
    t_prev = torch.tensor([95, 99], dtype=torch.long)
    out = sched.step_from_prediction(bands_pred, bands_t, t_prev, t)
    m = sched.m_sequence("mid")
    for band in bands_pred:
        m_p = m[t_prev].view(-1, 1, 1, 1)
        expected = (
            (1.0 - m_p) * bands_pred[band].to(torch.float64)
            + m_p * bands_t[band].to(torch.float64)
        )
        assert torch.allclose(out[band].to(torch.float64), expected, rtol=1e-5, atol=1e-6)
    # t_prev==0 stays pinned to ẑ0 even from the degenerate start
    out0 = sched.step_from_prediction(bands_pred, bands_t, torch.zeros(2, dtype=torch.long), t)
    for band in bands_pred:
        assert torch.equal(out0[band], bands_pred[band])


def test_step_ancestral_degenerate_at_t_equals_T_samples_posterior():
    # m_t=1: ancestral branch draws from the closed-form posterior —
    # var=2σ²m_p(1−m_p) (conditioning on the pinned endpoint z_T=e leaves the
    # full s-marginal variance; [审计] §3.2 var=2σ²m_s(m_t−m_s)/m_t at m_t=1).
    num_t = 100
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=num_t))
    gen = torch.Generator().manual_seed(47)
    bands_pred = haar_forward2(torch.randn(2, 1, 8, 8, generator=gen))
    bands_t = haar_forward2(torch.randn(2, 1, 8, 8, generator=gen))
    noise = {b: torch.randn(v.shape, generator=gen) for b, v in bands_pred.items()}
    t = torch.full((2,), num_t, dtype=torch.long)
    t_prev = torch.tensor([95, 99], dtype=torch.long)
    out = sched.step_from_prediction(bands_pred, bands_t, t_prev, t, noise=noise)
    m = sched.m_sequence("mid")
    for band in bands_pred:
        m_p = m[t_prev].view(-1, 1, 1, 1)
        mean = (
            (1.0 - m_p) * bands_pred[band].to(torch.float64)
            + m_p * bands_t[band].to(torch.float64)
        )
        std = torch.sqrt(2.0 * m_p * (1.0 - m_p))
        expected = mean + std * noise[band].to(torch.float64)
        assert torch.allclose(out[band].to(torch.float64), expected, rtol=1e-5, atol=1e-6)


def test_step_degenerate_at_t_zero_pins_z0_pred_both_modes():
    # m_t=0 (t=0 ⟹ t_prev==t==0): ε̂ not invertible; the t_prev==0 pin
    # returns ẑ0 exactly in deterministic AND ancestral mode (finite, no
    # masked division, no NaN).
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=50))
    gen = torch.Generator().manual_seed(49)
    bands_pred = haar_forward2(torch.randn(2, 1, 8, 8, generator=gen))
    bands_t = haar_forward2(torch.randn(2, 1, 8, 8, generator=gen))
    noise = {b: torch.randn(v.shape, generator=gen) for b, v in bands_pred.items()}
    t0 = torch.zeros(2, dtype=torch.long)
    for kwargs in ({}, {"noise": noise}):
        out = sched.step_from_prediction(bands_pred, bands_t, t0, t0, **kwargs)
        for band in bands_pred:
            assert torch.equal(out[band], bands_pred[band])


def test_step_deterministic_near_endpoint_T_minus_one_uses_eps_hat():
    # t=T−1 is NOT degenerate (m_t=1−1/T sits ~1e-2 from 1, far beyond the
    # 1e-12 tolerance): the deterministic ε̂-reuse path still applies and the
    # hand formula μ_p+(σ_p/σ_t)(z_t−μ_t) holds right next to the endpoint —
    # the explicit branch fires only at m_t∈{0,1}.
    num_t = 100
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=num_t))
    gen = torch.Generator().manual_seed(50)
    bands_pred = haar_forward2(torch.randn(2, 1, 8, 8, generator=gen))
    bands_t = haar_forward2(torch.randn(2, 1, 8, 8, generator=gen))
    t = torch.full((2,), num_t - 1, dtype=torch.long)
    t_prev = torch.tensor([95, 98], dtype=torch.long)
    out = sched.step_from_prediction(bands_pred, bands_t, t_prev, t)
    m = sched.m_sequence("mid")
    for band in bands_pred:
        m_t = m[t].view(-1, 1, 1, 1)
        m_p = m[t_prev].view(-1, 1, 1, 1)
        z0p = bands_pred[band].to(torch.float64)
        zt = bands_t[band].to(torch.float64)
        mu_t, mu_p = (1.0 - m_t) * z0p, (1.0 - m_p) * z0p
        s_t = torch.sqrt(2.0 * m_t * (1.0 - m_t))
        s_p = torch.sqrt(2.0 * m_p * (1.0 - m_p))
        expected = mu_p + (s_p / s_t) * (zt - mu_t)
        assert torch.allclose(out[band].to(torch.float64), expected, rtol=1e-5, atol=1e-6)


# ---- v1.0g: forward_transition consistency ([审计] §3.2) ------------------


def test_forward_transition_monte_carlo_matches_q_marginal():
    # [AUDIT 5 §3.2 / DESIGN §4 v1.0g] Markov consistency: sampling
    # z_s~q(z_s|z0) via q_marginal and then transitioning s→t with the closed
    # form must reproduce the q_marginal(z0,t) moments empirically.
    n_samples = 200_000
    z0_value, e_value = 1.7, -0.9
    cases = [
        # (schedule, group, s, t, mode, endpoints, seed)
        (BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=100)), "LH1", 30, 70, "zeros", None, 51),
        (BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=100)), "LH1", 10, 90, "zeros", None, 52),
        (
            BandwiseBridgeSchedule(
                BandwiseScheduleConfig(num_timesteps=100, kappa=1.0), contract=_StubContract()
            ),
            "high", 25, 75, "zeros", None, 53,
        ),
        (
            BandwiseBridgeSchedule(
                BandwiseScheduleConfig(num_timesteps=100, endpoint_mode="ct_minus_mean")
            ),
            "LH1", 20, 80, "ct",
            {"LH1": torch.full((n_samples, 1, 1, 1), e_value, dtype=torch.float64)}, 54,
        ),
    ]
    for sched, group, s, t, mode, endpoints, seed in cases:
        gen = torch.Generator().manual_seed(seed)
        z0 = {"LH1": torch.full((n_samples, 1, 1, 1), z0_value, dtype=torch.float64)}
        t_s = torch.full((n_samples,), s, dtype=torch.long)
        tv = torch.full((n_samples,), t, dtype=torch.long)
        eps_s = {"LH1": torch.randn(z0["LH1"].shape, generator=gen, dtype=torch.float64)}
        z_s, _ = sched.q_marginal(z0, t_s, eps_s, endpoint_bands=endpoints)
        mean_b, var_b = sched.forward_transition(z_s, t_s, tv, endpoint_bands=endpoints)
        eps_t = {"LH1": torch.randn(z0["LH1"].shape, generator=gen, dtype=torch.float64)}
        z_t = mean_b["LH1"] + torch.sqrt(var_b["LH1"]).view(-1, 1, 1, 1) * eps_t["LH1"]
        m_t = sched.m_sequence(group)[t].item()
        ref_mean = (1.0 - m_t) * z0_value + (m_t * e_value if mode == "ct" else 0.0)
        ref_var = 2.0 * m_t * (1.0 - m_t)   # σ_b=1 ⇒ var=σ_b²·2m(1−m)
        tol_mean = 6.0 * math.sqrt(ref_var / n_samples) + 1e-3
        tol_var = 6.0 * ref_var * math.sqrt(2.0 / n_samples) + 2e-3
        assert abs(z_t.mean().item() - ref_mean) < tol_mean
        assert abs(z_t.var(unbiased=True).item() - ref_var) < tol_var


def test_forward_transition_posterior_bayes_mutual_inverse():
    # [AUDIT 5 §3.2] Bayes mutual inverse of transition and posterior: for
    # the jointly Gaussian (z_s, z_t) pair the composition of the two
    # conditional-mean maps is the exact MMSE shrinkage
    #   posterior_mean(transition_mean(z_s)) = μ_s + (1−v_post/v_s)·(z_s−μ_s)
    # with v_post=2σ²m_s(m_t−m_s)/m_t and v_s=2σ²m_s(1−m_s) (corr²=1−v_post/v_s;
    # a literal identity would need corr²=1, impossible for s<t).  Matching
    # the predicted shrinkage certifies that the transition gain
    # (1−m_t)/(1−m_s), the posterior coefficient m_s/m_t and the posterior
    # variance all derive from ONE joint law — any mismatched pairing breaks it.
    cases = [
        (BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=100)), "LH1", 30, 70),
        (
            BandwiseBridgeSchedule(
                BandwiseScheduleConfig(num_timesteps=100, kappa=1.0), contract=_StubContract()
            ),
            "high", 20, 80,
        ),
    ]
    gen = torch.Generator().manual_seed(55)
    z0 = torch.randn(1, 1, 2, 2, generator=gen, dtype=torch.float64)
    z_s = torch.randn(1, 1, 2, 2, generator=gen, dtype=torch.float64)
    for sched, group, s, t in cases:
        t_s = torch.tensor([s], dtype=torch.long)
        tv = torch.tensor([t], dtype=torch.long)
        mean_b, _ = sched.forward_transition({"LH1": z_s}, t_s, tv)
        pm, pv = sched.posterior({"LH1": z0}, {"LH1": mean_b["LH1"]}, t_s, tv)
        m_seq = sched.m_sequence(group)
        m_s, m_t = m_seq[s].item(), m_seq[t].item()
        mu_s = (1.0 - m_s) * z0          # e=0
        v_post = pv["LH1"]
        v_s = 2.0 * m_s * (1.0 - m_s)    # σ_b=1
        expected = mu_s + (1.0 - v_post / v_s) * (z_s - mu_s)
        assert torch.allclose(pm["LH1"], expected, rtol=1e-8, atol=1e-10)


def test_vp_posterior_raises_not_implemented():
    sched = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=20, forward_mode="vp_bandwise")
    )
    bands = haar_forward2(torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(9)))
    with pytest.raises(NotImplementedError):
        sched.posterior(
            bands,
            bands,
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
        )
