"""FR-6.4 (DESIGN S10 row 3): M1 math verification for the RC-BRD schedule.

Runs the full [计划] §9 Stage-M1 checklist on synthetic tensors:
- kappa=0 equivalence with the scalar residual bridge (m_sequence == t/T,
  q_marginal vs BBDMBridgeSchedule(sigma_scale=1.0), step_from_prediction
  vs _bbdm_ddim_step, both across a Haar round trip);
- per-group clock monotonicity and m[0]=0 / m[T]=1 endpoints (kappa=1);
- bridge posterior variance non-negativity (and ==0 at t_prev==t);
- Haar round trip < 1e-5 (fp32); fixed-seed reproducibility (bitwise);
- no NaN/Inf; contract permutations (A4 group swap / A5 lambda-grid
  reversal) moving only the expected dimensions.

CLI (DESIGN S10): --num-samples 16 --seed 0 --dtype float32 -> stdout
checklist + results/rc_brd_math_verify/report.json; any failed check exits
1. Logic lives in importable functions (DESIGN S12.5): run_verification(cfg)
returns the report dict.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model.noise.base import BBDMBridgeSchedule  # noqa: E402
from src.model.rc_brd import (  # noqa: E402
    BandwiseBridgeSchedule,
    BandwiseScheduleConfig,
    RecoverabilityContract,
    haar_forward2,
    haar_inverse2,
    roundtrip_error,
)
from src.model.slmf_bbdm import _bbdm_ddim_step  # noqa: E402

REPORT_PATH = _REPO_ROOT / "results" / "rc_brd_math_verify" / "report.json"
# [DESIGN §4] equivalence tolerances per dtype. The scalar reference
# (BBDMBridgeSchedule) stores its clock in float32, so float64 inputs cannot
# beat ~1e-7 relative error; both dtypes use the pinned fp32 tolerances.
DTYPE_TOLERANCES = {"float32": (1e-5, 1e-6), "float64": (1e-5, 1e-6)}
HAAR_ROUNDTRIP_LIMITS = {"float32": 1e-5, "float64": 1e-9}  # [计划] §9 (按 dtype)
KAPPA = 1.0        # warped-clock checks ([计划] §3.6); kappa=0 checks use 0.0
_M_TOL = 1e-12     # [计划] §9: kappa=0 clock identity < 1e-12
_PERM_TOL = 1e-9   # permutation scoping / lambda-mirror symmetry tolerance
ScheduleFactory = Callable[..., BandwiseBridgeSchedule]


@dataclass(frozen=True)
class VerifyConfig:
    """M1 verification inputs; CLI defaults follow DESIGN S10 row 3."""

    num_samples: int = 16
    seed: int = 0
    dtype: str = "float32"
    num_timesteps: int = 200   # small smoke T ([计划] §9: 8-16 samples)
    image_size: int = 32       # H=W divisible by 4 (two-level Haar, DESIGN §2)
    report_path: Path = REPORT_PATH

    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == "float32" else torch.float64


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _entry(passed: bool, metric: float, threshold: float, detail: str) -> dict:
    return {"passed": bool(passed), "metric": float(metric),
            "threshold": float(threshold), "detail": detail}


def _interior_timesteps(num_timesteps: int) -> list[tuple[int, int]]:
    """Interior (t, t_prev) pairs avoiding endpoint degeneracies."""
    step = max(num_timesteps // 10, 1)
    pairs = []
    for t in (num_timesteps * f // 5 for f in (1, 2, 3, 4)):
        if t - step >= 1:
            pairs.append((t, t - step))
    return pairs


def synthetic_contract() -> RecoverabilityContract:
    """Synthetic frozen contract for the kappa!=0 checks ([计划] §3.5 schema)."""
    payload = {
        "fold_id": "m1_synthetic",
        "band_groups": {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"],
                        "high": ["LH1", "HL1", "HH1"]},
        "log_snr_grid": [-10.0, 0.0, 10.0],
        "c_values": {"low": [0.2, 0.3, 0.4], "mid": [0.4, 0.5, 0.6],
                     "high": [0.6, 0.7, 0.8]},
        "mean_checkpoint_sha256": "a" * 64,
        "b_active": ["low", "mid", "high"],
        "psd_floors": {"low": 1e-4, "mid": 2e-4, "high": 5e-4},
        "size_thresholds": {"small_lesion_q25": 12.0},
        "kappa_grid": [0.0, 1.0],
        "s_ref": 1.0,
        "support_mode": "floor_gated",
        "eta_max": 0.8,
        "floor_rho": 0.1,
    }
    return RecoverabilityContract.from_payload(payload)


def _random_image(cfg: VerifyConfig, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(cfg.num_samples, 1, cfg.image_size, cfg.image_size,
                       generator=gen, dtype=cfg.torch_dtype())


# ---------------------------------------------------------------------------
# Individual M1 checks ([计划] §9)
# ---------------------------------------------------------------------------

def _check_haar_roundtrip(cfg: VerifyConfig) -> dict:
    gen = torch.Generator().manual_seed(cfg.seed)
    worst = 0.0
    for _ in range(3):
        worst = max(worst, roundtrip_error(_random_image(cfg, gen)))
    limit = HAAR_ROUNDTRIP_LIMITS[cfg.dtype]
    return _entry(worst < limit, worst, limit, "max |W^-1 W x - x| over 3 batches")


def _check_kappa0_clock(cfg: VerifyConfig, factory: ScheduleFactory) -> dict:
    sched = factory(BandwiseScheduleConfig(num_timesteps=cfg.num_timesteps), None)
    expected = torch.arange(cfg.num_timesteps + 1, dtype=torch.float64) / cfg.num_timesteps
    worst = max(float((sched.m_sequence(b) - expected).abs().max()) for b in ("LL2", "LH1"))
    return _entry(worst < _M_TOL, worst, _M_TOL, "kappa=0: m_sequence == t/T")


def _check_kappa0_marginal(cfg: VerifyConfig) -> dict:
    """kappa=0 q_marginal vs BBDMBridgeSchedule(sigma_scale=1.0), same eps."""
    rtol, atol = DTYPE_TOLERANCES[cfg.dtype]
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    z0 = _random_image(cfg, gen)
    eps = torch.randn(z0.shape, generator=gen, dtype=z0.dtype)
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=cfg.num_timesteps))
    scalar = BBDMBridgeSchedule(num_train_timesteps=cfg.num_timesteps, sigma_scale=1.0)
    z0_bands, eps_bands = haar_forward2(z0), haar_forward2(eps)
    zeros = torch.zeros_like(z0)
    worst, passed = 0.0, True
    for t, _ in _interior_timesteps(cfg.num_timesteps):
        tv = torch.full((z0.shape[0],), t, dtype=torch.int64)
        zt_bands, _target = sched.q_marginal(z0_bands, tv, eps_bands=eps_bands)
        zt_bandwise = haar_inverse2(zt_bands)
        zt_scalar = scalar.add_noise(z0, eps, tv, condition=None, x_source=zeros)
        passed &= bool(torch.allclose(zt_bandwise, zt_scalar, rtol=rtol, atol=atol))
        worst = max(worst, float((zt_bandwise - zt_scalar).abs().max()))
    return _entry(passed, worst, atol, "max |q_marginal(k=0) - BBDM add_noise|")


def _check_kappa0_step(cfg: VerifyConfig) -> dict:
    """kappa=0 step_from_prediction vs _bbdm_ddim_step (Haar round trip)."""
    rtol, atol = DTYPE_TOLERANCES[cfg.dtype]
    gen = torch.Generator().manual_seed(cfg.seed + 2)
    z0 = _random_image(cfg, gen)
    z0_pred = z0 * 0.9 + 0.1 * torch.randn(z0.shape, generator=gen, dtype=z0.dtype)
    eps = torch.randn(z0.shape, generator=gen, dtype=z0.dtype)
    sched = BandwiseBridgeSchedule(BandwiseScheduleConfig(num_timesteps=cfg.num_timesteps))
    scalar = BBDMBridgeSchedule(num_train_timesteps=cfg.num_timesteps, sigma_scale=1.0)
    z0_bands = haar_forward2(z0)
    pred_bands, eps_bands = haar_forward2(z0_pred), haar_forward2(eps)
    zeros = torch.zeros_like(z0)
    worst, passed = 0.0, True
    for t, tp in _interior_timesteps(cfg.num_timesteps):
        tv = torch.full((z0.shape[0],), t, dtype=torch.int64)
        tpv = torch.full((z0.shape[0],), tp, dtype=torch.int64)
        zt_bands, _ = sched.q_marginal(z0_bands, tv, eps_bands=eps_bands)
        zt_image = haar_inverse2(zt_bands)
        out_bandwise = haar_inverse2(sched.step_from_prediction(
            pred_bands, haar_forward2(zt_image), tpv, tv))
        out_scalar = _bbdm_ddim_step(scalar, zt_image, z0_pred, zeros, tv, tpv)
        passed &= bool(torch.allclose(out_bandwise, out_scalar, rtol=rtol, atol=atol))
        worst = max(worst, float((out_bandwise - out_scalar).abs().max()))
    # [计划] §9: first reverse step from the t=T endpoint stays finite.
    tv_max = torch.full((z0.shape[0],), cfg.num_timesteps, dtype=torch.int64)
    tv_prev = torch.full((z0.shape[0],), cfg.num_timesteps - 1, dtype=torch.int64)
    zt_end, _ = sched.q_marginal(z0_bands, tv_max, eps_bands=eps_bands)
    first = sched.step_from_prediction(pred_bands, zt_end, tv_prev, tv_max)
    finite = all(bool(torch.isfinite(v).all()) for v in first.values())
    return _entry(passed and finite, worst, atol, "max |step(k=0) - _bbdm_ddim_step|")


def _check_clock_properties(cfg: VerifyConfig, contract: RecoverabilityContract,
                            factory: ScheduleFactory) -> dict:
    sched = factory(BandwiseScheduleConfig(num_timesteps=cfg.num_timesteps, kappa=KAPPA),
                    contract)
    ok, min_diff = True, float("inf")
    for group in ("low", "mid", "high"):
        m = sched.m_sequence(group)
        ok &= m[0].item() == 0.0 and m[-1].item() == 1.0  # [计划] §3.6 endpoints
        diffs = m[1:] - m[:-1]
        ok &= bool((diffs > 0).all())                     # strictly monotone clock
        min_diff = min(min_diff, float(diffs.min()))
    return _entry(ok, min_diff, 0.0, "kappa=1: m[0]=0, m[T]=1, strictly increasing")


def _check_posterior(cfg: VerifyConfig, contract: RecoverabilityContract,
                     factory: ScheduleFactory) -> dict:
    sched = factory(BandwiseScheduleConfig(num_timesteps=cfg.num_timesteps, kappa=KAPPA),
                    contract)
    gen = torch.Generator().manual_seed(cfg.seed + 3)
    z0_bands = haar_forward2(_random_image(cfg, gen))
    eps_bands = haar_forward2(torch.randn(
        (cfg.num_samples, 1, cfg.image_size, cfg.image_size),
        generator=gen, dtype=cfg.torch_dtype()))
    ok, min_var = True, float("inf")
    for t, tp in _interior_timesteps(cfg.num_timesteps):
        tv = torch.full((cfg.num_samples,), t, dtype=torch.int64)
        tpv = torch.full((cfg.num_samples,), tp, dtype=torch.int64)
        zt_bands, _ = sched.q_marginal(z0_bands, tv, eps_bands=eps_bands)
        _, var_bands = sched.posterior(z0_bands, zt_bands, tpv, tv)
        for var in var_bands.values():
            ok &= bool((var >= 0).all()) and bool(torch.isfinite(var).all())
            min_var = min(min_var, float(var.min()))
        _, var_eq = sched.posterior(z0_bands, zt_bands, tv, tv)  # t_prev == t
        ok &= all(bool((v == 0).all()) for v in var_eq.values())
    return _entry(ok, min_var, 0.0, "min posterior variance (>=0; ==0 when t_prev==t)")


def _pipeline_fingerprint(cfg: VerifyConfig, contract: RecoverabilityContract) -> torch.Tensor:
    """Deterministic kappa=0 + kappa=1 forward/posterior/step pipeline."""
    gen = torch.Generator().manual_seed(cfg.seed + 101)
    z0 = _random_image(cfg, gen)
    eps = torch.randn(z0.shape, generator=gen, dtype=z0.dtype)
    z0_bands, eps_bands = haar_forward2(z0), haar_forward2(eps)
    tv = torch.full((cfg.num_samples,), cfg.num_timesteps // 2, dtype=torch.int64)
    tpv = torch.full((cfg.num_samples,), cfg.num_timesteps // 4, dtype=torch.int64)
    parts = [z0.to(torch.float32).flatten(), eps.to(torch.float32).flatten()]
    for kappa, bound in ((0.0, None), (KAPPA, contract)):
        sched = BandwiseBridgeSchedule(
            BandwiseScheduleConfig(num_timesteps=cfg.num_timesteps, kappa=kappa), bound)
        zt_bands, _ = sched.q_marginal(z0_bands, tv, eps_bands=eps_bands)
        step = sched.step_from_prediction(z0_bands, zt_bands, tpv, tv)
        mean_b, var_b = sched.posterior(z0_bands, zt_bands, tpv, tv)
        for band_dict in (zt_bands, step, mean_b, var_b):
            parts.append(torch.cat([t.flatten() for t in band_dict.values()])
                         .to(torch.float32))
    return torch.cat(parts)


def _check_permutation_scoping(cfg: VerifyConfig) -> dict:
    """A4/A5 contract permutations move only the expected dimensions.

    A4 (group swap): the untouched low group's clock stays bitwise identical
    and the swapped groups exchange clocks exactly.  A5 (lambda-grid
    reversal = c order mirrored in lambda, grid kept increasing for
    interpolation): every mirrored group still yields a valid clock
    (m[0]=0, m[T]=1, strictly monotone) and the reversal has a real effect.
    Under the v1.5 clamped log-SNR clock lookup lambda0(u)=log((1-u)/(nu^2 u))
    (DESIGN v1.0f) the stricter mirror identity m_rev(u)=1-m(1-u) is no
    longer exact (nu^2 != 1 breaks the u-mirror antisymmetry by a constant
    log-offset), so validity + effect are the pinned invariants."""
    base = synthetic_contract()
    sched_base = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=cfg.num_timesteps, kappa=KAPPA), base)
    m_base = {g: sched_base.m_sequence(g) for g in ("low", "mid", "high")}
    payload = base.to_payload()
    # A4 ([PRD] §3): swap the mid/high c-curves; the low group must not move.
    swapped_c = {g: list(v) for g, v in payload["c_values"].items()}
    swapped_c["mid"], swapped_c["high"] = list(payload["c_values"]["high"]), \
        list(payload["c_values"]["mid"])
    sched_swap = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=cfg.num_timesteps, kappa=KAPPA),
        RecoverabilityContract.from_payload({**payload, "c_values": swapped_c}))
    low_still = float((sched_swap.m_sequence("low") - m_base["low"]).abs().max())
    mid_takes_high = float((sched_swap.m_sequence("mid") - m_base["high"]).abs().max())
    high_takes_mid = float((sched_swap.m_sequence("high") - m_base["mid"]).abs().max())
    # A5: lambda-grid reversal == c order mirrored in lambda (grid must stay
    # increasing for interpolation); valid clock + non-vacuous effect.
    sched_rev = BandwiseBridgeSchedule(
        BandwiseScheduleConfig(num_timesteps=cfg.num_timesteps, kappa=KAPPA),
        RecoverabilityContract.from_payload({
            **payload,
            "c_values": {g: list(reversed(v)) for g, v in payload["c_values"].items()},
        }))
    effect = 0.0
    a5_valid = True
    for group in m_base:
        m_rev = sched_rev.m_sequence(group)
        effect = max(effect, float((m_rev - m_base[group]).abs().max()))
        diffs = m_rev[1:] - m_rev[:-1]
        a5_valid &= m_rev[0].item() == 0.0 and m_rev[-1].item() == 1.0
        a5_valid &= bool((diffs > 0).all())
    ok = (low_still < _M_TOL and mid_takes_high < _PERM_TOL
          and high_takes_mid < _PERM_TOL and a5_valid and effect > 0.0)
    worst = max(low_still, mid_takes_high, high_takes_mid)
    return _entry(ok, worst, _PERM_TOL,
                  "A4 swap moves only mid/high (exact); A5 lambda reversal "
                  "keeps every clock valid and changes it")


# ---------------------------------------------------------------------------
# Orchestration + CLI (DESIGN S10 row 3)
# ---------------------------------------------------------------------------

def run_verification(cfg: VerifyConfig,
                     schedule_factory: ScheduleFactory | None = None) -> dict:
    """Run the full M1 checklist; return the report dict ([计划] §9)."""
    if cfg.num_samples < 1 or cfg.num_timesteps < 10:
        raise ValueError("num_samples >= 1 and num_timesteps >= 10 are required")
    factory = schedule_factory or BandwiseBridgeSchedule
    contract = synthetic_contract()
    checks: dict[str, dict] = {}
    checks["haar_roundtrip"] = _check_haar_roundtrip(cfg)
    checks["kappa0_m_identity"] = _check_kappa0_clock(cfg, factory)
    checks["kappa0_q_marginal_scalar_equivalence"] = _check_kappa0_marginal(cfg)
    checks["kappa0_step_scalar_equivalence"] = _check_kappa0_step(cfg)
    checks["clock_monotone_endpoints"] = _check_clock_properties(cfg, contract, factory)
    checks["posterior_variance_nonnegative"] = _check_posterior(cfg, contract, factory)
    fp_a = _pipeline_fingerprint(cfg, contract)
    fp_b = _pipeline_fingerprint(cfg, contract)
    checks["fixed_seed_reproducibility"] = _entry(
        torch.equal(fp_a, fp_b), float((fp_a - fp_b).abs().max()), 0.0,
        "two seeded pipeline runs are bitwise identical")
    checks["no_nan_inf"] = _entry(bool(torch.isfinite(fp_a).all()),
                                  float(fp_a.abs().max()), float("inf"),
                                  "all pipeline tensors finite")
    checks["contract_permutation_scoping"] = _check_permutation_scoping(cfg)
    all_passed = all(entry["passed"] for entry in checks.values())
    return {
        "schema_version": 1,
        "config": {"num_samples": cfg.num_samples, "seed": cfg.seed,
                   "dtype": cfg.dtype, "num_timesteps": cfg.num_timesteps,
                   "image_size": cfg.image_size, "report_path": str(cfg.report_path)},
        "checks": checks,
        "all_passed": all_passed,
        "exit_code": 0 if all_passed else 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="M1 math verification of the RC-BRD schedule ([计划] §9)")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=sorted(DTYPE_TOLERANCES), default="float32")
    args = parser.parse_args(argv)
    cfg = VerifyConfig(num_samples=args.num_samples, seed=args.seed, dtype=args.dtype)
    report = run_verification(cfg)
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text(json.dumps(report, indent=2, sort_keys=True),
                               encoding="utf-8")
    for name, entry in report["checks"].items():
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"[{status}] {name}: {entry['detail']} "
              f"(metric={entry['metric']:.3e}, threshold={entry['threshold']:.3e})")
    print(f"all_passed={report['all_passed']} report={cfg.report_path}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
