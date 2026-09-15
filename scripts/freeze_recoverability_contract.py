"""FR-6.3 (DESIGN S10 row 4): freeze an R0 probe result into a contract.

Reads the probe JSON produced by run_recoverability_probe.py and freezes
[计划] §3.5:

    c_g(lambda_k) = clip(LCB^maxT_95(Delta_g(lambda_k)) / s_ref, 0, 1)

into a RecoverabilityContract payload (band_groups resolved from the probe
grid, mean checkpoint SHA computed with the canonical mean_weights_sha256
from --mean-checkpoint or taken verbatim from --mean-sha, psd_floors from
the probe data or --sigma-floor). The payload is validated fail-closed
(from_payload + validate, FR-2.2) and written with its self hash; any
validation failure exits 1.

AUDIT evidence-chain gates (fail-closed):
* only a PASSed probe freezes - pass_rule.passed must be exactly True
  (anything else, including a missing pass_rule, raises);
* b_active defaults to the groups with >=1 passing confirmatory cell
  (probe pass_rule.per_cell), NOT to all groups - groups outside b_active
  run the ordinary clock (effective_c == 0.5); an empty passing set refuses
  to freeze (close RC-BRD, fall back to D1).

CLI: --probe-result --fold --out --s-ref --kappa-grid (must contain 0.0;
s_ref must be > 0).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model.rc_brd import (  # noqa: E402
    BAND_NAMES,
    ContractViolationError,
    RecoverabilityContract,
    band_groups,
    mean_weights_sha256,
)

CONFIRMATORY_STRATUM = "small_lesion"  # [审计] §1.1: only confirmatory layer
DEFAULT_SIGMA_FLOOR = 1e-4
DEFAULT_SIZE_THRESHOLDS = {"small_lesion_q25": 25.0}  # until the frozen q25
# artifact lands; overridable via --size-thresholds.


def compute_c_values(probe: Mapping[str, Any], s_ref: float) -> dict[str, list[float]]:
    """c_g(lambda_k) = clip(LCB^maxT / s_ref, 0, 1) ([计划] §3.5)."""
    if s_ref <= 0.0:
        raise ValueError(f"s_ref must be > 0, got {s_ref}")
    delta = probe["delta"][CONFIRMATORY_STRATUM]
    n_lambda = len(probe["grid"]["log_snr_centers"])
    values: dict[str, list[float]] = {}
    for group in probe["grid"]["groups"]:
        row = []
        for k in range(n_lambda):
            lcb = float(delta[group][str(k)]["lcb95"])
            row.append(float(min(1.0, max(0.0, lcb / s_ref))))
        values[str(group)] = row
    return values


def resolve_band_groups(group_names: list[str]) -> dict[str, list[str]]:
    """Map the probe grid groups onto the canonical layouts ([计划] §3.2)."""
    if list(group_names) == ["low", "mid", "high"]:
        return band_groups(3)
    if list(group_names) == list(BAND_NAMES):
        return band_groups(7)
    raise ValueError(f"probe grid groups {group_names} match neither "
                     "band_groups(3) nor band_groups(7)")


def load_mean_sha(path: Path) -> str:
    """Canonical mean SHA from a checkpoint file (DESIGN v1.0b: the freeze
    side and the model-build side share mean_weights_sha256)."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, Mapping) and isinstance(obj.get("state_dict"), Mapping):
        obj = obj["state_dict"]
    if not isinstance(obj, Mapping) or not obj:
        raise ValueError(f"mean checkpoint {path} is not a non-empty state dict")
    return mean_weights_sha256(obj)


def freeze_contract(probe_result: Path, fold: str, out_path: Path, s_ref: float,
                    kappa_grid: tuple[float, ...], *, mean_checkpoint: Path | None = None,
                    mean_sha: str | None = None, sigma_floor: float = DEFAULT_SIGMA_FLOOR,
                    size_thresholds: dict[str, float] | None = None,
                    support_mode: str = "floor_gated", eta_max: float = 0.8,
                    floor_rho: float = 0.1,
                    band_powers: dict[str, float] | None = None,
                    band_power_split: str | None = None) -> dict:
    """Build, validate and write the frozen contract; returns the artifact."""
    probe = json.loads(Path(probe_result).read_text(encoding="utf-8"))
    if probe.get("status") == "insufficient_patients":
        raise ValueError("probe result has insufficient patients; refusing to freeze")
    if probe.get("fold") != fold:
        raise ValueError(f"fold mismatch: probe {probe.get('fold')!r} != --fold {fold!r}")
    # AUDIT evidence-chain gate: only a PASSed probe may freeze.  The old code
    # merely refused status == "insufficient_patients", so a probe whose
    # confirmatory cells FAILED could still be frozen.
    pass_rule = probe.get("pass_rule")
    if not isinstance(pass_rule, Mapping) or pass_rule.get("passed") is not True:
        raise ValueError(
            "probe result did not PASS the v1.5 cell rule (pass_rule.passed is "
            "not true); refusing to freeze - rerun the probe or close RC-BRD "
            "and fall back to D1")
    per_cell = pass_rule.get("per_cell") or {}
    if not isinstance(per_cell, Mapping):
        # Review finding 3: a malformed truthy per_cell must exit 1 with a
        # clean message, not an AttributeError traceback.
        raise ValueError("pass_rule.per_cell must be a mapping of cell "
                         "judgments when present")
    passing_groups = {str(cell).split("|", 1)[0] for cell, judgement in per_cell.items()
                      if isinstance(judgement, Mapping) and judgement.get("pass") is True}
    if not passing_groups:
        raise ValueError(
            "no band group passed any confirmatory cell; refusing to freeze - "
            "close RC-BRD and fall back to D1")
    groups_map = resolve_band_groups(list(probe["grid"]["groups"]))
    log_snr_grid = [float(v) for v in probe["grid"]["log_snr_centers"]]
    if not all(log_snr_grid[i] < log_snr_grid[i + 1]
               for i in range(len(log_snr_grid) - 1)):
        raise ValueError("probe log_snr_centers must be strictly increasing")
    c_values = compute_c_values(probe, s_ref)
    if mean_checkpoint is not None:
        sha = load_mean_sha(Path(mean_checkpoint))
    elif mean_sha is not None:
        sha = str(mean_sha)
    else:
        raise ValueError("one of --mean-checkpoint / --mean-sha is required")
    psd = probe.get("psd_floors") or {g: float(sigma_floor) for g in groups_map}
    thresholds = dict(size_thresholds) if size_thresholds else dict(
        probe.get("size_thresholds") or DEFAULT_SIZE_THRESHOLDS)
    # schema v2 band powers (clock math review correction #3; audit band-clock
    # fix): P_g must be the MEAN PER-COEFFICIENT power of the group's Haar
    # residual coefficients on the freeze split - never the group total
    # (Parseval 1:3:12/16 would inflate the high group ~12x).  Never default
    # silently: absent CLI/probe values freeze the explicit all-1.0 form,
    # which IS matched control A (power-blind), and say so on stderr.
    powers = band_powers if band_powers is not None else probe.get("band_powers")
    if powers:
        powers = {str(g): float(v) for g, v in dict(powers).items()}
    else:
        powers = {g: 1.0 for g in groups_map}
        print("freeze: band powers absent; freezing explicit P_g=1.0 for every "
              "group (power-blind control A, DESIGN_RC_BRD_clock_v2 §3)")
    missing_p = [g for g in groups_map if g not in powers]
    if missing_p:
        raise ValueError(f"band powers are missing groups {missing_p}")
    bad_p = [g for g in groups_map
             if not (math.isfinite(float(powers[g])) and float(powers[g]) > 0.0)]
    if bad_p:
        raise ValueError(f"band powers must be finite and > 0 for {bad_p}")
    split_note = str(band_power_split) if band_power_split else f"outer_train_{fold}"
    payload = {"fold_id": str(fold), "band_groups": groups_map,
               "log_snr_grid": log_snr_grid, "c_values": c_values,
               "mean_checkpoint_sha256": sha,
               # AUDIT evidence-chain gate: b_active contains ONLY groups with
               # >=1 passing confirmatory cell (canonical group order kept).
               # Groups outside b_active run the ordinary clock via
               # effective_c == 0.5 (contract.py v1.0g semantics).
               "b_active": [g for g in groups_map if g in passing_groups],
               "psd_floors": {str(g): float(v) for g, v in psd.items()},
               "size_thresholds": thresholds, "kappa_grid": [float(k) for k in kappa_grid],
               "s_ref": float(s_ref), "support_mode": str(support_mode),
               "eta_max": float(eta_max), "floor_rho": float(floor_rho),
               "band_powers": {str(g): float(powers[g]) for g in groups_map},
               "band_powers_split": split_note}
    contract = RecoverabilityContract.from_payload(payload)
    contract.validate()  # fail-closed (FR-2.2; [计划] §3.5)
    artifact = dict(payload)
    artifact["contract_sha256"] = contract.contract_sha256
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def _parse_kappa(text: str) -> tuple[float, ...]:
    values = tuple(float(v) for v in text.split(",") if v.strip())
    if not values:
        raise ValueError("--kappa-grid must list at least one value")
    return values


def _parse_powers(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--band-powers entries must be key=value, got {item!r}")
        out[key] = float(value)
    return out


def _load_band_powers_file(path: Path) -> tuple[dict[str, float], str | None]:
    """Read a sealed band-powers artifact (compute_band_powers.py output).

    Accepts the schema written by scripts/compute_band_powers.py: a JSON
    object with a non-empty 'band_powers' {group: P_g > 0} mapping and
    optional provenance labels ('split_label' preferred,
    'band_powers_split' accepted for hand-written artifacts).  Any other
    shape raises ValueError — freeze never guesses (fail-closed).
    Returns (powers, split_label_or_None).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"band powers file {path} is not a JSON object")
    powers = payload.get("band_powers")
    if not isinstance(powers, Mapping) or not powers:
        raise ValueError(
            f"band powers file {path} has no non-empty 'band_powers' mapping")
    out: dict[str, float] = {}
    for group, value in powers.items():
        try:
            out[str(group)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"band powers file {path}: group {group!r} "
                             f"value {value!r} is not numeric") from exc
    split = payload.get("split_label") or payload.get("band_powers_split")
    return out, (str(split) if split else None)


def _parse_thresholds(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--size-thresholds entries must be key=value, got {item!r}")
        out[key] = float(value)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an R0 probe result into a recoverability contract "
                    "([计划] §3.5)")
    parser.add_argument("--probe-result", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--s-ref", type=float, required=True)
    parser.add_argument("--kappa-grid", required=True)
    mean = parser.add_mutually_exclusive_group(required=True)
    mean.add_argument("--mean-checkpoint")
    mean.add_argument("--mean-sha")
    parser.add_argument("--sigma-floor", type=float, default=DEFAULT_SIGMA_FLOOR)
    parser.add_argument("--size-thresholds", action="append", default=[])
    parser.add_argument("--support-mode", choices=("stratified_mixture", "floor_gated"),
                        default="floor_gated")
    parser.add_argument("--eta-max", type=float, default=0.8)
    parser.add_argument("--floor-rho", type=float, default=0.1)
    parser.add_argument("--band-powers", action="append", default=[],
                        help="group=mean-per-coefficient Haar residual power "
                             "(repeatable; absent -> explicit all-1.0 control A)")
    parser.add_argument("--band-powers-file", default=None,
                        help="sealed band-powers artifact from scripts/"
                             "compute_band_powers.py (per-fold outer-train "
                             "P_g); used when no explicit --band-powers "
                             "pair is given")
    parser.add_argument("--band-power-split", default=None,
                        help="provenance label of the P_g estimation split "
                             "(default outer_train_<fold>, or the artifact's "
                             "split_label when --band-powers-file is used)")
    args = parser.parse_args(argv)
    try:
        kappa = _parse_kappa(args.kappa_grid)
    except ValueError as exc:
        print(f"freeze failed: {exc}", file=sys.stderr)
        return 1
    if args.s_ref <= 0.0:
        print(f"freeze failed: --s-ref must be > 0, got {args.s_ref}", file=sys.stderr)
        return 1
    if 0.0 not in kappa:  # D1 strict fallback must stay selectable (DESIGN §3)
        print(f"freeze failed: --kappa-grid must contain 0.0, got {list(kappa)}",
              file=sys.stderr)
        return 1
    # P_g precedence (fail-closed, never silent): explicit --band-powers
    # pairs > sealed --band-powers-file artifact > probe artifact domain >
    # explicit all-1.0 power-blind control A.
    file_powers: dict[str, float] | None = None
    file_split: str | None = None
    if args.band_powers_file:
        try:
            file_powers, file_split = _load_band_powers_file(
                Path(args.band_powers_file))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"freeze failed: {exc}", file=sys.stderr)
            return 1
    explicit_powers = _parse_powers(args.band_powers)
    band_powers = explicit_powers if explicit_powers else file_powers
    band_power_split = args.band_power_split or (file_split if file_powers else None)
    try:
        artifact = freeze_contract(
            Path(args.probe_result), args.fold, Path(args.out), args.s_ref, kappa,
            mean_checkpoint=Path(args.mean_checkpoint) if args.mean_checkpoint else None,
            mean_sha=args.mean_sha, sigma_floor=args.sigma_floor,
            size_thresholds=_parse_thresholds(args.size_thresholds),
            support_mode=args.support_mode, eta_max=args.eta_max,
            floor_rho=args.floor_rho,
            band_powers=band_powers,
            band_power_split=band_power_split)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            ContractViolationError) as exc:
        print(f"freeze failed: {exc}", file=sys.stderr)
        return 1
    print(f"froze contract fold={artifact['fold_id']} "
          f"sha={artifact['contract_sha256'][:12]}... -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
