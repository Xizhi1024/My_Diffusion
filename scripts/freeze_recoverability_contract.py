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
  to freeze (close RC-BRD, fall back to D1);
* a --band-powers-file artifact is only accepted when its fold, split
  (outer-train), split_label, MeanNet SHA, band_groups AND self hash all
  match the contract being frozen (provenance gate, schema v2); explicit
  --band-powers pairs and the sealed file are mutually exclusive so
  numbers and provenance label can never come from different sources.

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
    compute_contract_sha256,
    mean_weights_sha256,
)

CONFIRMATORY_STRATUM = "small_lesion"  # [审计] §1.1: only confirmatory layer
DEFAULT_SIGMA_FLOOR = 1e-4
DEFAULT_SIZE_THRESHOLDS = {"small_lesion_q25": 25.0}  # until the frozen q25
# artifact lands; overridable via --size-thresholds.
# The outer-train side of a fold is the manifest split named "train"
# (compute_band_powers.py --split default); val/test artifacts must never
# freeze into a contract (outer-train no-leakage claim, provenance audit).
OUTER_TRAIN_SPLIT = "train"


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
                    band_power_split: str | None = None,
                    band_powers_artifact: Mapping[str, Any] | None = None) -> dict:
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
    # Provenance gate (P0, fail-closed): a sealed band-powers artifact is
    # only accepted when its fold / split / MeanNet SHA / band_groups / self
    # hash ALL match the contract being frozen.  Values and label then come
    # from the SAME verified artifact — numeric overrides that silently
    # inherit the file's split label are structurally impossible (the CLI
    # makes --band-powers and --band-powers-file mutually exclusive).
    if band_powers_artifact is not None:
        _validate_band_powers_provenance(band_powers_artifact, Path("<band-powers-file>"),
                                         fold, groups_map, sha)
        band_powers = {str(g): float(v)
                       for g, v in dict(band_powers_artifact["band_powers"]).items()}
        band_power_split = str(band_powers_artifact["split_label"])
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


def _validate_band_powers_provenance(payload: Mapping[str, Any], path: Path,
                                     fold: str, groups_map: dict[str, list[str]],
                                     mean_sha: str) -> None:
    """Evidence-chain gate for a sealed band-powers artifact (fail-closed).

    Provenance audit (P0): the old loader read only band_powers +
    split_label, so a fold_0 or val-split P_g could silently freeze into
    any fold's contract.  Every check below raises ValueError on mismatch:

    * artifact_sha256 present and equal to the canonical-JSON sha256 of
      the de-hashed payload (the artifact was not edited after sealing);
    * fold == --fold (a fold_0 artifact never freezes fold_3);
    * split == "train" and split_label == outer_train_<fold> (the P_g
      estimation side really is the fold's outer-train, not val/test);
    * mean_checkpoint_sha256 == the contract's MeanNet SHA (the P_g were
      measured against the SAME frozen mean the contract pins);
    * band_groups identical to the contract's probe-derived groups.
    """
    stored = payload.get("artifact_sha256")
    if not isinstance(stored, str) or not stored:
        raise ValueError(
            f"band powers file {path} is missing artifact_sha256 (sealed "
            "schema v2 from compute_band_powers.py required; hand-written "
            "artifacts must use explicit --band-powers pairs instead)")
    dehashed = {k: v for k, v in payload.items() if k != "artifact_sha256"}
    recomputed = compute_contract_sha256(dehashed)
    if stored != recomputed:
        raise ValueError(
            f"band powers file {path}: artifact_sha256 mismatch (stored "
            f"{stored[:12]!r}... != recomputed {recomputed[:12]!r}...) — the "
            "artifact was modified after sealing")
    artifact_fold = payload.get("fold")
    if artifact_fold != fold:
        raise ValueError(
            f"band powers file {path}: fold mismatch (artifact {artifact_fold!r} "
            f"!= --fold {fold!r}); P_g are fold-specific — recompute with "
            "scripts/compute_band_powers.py --fold " + repr(fold))
    artifact_split = payload.get("split")
    if artifact_split != OUTER_TRAIN_SPLIT:
        raise ValueError(
            f"band powers file {path}: split {artifact_split!r} is not the "
            f"outer-train side ({OUTER_TRAIN_SPLIT!r}); freezing val/test "
            "powers would break the outer-train no-leakage claim")
    expected_label = f"outer_train_{fold}"
    artifact_label = payload.get("split_label")
    if artifact_label != expected_label:
        raise ValueError(
            f"band powers file {path}: split_label {artifact_label!r} != "
            f"expected {expected_label!r} (split/label provenance mismatch)")
    artifact_sha = payload.get("mean_checkpoint_sha256")
    if artifact_sha != mean_sha:
        raise ValueError(
            f"band powers file {path}: mean_checkpoint_sha256 mismatch "
            f"(artifact {str(artifact_sha)[:12]!r}... != contract "
            f"{str(mean_sha)[:12]!r}...) — P_g must be measured against the "
            "SAME frozen MeanNet the contract pins")
    artifact_groups = payload.get("band_groups")
    if not isinstance(artifact_groups, Mapping):
        raise ValueError(
            f"band powers file {path}: band_groups must be a mapping")
    normalized = {str(g): [str(b) for b in bands]
                  for g, bands in artifact_groups.items()}
    if normalized != {g: list(b) for g, b in groups_map.items()}:
        raise ValueError(
            f"band powers file {path}: band_groups {normalized} != contract "
            f"groups {dict(groups_map)} (P_g layout must match the contract)")


def _load_band_powers_file(path: Path) -> dict[str, Any]:
    """Read a sealed band-powers artifact (compute_band_powers.py output).

    Accepts the schema written by scripts/compute_band_powers.py: a JSON
    object with a non-empty 'band_powers' {group: P_g > 0} mapping plus
    the sealed provenance fields (fold / split / split_label /
    mean_checkpoint_sha256 / band_groups / artifact_sha256).  The full
    payload is returned — freeze_contract re-validates provenance once
    the contract's fold/groups/MeanNet SHA are known.  Any other shape
    raises ValueError — freeze never guesses (fail-closed).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"band powers file {path} is not a JSON object")
    powers = payload.get("band_powers")
    if not isinstance(powers, Mapping) or not powers:
        raise ValueError(
            f"band powers file {path} has no non-empty 'band_powers' mapping")
    for group, value in powers.items():
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"band powers file {path}: group {group!r} "
                             f"value {value!r} is not numeric") from exc
    return dict(payload)


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
                             "(repeatable; hand-set pairs — provenance label "
                             "comes from --band-power-split or the "
                             "outer-train default; mutually exclusive with "
                             "--band-powers-file)")
    parser.add_argument("--band-powers-file", default=None,
                        help="sealed band-powers artifact from scripts/"
                             "compute_band_powers.py (per-fold outer-train "
                             "P_g).  Values AND the split label come from "
                             "this verified artifact (fold/split/MeanNet-SHA/"
                             "band_groups/self-hash all checked; mutually "
                             "exclusive with --band-powers and "
                             "--band-power-split)")
    parser.add_argument("--band-power-split", default=None,
                        help="provenance label for EXPLICIT --band-powers "
                             "pairs only (default outer_train_<fold>); a "
                             "sealed --band-powers-file artifact keeps its "
                             "own verified split_label")
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
    # P_g source (fail-closed, single origin): EITHER explicit --band-powers
    # pairs (label from --band-power-split or the outer-train default) OR
    # the sealed --band-powers-file artifact (values AND label from the one
    # verified artifact).  Mixing the two would let hand-set numbers inherit
    # the file's split label — a provenance mismatch, rejected.  With
    # neither given, the probe artifact domain applies, else the explicit
    # all-1.0 power-blind control A.
    if args.band_powers and args.band_powers_file:
        print("freeze failed: --band-powers and --band-powers-file are mutually "
              "exclusive (P_g numbers and provenance must come from ONE source)",
              file=sys.stderr)
        return 1
    if args.band_powers_file and args.band_power_split:
        print("freeze failed: --band-power-split cannot override a sealed "
              "--band-powers-file artifact (its verified split_label is "
              "authoritative)", file=sys.stderr)
        return 1
    artifact_payload: dict[str, Any] | None = None
    if args.band_powers_file:
        try:
            artifact_payload = _load_band_powers_file(
                Path(args.band_powers_file))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"freeze failed: {exc}", file=sys.stderr)
            return 1
    try:
        explicit_powers = _parse_powers(args.band_powers)
    except ValueError as exc:
        print(f"freeze failed: {exc}", file=sys.stderr)
        return 1
    try:
        artifact = freeze_contract(
            Path(args.probe_result), args.fold, Path(args.out), args.s_ref, kappa,
            mean_checkpoint=Path(args.mean_checkpoint) if args.mean_checkpoint else None,
            mean_sha=args.mean_sha, sigma_floor=args.sigma_floor,
            size_thresholds=_parse_thresholds(args.size_thresholds),
            support_mode=args.support_mode, eta_max=args.eta_max,
            floor_rho=args.floor_rho,
            band_powers=explicit_powers if explicit_powers else None,
            band_power_split=args.band_power_split,
            band_powers_artifact=artifact_payload)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            ContractViolationError) as exc:
        print(f"freeze failed: {exc}", file=sys.stderr)
        return 1
    print(f"froze contract fold={artifact['fold_id']} "
          f"sha={artifact['contract_sha256'][:12]}... -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
