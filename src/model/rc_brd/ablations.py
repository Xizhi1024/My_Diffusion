"""RC-BRD ablation configuration generator (FR-7, DESIGN §6).

Builds the nine pre-registered mechanism-attribution arms ([计划] §6.2) in
their final adjudicated form ([裁决] 附录 C-2/C-3; DESIGN §6):

    ABLATION_ARMS = A1, A2, A3a, A3b, A4, A5, A6, A8, A9

The generator has two layers:

1. Config layer - build_ablation_config deep-copies the base config and
   applies dotted-path overrides via config_utils.apply_dotlist_overrides
   (DESIGN §6: "dotlist 语义复用"). New config keys defined by this module
   (consumed by the integration-side _build_rc_brd per DESIGN §8 v1.0f;
   documented for cross-batch alignment):

       modules.rc_brd.contract_transform   # "none"|"flat_c"|"group_rotation"|"grid_reversal"
       modules.rc_brd.a3_variant           # "budget_matched"|"density_matched"  (C8)
       modules.rc_brd.density_match        # "logsnr"  (A3b only)
       modules.rc_brd.loss_weighting       # "per_band_min_snr"  (A8)
       modules.rc_brd.min_snr_gamma        # 5.0  (A8 clamp γ)

2. Contract layer - pure transforms implementing the A3/A4/A5 arm
   semantics on a loaded RecoverabilityContract. The config only carries
   the transform *tag* (contract_transform); apply_contract_transform
   dispatches that tag to the matching contract rewrite for whichever path
   consumes it (contract-freezing script or the integration-side
   _build_rc_brd per DESIGN §8 v1.0f), keeping this generator a pure
   dict -> dict function whose output is hash-freezable (FR-7.2).

Hashing: ablation_config_hash reuses the canonical-JSON sha256 convention
of contract.compute_contract_sha256 (DESIGN §3), so ablation generation
rules are sealable before the first outer-test unsealing ([计划] §0.4).

Sources
-------
- arm semantics (adjudicated final form): DESIGN §6; [裁决] 附录 C #2/#3/#4;
  [计划] §6.2 (original A1-A9 definitions)
- C8 a3_variant pairing and default: PRD §3 C8 (budget_matched ↔ A3a,
  density_matched ↔ A3b, default density_matched)
- A8 per-band Min-SNR-γ closed form min{SNR_b(t), γ}: [地图] §10.1 (Min-SNR
  #07, "clamped signal-to-noise ratios"); γ default 5.0 follows the repo's
  existing min_snr_gamma default ([地图] §2.1 #07)
- A9 endpoint swap zeros ↔ ct_minus_mean: DESIGN §6; C7; [裁决] 附录 C #4
  (this repo pins A9 to the endpoint swap, not the plan's input-scaling
  arm)
- A3 flat c ≡ 0.5: [审计] §3.3 (c0 = 0.5 ⇔ ordinary clock, ρ_b ≡ 1)
"""

from __future__ import annotations

from typing import Any, Mapping

from src.model.config_utils import apply_dotlist_overrides

from .contract import RecoverabilityContract, compute_contract_sha256
from .schedule import ENDPOINT_MODES

# [裁决] C-2/C-3 最终形态（DESIGN §6 逐字）：A7 被删除，A8/A9 按裁决重定义。
ABLATION_ARMS: tuple[str, ...] = (
    "A1", "A2", "A3a", "A3b", "A4", "A5", "A6", "A8", "A9",
)
# clock × specialist 2×2 factorial (mechanism-attribution directive: the
# RC arm stacks TWO independent mechanisms — the contracted band clock
# (κ≠0 time-change) and the bounded reverse specialist — and A1/A2 each
# toggle only one, leaving two cells of the matrix unmeasured.  The
# factorial names all four cells explicitly:
#   clock_off_specialist_off  — pure scalar D1 residual through rc_brd
#   clock_off_specialist_on   — specialist only   (≡ A1 semantics)
#   clock_on_specialist_off   — clock only        (≡ A2 semantics)
#   clock_on_specialist_on    — full RC mainline  (base κ, head on)
# 'clock' off means κ=0 (the time-changed clock degenerates to m≡t/T
# exactly; the band_snr query axis stays available for head gating);
# 'clock' on keeps the base config κ (must be ≠0, fail-closed).
CLOCK_SPECIALIST_ARMS: tuple[str, ...] = (
    "clock_off_specialist_off",
    "clock_off_specialist_on",
    "clock_on_specialist_off",
    "clock_on_specialist_on",
)
# band-SNR clock × loss-weighting pair (mechanism-attribution directive,
# provenance audit follow-up): the band_snr production config changes
# THREE things at once vs base RC — clock_mode, contract/P_g and
# loss_weighting — so a band_snr win over base-RC is not attributable to
# the clock alone.  These two arms hold the band-SNR clock fixed and
# toggle ONLY the loss, making the third mechanism measurable:
#   band_snr_clock_uniform_loss    — band_snr clock, uniform loss (isolates
#                                    the clock change vs base-RC)
#   band_snr_clock_band_snr_loss   — band_snr clock + per-band Min-SNR loss
#                                    (the full v2 stack; equals the
#                                    rc_brd_prod_rc_band_snr mainline cell)
# Both keep the base config's kappa and specialist settings verbatim —
# they decompose the clock_mode/loss factors, orthogonal to the κ ×
# specialist 2×2 above (which stays on the base clock_mode).
BAND_SNR_LOSS_ARMS: tuple[str, ...] = (
    "band_snr_clock_uniform_loss",
    "band_snr_clock_band_snr_loss",
)
# C8（PRD §3）：A3 sham 强度的两个变体。
A3_VARIANTS: tuple[str, ...] = ("budget_matched", "density_matched")
DEFAULT_A3_VARIANT: str = "density_matched"  # C8 默认（PRD §3）

# Contract transform tags carried by the generated configs (A3/A4/A5).
CONTRACT_TRANSFORMS: tuple[str, ...] = ("none", "flat_c", "group_rotation", "grid_reversal")

# C7（PRD §3）：A9 在 ENDPOINT_MODES（.schedule 单一来源，DESIGN §4）间互换。
_ENDPOINT_FLIP: dict[str, str] = {"zeros": "ct_minus_mean", "ct_minus_mean": "zeros"}
_DEFAULT_ENDPOINT_MODE: str = "zeros"  # DESIGN §8 schema 默认；[计划] §3.6 主线 e_r=0

_FLAT_C: float = 0.5            # [审计] §3.3: c0=0.5 ⇔ 2c−1=0 ⇔ ordinary clock
_MIN_SNR_GAMMA: float = 5.0     # A8 截断 γ；对齐仓库现行 min_snr_gamma 默认（[地图] §2.1 #07）
_DENSITY_MATCH_LOGSNR: str = "logsnr"  # A3b logSNR 密度匹配标记（[地图] §10.1 Laplace #20）
_LOSS_WEIGHTING_MIN_SNR: str = "per_band_min_snr"  # A8（[地图] §10.1 A8 行）

# A6（[计划] §6.2 "删除 lesion/cold/false-hotspot loss"）：匹配 token 与权重叶规则。
_LESION_SAFETY_TOKENS: tuple[str, ...] = ("lesion", "cold", "hot")
_WEIGHT_LEAF_SUFFIX: str = "weight"  # 只置零权重叶子（weight / *_weight）

# C8 臂名 ↔ 变体一一配对：A3a 固定 budget_matched，A3b 固定 density_matched。
_ARM_REQUIRED_A3_VARIANT: dict[str, str] = {"A3a": "budget_matched", "A3b": "density_matched"}
_A3_ARMS: tuple[str, ...] = ("A3a", "A3b")


# ---------------------------------------------------------------------------
# Contract-layer transforms (A3 / A4 / A5 semantics)
# ---------------------------------------------------------------------------

def _rebuild_validated(payload: Mapping[str, Any]) -> RecoverabilityContract:
    """Rebuild a contract from a transformed payload, fail-closed.

    from_payload recomputes the self hash (DESIGN §3); validate() then
    rejects any transform that broke a domain invariant (grid ordering,
    c ranges, key coverage).
    """
    rebuilt = RecoverabilityContract.from_payload(dict(payload))
    rebuilt.validate()
    return rebuilt


def flatten_contract(contract: RecoverabilityContract) -> RecoverabilityContract:
    """A3 (a/b common): flat contract with c ≡ 0.5 (DESIGN §6; [裁决] C-2).

    Only c_values is rewritten to [0.5] * len(log_snr_grid) for every
    group. With c ≡ 0.5 the clock weight is ρ_b = exp{κ(2c̃−1)} = 1 for
    every κ ([审计] §3.3 c0 definition), so the time-changed clock
    degenerates to the uniform one - the corruption-budget-matched sham
    baseline of C8.
    """
    payload = contract.to_payload()
    n_points = len(payload["log_snr_grid"])
    payload["c_values"] = {group: [_FLAT_C] * n_points for group in payload["c_values"]}
    return _rebuild_validated(payload)


def permute_contract_groups(contract: RecoverabilityContract) -> RecoverabilityContract:
    """A4: cyclic rotation of group-level contract evidence (DESIGN §6).

    Permutation table (deterministic, hence hash-stable): with G the group
    names in lexicographic order, group G[i] receives the evidence
    (c_values and psd_floors) of G[(i + 1) % n]. For the default 3-group
    layout the sorted order is ["high", "low", "mid"], i.e. high<-low,
    low<-mid, mid<-high. With a single group (n=1) the rotation is the
    identity. Structural fields (band_groups, b_active, grids, identity
    fields) are untouched, so the result re-validates and the arm isolates
    "recoverability evidence attached to the wrong group" ([计划] §6.2 A4
    shuffled contract).
    """
    payload = contract.to_payload()
    order = sorted(payload["band_groups"])
    n = len(order)
    payload["c_values"] = {
        group: list(payload["c_values"][order[(i + 1) % n]])
        for i, group in enumerate(order)
    }
    payload["psd_floors"] = {
        group: float(payload["psd_floors"][order[(i + 1) % n]])
        for i, group in enumerate(order)
    }
    return _rebuild_validated(payload)


def reverse_contract_grid(contract: RecoverabilityContract) -> RecoverabilityContract:
    """A5: mirror the contract c(λ) curve on the logSNR axis (DESIGN §6).

    New grid Λ'[i] = −Λ[n−1−i] (strictly increasing again, because Λ is),
    and c'[g][i] = c[g][n−1−i]: every grid point keeps its c value while
    its logSNR coordinate is negated. The recoverability↔logSNR association
    is therefore reversed while the c magnitudes are preserved - the sham
    required to attribute the effect to the direction of the association.
    All other fields are untouched.
    """
    payload = contract.to_payload()
    payload["log_snr_grid"] = [-x for x in reversed(payload["log_snr_grid"])]
    payload["c_values"] = {
        group: list(reversed(values)) for group, values in payload["c_values"].items()
    }
    return _rebuild_validated(payload)


def apply_contract_transform(contract: RecoverabilityContract,
                             transform: str) -> RecoverabilityContract:
    """Dispatch a config-tagged contract transform (A3/A4/A5; DESIGN §6/§8).

    The generated configs carry modules.rc_brd.contract_transform (consumed
    by the integration-side _build_rc_brd per DESIGN §8 v1.0f); this
    dispatcher maps the tag to the matching contract rewrite.
    "none" returns the contract unchanged; unknown tags raise ValueError
    (fail-closed, PRD NFR-5).
    """
    if transform == "none":
        return contract
    if transform == "flat_c":
        return flatten_contract(contract)
    if transform == "group_rotation":
        return permute_contract_groups(contract)
    if transform == "grid_reversal":
        return reverse_contract_grid(contract)
    raise ValueError(
        f"unknown contract transform {transform!r}; expected one of {CONTRACT_TRANSFORMS}")


# ---------------------------------------------------------------------------
# Config layer: arm -> dotlist overrides
# ---------------------------------------------------------------------------

def _is_numeric_weight(value: Any) -> bool:
    """True for int/float leaves (bools excluded) - the only zeroable kind."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _iter_weight_leaves(node: Mapping[str, Any], prefix: str):
    """Yield (dotted_path, leaf_key, value) for every leaf under node."""
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _iter_weight_leaves(value, dotted)
        else:
            yield dotted, str(key), value


def _lesion_safety_weight_overrides(config: Mapping[str, Any]) -> dict[str, float]:
    """A6: zero the existing lesion-safety loss weights (DESIGN §6; [计划] §6.2).

    Matching rule (deterministic, documented for hash stability): a leaf
    under "losses" is zeroed iff (a) its leaf key ends with "weight"
    (covers weight / cold_weight / distance_weight) and (b) its lowercase
    dotted path contains one of the tokens lesion / cold / hot. Only keys
    that already exist in the base config are written - no phantom keys
    are invented (batch instruction); non-weight leaves (enabled,
    active_tau_max, ...) and weights outside losses (e.g. model.base_loss)
    are never touched.
    """
    losses = config.get("losses", {})
    overrides: dict[str, float] = {}
    if not isinstance(losses, Mapping):
        return overrides
    for dotted, leaf_key, value in _iter_weight_leaves(losses, "losses"):
        if not _is_numeric_weight(value):
            continue
        path_lower = dotted.lower()
        leaf_lower = leaf_key.lower()
        if leaf_lower.endswith(_WEIGHT_LEAF_SUFFIX) and any(
                token in path_lower for token in _LESION_SAFETY_TOKENS):
            overrides[dotted] = 0.0
    return overrides


def _flip_endpoint_mode(config: Mapping[str, Any]) -> str:
    """A9: swap endpoint_mode zeros ↔ ct_minus_mean (DESIGN §6; C7).

    Reads modules.rc_brd.endpoint_mode (default "zeros" per DESIGN §8
    schema / [计划] §3.6) and returns the opposite mode. Any other value
    raises ValueError - the swap is only defined on the C7 enum.
    """
    modules = config.get("modules", {})
    rc_brd = modules.get("rc_brd", {}) if isinstance(modules, Mapping) else {}
    if not isinstance(rc_brd, Mapping):
        rc_brd = {}
    mode = rc_brd.get("endpoint_mode", _DEFAULT_ENDPOINT_MODE)
    flipped = _ENDPOINT_FLIP.get(str(mode))
    if flipped is None:
        raise ValueError(
            f"endpoint_mode {mode!r} cannot be swapped; expected one of "
            f"{ENDPOINT_MODES} (C7)")
    return flipped


def _base_rc_brd_config(base_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The modules.rc_brd mapping of the base config (empty if absent)."""
    modules = base_config.get("modules", {})
    rc_brd = modules.get("rc_brd", {}) if isinstance(modules, Mapping) else {}
    return rc_brd if isinstance(rc_brd, Mapping) else {}


def _four_arm_overrides(base_config: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Dotlist overrides of one clock × specialist factorial cell.

    clock off -> kappa=0 (exact scalar-D1 degeneration of the clock);
    clock on  -> the base config kappa, which MUST be non-zero (a zero
    base kappa would make the clock-on cell identical to clock-off and
    the factorial silently degenerate — fail-closed ValueError).
    The specialist flag follows the arm suffix verbatim.
    """
    rc_brd = _base_rc_brd_config(base_config)
    base_kappa = float(rc_brd.get("kappa", 0.0) or 0.0)
    clock_on = arm.startswith("clock_on")
    specialist_on = arm.endswith("specialist_on")
    overrides: dict[str, Any] = {"modules.rc_brd.enabled": True}
    if clock_on:
        if base_kappa == 0.0:
            raise ValueError(
                f"four-arm cell {arm!r} needs a base config with"
                " modules.rc_brd.kappa != 0 (the RC mainline arm); a"
                " kappa=0 base makes clock_on indistinguishable from"
                " clock_off")
        overrides["modules.rc_brd.kappa"] = base_kappa
    else:
        overrides["modules.rc_brd.kappa"] = 0.0
    overrides["modules.rc_brd.specialist.enabled"] = specialist_on
    return overrides


def _band_snr_loss_overrides(base_config: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Dotlist overrides of one band-SNR-clock × loss-weighting arm.

    Both arms pin clock_mode='band_snr' (requires a v2 contract with
    band_powers at model-build time — the schedule guard raises otherwise);
    the specialist flag and kappa follow the base config verbatim so these
    arms never confound the κ × specialist factorial.  The loss axis is the
    ONLY difference between the two arms (uniform vs per_band_min_snr).
    """
    overrides: dict[str, Any] = {
        "modules.rc_brd.enabled": True,
        "modules.rc_brd.clock_mode": "band_snr",
    }
    if arm == "band_snr_clock_uniform_loss":
        # Explicit (not merely inherited): the diff must show the loss axis.
        overrides["modules.rc_brd.loss_weighting"] = "uniform"
    else:  # band_snr_clock_band_snr_loss
        overrides["modules.rc_brd.loss_weighting"] = _LOSS_WEIGHTING_MIN_SNR
        overrides["modules.rc_brd.min_snr_gamma"] = _MIN_SNR_GAMMA
    return overrides


def _check_arm_and_variant(arm: str, a3_variant: str) -> None:
    """Validate arm identity and the C8 a3_variant pairing (fail-closed)."""
    known_arms = ABLATION_ARMS + CLOCK_SPECIALIST_ARMS + BAND_SNR_LOSS_ARMS
    if arm not in known_arms:
        raise ValueError(
            f"unknown ablation arm {arm!r}; expected one of {known_arms}")
    if a3_variant not in A3_VARIANTS:
        raise ValueError(
            f"unknown a3_variant {a3_variant!r}; expected one of {A3_VARIANTS} (C8)")
    if arm in _A3_ARMS:
        required = _ARM_REQUIRED_A3_VARIANT[arm]
        if a3_variant != required:
            raise ValueError(
                f"arm {arm!r} is pinned to a3_variant {required!r} by C8; got "
                f"{a3_variant!r}. Use build_ablation_config(base, {arm!r}, "
                f"a3_variant={required!r})")
    elif a3_variant != DEFAULT_A3_VARIANT:
        raise ValueError(
            f"a3_variant {a3_variant!r} only applies to arms {_A3_ARMS}; arm "
            f"{arm!r} must keep the default {DEFAULT_A3_VARIANT!r} (C8)")


def _arm_overrides(base_config: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Dotlist overrides implementing the arm semantics (DESIGN §6).

    Every arm turns the RC-BRD mainline on (modules.rc_brd.enabled=true):
    the ablations remove/permute one mechanism of M ([计划] §6.2 "消融只能
    删除、置换或匹配既有机制"), they are not scalar-baseline runs.
    """
    overrides: dict[str, Any] = {"modules.rc_brd.enabled": True}
    if arm == "A1":
        # κ=0 → forward degenerates to the scalar residual bridge (D1);
        # specialist stays on so its isolated contribution is measurable.
        overrides["modules.rc_brd.kappa"] = 0.0
        overrides["modules.rc_brd.specialist.enabled"] = True
    elif arm == "A2":
        # Forward bandwise clock kept (base κ); head bypassed identically.
        overrides["modules.rc_brd.specialist.enabled"] = False
    elif arm == "A3a":
        # Flat c≡0.5 + corruption-budget alignment (C8 取值 A).
        overrides["modules.rc_brd.contract_transform"] = "flat_c"
        overrides["modules.rc_brd.a3_variant"] = "budget_matched"
    elif arm == "A3b":
        # Flat c≡0.5 + logSNR density matching ([地图] §10.1 Laplace #20).
        overrides["modules.rc_brd.contract_transform"] = "flat_c"
        overrides["modules.rc_brd.a3_variant"] = "density_matched"
        overrides["modules.rc_brd.density_match"] = _DENSITY_MATCH_LOGSNR
    elif arm == "A4":
        overrides["modules.rc_brd.contract_transform"] = "group_rotation"
    elif arm == "A5":
        overrides["modules.rc_brd.contract_transform"] = "grid_reversal"
    elif arm == "A6":
        overrides.update(_lesion_safety_weight_overrides(base_config))
    elif arm == "A8":
        # κ=0 + per-band Min-SNR-γ loss weighting, closed form min{SNR_b(t), γ},
        # unmeasured and unfrozen ([地图] §10.1 A8 行).
        overrides["modules.rc_brd.kappa"] = 0.0
        overrides["modules.rc_brd.loss_weighting"] = _LOSS_WEIGHTING_MIN_SNR
        overrides["modules.rc_brd.min_snr_gamma"] = _MIN_SNR_GAMMA
    elif arm == "A9":
        overrides["modules.rc_brd.endpoint_mode"] = _flip_endpoint_mode(base_config)
    else:  # pragma: no cover - guarded by _check_arm_and_variant
        raise ValueError(f"unhandled ablation arm {arm!r}")
    return overrides


def build_ablation_config(base_config: dict, arm: str, *,
                          a3_variant: str = DEFAULT_A3_VARIANT) -> dict:
    """Return the deep-copied config with the arm overrides applied.

    Signature pinned by DESIGN §6. The base config is never mutated
    (apply_dotlist_overrides deep-copies first). Arm semantics
    (DESIGN §6 / [裁决] 附录 C final form):

    * A1: rc_brd on, κ=0 (scalar forward degenerate) + specialist on
    * A2: rc_brd on (bandwise forward) + specialist off (identity bypass)
    * A3a: contract_transform=flat_c + a3_variant=budget_matched
    * A3b: contract_transform=flat_c + a3_variant=density_matched +
      density_match=logsnr
    * A4: contract_transform=group_rotation (cyclic group-evidence rotation)
    * A5: contract_transform=grid_reversal (logSNR-axis mirror of c(λ))
    * A6: existing lesion-safety loss weights zeroed (losses.*lesion*/cold/
      hot weight leaves; existing keys only)
    * A8: κ=0 + loss_weighting=per_band_min_snr + min_snr_gamma=5.0
    * A9: endpoint_mode swapped zeros ↔ ct_minus_mean

    C8 pairing: A3a requires a3_variant="budget_matched" and A3b requires
    a3_variant="density_matched" (the default); any other arm must keep the
    default. Unknown arms or variants raise ValueError.

    Additionally accepts the clock × specialist 2×2 factorial arms
    CLOCK_SPECIALIST_ARMS (clock_{off,on}_specialist_{off,on}):

    * clock_off_specialist_off: κ=0 + specialist off (pure scalar-D1
      twin through rc_brd) — the cell A1/A2 never measured
    * clock_off_specialist_on:  κ=0 + specialist on (≡ A1 semantics)
    * clock_on_specialist_off:  base κ + specialist off (≡ A2 semantics)
    * clock_on_specialist_on:   base κ + specialist on (RC mainline cell)

    clock_on cells require a base config with modules.rc_brd.kappa != 0
    (fail-closed ValueError otherwise).

    Additionally accepts the band-SNR clock × loss-weighting arms
    BAND_SNR_LOSS_ARMS (band_snr_clock_{uniform,band_snr}_loss): both pin
    clock_mode='band_snr'; the loss axis (uniform vs per_band_min_snr) is
    the only difference.  kappa and specialist follow the base config so
    these arms stay orthogonal to the κ × specialist factorial (which
    itself stays on the base clock_mode).
    """
    _check_arm_and_variant(arm, a3_variant)
    if arm in CLOCK_SPECIALIST_ARMS:
        overrides = _four_arm_overrides(base_config, arm)
    elif arm in BAND_SNR_LOSS_ARMS:
        overrides = _band_snr_loss_overrides(base_config, arm)
    else:
        overrides = _arm_overrides(base_config, arm)
    return apply_dotlist_overrides(dict(base_config), overrides)


def ablation_config_hash(config: dict) -> str:
    """Canonical-JSON sha256 of a generated config (FR-7.2; DESIGN §6).

    Reuses the serialization convention of contract.compute_contract_sha256
    (DESIGN §3): json.dumps(config, sort_keys=True, separators=(",", ":"))
    hashed with sha256 - key-order independent and therefore stable across
    dict round trips, which lets the generation rule be sealed before the
    first outer-test unsealing ([计划] §0.4).
    """
    return compute_contract_sha256(config)