"""RC-BRD: Recoverability-Contracted Bandwise Residual Diffusion.

Paper-level name: RC-SRBB (see docs/prd/PRD_RC_BRD_v1.md).
Public API is pinned by docs/prd/DESIGN_RC_BRD_v1.md (section 7).
"""

from .wavelet import (
    BAND_NAMES,
    DETAIL_BANDS,
    band_groups,
    haar_forward2,
    haar_inverse2,
    roundtrip_error,
)
from .contract import (
    CONTRACT_SCHEMA_VERSION,
    SUPPORT_MODES,
    ContractViolationError,
    RecoverabilityContract,
    compute_contract_sha256,
    expand_group_to_bands,
    mean_weights_sha256,
)
from .schedule import (
    ENDPOINT_MODES,
    FORWARD_MODES,
    BandwiseBridgeSchedule,
    BandwiseScheduleConfig,
)
from .head import (
    CTFeatureToken,
    SpecialistConfig,
    BoundedSpecialistHead,
    background_delta_energy,
)
from .ablations import (
    A3_VARIANTS,
    ABLATION_ARMS,
    CONTRACT_TRANSFORMS,
    ablation_config_hash,
    apply_contract_transform,
    build_ablation_config,
)

__all__ = [
    "BAND_NAMES",
    "DETAIL_BANDS",
    "band_groups",
    "haar_forward2",
    "haar_inverse2",
    "roundtrip_error",
    "CONTRACT_SCHEMA_VERSION",
    "SUPPORT_MODES",
    "ContractViolationError",
    "RecoverabilityContract",
    "compute_contract_sha256",
    "expand_group_to_bands",
    "mean_weights_sha256",
    "ENDPOINT_MODES",
    "FORWARD_MODES",
    "BandwiseBridgeSchedule",
    "BandwiseScheduleConfig",
    "SpecialistConfig",
    "BoundedSpecialistHead",
    "CTFeatureToken",
    "background_delta_energy",
    "A3_VARIANTS",
    "ABLATION_ARMS",
    "CONTRACT_TRANSFORMS",
    "ablation_config_hash",
    "apply_contract_transform",
    "build_ablation_config",
]
