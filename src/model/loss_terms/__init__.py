"""Loss term registry and base – time-varying activation gating."""

from .base import DisabledLossTerm, tau_gate
from .topk import TopKLesionLoss
from .frequency import FocalFrequencyLoss
from .roi_suv import ROISUVLoss
from .false_hotspot import FalseHotspotLoss
from .heteroscedastic import HeteroscedasticNLLLoss
from .hotspot import HotspotPriorLoss
from .patch_nce import PatchNCELoss
from .lesion_roi import LesionROIL1Loss, OutsidePeakRankingLoss
from .boundary_frequency import BoundaryFrequencyLoss
from .frequency_gate_tv import FrequencyGateTVLoss
from .normalized_lesion_peak import NormalizedLesionPeakLoss
from .spectral_router import SpectralRouterRegularizationLoss
from .perceptual_x0 import LesionAwarePerceptualX0Loss, PETFeatureEncoder
