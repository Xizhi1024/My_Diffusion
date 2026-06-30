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
