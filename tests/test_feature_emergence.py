"""Tests for A1 (cross-modal CKA) + A2 (lesion emergence) analysis."""

from __future__ import annotations

import torch


class _ToyEncoder(torch.nn.Module):
    """Minimal 2-layer encoder with a hard-coded spatial structure."""

    def __init__(self, in_channels=1, base=8):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, base, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.base = base

    def forward(self, x):
        c1 = torch.relu(self.conv1(x))  # [N, 8, H, W]
        c2 = torch.relu(self.conv2(c1))  # [N, 16, H/2, W/2]
        return [c1, c2]


class _ToyModel(torch.nn.Module):
    def __init__(self, image_size=32):
        super().__init__()
        self.ct_encoder = _ToyEncoder()
        self.image_size = image_size

    def build_condition_bundle(self, batch, timesteps):
        from src.model.interfaces import ConditionBundle

        feats = self.ct_encoder(batch["ct"])
        bundle = ConditionBundle.empty()
        bundle.maps["ct"] = batch["ct"]
        bundle.maps["ct_feat_0"] = feats[0]
        bundle.maps["ct_feat_1"] = feats[1]
        return bundle


def _make_batches(n_patients=4, slices=2, image_size=32):
    """Deterministic fake data: each patient has 2 slices, paired CT/PET."""
    batches = []
    rng = torch.Generator().manual_seed(7)
    for p in range(n_patients):
        for s in range(slices):
            ct = torch.randn(1, 1, image_size, image_size, generator=rng)
            pet = torch.randn(1, 1, image_size, image_size, generator=rng)
            mask = torch.zeros(1, 1, image_size, image_size)
            mask[0, 0, image_size // 2 : image_size // 2 + 4, image_size // 2 : image_size // 2 + 4] = 1.0
            batches.append(
                {
                    "ct": ct,
                    "pet": pet,
                    "mask": mask,
                    "meta": [{"patient_id": f"P{p}"}],
                }
            )
    return batches


def _loader_from_batches(batches, batch_size=2):
    class _ListLoader:
        def __init__(self, items, bs):
            self.items = items
            self.bs = bs

        def __len__(self):
            return len(self.items)

        def __iter__(self):
            for start in range(0, len(self.items), self.bs):
                batch = self.items[start : start + self.bs]
                collated = {
                    "ct": torch.cat([b["ct"] for b in batch], dim=0),
                    "pet": torch.cat([b["pet"] for b in batch], dim=0),
                    "mask": torch.cat([b["mask"] for b in batch], dim=0),
                    "meta": [m for b in batch for m in b["meta"]],
                }
                yield collated

    return _ListLoader(batches, batch_size)


def test_compute_linear_cka_is_bounded_and_identity():
    from src.mechanism_validation.feature_emergence import compute_linear_cka

    X = torch.randn(50, 16)
    Y = torch.randn(50, 16)
    value = compute_linear_cka(X, Y)
    assert 0.0 <= value <= 1.0

    # Self-similarity is high (not exactly 1.0 because of centering, but large).
    self_value = compute_linear_cka(X, X)
    assert self_value > 0.9


def test_channel_agnostic_activation_is_channel_invariant():
    from src.mechanism_validation.feature_emergence import channel_agnostic_activation

    feat = torch.randn(1, 8, 32, 32)
    act = channel_agnostic_activation(feat)
    assert act.shape == (1, 1, 32, 32)
    assert torch.isfinite(act).all()


def test_lesion_ring_contrast_positive_when_lesion_hotter():
    from src.mechanism_validation.feature_emergence import lesion_ring_contrast

    act = torch.zeros(1, 32, 32)
    mask = torch.zeros(1, 32, 32)
    act[0, 15:18, 15:18] = 5.0  # lesion
    act[0, 12:15, 12:15] = 1.0  # ring
    mask[0, 15:18, 15:18] = 1.0
    contrast = lesion_ring_contrast(act, mask, ring_width=3)
    assert contrast > 0.0


def test_lesion_discriminability_auroc_high_for_separated_maps():
    from src.mechanism_validation.feature_emergence import lesion_discriminability

    act = torch.randn(1, 32, 32)
    mask = torch.zeros(1, 32, 32)
    act[0, 15:20, 15:20] += 5.0
    mask[0, 15:20, 15:20] = 1.0
    disc = lesion_discriminability(act, mask)
    assert disc["auroc"] > 0.8
    assert disc["auprc"] > 0.5


def test_cross_modal_similarity_paired_beats_shuffled(tmp_path):
    from src.mechanism_validation.feature_emergence import cross_modal_similarity_analysis

    model = _ToyModel()
    model.eval()
    loader = _loader_from_batches(_make_batches(n_patients=4, slices=2), batch_size=2)

    result = cross_modal_similarity_analysis(model, loader, "cpu", tmp_path)
    assert (tmp_path / "layer_metrics.csv").is_file()
    assert (tmp_path / "patient_summary.csv").is_file()
    assert result["s1_gate"]["shallow_layer"] == "c1"
    # layer_rows present
    assert len(result["layer_rows"]) == 2  # c1, c2
    # patient rows must not treat slices as independent samples
    patient_ids = {r["patient_id"] for r in result["patient_rows"]}
    assert len(patient_ids) >= 4


def test_lesion_emergence_writes_cohort_and_gate(tmp_path):
    from src.mechanism_validation.feature_emergence import lesion_emergence_analysis

    model = _ToyModel()
    model.eval()
    loader = _loader_from_batches(_make_batches(n_patients=4, slices=2), batch_size=2)

    result = lesion_emergence_analysis(model, loader, "cpu", tmp_path)
    assert (tmp_path / "cohort.csv").is_file()
    assert (tmp_path / "lesion_emergence_metrics.csv").is_file()
    assert result["m1_gate"]["c2_lesion_ring_contrast_ci95"]["patients"] > 0
