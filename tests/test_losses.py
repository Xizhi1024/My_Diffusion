import torch
from src.model.losses import (
    WeightedPETLoss,
    GradientLoss,
    FocalFrequencyLoss,
    QuantitativeConstraintLoss,
    CombinedDiffusionLoss,
)


def test_weighted_pet_loss():
    """WeightedPETLoss should prioritize hotspot errors."""
    loss_fn = WeightedPETLoss(threshold=0.3, high_weight=10.0)

    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:30, 20:30] = 0.8
    target[:, :, 50:55, 50:55] = 0.9

    pred_perfect = target.clone()
    loss_perfect = loss_fn(pred_perfect, target)

    pred_noisy = target + torch.randn_like(target) * 0.1
    loss_noisy = loss_fn(pred_noisy, target)

    pred_wrong = torch.zeros_like(target)
    loss_wrong = loss_fn(pred_wrong, target)

    assert loss_perfect < loss_noisy < loss_wrong


def test_gradient_loss():
    """GradientLoss should penalize blurry edges more than exact matches."""
    loss_fn = GradientLoss()

    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:40, 20:40] = 1.0

    pred_perfect = target.clone()
    loss_perfect = loss_fn(pred_perfect, target)

    pred_blurry = torch.nn.functional.avg_pool2d(target, 5, 1, 2)
    loss_blurry = loss_fn(pred_blurry, target)

    assert loss_perfect < loss_blurry


def test_focal_frequency_loss():
    """FocalFrequencyLoss should be near-zero for identical tensors."""
    loss_fn = FocalFrequencyLoss()

    target = torch.randn(2, 1, 64, 64)
    pred_perfect = target.clone()
    pred_different = torch.randn(2, 1, 64, 64)

    loss_perfect = loss_fn(pred_perfect, target)
    loss_different = loss_fn(pred_different, target)

    assert loss_perfect < loss_different


def test_combined_loss():
    """CombinedDiffusionLoss should return all enabled components."""
    loss_fn = CombinedDiffusionLoss(
        noise_weight=1.0,
        pet_weight=0.5,
        gradient_weight=0.1,
        frequency_weight=0.1,
    )

    pred_noise = torch.randn(2, 4, 32, 32)
    target_noise = torch.randn(2, 4, 32, 32)
    pred_x0 = torch.rand(2, 1, 64, 64)
    target_x0 = torch.rand(2, 1, 64, 64)

    total_loss, loss_dict = loss_fn(pred_noise, target_noise, pred_x0, target_x0)

    assert total_loss.item() > 0
    assert 'noise_loss' in loss_dict
    assert 'pet_loss' in loss_dict


def test_gradient_flow():
    """All learnable inputs should receive gradients."""
    loss_fn = CombinedDiffusionLoss()

    pred_noise = torch.randn(1, 4, 16, 16, requires_grad=True)
    target_noise = torch.randn(1, 4, 16, 16)
    pred_x0 = torch.randn(1, 1, 32, 32, requires_grad=True)
    target_x0 = torch.randn(1, 1, 32, 32)

    total_loss, _ = loss_fn(pred_noise, target_noise, pred_x0, target_x0)
    total_loss.backward()

    assert pred_noise.grad is not None
    assert pred_x0.grad is not None


def test_quantitative_constraint_loss_zero_for_identical_inputs():
    loss_fn = QuantitativeConstraintLoss(
        hotspot_threshold=0.2,
        topk_percent=0.02,
    )
    x = torch.rand(2, 1, 32, 32)
    loss, items = loss_fn(x, x)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6
    assert set(items.keys()) == {'mean', 'integral', 'hotspot', 'peak'}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
