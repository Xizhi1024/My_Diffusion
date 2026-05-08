import torch
from src.model.encoder import MultiScaleStem, DualStreamCTEncoder


def test_multi_scale_stem():
    """MultiScaleStem should preserve spatial size and set output channels."""
    stem = MultiScaleStem(in_channels=1, base_channels=64)

    x = torch.randn(2, 1, 128, 128)
    out = stem(x)

    assert out.shape == (2, 64, 128, 128)


def test_dual_stream_encoder():
    """DualStreamCTEncoder should output the requested channel count."""
    encoder = DualStreamCTEncoder(
        in_channels=1,
        base_channels=32,
        out_channels=64,
        num_heads=4,
        num_res_blocks=2,
    )

    x = torch.randn(2, 1, 128, 128)
    out = encoder(x)

    assert out.shape == (2, 64, 128, 128)


def test_gradient_flow():
    """Encoder parameters and inputs should receive gradients."""
    encoder = DualStreamCTEncoder(
        in_channels=1,
        base_channels=16,
        out_channels=16,
        num_heads=2,
        num_res_blocks=1,
        groups=4,
    )
    x = torch.randn(1, 1, 64, 64, requires_grad=True)

    out = encoder(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    has_grad = all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert has_grad


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
