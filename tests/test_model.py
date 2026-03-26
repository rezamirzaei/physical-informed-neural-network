"""Comprehensive tests for the PINN model architecture components."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from physics_informed_neural_network.config import NetworkConfig
from physics_informed_neural_network.model import (
    AdaptiveLossWeights,
    BurgersPINN,
    FourierFeatureEmbedding,
    ResidualBlock,
)


class TestFourierFeatureEmbedding:
    """Verify random Fourier feature embedding."""

    def test_output_shape(self) -> None:
        emb = FourierFeatureEmbedding(input_dim=2, n_features=64, scale=1.0)
        x = torch.randn(10, 2)
        out = emb(x)
        assert out.shape == (10, 128)  # 64 sin + 64 cos

    def test_output_dim_property(self) -> None:
        emb = FourierFeatureEmbedding(input_dim=2, n_features=32, scale=2.0)
        assert emb.output_dim == 64

    def test_deterministic_after_creation(self) -> None:
        emb = FourierFeatureEmbedding(input_dim=2, n_features=16, scale=1.0)
        x = torch.randn(5, 2)
        out1 = emb(x)
        out2 = emb(x)
        assert torch.allclose(out1, out2)


class TestResidualBlock:
    """Verify skip-connection residual block."""

    def test_output_shape(self) -> None:
        block = ResidualBlock(dim=32, activation=nn.Tanh())
        x = torch.randn(10, 32)
        out = block(x)
        assert out.shape == (10, 32)

    def test_skip_connection_passthrough(self) -> None:
        """With zero weights, output should roughly equal input."""
        block = ResidualBlock(dim=16, activation=nn.Tanh())
        with torch.no_grad():
            block.linear1.weight.zero_()
            block.linear1.bias.zero_()
            block.linear2.weight.zero_()
            block.linear2.bias.zero_()
        x = torch.randn(5, 16)
        out = block(x)
        # tanh(0) = 0, so residual path is 0, output = x + 0 = x
        assert torch.allclose(out, x, atol=1e-6)


class TestBurgersPINN:
    """Verify full PINN model."""

    def test_forward_shape(self) -> None:
        config = NetworkConfig()
        model = BurgersPINN(config)
        xt = torch.randn(20, 2)
        out = model(xt)
        assert out.shape == (20, 1)

    def test_count_parameters(self) -> None:
        config = NetworkConfig()
        model = BurgersPINN(config)
        assert model.count_parameters() > 0

    def test_architecture_string(self) -> None:
        config = NetworkConfig()
        model = BurgersPINN(config)
        s = model.architecture_string()
        assert "BurgersPINN" in s
        assert str(config.hidden_dim) in s

    def test_gradient_flow(self) -> None:
        """Verify that gradients propagate through the full model."""
        config = NetworkConfig(hidden_dim=16, hidden_layers=2, fourier_features=8)
        model = BurgersPINN(config)
        xt = torch.randn(10, 2, requires_grad=True)
        out = model(xt)
        loss = out.sum()
        loss.backward()
        assert xt.grad is not None
        assert torch.isfinite(xt.grad).all()


class TestAdaptiveLossWeights:
    """Verify learnable loss weights."""

    def test_weights_sum(self) -> None:
        alw = AdaptiveLossWeights(1.0, 1.0, 1.0, 1.0)
        weights = alw()
        total = sum(weights.values())
        assert abs(float(total) - 4.0) < 1e-5  # 4 terms

    def test_positive_weights(self) -> None:
        alw = AdaptiveLossWeights(1.0, 0.1, 10.0, 0.5)
        weights = alw()
        for v in weights.values():
            assert float(v) > 0

    def test_snapshot(self) -> None:
        alw = AdaptiveLossWeights(1.0, 1.0, 1.0, 1.0)
        snap = alw.snapshot()
        assert set(snap.keys()) == set(AdaptiveLossWeights.TERM_NAMES)
        assert all(isinstance(v, float) for v in snap.values())

    def test_logit_initialization(self) -> None:
        alw = AdaptiveLossWeights(2.0, 3.0, 4.0, 5.0)
        expected = [math.log(2.0), math.log(3.0), math.log(4.0), math.log(5.0)]
        for actual, exp in zip(alw.logits.tolist(), expected):
            assert abs(actual - exp) < 1e-5

