"""Comprehensive tests for the physics residual module."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from physics_informed_neural_network.physics import BurgersResidual


class TestBurgersResidual:
    """Verify the Burgers PDE residual implementation."""

    def test_initial_condition(self) -> None:
        pde = BurgersResidual(viscosity=0.01)
        x = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0])
        ic = pde.initial_condition(x)
        expected = -torch.sin(torch.pi * x)
        assert torch.allclose(ic, expected, atol=1e-6)

    def test_boundary_value(self) -> None:
        pde = BurgersResidual(viscosity=0.01)
        xt = torch.tensor([[-1.0, 0.5], [1.0, 0.3]])
        bv = pde.boundary_value(xt)
        assert bv.shape == (2, 1)
        assert torch.allclose(bv, torch.zeros(2, 1))

    def test_zero_residual_for_constant_field(self) -> None:
        """A model trained to output ~0 should have residual ~0."""
        pde = BurgersResidual(viscosity=0.01)

        # A small model with tanh ensures the computational graph supports 2nd derivatives
        model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        # Initialize output layer to near-zero so output ≈ 0
        with torch.no_grad():
            model[2].weight.zero_()
            model[2].bias.zero_()

        collocation = torch.rand(50, 2)
        residual = pde.residual(model, collocation)
        assert residual.shape == (50, 1)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)

    def test_residual_shape(self) -> None:
        pde = BurgersResidual(viscosity=0.1)

        model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
        collocation = torch.rand(30, 2)
        residual = pde.residual(model, collocation)
        assert residual.shape == (30, 1)
        assert torch.isfinite(residual).all()

    def test_residual_for_linear_solution(self) -> None:
        """A model outputting approximately u=x should have residual ≈ x (since u_xx≈0)."""
        pde = BurgersResidual(viscosity=0.5)

        # Build a model that approximates u ≈ x using a narrow tanh
        # The key: we need the graph to support 2nd derivatives
        model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
        with torch.no_grad():
            # Set first layer to pass through x coordinate weakly
            model[0].weight.zero_()
            model[0].weight[0, 0] = 0.01  # small so tanh ≈ linear
            model[0].bias.zero_()
            # Set output layer to amplify back
            model[2].weight.zero_()
            model[2].weight[0, 0] = 100.0  # 100 * tanh(0.01*x) ≈ x
            model[2].bias.zero_()

        collocation = torch.tensor([[0.5, 0.3], [0.0, 0.5], [-0.5, 0.7]], dtype=torch.float32)
        residual = pde.residual(model, collocation)
        # u ≈ x, u_t ≈ 0, u_x ≈ 1, u_xx ≈ 0 ⇒ r ≈ x
        expected = collocation[:, 0:1]
        assert torch.allclose(residual, expected, atol=0.05)  # approximate





