"""Tests for the 2-D Darcy-flow FNO pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from physics_informed_neural_network.neural_operator.config import (
    DarcyExperimentConfig,
    DarcyProblemConfig,
    FourierNeuralOperator2dConfig,
)
from physics_informed_neural_network.neural_operator.data_2d import (
    DarcyDataset,
    build_darcy_splits,
    sample_grf_diffusivity,
    solve_darcy_2d,
)
from physics_informed_neural_network.neural_operator.model_2d import FourierNeuralOperator2d
from physics_informed_neural_network.neural_operator.presets import build_darcy_smoke_test_config
from physics_informed_neural_network.neural_operator.pipeline_2d import run_darcy_experiment


class TestGRFSampling:
    """Verify Gaussian Random Field sampling properties."""

    def test_piecewise_constant_values(self):
        problem = DarcyProblemConfig()
        rng = np.random.default_rng(42)
        fields = sample_grf_diffusivity(5, 29, problem, rng)
        unique_values = set(np.unique(fields))
        assert unique_values == {problem.a_lo, problem.a_hi}

    def test_smooth_field_positive(self):
        problem = DarcyProblemConfig(use_piecewise_constant=False)
        rng = np.random.default_rng(42)
        fields = sample_grf_diffusivity(5, 29, problem, rng)
        assert np.all(fields > 0)

    def test_shape(self):
        problem = DarcyProblemConfig()
        rng = np.random.default_rng(42)
        fields = sample_grf_diffusivity(10, 33, problem, rng)
        assert fields.shape == (10, 33, 33)


class TestDarcySolver:
    """Verify the 2-D Darcy finite-difference solver."""

    def test_boundary_conditions(self):
        N = 21
        a = np.ones((N, N)) * 5.0
        f = np.ones((N, N))
        h = 1.0 / (N - 1)
        u = solve_darcy_2d(a, f, h)
        # Dirichlet BCs: u = 0 on boundary
        assert np.allclose(u[0, :], 0.0)
        assert np.allclose(u[-1, :], 0.0)
        assert np.allclose(u[:, 0], 0.0)
        assert np.allclose(u[:, -1], 0.0)

    def test_solution_positive_interior(self):
        """For constant a > 0 and constant f > 0, solution should be positive in interior."""
        N = 21
        a = np.ones((N, N)) * 5.0
        f = np.ones((N, N))
        h = 1.0 / (N - 1)
        u = solve_darcy_2d(a, f, h)
        assert np.all(u[1:-1, 1:-1] > 0)


class TestFNO2dModel:
    """Verify forward pass shape and resolution transfer."""

    def test_forward_shape(self):
        config = FourierNeuralOperator2dConfig(width=16, modes_x=6, modes_y=6, layers=2, padding=3)
        model = FourierNeuralOperator2d(config)
        x = torch.randn(4, 29, 29, 4)
        out = model(x)
        assert out.shape == (4, 29, 29, 1)

    def test_resolution_transfer(self):
        config = FourierNeuralOperator2dConfig(width=16, modes_x=6, modes_y=6, layers=2, padding=3)
        model = FourierNeuralOperator2d(config)
        # Train resolution
        x1 = torch.randn(2, 29, 29, 4)
        out1 = model(x1)
        assert out1.shape == (2, 29, 29, 1)
        # Higher resolution (same model)
        x2 = torch.randn(2, 57, 57, 4)
        out2 = model(x2)
        assert out2.shape == (2, 57, 57, 1)


class TestDarcyPipelineSmoke:
    """End-to-end smoke test for the 2-D Darcy pipeline."""

    def test_smoke(self):
        config = build_darcy_smoke_test_config()
        experiment = run_darcy_experiment(config)
        assert experiment.native_prediction.shape[0] == config.data.test_samples
        assert experiment.refined_prediction.shape[0] == config.data.test_samples
        # The model should produce finite predictions
        assert np.all(np.isfinite(experiment.native_prediction))
        assert np.all(np.isfinite(experiment.refined_prediction))

