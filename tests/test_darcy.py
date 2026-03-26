"""Comprehensive tests for the 2-D Darcy-flow FNO pipeline."""

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
from physics_informed_neural_network.neural_operator.model_2d import (
    FourierNeuralOperator2d,
    SpectralConv2d,
)
from physics_informed_neural_network.neural_operator.presets import build_darcy_smoke_test_config
from physics_informed_neural_network.neural_operator.pipeline_2d import run_darcy_experiment
from physics_informed_neural_network.neural_operator.training_2d import (
    TensorNormalizer2d,
    compute_error_metrics_2d,
)


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

    def test_symmetry_with_symmetric_input(self):
        """Symmetric diffusivity and forcing should produce symmetric solution."""
        N = 21
        a = np.ones((N, N)) * 3.0
        f = np.ones((N, N))
        h = 1.0 / (N - 1)
        u = solve_darcy_2d(a, f, h)
        assert np.allclose(u, u.T, atol=1e-10)


class TestTensorNormalizer2d:
    """Verify 2-D tensor normalizer."""

    def test_round_trip(self):
        data = torch.randn(8, 15, 15, 2)
        normalizer = TensorNormalizer2d.fit(data, dims=(0, 1, 2))
        recovered = normalizer.inverse(normalizer.transform(data))
        assert torch.allclose(data, recovered, atol=1e-5)

    def test_device_transfer(self):
        data = torch.randn(4, 10, 10, 1)
        normalizer = TensorNormalizer2d.fit(data)
        moved = normalizer.to(torch.device("cpu"))
        assert moved.mean.device.type == "cpu"


class TestSpectralConv2d:
    """Verify 2-D spectral convolution."""

    def test_output_shape(self):
        layer = SpectralConv2d(in_channels=8, out_channels=8, modes_x=4, modes_y=4)
        x = torch.randn(2, 8, 16, 16)
        out = layer(x)
        assert out.shape == (2, 8, 16, 16)

    def test_resolution_transfer(self):
        layer = SpectralConv2d(in_channels=4, out_channels=4, modes_x=3, modes_y=3)
        x1 = torch.randn(1, 4, 16, 16)
        x2 = torch.randn(1, 4, 32, 32)
        assert layer(x1).shape == (1, 4, 16, 16)
        assert layer(x2).shape == (1, 4, 32, 32)


class TestComputeErrorMetrics2d:
    """Verify 2-D error metrics."""

    def test_zero_error(self):
        a = np.ones((3, 10, 10))
        metrics = compute_error_metrics_2d(a, a)
        assert metrics.mse == 0.0
        assert metrics.relative_l2 < 1e-10

    def test_known_error(self):
        pred = np.zeros((1, 5, 5))
        true = np.ones((1, 5, 5))
        metrics = compute_error_metrics_2d(pred, true)
        assert metrics.mse == 1.0
        assert metrics.max_absolute_error == 1.0


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
        x1 = torch.randn(2, 29, 29, 4)
        out1 = model(x1)
        assert out1.shape == (2, 29, 29, 1)
        x2 = torch.randn(2, 57, 57, 4)
        out2 = model(x2)
        assert out2.shape == (2, 57, 57, 1)

    def test_architecture_string(self):
        config = FourierNeuralOperator2dConfig(width=32, modes_x=8, modes_y=8, layers=3, padding=5)
        model = FourierNeuralOperator2d(config)
        s = model.architecture_string()
        assert "32" in s
        assert "FourierNeuralOperator2d" in s

    def test_count_parameters(self):
        config = FourierNeuralOperator2dConfig(width=16, modes_x=4, modes_y=4, layers=2, padding=2)
        model = FourierNeuralOperator2d(config)
        assert model.count_parameters() > 0


class TestDarcyDataset:
    """Verify Darcy dataset properties."""

    def test_features_shape(self):
        config = build_darcy_smoke_test_config()
        splits = build_darcy_splits(config)
        features = splits.train.features()
        n = config.data.train_samples
        r = config.data.train_resolution
        assert features.shape[0] == n
        assert features.shape[1] == r
        assert features.shape[2] == r

    def test_targets_shape(self):
        config = build_darcy_smoke_test_config()
        splits = build_darcy_splits(config)
        targets = splits.train.targets()
        n = config.data.train_samples
        r = config.data.train_resolution
        assert targets.shape == (n, r, r, 1)


class TestDarcyPipelineSmoke:
    """End-to-end smoke test for the 2-D Darcy pipeline."""

    def test_smoke(self):
        config = build_darcy_smoke_test_config()
        experiment = run_darcy_experiment(config)
        assert experiment.native_prediction.shape[0] == config.data.test_samples
        assert experiment.refined_prediction.shape[0] == config.data.test_samples
        assert np.all(np.isfinite(experiment.native_prediction))
        assert np.all(np.isfinite(experiment.refined_prediction))
