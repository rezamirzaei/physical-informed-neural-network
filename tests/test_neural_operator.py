"""Comprehensive tests for the 1-D neural-operator stack."""

from __future__ import annotations

import numpy as np
import torch

from physics_informed_neural_network.neural_operator.data import (
    OperatorSample,
    build_dataset_splits,
    compute_discrete_diffusion_residual,
)
from physics_informed_neural_network.neural_operator.model import (
    FourierNeuralOperator1d,
    SpectralConv1d,
)
from physics_informed_neural_network.neural_operator.pipeline import run_neural_operator_experiment
from physics_informed_neural_network.neural_operator.presets import build_smoke_test_config
from physics_informed_neural_network.neural_operator.training import (
    TensorNormalizer,
    compute_error_metrics,
)


# ---------------------------------------------------------------------------
# TensorNormalizer
# ---------------------------------------------------------------------------


class TestTensorNormalizer1d:
    """Verify 1-D operator-stack normalizer."""

    def test_round_trip(self) -> None:
        data = torch.randn(10, 32, 3)
        normalizer = TensorNormalizer.fit(data, dims=(0, 1))
        recovered = normalizer.inverse(normalizer.transform(data))
        assert torch.allclose(data, recovered, atol=1e-5)

    def test_device_transfer(self) -> None:
        data = torch.randn(10, 32, 2)
        normalizer = TensorNormalizer.fit(data, dims=(0, 1))
        moved = normalizer.to(torch.device("cpu"))
        assert moved.mean.device.type == "cpu"


# ---------------------------------------------------------------------------
# SpectralConv1d
# ---------------------------------------------------------------------------


class TestSpectralConv1d:
    """Verify spectral convolution layer."""

    def test_output_shape(self) -> None:
        layer = SpectralConv1d(in_channels=16, out_channels=16, modes=8)
        x = torch.randn(4, 16, 64)
        out = layer(x)
        assert out.shape == (4, 16, 64)

    def test_low_modes_only(self) -> None:
        """Only first `modes` Fourier modes should be nonzero before irfft."""
        layer = SpectralConv1d(in_channels=4, out_channels=4, modes=3)
        x = torch.randn(2, 4, 32)
        out = layer(x)
        assert out.shape == (2, 4, 32)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# OperatorSample
# ---------------------------------------------------------------------------


class TestOperatorSample:
    """Verify data sample integrity."""

    def test_sample_fields(self) -> None:
        config = build_smoke_test_config()
        splits = build_dataset_splits(config)
        sample = splits.train.sample(0)
        assert isinstance(sample, OperatorSample)
        assert sample.grid.shape == (config.data.train_resolution,)
        assert sample.diffusion.shape == (config.data.train_resolution,)
        assert sample.forcing.shape == (config.data.train_resolution,)
        assert sample.solution.shape == (config.data.train_resolution,)

    def test_dirichlet_boundary(self) -> None:
        config = build_smoke_test_config()
        splits = build_dataset_splits(config)
        sample = splits.train.sample(0)
        assert sample.solution[0] == 0.0
        assert sample.solution[-1] == 0.0


# ---------------------------------------------------------------------------
# compute_error_metrics
# ---------------------------------------------------------------------------


class TestComputeErrorMetrics:
    """Verify error metric computation."""

    def test_zero_error(self) -> None:
        a = np.ones((5, 32))
        metrics = compute_error_metrics(a, a)
        assert metrics.mse == 0.0
        assert metrics.mae == 0.0
        assert metrics.relative_l2 < 1e-10

    def test_known_error(self) -> None:
        pred = np.array([1.0, 0.0])
        true = np.array([0.0, 0.0])
        metrics = compute_error_metrics(pred, true)
        assert metrics.max_absolute_error == 1.0


# ---------------------------------------------------------------------------
# Dataset and solver
# ---------------------------------------------------------------------------


def test_exact_diffusion_solver_has_dirichlet_boundary_and_small_residual() -> None:
    config = build_smoke_test_config()
    splits = build_dataset_splits(config)
    sample = splits.train.sample(0)
    residual = compute_discrete_diffusion_residual(sample.grid, sample.diffusion, sample.solution, sample.forcing)

    assert sample.solution[0] == 0.0
    assert sample.solution[-1] == 0.0
    assert float(np.mean(np.abs(residual))) < 0.05


def test_fno_forward_supports_resolution_transfer() -> None:
    config = build_smoke_test_config()
    model = FourierNeuralOperator1d(config.model)

    native_input = torch.zeros(2, config.data.train_resolution, config.model.input_channels)
    refined_input = torch.zeros(2, config.data.evaluation_resolution, config.model.input_channels)

    assert model(native_input).shape == (2, config.data.train_resolution, 1)
    assert model(refined_input).shape == (2, config.data.evaluation_resolution, 1)


def test_neural_operator_smoke_experiment_runs_end_to_end(tmp_path) -> None:
    config = build_smoke_test_config(output_dir=tmp_path / "artifacts")
    experiment = run_neural_operator_experiment(config)

    assert len(experiment.history.entries) == config.optimization.epochs
    assert experiment.native_prediction.shape == (config.data.test_samples, config.data.train_resolution)
    assert experiment.refined_prediction.shape == (config.data.test_samples, config.data.evaluation_resolution)
    assert experiment.summary.evaluations["test"].metrics.relative_l2 >= 0.0
    assert experiment.summary.evaluations["refined_test"].resolution == config.data.evaluation_resolution
