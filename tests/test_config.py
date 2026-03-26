"""Comprehensive tests for Pydantic configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from physics_informed_neural_network.config import (
    ArtifactConfig,
    DataConfig,
    LossWeights,
    NetworkConfig,
    PDEConfig,
    ProjectConfig,
    TrainingConfig,
)


class TestPDEConfig:
    """Verify PDE domain and physics constraints."""

    def test_defaults_are_valid(self) -> None:
        config = PDEConfig()
        assert config.viscosity > 0
        assert config.x_max > config.x_min
        assert config.t_max > config.t_min

    def test_rejects_invalid_spatial_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PDEConfig(x_min=1.0, x_max=-1.0)

    def test_rejects_invalid_temporal_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PDEConfig(t_min=1.0, t_max=0.0)

    def test_rejects_equal_spatial_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PDEConfig(x_min=0.0, x_max=0.0)

    def test_rejects_zero_viscosity(self) -> None:
        with pytest.raises(ValidationError):
            PDEConfig(viscosity=0.0)

    def test_rejects_negative_viscosity(self) -> None:
        with pytest.raises(ValidationError):
            PDEConfig(viscosity=-0.01)

    def test_custom_domain(self) -> None:
        config = PDEConfig(x_min=-2.0, x_max=2.0, t_min=0.0, t_max=5.0, viscosity=0.1)
        assert config.x_min == -2.0
        assert config.t_max == 5.0

    def test_validate_assignment(self) -> None:
        config = PDEConfig()
        with pytest.raises(ValidationError):
            config.x_max = config.x_min - 1.0


class TestDataConfig:
    """Verify data-generation configuration."""

    def test_defaults(self) -> None:
        config = DataConfig()
        assert config.nx > 0
        assert config.nt > 0
        assert config.n_observed > 0
        assert config.noise_std >= 0.0

    def test_rejects_zero_grid(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(nx=0)

    def test_rejects_negative_noise(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(noise_std=-0.1)


class TestNetworkConfig:
    """Verify architecture configuration."""

    def test_defaults(self) -> None:
        config = NetworkConfig()
        assert config.hidden_dim > 0
        assert config.hidden_layers > 0
        assert config.fourier_features > 0
        assert config.fourier_scale > 0

    def test_supported_activations(self) -> None:
        for act in ("tanh", "sin", "gelu"):
            config = NetworkConfig(activation=act)
            assert config.activation == act

    def test_rejects_bad_activation(self) -> None:
        with pytest.raises(ValidationError):
            NetworkConfig(activation="relu")


class TestLossWeights:
    """Verify loss weight configuration."""

    def test_defaults_positive(self) -> None:
        config = LossWeights()
        assert config.pde_residual > 0
        assert config.boundary > 0
        assert config.initial_condition > 0
        assert config.data > 0

    def test_rejects_zero_weight(self) -> None:
        with pytest.raises(ValidationError):
            LossWeights(pde_residual=0.0)


class TestTrainingConfig:
    """Verify training loop configuration."""

    def test_defaults(self) -> None:
        config = TrainingConfig()
        assert config.adam_epochs > 0
        assert config.adam_learning_rate > 0
        assert config.n_collocation > 0

    def test_lbfgs_can_be_zero(self) -> None:
        config = TrainingConfig(lbfgs_iterations=0)
        assert config.lbfgs_iterations == 0

    def test_rejects_negative_lbfgs(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(lbfgs_iterations=-1)

    def test_scheduler_options(self) -> None:
        for sched in ("cosine", "step", "none"):
            config = TrainingConfig(scheduler=sched)
            assert config.scheduler == sched

    def test_warmup_can_be_zero(self) -> None:
        config = TrainingConfig(warmup_steps=0)
        assert config.warmup_steps == 0


class TestArtifactConfig:
    """Verify artifact configuration."""

    def test_defaults(self) -> None:
        config = ArtifactConfig()
        assert config.prediction_nx > 0
        assert config.prediction_nt > 0


class TestProjectConfig:
    """Verify top-level configuration composition."""

    def test_defaults(self) -> None:
        config = ProjectConfig()
        assert config.pde.viscosity > 0
        assert config.data.nx > 0
        assert config.network.hidden_dim > 0
        assert config.training.adam_epochs > 0

    def test_rejects_invalid_nested_config(self) -> None:
        with pytest.raises(ValidationError):
            ProjectConfig(pde={"x_min": 1.0, "x_max": -1.0})

    def test_deep_copy(self) -> None:
        config = ProjectConfig()
        copy = config.model_copy(deep=True)
        copy.pde.viscosity = 0.5
        assert config.pde.viscosity != 0.5

