"""Reusable presets for neural-operator smoke tests and notebook demos."""

from __future__ import annotations

from pathlib import Path

from .config import DarcyExperimentConfig, NeuralOperatorExperimentConfig


def apply_smoke_test_preset(
    config: NeuralOperatorExperimentConfig,
    output_dir: str | Path | None = None,
) -> NeuralOperatorExperimentConfig:
    """Return a fast-running preset suitable for CI and local verification."""
    config = config.model_copy(deep=True)

    config.data.train_samples = 48
    config.data.validation_samples = 16
    config.data.test_samples = 16
    config.data.train_resolution = 32
    config.data.evaluation_resolution = 64

    config.model.width = 24
    config.model.modes = 8
    config.model.layers = 3
    config.model.padding = 4

    config.optimization.device = "cpu"
    config.optimization.batch_size = 8
    config.optimization.epochs = 6
    config.optimization.scheduler_step = 3
    config.optimization.log_every = 3

    config.artifacts.save_artifacts = False
    config.artifacts.output_dir = (
        Path(output_dir) if output_dir is not None else Path("artifacts/neural_operator_smoke")
    )
    return config


def build_smoke_test_config(output_dir: str | Path | None = None) -> NeuralOperatorExperimentConfig:
    """Create a fresh smoke-test configuration."""
    return apply_smoke_test_preset(NeuralOperatorExperimentConfig(), output_dir=output_dir)


def apply_tutorial_preset(
    config: NeuralOperatorExperimentConfig,
    output_dir: str | Path | None = None,
) -> NeuralOperatorExperimentConfig:
    """Return a notebook-friendly preset with moderate training cost."""
    config = config.model_copy(deep=True)

    config.data.train_samples = 256
    config.data.validation_samples = 64
    config.data.test_samples = 64
    config.data.train_resolution = 64
    config.data.evaluation_resolution = 128

    config.model.width = 48
    config.model.modes = 16
    config.model.layers = 4
    config.model.padding = 8

    config.optimization.device = "cpu"
    config.optimization.batch_size = 32
    config.optimization.epochs = 80
    config.optimization.scheduler_step = 25
    config.optimization.log_every = 10

    config.artifacts.save_artifacts = False
    config.artifacts.output_dir = Path(output_dir) if output_dir is not None else Path("artifacts/neural_operator")
    return config


def build_tutorial_config(output_dir: str | Path | None = None) -> NeuralOperatorExperimentConfig:
    """Create a fresh tutorial configuration."""
    return apply_tutorial_preset(NeuralOperatorExperimentConfig(), output_dir=output_dir)


# ---------------------------------------------------------------------------
# 2-D Darcy flow presets
# ---------------------------------------------------------------------------

def apply_darcy_smoke_test_preset(
    config: DarcyExperimentConfig,
    output_dir: str | Path | None = None,
) -> DarcyExperimentConfig:
    """Fast-running 2-D Darcy config for CI/local verification."""
    config = config.model_copy(deep=True)

    config.data.train_samples = 32
    config.data.validation_samples = 8
    config.data.test_samples = 8
    config.data.grf_resolution = 29
    config.data.train_resolution = 15
    config.data.evaluation_resolution = 22

    config.model.width = 16
    config.model.modes_x = 6
    config.model.modes_y = 6
    config.model.layers = 3
    config.model.padding = 3

    config.optimization.device = "cpu"
    config.optimization.batch_size = 8
    config.optimization.epochs = 4
    config.optimization.scheduler_step = 2
    config.optimization.log_every = 2

    config.artifacts.save_artifacts = False
    config.artifacts.output_dir = (
        Path(output_dir) if output_dir is not None else Path("artifacts/darcy_smoke")
    )
    return config


def build_darcy_smoke_test_config(output_dir: str | Path | None = None) -> DarcyExperimentConfig:
    """Create a fresh 2-D Darcy smoke-test configuration."""
    return apply_darcy_smoke_test_preset(DarcyExperimentConfig(), output_dir=output_dir)


def apply_darcy_tutorial_preset(
    config: DarcyExperimentConfig,
    output_dir: str | Path | None = None,
) -> DarcyExperimentConfig:
    """Notebook-friendly 2-D Darcy preset with moderate training cost."""
    config = config.model_copy(deep=True)

    config.data.train_samples = 200
    config.data.validation_samples = 50
    config.data.test_samples = 50
    config.data.grf_resolution = 61
    config.data.train_resolution = 29
    config.data.evaluation_resolution = 43

    config.model.width = 32
    config.model.modes_x = 12
    config.model.modes_y = 12
    config.model.layers = 4
    config.model.padding = 9

    config.optimization.device = "cpu"
    config.optimization.batch_size = 20
    config.optimization.epochs = 100
    config.optimization.learning_rate = 1e-3
    config.optimization.scheduler_step = 30
    config.optimization.scheduler_gamma = 0.5
    config.optimization.log_every = 10

    config.artifacts.save_artifacts = False
    config.artifacts.output_dir = (
        Path(output_dir) if output_dir is not None else Path("artifacts/darcy_tutorial")
    )
    return config


def build_darcy_tutorial_config(output_dir: str | Path | None = None) -> DarcyExperimentConfig:
    """Create a fresh 2-D Darcy tutorial configuration."""
    return apply_darcy_tutorial_preset(DarcyExperimentConfig(), output_dir=output_dir)


