"""Reusable presets for KAN smoke tests and notebook demonstrations."""

from __future__ import annotations

from pathlib import Path

from .config import KANExperimentConfig


def apply_smoke_test_preset(
    config: KANExperimentConfig,
    output_dir: str | Path | None = None,
) -> KANExperimentConfig:
    """Return a fast-running KAN configuration for tests and quick verification."""
    config = config.model_copy(deep=True)

    config.data.train_nx = 17
    config.data.train_nt = 17
    config.data.validation_nx = 21
    config.data.validation_nt = 21
    config.data.test_nx = 25
    config.data.test_nt = 25
    config.data.evaluation_nx = 33
    config.data.evaluation_nt = 33

    config.model.hidden_widths = (10, 10)
    config.model.num_knots = 9

    config.optimization.device = "cpu"
    config.optimization.batch_size = 128
    config.optimization.epochs = 8
    config.optimization.scheduler_step = 4
    config.optimization.log_every = 4

    config.artifacts.save_artifacts = False
    config.artifacts.output_dir = Path(output_dir) if output_dir is not None else Path("artifacts/kan_smoke")
    return config


def build_smoke_test_config(output_dir: str | Path | None = None) -> KANExperimentConfig:
    """Create a fresh KAN smoke-test configuration."""
    return apply_smoke_test_preset(KANExperimentConfig(), output_dir=output_dir)


def apply_tutorial_preset(
    config: KANExperimentConfig,
    output_dir: str | Path | None = None,
) -> KANExperimentConfig:
    """Return a notebook-friendly KAN configuration."""
    config = config.model_copy(deep=True)

    config.data.train_nx = 33
    config.data.train_nt = 25
    config.data.validation_nx = 49
    config.data.validation_nt = 33
    config.data.test_nx = 65
    config.data.test_nt = 41
    config.data.evaluation_nx = 129
    config.data.evaluation_nt = 81

    config.model.hidden_widths = (24, 24)
    config.model.num_knots = 17

    config.optimization.device = "cpu"
    config.optimization.batch_size = 256
    config.optimization.epochs = 220
    config.optimization.learning_rate = 2e-3
    config.optimization.scheduler_step = 70
    config.optimization.scheduler_gamma = 0.65
    config.optimization.log_every = 20
    config.optimization.patience = 30
    config.optimization.min_delta = 1e-7

    config.artifacts.save_artifacts = False
    config.artifacts.output_dir = Path(output_dir) if output_dir is not None else Path("artifacts/kan_tutorial")
    return config


def build_tutorial_config(output_dir: str | Path | None = None) -> KANExperimentConfig:
    """Create a fresh KAN tutorial configuration."""
    return apply_tutorial_preset(KANExperimentConfig(), output_dir=output_dir)
