"""Tests for PINN plotting functions and presets."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from physics_informed_neural_network import ProjectConfig, run_experiment
from physics_informed_neural_network.data import evaluate_reference_solution
from physics_informed_neural_network.plotting import (
    apply_plot_style,
    plot_comparison,
    plot_loss_history,
    plot_pointwise_error,
    plot_reference_solution,
    plot_residual_distribution,
    plot_time_slices,
)
from physics_informed_neural_network.presets import apply_smoke_test_preset, build_smoke_test_config


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    """Verify PINN configuration presets."""

    def test_smoke_test_config_is_valid(self) -> None:
        config = build_smoke_test_config()
        assert config.training.adam_epochs > 0
        assert config.training.device == "cpu"
        assert config.training.lbfgs_iterations == 0

    def test_smoke_test_with_custom_output_dir(self, tmp_path) -> None:
        config = build_smoke_test_config(output_dir=tmp_path / "custom")
        assert config.artifacts.output_dir == tmp_path / "custom"

    def test_apply_preset_preserves_pde(self) -> None:
        config = ProjectConfig()
        original_viscosity = config.pde.viscosity
        smoke = apply_smoke_test_preset(config)
        assert smoke.pde.viscosity == original_viscosity

    def test_apply_preset_deep_copies(self) -> None:
        original = ProjectConfig()
        smoke = apply_smoke_test_preset(original)
        smoke.training.adam_epochs = 999
        assert original.training.adam_epochs != 999


# ---------------------------------------------------------------------------
# Plotting (smoke tests)
# ---------------------------------------------------------------------------


class TestPlotting:
    """Verify all PINN plotting functions produce figures."""

    @pytest.fixture()
    def experiment(self, tmp_path):
        config = build_smoke_test_config(output_dir=tmp_path / "artifacts")
        config.artifacts.save_artifacts = False
        return run_experiment(config)

    def test_plot_reference_solution(self, experiment) -> None:
        apply_plot_style()
        fig = plot_reference_solution(experiment.reference)
        assert fig is not None

    def test_plot_comparison(self, experiment) -> None:
        apply_plot_style()
        u_ref = evaluate_reference_solution(
            experiment.x_pred, experiment.t_pred,
            experiment.summary.viscosity,
        )
        fig = plot_comparison(experiment.x_pred, experiment.t_pred, u_ref, experiment.u_pred)
        assert fig is not None

    def test_plot_time_slices(self, experiment) -> None:
        apply_plot_style()
        u_ref = evaluate_reference_solution(
            experiment.x_pred, experiment.t_pred,
            experiment.summary.viscosity,
        )
        fig = plot_time_slices(experiment.x_pred, experiment.t_pred, u_ref, experiment.u_pred)
        assert fig is not None

    def test_plot_time_slices_single(self, experiment) -> None:
        apply_plot_style()
        u_ref = evaluate_reference_solution(
            experiment.x_pred, experiment.t_pred,
            experiment.summary.viscosity,
        )
        fig = plot_time_slices(experiment.x_pred, experiment.t_pred, u_ref, experiment.u_pred, time_fractions=(0.5,))
        assert fig is not None

    def test_plot_pointwise_error(self, experiment) -> None:
        apply_plot_style()
        u_ref = evaluate_reference_solution(
            experiment.x_pred, experiment.t_pred,
            experiment.summary.viscosity,
        )
        fig = plot_pointwise_error(experiment.x_pred, experiment.t_pred, u_ref, experiment.u_pred)
        assert fig is not None

    def test_plot_loss_history(self, experiment) -> None:
        apply_plot_style()
        fig = plot_loss_history(experiment.history)
        assert fig is not None

    def test_plot_residual_distribution(self) -> None:
        apply_plot_style()
        residuals = np.random.randn(200)
        fig = plot_residual_distribution(residuals)
        assert fig is not None


