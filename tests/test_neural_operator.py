from __future__ import annotations

import numpy as np
import torch

from physics_informed_neural_network.neural_operator.data import (
    build_dataset_splits,
    compute_discrete_diffusion_residual,
)
from physics_informed_neural_network.neural_operator.model import FourierNeuralOperator1d
from physics_informed_neural_network.neural_operator.pipeline import run_neural_operator_experiment
from physics_informed_neural_network.neural_operator.presets import build_smoke_test_config


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
