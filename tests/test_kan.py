from __future__ import annotations

import numpy as np
import torch

from physics_informed_neural_network.kan.model import KolmogorovArnoldNetwork, PiecewiseLinearBasis
from physics_informed_neural_network.kan.pipeline import run_kan_experiment
from physics_informed_neural_network.kan.presets import build_smoke_test_config


def test_piecewise_linear_basis_forms_partition_of_unity() -> None:
    basis = PiecewiseLinearBasis(num_knots=9, domain_min=-1.0, domain_max=1.0)
    inputs = torch.linspace(-1.0, 1.0, steps=17).reshape(-1, 1)
    values = basis(inputs)

    assert values.shape == (17, 1, 9)
    assert torch.allclose(values.sum(dim=-1), torch.ones_like(inputs), atol=1e-6)


def test_kan_forward_and_edge_visualization_shapes() -> None:
    config = build_smoke_test_config()
    model = KolmogorovArnoldNetwork(config.model)

    batch = torch.zeros(5, config.model.input_dim)
    prediction = model(batch)
    sample_axis = torch.linspace(-1.0, 1.0, steps=32)
    edge_responses = model.evaluate_first_layer_edges(sample_axis, output_indices=(0, 1))

    assert prediction.shape == (5, 1)
    assert sorted(edge_responses.keys()) == [0, 1]
    assert edge_responses[0][0].shape == (32,)
    assert edge_responses[0][1].shape == (32,)


def test_kan_smoke_experiment_runs_end_to_end(tmp_path) -> None:
    config = build_smoke_test_config(output_dir=tmp_path / "artifacts")
    experiment = run_kan_experiment(config)

    assert len(experiment.history.entries) == config.optimization.epochs
    assert experiment.test_prediction.shape == (experiment.datasets.test.n_points,)
    assert experiment.evaluation_prediction.shape == (
        experiment.datasets.evaluation.nt,
        experiment.datasets.evaluation.nx,
    )
    assert experiment.summary.evaluation_metrics.relative_l2 >= 0.0
    assert experiment.summary.residual_metrics.mean_absolute_residual >= 0.0
    assert np.isfinite(experiment.residuals).all()
