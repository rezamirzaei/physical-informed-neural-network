"""Comprehensive tests for the KAN (Kolmogorov-Arnold Network) stack."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from physics_informed_neural_network.kan.config import (
    KANExperimentConfig,
    PiecewiseLinearKANConfig,
)
from physics_informed_neural_network.kan.data import (
    BurgersGridDataset,
    CoordinateNormalizer,
    KANDatasetSplits,
    build_dataset_splits,
)
from physics_informed_neural_network.kan.model import (
    KANLayer,
    KolmogorovArnoldNetwork,
    MLPBaseline,
    PiecewiseLinearBasis,
)
from physics_informed_neural_network.kan.pipeline import run_kan_experiment
from physics_informed_neural_network.kan.plotting import (
    plot_3d_surface,
    plot_edge_functions,
    plot_kan_vs_mlp_comparison,
    plot_pointwise_error_heatmap,
    plot_residual_distribution,
    plot_solution_comparison,
    plot_time_slices,
    plot_training_history,
)
from physics_informed_neural_network.kan.presets import build_smoke_test_config
from physics_informed_neural_network.kan.schemas import KANErrorMetrics, ResidualMetrics
from physics_informed_neural_network.kan.training import (
    KANTrainer,
    TensorNormalizer,
    compute_error_metrics,
    compute_residual_metrics,
)


# ---------------------------------------------------------------------------
# PiecewiseLinearBasis
# ---------------------------------------------------------------------------


class TestPiecewiseLinearBasis:
    """Verify hat-function basis correctness."""

    def test_partition_of_unity(self) -> None:
        basis = PiecewiseLinearBasis(num_knots=9, domain_min=-1.0, domain_max=1.0)
        inputs = torch.linspace(-1.0, 1.0, steps=17).reshape(-1, 1)
        values = basis(inputs)
        assert values.shape == (17, 1, 9)
        assert torch.allclose(values.sum(dim=-1), torch.ones_like(inputs), atol=1e-6)

    def test_non_negative(self) -> None:
        basis = PiecewiseLinearBasis(num_knots=5, domain_min=0.0, domain_max=1.0)
        inputs = torch.linspace(-0.5, 1.5, steps=50).reshape(-1, 1)
        values = basis(inputs)
        assert (values >= -1e-7).all()

    def test_boundary_clamping(self) -> None:
        basis = PiecewiseLinearBasis(num_knots=5, domain_min=0.0, domain_max=1.0)
        out_of_range = torch.tensor([[-1.0], [2.0]])
        values = basis(out_of_range)
        assert torch.allclose(values.sum(dim=-1), torch.ones(2, 1), atol=1e-6)

    def test_rejects_bad_knots(self) -> None:
        with pytest.raises(ValueError, match="num_knots must be at least 2"):
            PiecewiseLinearBasis(num_knots=1)

    def test_rejects_bad_domain(self) -> None:
        with pytest.raises(ValueError, match="domain_max must exceed"):
            PiecewiseLinearBasis(num_knots=5, domain_min=1.0, domain_max=0.0)


# ---------------------------------------------------------------------------
# CoordinateNormalizer
# ---------------------------------------------------------------------------


class TestCoordinateNormalizer:
    """Verify affine coordinate normalization."""

    def test_round_trip_numpy(self) -> None:
        normalizer = CoordinateNormalizer(
            minimum=np.array([-1.0, 0.0]),
            maximum=np.array([1.0, 1.0]),
        )
        original = np.array([[0.0, 0.5], [-1.0, 0.0], [1.0, 1.0]])
        transformed = normalizer.transform_numpy(original)
        assert abs(transformed[0, 0] - 0.0) < 1e-12
        assert abs(transformed[0, 1] - 0.0) < 1e-12
        assert abs(transformed[1, 0] - (-1.0)) < 1e-12
        assert abs(transformed[2, 0] - 1.0) < 1e-12

    def test_round_trip_tensor(self) -> None:
        normalizer = CoordinateNormalizer(
            minimum=np.array([-1.0, 0.0]),
            maximum=np.array([1.0, 1.0]),
        )
        original = torch.tensor([[0.0, 0.5], [-1.0, 0.0]], dtype=torch.float64)
        transformed = normalizer.transform_tensor(original)
        assert transformed.shape == original.shape
        assert abs(float(transformed[0, 0])) < 1e-12

    def test_from_pde_config(self) -> None:
        from physics_informed_neural_network.config import PDEConfig
        pde = PDEConfig()
        normalizer = CoordinateNormalizer.from_pde(pde)
        assert normalizer.minimum[0] == pde.x_min
        assert normalizer.maximum[1] == pde.t_max


# ---------------------------------------------------------------------------
# TensorNormalizer
# ---------------------------------------------------------------------------


class TestTensorNormalizer:
    """Verify affine target normalization with round-trip."""

    def test_round_trip(self) -> None:
        data = torch.randn(100, 1)
        normalizer = TensorNormalizer.fit(data)
        transformed = normalizer.transform(data)
        recovered = normalizer.inverse(transformed)
        assert torch.allclose(data, recovered, atol=1e-5)

    def test_zero_mean_unit_variance(self) -> None:
        data = torch.randn(1000, 1) * 5.0 + 3.0
        normalizer = TensorNormalizer.fit(data)
        transformed = normalizer.transform(data)
        assert abs(float(transformed.mean())) < 0.1
        assert abs(float(transformed.std()) - 1.0) < 0.1


# ---------------------------------------------------------------------------
# KANLayer
# ---------------------------------------------------------------------------


class TestKANLayer:
    """Verify a single KAN layer."""

    def test_forward_shape(self) -> None:
        config = PiecewiseLinearKANConfig(hidden_widths=(8,), num_knots=5)
        layer = KANLayer(in_features=2, out_features=8, config=config, input_activation="identity")
        output = layer(torch.randn(10, 2))
        assert output.shape == (10, 8)

    def test_edge_contributions_shape(self) -> None:
        config = PiecewiseLinearKANConfig(hidden_widths=(8,), num_knots=5)
        layer = KANLayer(in_features=2, out_features=8, config=config, input_activation="identity")
        contributions = layer.edge_contributions(torch.randn(10, 2))
        assert contributions.shape == (10, 8, 2)

    def test_evaluate_edge_function(self) -> None:
        config = PiecewiseLinearKANConfig(hidden_widths=(8,), num_knots=5)
        layer = KANLayer(in_features=2, out_features=8, config=config, input_activation="identity")
        samples = torch.linspace(-1, 1, 20)
        result = layer.evaluate_edge_function(0, 0, samples)
        assert result.shape == (20,)
        assert torch.isfinite(result).all()

    def test_edge_function_out_of_bounds(self) -> None:
        config = PiecewiseLinearKANConfig(hidden_widths=(4,), num_knots=5)
        layer = KANLayer(in_features=2, out_features=4, config=config, input_activation="identity")
        with pytest.raises(IndexError):
            layer.evaluate_edge_function(5, 0, torch.linspace(-1, 1, 10))
        with pytest.raises(IndexError):
            layer.evaluate_edge_function(0, 10, torch.linspace(-1, 1, 10))


# ---------------------------------------------------------------------------
# KolmogorovArnoldNetwork
# ---------------------------------------------------------------------------


class TestKolmogorovArnoldNetwork:
    """Verify full KAN model."""

    def test_forward_and_edge_visualization_shapes(self) -> None:
        config = build_smoke_test_config()
        model = KolmogorovArnoldNetwork(config.model)
        batch = torch.zeros(5, config.model.input_dim)
        prediction = model(batch)
        sample_axis = torch.linspace(-1.0, 1.0, steps=32)
        edge_responses = model.evaluate_first_layer_edges(sample_axis, output_indices=(0, 1))
        assert prediction.shape == (5, 1)
        assert sorted(edge_responses.keys()) == [0, 1]
        assert edge_responses[0][0].shape == (32,)

    def test_deeper_architecture(self) -> None:
        config = PiecewiseLinearKANConfig(hidden_widths=(8, 8, 8), num_knots=5)
        model = KolmogorovArnoldNetwork(config)
        output = model(torch.randn(4, 2))
        assert output.shape == (4, 1)
        assert model.count_parameters() > 0

    def test_wrong_input_dim(self) -> None:
        config = PiecewiseLinearKANConfig(hidden_widths=(8,), num_knots=5)
        model = KolmogorovArnoldNetwork(config)
        with pytest.raises(ValueError):
            model(torch.randn(4, 3))

    def test_architecture_string(self) -> None:
        config = PiecewiseLinearKANConfig(hidden_widths=(12, 12), num_knots=9)
        model = KolmogorovArnoldNetwork(config)
        s = model.architecture_string()
        assert "12,12" in s
        assert "knots=9" in s


# ---------------------------------------------------------------------------
# MLPBaseline
# ---------------------------------------------------------------------------


class TestMLPBaseline:
    """Verify the MLP baseline model."""

    def test_forward_shape(self) -> None:
        mlp = MLPBaseline(input_dim=2, hidden_dim=16, depth=3, activation="silu")
        output = mlp(torch.randn(10, 2))
        assert output.shape == (10, 1)

    def test_parameter_count(self) -> None:
        mlp = MLPBaseline(input_dim=2, hidden_dim=16, depth=3)
        assert mlp.count_parameters() > 0

    def test_architecture_string(self) -> None:
        mlp = MLPBaseline(input_dim=2, hidden_dim=32, depth=2)
        s = mlp.architecture_string()
        assert "MLPBaseline" in s
        assert "32" in s

    def test_wrong_input(self) -> None:
        mlp = MLPBaseline(input_dim=2, hidden_dim=16, depth=2)
        with pytest.raises(ValueError):
            mlp(torch.randn(4, 5))

    def test_bad_activation(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            MLPBaseline(activation="leaky_relu")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


class TestMetrics:
    """Verify error and residual metrics on known inputs."""

    def test_compute_error_metrics_identical(self) -> None:
        prediction = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        metrics = compute_error_metrics(prediction, target)
        assert metrics.mse == 0.0
        assert metrics.mae == 0.0
        assert metrics.relative_l2 < 1e-10
        assert metrics.max_absolute_error == 0.0

    def test_compute_error_metrics_known_diff(self) -> None:
        prediction = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 4.0])
        metrics = compute_error_metrics(prediction, target)
        assert abs(metrics.mse - 1.0 / 3.0) < 1e-10
        assert abs(metrics.mae - 1.0 / 3.0) < 1e-10
        assert metrics.max_absolute_error == 1.0

    def test_compute_residual_metrics(self) -> None:
        residuals = np.array([0.1, -0.2, 0.3, -0.4])
        metrics = compute_residual_metrics(residuals)
        assert abs(metrics.mean_absolute_residual - 0.25) < 1e-10
        assert metrics.max_absolute_residual == 0.4
        assert metrics.root_mean_square_residual > 0


# ---------------------------------------------------------------------------
# BurgersGridDataset
# ---------------------------------------------------------------------------


class TestBurgersGridDataset:
    """Verify dataset shapes and properties."""

    def test_properties(self) -> None:
        config = build_smoke_test_config()
        splits = build_dataset_splits(config)
        ds = splits.train
        assert ds.nx == config.data.train_nx
        assert ds.nt == config.data.train_nt
        assert ds.n_points == ds.nx * ds.nt
        assert ds.solution.shape == (ds.nt, ds.nx)

    def test_coordinates_shape(self) -> None:
        config = build_smoke_test_config()
        splits = build_dataset_splits(config)
        coords = splits.train.coordinates()
        assert coords.shape == (splits.train.n_points, 2)

    def test_normalized_coordinates_in_range(self) -> None:
        config = build_smoke_test_config()
        splits = build_dataset_splits(config)
        norm_coords = splits.train.normalized_coordinates()
        assert norm_coords.min() >= -1.0 - 1e-6
        assert norm_coords.max() <= 1.0 + 1e-6

    def test_targets_shape(self) -> None:
        config = build_smoke_test_config()
        splits = build_dataset_splits(config)
        targets = splits.train.targets()
        assert targets.shape == (splits.train.n_points, 1)

    def test_reshape_prediction(self) -> None:
        config = build_smoke_test_config()
        splits = build_dataset_splits(config)
        ds = splits.train
        flat = np.zeros(ds.n_points)
        reshaped = ds.reshape_prediction(flat)
        assert reshaped.shape == (ds.nt, ds.nx)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Verify Pydantic config validators."""

    def test_rejects_bad_num_knots(self) -> None:
        with pytest.raises(ValidationError):
            PiecewiseLinearKANConfig(num_knots=1)

    def test_rejects_bad_domain(self) -> None:
        with pytest.raises(ValidationError):
            PiecewiseLinearKANConfig(spline_domain_min=1.0, spline_domain_max=-1.0)

    def test_rejects_empty_hidden(self) -> None:
        with pytest.raises(ValidationError):
            PiecewiseLinearKANConfig(hidden_widths=())

    def test_rejects_wrong_input_dim(self) -> None:
        with pytest.raises(ValidationError):
            PiecewiseLinearKANConfig(input_dim=5)

    def test_rejects_coarse_eval_grid(self) -> None:
        with pytest.raises(ValidationError):
            KANExperimentConfig(
                data={"train_nx": 100, "train_nt": 100, "evaluation_nx": 10, "evaluation_nt": 10}
            )


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Verify early stopping behavior."""

    def test_early_stopping_triggers(self) -> None:
        config = build_smoke_test_config()
        config.optimization.epochs = 200
        config.optimization.patience = 3
        config.optimization.min_delta = 1e-12
        experiment = run_kan_experiment(config)
        assert len(experiment.history.entries) < 200

    def test_no_early_stopping_when_disabled(self) -> None:
        config = build_smoke_test_config()
        config.optimization.epochs = 8
        config.optimization.patience = 0
        experiment = run_kan_experiment(config)
        assert len(experiment.history.entries) == 8


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Plotting (smoke tests — verify they return a Figure without error)
# ---------------------------------------------------------------------------


class TestPlotting:
    """Verify all plotting functions produce Figures."""

    @pytest.fixture()
    def experiment(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        config = build_smoke_test_config(output_dir=tmp_path / "artifacts")
        return run_kan_experiment(config)

    def test_plot_training_history(self, experiment) -> None:
        fig = plot_training_history(experiment.history)
        assert fig is not None

    def test_plot_solution_comparison(self, experiment) -> None:
        fig = plot_solution_comparison(
            experiment.datasets.evaluation, experiment.evaluation_prediction, title="Test"
        )
        assert fig is not None

    def test_plot_time_slices(self, experiment) -> None:
        fig = plot_time_slices(experiment.datasets.evaluation, experiment.evaluation_prediction)
        assert fig is not None

    def test_plot_residual_distribution(self, experiment) -> None:
        fig = plot_residual_distribution(experiment.residuals)
        assert fig is not None

    def test_plot_3d_surface(self, experiment) -> None:
        fig = plot_3d_surface(experiment.datasets.evaluation, experiment.evaluation_prediction)
        assert fig is not None

    def test_plot_pointwise_error_heatmap(self, experiment) -> None:
        fig = plot_pointwise_error_heatmap(
            experiment.datasets.evaluation, experiment.evaluation_prediction
        )
        assert fig is not None

    def test_plot_edge_functions(self, experiment) -> None:
        sample_axis = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
        sample_tensor = torch.tensor(sample_axis)
        edge_responses_torch = experiment.model.evaluate_first_layer_edges(
            sample_tensor, output_indices=(0, 1)
        )
        edge_responses = {
            n: {i: r.detach().cpu().numpy() for i, r in edges.items()}
            for n, edges in edge_responses_torch.items()
        }
        fig = plot_edge_functions(sample_axis, edge_responses)
        assert fig is not None

    def test_plot_kan_vs_mlp_comparison(self, experiment) -> None:
        from physics_informed_neural_network.kan.model import MLPBaseline
        from physics_informed_neural_network.kan.training import KANTrainer

        mlp = MLPBaseline(input_dim=2, hidden_dim=8, depth=2, activation="silu")
        trainer = KANTrainer(
            model=mlp,
            config=experiment.config.optimization,
            device=torch.device("cpu"),
            coordinate_normalizer=experiment.datasets.train.normalizer,
        )
        trainer.fit(experiment.datasets.train, experiment.datasets.validation)
        mlp_pred = trainer.predict_dataset(experiment.datasets.evaluation)
        mlp_grid = experiment.datasets.evaluation.reshape_prediction(mlp_pred)

        fig = plot_kan_vs_mlp_comparison(
            experiment.datasets.evaluation,
            experiment.evaluation_prediction,
            mlp_grid,
        )
        assert fig is not None


