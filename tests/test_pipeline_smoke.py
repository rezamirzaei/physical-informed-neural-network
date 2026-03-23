from __future__ import annotations

from physics_informed_neural_network import run_experiment
from physics_informed_neural_network.presets import build_smoke_test_config


def test_smoke_experiment_runs_end_to_end(tmp_path) -> None:
    config = build_smoke_test_config(output_dir=tmp_path / "artifacts")
    config.artifacts.save_artifacts = False

    experiment = run_experiment(config)

    assert experiment.u_pred.shape == (config.artifacts.prediction_nt, config.artifacts.prediction_nx)
    assert len(experiment.history.entries) >= 2
    assert experiment.summary.trainable_parameters > 0
    assert experiment.summary.device == "cpu"
    assert experiment.summary.metrics.l2_relative_error >= 0.0
