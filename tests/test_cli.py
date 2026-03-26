from __future__ import annotations

import pytest

from physics_informed_neural_network.cli import build_config, build_parser


def test_build_config_applies_cli_overrides_to_smoke_preset(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--smoke-test",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--adam-epochs",
            "7",
            "--lbfgs-iterations",
            "0",
            "--n-collocation",
            "111",
            "--n-boundary",
            "12",
            "--n-initial",
            "13",
            "--n-observed",
            "99",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--no-artifacts",
        ]
    )

    config = build_config(args)

    assert config.training.device == "cpu"
    assert config.data.seed == 123
    assert config.training.seed == 123
    assert config.training.adam_epochs == 7
    assert config.training.lbfgs_iterations == 0
    assert config.training.n_collocation == 111
    assert config.training.n_boundary == 12
    assert config.training.n_initial == 13
    assert config.data.n_observed == 99
    assert config.artifacts.output_dir == tmp_path / "artifacts"
    assert config.artifacts.save_artifacts is False


def test_parser_rejects_non_positive_epoch_override() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--adam-epochs", "0"])
