"""Command-line interface for training and evaluating the Burgers PINN."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from . import __version__
from .config import ProjectConfig
from .pipeline import run_experiment
from .presets import build_smoke_test_config


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="burgers-pinn",
        description="Train and evaluate a physics-informed neural network for the 1-D viscous Burgers equation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast configuration intended for local verification and CI.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Override the execution device.",
    )
    parser.add_argument("--seed", type=_positive_int, help="Override the random seed.")
    parser.add_argument("--adam-epochs", type=_positive_int, help="Override the number of Adam epochs.")
    parser.add_argument("--lbfgs-iterations", type=_non_negative_int, help="Override the number of L-BFGS iterations.")
    parser.add_argument("--n-collocation", type=_positive_int, help="Override the number of interior collocation points.")
    parser.add_argument("--n-boundary", type=_positive_int, help="Override the number of boundary points per edge.")
    parser.add_argument("--n-initial", type=_positive_int, help="Override the number of initial-condition points.")
    parser.add_argument("--n-observed", type=_positive_int, help="Override the number of observed data points.")
    parser.add_argument("--output-dir", type=Path, help="Artifact output directory.")
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Disable writing CSV, JSON, and plot artifacts to disk.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only the final experiment summary as JSON.",
    )
    return parser


def _apply_overrides(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    if args.device is not None:
        config.training.device = args.device

    if args.seed is not None:
        config.data.seed = args.seed
        config.training.seed = args.seed

    if args.adam_epochs is not None:
        config.training.adam_epochs = args.adam_epochs

    if args.lbfgs_iterations is not None:
        config.training.lbfgs_iterations = args.lbfgs_iterations

    if args.n_collocation is not None:
        config.training.n_collocation = args.n_collocation

    if args.n_boundary is not None:
        config.training.n_boundary = args.n_boundary

    if args.n_initial is not None:
        config.training.n_initial = args.n_initial

    if args.n_observed is not None:
        config.data.n_observed = args.n_observed

    if args.output_dir is not None:
        config.artifacts.output_dir = args.output_dir

    if args.no_artifacts:
        config.artifacts.save_artifacts = False

    return config


def build_config(args: argparse.Namespace) -> ProjectConfig:
    config = build_smoke_test_config(output_dir=args.output_dir) if args.smoke_test else ProjectConfig()
    return _apply_overrides(config, args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config(args)
    except ValidationError as exc:
        parser.error(str(exc))
    experiment = run_experiment(config)
    summary_json = experiment.summary.model_dump_json(indent=2)

    if args.json:
        print(summary_json)
    else:
        print("\nExperiment summary")
        print(summary_json)

    return 0
