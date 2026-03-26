"""Reusable neural-operator components for operator-learning tutorials."""

from .config import (
    FourierNeuralOperatorConfig,
    NeuralOperatorExperimentConfig,
    OperatorArtifactConfig,
    OperatorDatasetConfig,
    OperatorOptimizationConfig,
    OperatorProblemConfig,
)
from .data import (
    OperatorDataset,
    OperatorDatasetSplits,
    OperatorSample,
    build_dataset_splits,
    build_operator_dataset,
    compute_discrete_diffusion_residual,
    create_uniform_grid,
    evaluate_field_draw,
    sample_field_draws,
    solve_dirichlet_diffusion_1d,
)
from .model import FourierNeuralOperator1d
from .pipeline import NeuralOperatorExperiment, run_neural_operator_experiment
from .plotting import (
    apply_plot_style,
    plot_dataset_examples,
    plot_frequency_spectrum,
    plot_prediction_comparison,
    plot_resolution_metrics,
    plot_training_history,
)
from .presets import (
    apply_smoke_test_preset,
    apply_tutorial_preset,
    build_smoke_test_config,
    build_tutorial_config,
)
from .schemas import (
    NeuralOperatorExperimentSummary,
    OperatorErrorMetrics,
    OperatorTrainingHistory,
    ResolutionEvaluation,
)
from .training import NeuralOperatorTrainer, compute_error_metrics

__all__ = [
    "FourierNeuralOperatorConfig",
    "NeuralOperatorExperimentConfig",
    "NeuralOperatorExperiment",
    "NeuralOperatorExperimentSummary",
    "OperatorArtifactConfig",
    "OperatorDataset",
    "OperatorDatasetConfig",
    "OperatorDatasetSplits",
    "OperatorErrorMetrics",
    "OperatorOptimizationConfig",
    "OperatorProblemConfig",
    "OperatorSample",
    "OperatorTrainingHistory",
    "ResolutionEvaluation",
    "FourierNeuralOperator1d",
    "NeuralOperatorTrainer",
    "apply_plot_style",
    "apply_smoke_test_preset",
    "apply_tutorial_preset",
    "build_dataset_splits",
    "build_operator_dataset",
    "build_smoke_test_config",
    "build_tutorial_config",
    "compute_discrete_diffusion_residual",
    "compute_error_metrics",
    "create_uniform_grid",
    "evaluate_field_draw",
    "plot_dataset_examples",
    "plot_frequency_spectrum",
    "plot_prediction_comparison",
    "plot_resolution_metrics",
    "plot_training_history",
    "run_neural_operator_experiment",
    "sample_field_draws",
    "solve_dirichlet_diffusion_1d",
]
