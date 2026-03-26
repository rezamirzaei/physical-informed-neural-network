"""Reusable neural-operator components for operator-learning tutorials."""

from .config import (
    DarcyDatasetConfig,
    DarcyExperimentConfig,
    DarcyProblemConfig,
    FourierNeuralOperator2dConfig,
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
from .data_2d import (
    DarcyDataset,
    DarcyDatasetSplits,
    DarcySample,
    build_darcy_splits,
    solve_darcy_2d,
)
from .model import FourierNeuralOperator1d
from .model_2d import FourierNeuralOperator2d
from .pipeline import NeuralOperatorExperiment, run_neural_operator_experiment
from .pipeline_2d import DarcyExperiment, run_darcy_experiment
from .plotting import (
    apply_plot_style,
    plot_dataset_examples,
    plot_frequency_spectrum,
    plot_prediction_comparison,
    plot_resolution_metrics,
    plot_training_history,
)
from .plotting_2d import (
    plot_darcy_3d_surface,
    plot_darcy_cross_sections,
    plot_darcy_dataset_examples,
    plot_darcy_error_distribution,
    plot_darcy_prediction_comparison,
    plot_darcy_resolution_metrics,
)
from .presets import (
    apply_darcy_smoke_test_preset,
    apply_darcy_tutorial_preset,
    apply_smoke_test_preset,
    apply_tutorial_preset,
    build_darcy_smoke_test_config,
    build_darcy_tutorial_config,
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
from .training_2d import DarcyTrainer, compute_error_metrics_2d

__all__ = [
    # 1D configs
    "FourierNeuralOperatorConfig",
    "NeuralOperatorExperimentConfig",
    "OperatorArtifactConfig",
    "OperatorDatasetConfig",
    "OperatorOptimizationConfig",
    "OperatorProblemConfig",
    # 2D configs
    "DarcyDatasetConfig",
    "DarcyExperimentConfig",
    "DarcyProblemConfig",
    "FourierNeuralOperator2dConfig",
    # 1D data
    "OperatorDataset",
    "OperatorDatasetSplits",
    "OperatorSample",
    "build_dataset_splits",
    "build_operator_dataset",
    "compute_discrete_diffusion_residual",
    "create_uniform_grid",
    "evaluate_field_draw",
    "sample_field_draws",
    "solve_dirichlet_diffusion_1d",
    # 2D data
    "DarcyDataset",
    "DarcyDatasetSplits",
    "DarcySample",
    "build_darcy_splits",
    "solve_darcy_2d",
    # Models
    "FourierNeuralOperator1d",
    "FourierNeuralOperator2d",
    # Pipelines
    "NeuralOperatorExperiment",
    "DarcyExperiment",
    "run_neural_operator_experiment",
    "run_darcy_experiment",
    # 1D plotting
    "apply_plot_style",
    "plot_dataset_examples",
    "plot_frequency_spectrum",
    "plot_prediction_comparison",
    "plot_resolution_metrics",
    "plot_training_history",
    # 2D plotting
    "plot_darcy_3d_surface",
    "plot_darcy_cross_sections",
    "plot_darcy_dataset_examples",
    "plot_darcy_error_distribution",
    "plot_darcy_prediction_comparison",
    "plot_darcy_resolution_metrics",
    # Presets
    "apply_darcy_smoke_test_preset",
    "apply_darcy_tutorial_preset",
    "apply_smoke_test_preset",
    "apply_tutorial_preset",
    "build_darcy_smoke_test_config",
    "build_darcy_tutorial_config",
    "build_smoke_test_config",
    "build_tutorial_config",
    # Schemas
    "NeuralOperatorExperimentSummary",
    "OperatorErrorMetrics",
    "OperatorTrainingHistory",
    "ResolutionEvaluation",
    # Training
    "NeuralOperatorTrainer",
    "DarcyTrainer",
    "compute_error_metrics",
    "compute_error_metrics_2d",
]
