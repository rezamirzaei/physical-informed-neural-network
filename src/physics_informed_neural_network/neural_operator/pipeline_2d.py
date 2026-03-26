"""End-to-end pipeline for the 2-D Darcy-flow FNO experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DarcyExperimentConfig
from .data_2d import DarcyDatasetSplits, build_darcy_splits
from .model_2d import FourierNeuralOperator2d
from .schemas import NeuralOperatorExperimentSummary, OperatorTrainingHistory, ResolutionEvaluation
from .training_2d import DarcyTrainer
from ..utils import ensure_directory, select_device, set_global_seed


@dataclass(slots=True)
class DarcyExperiment:
    """Container for 2-D Darcy experiment results."""

    config: DarcyExperimentConfig
    datasets: DarcyDatasetSplits
    trainer: DarcyTrainer
    model: FourierNeuralOperator2d
    history: OperatorTrainingHistory
    native_prediction: np.ndarray
    refined_prediction: np.ndarray
    summary: NeuralOperatorExperimentSummary
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def _save_darcy_artifacts(
    config: DarcyExperimentConfig,
    history: OperatorTrainingHistory,
    summary: NeuralOperatorExperimentSummary,
) -> dict[str, Path]:
    output_dir = ensure_directory(config.artifacts.output_dir)
    history_path = output_dir / "training_history_2d.csv"
    summary_path = output_dir / "summary_2d.json"

    history.to_frame().to_csv(history_path, index=False)
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    return {"training_history": history_path, "summary": summary_path}


def run_darcy_experiment(config: DarcyExperimentConfig) -> DarcyExperiment:
    """Run the full 2-D Darcy-flow FNO experiment."""
    set_global_seed(config.optimization.seed)
    device = select_device(config.optimization.device)
    print(f"Device: {device}")

    print("Building 2-D Darcy datasets …")
    datasets = build_darcy_splits(config)
    print(
        "Datasets:",
        f"train={datasets.train.n_samples}×{datasets.train.resolution}²",
        f"val={datasets.validation.n_samples}×{datasets.validation.resolution}²",
        f"test={datasets.test.n_samples}×{datasets.test.resolution}²",
        f"refined={datasets.refined_test.n_samples}×{datasets.refined_test.resolution}²",
    )

    model = FourierNeuralOperator2d(config.model).to(device)
    trainer = DarcyTrainer(model=model, config=config.optimization, device=device)
    print(f"Model: {model.architecture_string()} ({model.count_parameters():,} parameters)")

    history = trainer.fit(datasets.train, datasets.validation)

    native_prediction = trainer.predict_dataset(datasets.test)
    refined_prediction = trainer.predict_dataset(datasets.refined_test)

    native_metrics = trainer.evaluate_dataset(datasets.test)
    refined_metrics = trainer.evaluate_dataset(datasets.refined_test)

    summary = NeuralOperatorExperimentSummary(
        architecture=model.architecture_string(),
        trainable_parameters=model.count_parameters(),
        device=str(device),
        train_resolution=datasets.train.resolution,
        evaluation_resolution=datasets.refined_test.resolution,
        train_samples=datasets.train.n_samples,
        validation_samples=datasets.validation.n_samples,
        test_samples=datasets.test.n_samples,
        evaluations={
            "test": ResolutionEvaluation(split="test", resolution=datasets.test.resolution, metrics=native_metrics),
            "refined_test": ResolutionEvaluation(
                split="refined_test", resolution=datasets.refined_test.resolution, metrics=refined_metrics,
            ),
        },
    )

    artifact_paths: dict[str, Path] = {}
    if config.artifacts.save_artifacts:
        artifact_paths = _save_darcy_artifacts(config=config, history=history, summary=summary)

    return DarcyExperiment(
        config=config,
        datasets=datasets,
        trainer=trainer,
        model=model,
        history=history,
        native_prediction=native_prediction,
        refined_prediction=refined_prediction,
        summary=summary,
        artifact_paths=artifact_paths,
    )

