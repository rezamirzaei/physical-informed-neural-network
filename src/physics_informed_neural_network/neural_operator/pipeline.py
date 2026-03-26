"""End-to-end pipeline for the neural-operator tutorial experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import NeuralOperatorExperimentConfig
from .data import OperatorDatasetSplits, build_dataset_splits
from .model import FourierNeuralOperator1d
from .schemas import NeuralOperatorExperimentSummary, OperatorTrainingHistory, ResolutionEvaluation
from .training import NeuralOperatorTrainer
from ..utils import ensure_directory, select_device, set_global_seed


@dataclass(slots=True)
class NeuralOperatorExperiment:
    """Container for datasets, predictions, and summary statistics."""

    config: NeuralOperatorExperimentConfig
    datasets: OperatorDatasetSplits
    trainer: NeuralOperatorTrainer
    model: FourierNeuralOperator1d
    history: OperatorTrainingHistory
    native_prediction: np.ndarray
    refined_prediction: np.ndarray
    summary: NeuralOperatorExperimentSummary
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def _save_artifacts(
    config: NeuralOperatorExperimentConfig,
    history: OperatorTrainingHistory,
    summary: NeuralOperatorExperimentSummary,
) -> dict[str, Path]:
    output_dir = ensure_directory(config.artifacts.output_dir)
    history_path = output_dir / "training_history.csv"
    summary_path = output_dir / "summary.json"

    history.to_frame().to_csv(history_path, index=False)
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    return {
        "training_history": history_path,
        "summary": summary_path,
    }


def run_neural_operator_experiment(config: NeuralOperatorExperimentConfig) -> NeuralOperatorExperiment:
    """Run data generation, training, native evaluation, and resolution-transfer evaluation."""
    set_global_seed(config.optimization.seed)
    device = select_device(config.optimization.device)
    print(f"Device: {device}")

    datasets = build_dataset_splits(config)
    print(
        "Datasets:",
        f"train={datasets.train.n_samples}×{datasets.train.resolution}",
        f"validation={datasets.validation.n_samples}×{datasets.validation.resolution}",
        f"test={datasets.test.n_samples}×{datasets.test.resolution}",
        f"refined_test={datasets.refined_test.n_samples}×{datasets.refined_test.resolution}",
    )

    model = FourierNeuralOperator1d(config.model).to(device)
    trainer = NeuralOperatorTrainer(model=model, config=config.optimization, device=device)
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
            "test": ResolutionEvaluation(
                split="test",
                resolution=datasets.test.resolution,
                metrics=native_metrics,
            ),
            "refined_test": ResolutionEvaluation(
                split="refined_test",
                resolution=datasets.refined_test.resolution,
                metrics=refined_metrics,
            ),
        },
    )

    artifact_paths: dict[str, Path] = {}
    if config.artifacts.save_artifacts:
        artifact_paths = _save_artifacts(config=config, history=history, summary=summary)

    return NeuralOperatorExperiment(
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
