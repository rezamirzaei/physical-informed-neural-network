"""End-to-end KAN experiment pipeline for exact Burgers regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import ensure_directory, select_device, set_global_seed
from .config import KANExperimentConfig
from .data import KANDatasetSplits, build_dataset_splits
from .model import KolmogorovArnoldNetwork
from .schemas import KANExperimentSummary, KANTrainingHistory
from .training import KANTrainer, compute_residual_metrics


@dataclass(slots=True)
class KANExperiment:
    """Container for datasets, model, predictions, and experiment summary."""

    config: KANExperimentConfig
    datasets: KANDatasetSplits
    trainer: KANTrainer
    model: KolmogorovArnoldNetwork
    history: KANTrainingHistory
    test_prediction: np.ndarray
    evaluation_prediction: np.ndarray
    residuals: np.ndarray
    summary: KANExperimentSummary
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def _save_artifacts(
    config: KANExperimentConfig,
    history: KANTrainingHistory,
    summary: KANExperimentSummary,
    evaluation_dataset,
    evaluation_prediction: np.ndarray,
) -> dict[str, Path]:
    output_dir = ensure_directory(config.artifacts.output_dir)
    history_path = output_dir / "training_history.csv"
    summary_path = output_dir / "summary.json"
    prediction_path = output_dir / "evaluation_prediction.csv"

    history.to_frame().to_csv(history_path, index=False)
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    grid_x, grid_t = evaluation_dataset.mesh()
    pd.DataFrame(
        {
            "x": grid_x.ravel(),
            "t": grid_t.ravel(),
            "u_exact": evaluation_dataset.solution.ravel(),
            "u_pred": evaluation_prediction.ravel(),
        }
    ).to_csv(prediction_path, index=False)

    return {
        "training_history": history_path,
        "summary": summary_path,
        "evaluation_prediction": prediction_path,
    }


def run_kan_experiment(config: KANExperimentConfig) -> KANExperiment:
    """Run data generation, supervised KAN training, and residual-based verification."""
    set_global_seed(config.optimization.seed)
    device = select_device(config.optimization.device)
    print(f"Device: {device}")

    datasets = build_dataset_splits(config)
    print(
        "Datasets:",
        f"train={datasets.train.nt}x{datasets.train.nx} ({datasets.train.n_points} points)",
        f"validation={datasets.validation.nt}x{datasets.validation.nx} ({datasets.validation.n_points} points)",
        f"test={datasets.test.nt}x{datasets.test.nx} ({datasets.test.n_points} points)",
        f"evaluation={datasets.evaluation.nt}x{datasets.evaluation.nx} ({datasets.evaluation.n_points} points)",
    )

    model = KolmogorovArnoldNetwork(config.model).to(device)
    trainer = KANTrainer(
        model=model,
        config=config.optimization,
        device=device,
        coordinate_normalizer=datasets.train.normalizer,
    )
    print(f"Model: {model.architecture_string()} ({model.count_parameters():,} parameters)")

    history = trainer.fit(datasets.train, datasets.validation)
    test_prediction = trainer.predict_dataset(datasets.test)
    evaluation_prediction = trainer.predict_dataset(datasets.evaluation)
    evaluation_grid_prediction = datasets.evaluation.reshape_prediction(evaluation_prediction)

    test_metrics = trainer.evaluate_dataset(datasets.test)
    evaluation_metrics = trainer.evaluate_dataset(datasets.evaluation)
    residuals = trainer.compute_pde_residuals(config.pde, datasets.evaluation.coordinates())
    residual_metrics = compute_residual_metrics(residuals)

    summary = KANExperimentSummary(
        architecture=model.architecture_string(),
        trainable_parameters=model.count_parameters(),
        device=str(device),
        train_points=datasets.train.n_points,
        validation_points=datasets.validation.n_points,
        test_points=datasets.test.n_points,
        evaluation_points=datasets.evaluation.n_points,
        train_grid=(datasets.train.nx, datasets.train.nt),
        evaluation_grid=(datasets.evaluation.nx, datasets.evaluation.nt),
        test_metrics=test_metrics,
        evaluation_metrics=evaluation_metrics,
        residual_metrics=residual_metrics,
    )

    artifact_paths: dict[str, Path] = {}
    if config.artifacts.save_artifacts:
        artifact_paths = _save_artifacts(
            config=config,
            history=history,
            summary=summary,
            evaluation_dataset=datasets.evaluation,
            evaluation_prediction=evaluation_grid_prediction,
        )

    return KANExperiment(
        config=config,
        datasets=datasets,
        trainer=trainer,
        model=model,
        history=history,
        test_prediction=test_prediction,
        evaluation_prediction=evaluation_grid_prediction,
        residuals=residuals,
        summary=summary,
        artifact_paths=artifact_paths,
    )
