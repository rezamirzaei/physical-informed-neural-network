"""Visualization helpers for the neural-operator tutorial notebook."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .data import OperatorDataset, OperatorSample
from .schemas import NeuralOperatorExperimentSummary


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "figure.facecolor": "#f7f5f2",
            "axes.facecolor": "#fffdfa",
            "axes.edgecolor": "#23303b",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.color": "#7d8b99",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def plot_dataset_examples(dataset: OperatorDataset, sample_indices: tuple[int, ...] = (0, 1, 2)) -> Figure:
    """Plot diffusivity, forcing, and solution examples from a dataset."""
    n_rows = len(sample_indices)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])

    for row, sample_index in enumerate(sample_indices):
        sample = dataset.sample(sample_index)
        axes[row, 0].plot(sample.grid, sample.diffusion, color="#12355b", linewidth=2.0)
        axes[row, 0].set_title(f"Sample {sample_index} — diffusion a(x)")
        axes[row, 0].set_ylabel("a(x)")

        axes[row, 1].plot(sample.grid, sample.forcing, color="#bc6c25", linewidth=2.0)
        axes[row, 1].set_title("forcing f(x)")

        axes[row, 2].plot(sample.grid, sample.solution, color="#2d6a4f", linewidth=2.0)
        axes[row, 2].set_title("solution u(x)")

    for axis in axes[-1]:
        axis.set_xlabel("x")

    fig.suptitle(f"{dataset.name.replace('_', ' ').title()} examples", fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


def plot_training_history(history) -> Figure:
    """Plot optimization loss and validation relative error."""
    frame = history.to_frame()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(frame["epoch"], frame["train_loss"], color="#12355b", linewidth=2.2, label="Train loss")
    axes[0].plot(frame["epoch"], frame["validation_loss"], color="#bc6c25", linewidth=2.0, label="Validation MSE")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Optimization history")
    axes[0].legend()

    axes[1].plot(
        frame["epoch"],
        frame["validation_relative_l2"],
        color="#2d6a4f",
        linewidth=2.2,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative L2")
    axes[1].set_title("Validation relative error")

    fig.tight_layout()
    return fig


def plot_prediction_comparison(sample: OperatorSample, prediction: np.ndarray, title: str) -> Figure:
    """Compare one predicted solution against the exact target."""
    error = np.abs(prediction - sample.solution)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(sample.grid, sample.diffusion, color="#12355b", linewidth=2.0, label="a(x)")
    axes[0].plot(sample.grid, sample.forcing, color="#bc6c25", linewidth=2.0, label="f(x)")
    axes[0].set_title("Inputs")
    axes[0].set_xlabel("x")
    axes[0].legend()

    axes[1].plot(sample.grid, sample.solution, color="#111111", linewidth=2.2, label="Exact")
    axes[1].plot(sample.grid, prediction, color="#9b2226", linewidth=2.0, linestyle="--", label="Prediction")
    axes[1].set_title("Operator output")
    axes[1].set_xlabel("x")
    axes[1].legend()

    axes[2].plot(sample.grid, error, color="#6a040f", linewidth=2.0)
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")

    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    return fig


def plot_resolution_metrics(summary: NeuralOperatorExperimentSummary) -> Figure:
    """Visualize native-vs-refined evaluation metrics."""
    labels = []
    relative_l2 = []
    mae = []
    for key in ("test", "refined_test"):
        evaluation = summary.evaluations[key]
        labels.append(f"{evaluation.split}\n(n={evaluation.resolution})")
        relative_l2.append(evaluation.metrics.relative_l2)
        mae.append(evaluation.metrics.mae)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, relative_l2, width=width, color="#12355b", label="Relative L2")
    ax.bar(x + width / 2, mae, width=width, color="#bc6c25", label="MAE")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Metric value")
    ax.set_title("Resolution-transfer evaluation")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_frequency_spectrum(grid: np.ndarray, truth: np.ndarray, prediction: np.ndarray, title: str) -> Figure:
    """Compare exact and predicted spectral energy on one sample."""
    spacing = float(grid[1] - grid[0])
    wavenumbers = np.fft.rfftfreq(len(grid), d=spacing)
    truth_spectrum = np.abs(np.fft.rfft(truth))
    prediction_spectrum = np.abs(np.fft.rfft(prediction))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(wavenumbers, truth_spectrum, color="#111111", linewidth=2.2, label="Exact")
    ax.plot(wavenumbers, prediction_spectrum, color="#9b2226", linewidth=2.0, linestyle="--", label="Prediction")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
