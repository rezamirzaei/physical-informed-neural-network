"""Publication-quality visualizations for the 1-D neural-operator tutorial notebook.

Uses a unified professional style with LaTeX-like math, consistent palette,
and carefully designed layouts.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .data import OperatorDataset, OperatorSample
from .schemas import NeuralOperatorExperimentSummary

# Consistent colour palette
_C = {
    "navy": "#12355b",
    "forest": "#2d6a4f",
    "amber": "#bc6c25",
    "crimson": "#9b2226",
    "dark": "#111111",
    "rust": "#6a040f",
    "teal": "#0077b6",
}


def apply_plot_style() -> None:
    """Apply a publication-quality matplotlib style."""
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "figure.facecolor": "#f8f6f3",
            "figure.dpi": 120,
            "axes.facecolor": "#fffdf9",
            "axes.edgecolor": "#23303b",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.color": "#7d8b99",
            "grid.linewidth": 0.5,
            "axes.titlesize": 14,
            "axes.titleweight": "medium",
            "axes.labelsize": 12,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.size": 10,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )


def plot_dataset_examples(dataset: OperatorDataset, sample_indices: tuple[int, ...] = (0, 1, 2)) -> Figure:
    """Plot diffusivity, forcing, and solution examples from a dataset."""
    n_rows = len(sample_indices)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.6 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])

    for row, sample_index in enumerate(sample_indices):
        sample = dataset.sample(sample_index)

        axes[row, 0].plot(sample.grid, sample.diffusion, color=_C["navy"], linewidth=2.2)
        axes[row, 0].fill_between(sample.grid, sample.diffusion, alpha=0.08, color=_C["navy"])
        axes[row, 0].set_title(f"Sample {sample_index} — diffusivity $a(x)$")
        axes[row, 0].set_ylabel("$a(x)$")

        axes[row, 1].plot(sample.grid, sample.forcing, color=_C["amber"], linewidth=2.2)
        axes[row, 1].fill_between(sample.grid, sample.forcing, alpha=0.08, color=_C["amber"])
        axes[row, 1].set_title("Forcing $f(x)$")

        axes[row, 2].plot(sample.grid, sample.solution, color=_C["forest"], linewidth=2.2)
        axes[row, 2].fill_between(sample.grid, sample.solution, alpha=0.08, color=_C["forest"])
        axes[row, 2].set_title("Solution $u(x)$")

    for axis in axes[-1]:
        axis.set_xlabel("$x$")

    fig.suptitle(
        f"{dataset.name.replace('_', ' ').title()} — dataset examples",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_training_history(history) -> Figure:
    """Plot optimization loss and validation relative error with annotations."""
    frame = history.to_frame()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(frame["epoch"], frame["train_loss"], color=_C["navy"], linewidth=2.4, label="Train loss")
    axes[0].plot(frame["epoch"], frame["validation_loss"], color=_C["amber"], linewidth=2.0, label="Validation MSE")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Optimization history")
    axes[0].legend()
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_rel = axes[1]
    ax_rel.plot(frame["epoch"], frame["validation_relative_l2"], color=_C["forest"], linewidth=2.4)
    ax_rel.set_xlabel("Epoch")
    ax_rel.set_ylabel("Relative $L^2$ error")
    ax_rel.set_title("Validation relative error")
    ax_rel.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Annotate final value
    final_val = frame["validation_relative_l2"].iloc[-1]
    ax_rel.annotate(
        f"{final_val:.4f}",
        xy=(frame["epoch"].iloc[-1], final_val),
        xytext=(-50, 15), textcoords="offset points",
        fontsize=11, fontweight="bold", color=_C["crimson"],
        arrowprops=dict(arrowstyle="->", color=_C["crimson"], lw=1.5),
    )

    # Learning rate on twin axis
    ax_lr = ax_rel.twinx()
    ax_lr.plot(frame["epoch"], frame["learning_rate"], color=_C["teal"], linewidth=1.2, alpha=0.5, linestyle=":")
    ax_lr.set_ylabel("Learning rate", color=_C["teal"], alpha=0.6)
    ax_lr.tick_params(axis="y", colors=_C["teal"])
    ax_lr.spines["right"].set_visible(True)
    ax_lr.spines["right"].set_color(_C["teal"])
    ax_lr.spines["right"].set_alpha(0.4)

    fig.tight_layout()
    return fig


def plot_prediction_comparison(sample: OperatorSample, prediction: np.ndarray, title: str) -> Figure:
    """Compare one predicted solution against the exact target with error band."""
    error = np.abs(prediction - sample.solution)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Inputs
    axes[0].plot(sample.grid, sample.diffusion, color=_C["navy"], linewidth=2.2, label="$a(x)$")
    axes[0].plot(sample.grid, sample.forcing, color=_C["amber"], linewidth=2.0, label="$f(x)$")
    axes[0].fill_between(sample.grid, sample.diffusion, alpha=0.06, color=_C["navy"])
    axes[0].set_title("Inputs")
    axes[0].set_xlabel("$x$")
    axes[0].legend()

    # Operator output
    axes[1].plot(sample.grid, sample.solution, color=_C["dark"], linewidth=2.4, label="Exact $u$")
    axes[1].plot(sample.grid, prediction, color=_C["crimson"], linewidth=2.0, linestyle="--", label="FNO $\\hat{u}$")
    axes[1].fill_between(sample.grid, sample.solution, prediction, alpha=0.12, color=_C["crimson"])
    axes[1].set_title("Operator output")
    axes[1].set_xlabel("$x$")
    axes[1].legend()

    # Error
    axes[2].fill_between(sample.grid, error, alpha=0.25, color=_C["rust"])
    axes[2].plot(sample.grid, error, color=_C["rust"], linewidth=2.0)
    axes[2].set_title(f"Absolute error  (max = {error.max():.2e})")
    axes[2].set_xlabel("$x$")
    axes[2].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


def plot_resolution_metrics(summary: NeuralOperatorExperimentSummary) -> Figure:
    """Visualize native-vs-refined evaluation metrics as a grouped bar chart."""
    labels, relative_l2, mae = [], [], []
    for key in ("test", "refined_test"):
        ev = summary.evaluations[key]
        labels.append(f"{ev.split.replace('_', ' ').title()}\n($n = {ev.resolution}$)")
        relative_l2.append(ev.metrics.relative_l2)
        mae.append(ev.metrics.mae)

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, relative_l2, width, color=_C["navy"], label="Relative $L^2$", edgecolor="white")
    bars2 = ax.bar(x + width / 2, mae, width, color=_C["amber"], label="MAE", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Metric value")
    ax.set_title("1-D Resolution-transfer evaluation", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_frequency_spectrum(grid: np.ndarray, truth: np.ndarray, prediction: np.ndarray, title: str) -> Figure:
    """Compare exact and predicted spectral energy on one sample."""
    spacing = float(grid[1] - grid[0])
    wavenumbers = np.fft.rfftfreq(len(grid), d=spacing)
    truth_spectrum = np.abs(np.fft.rfft(truth))
    pred_spectrum = np.abs(np.fft.rfft(prediction))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

    ax1.plot(wavenumbers, truth_spectrum, color=_C["dark"], linewidth=2.2, label="Exact")
    ax1.plot(wavenumbers, pred_spectrum, color=_C["crimson"], linewidth=2.0, linestyle="--", label="FNO")
    ax1.set_yscale("log")
    ax1.set_xlabel("Wavenumber $k$")
    ax1.set_ylabel("$|\\hat{u}(k)|$")
    ax1.set_title("Spectral magnitude")
    ax1.legend()

    # Spectral error ratio
    ratio = np.abs(pred_spectrum - truth_spectrum) / (truth_spectrum + 1e-12)
    ax2.semilogy(wavenumbers, ratio, color=_C["rust"], linewidth=2.0)
    ax2.set_xlabel("Wavenumber $k$")
    ax2.set_ylabel("Relative spectral error")
    ax2.set_title("Per-mode relative error")
    ax2.axhline(0.1, color=_C["teal"], linewidth=1, linestyle=":", alpha=0.6, label="10% reference")
    ax2.legend()

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig
