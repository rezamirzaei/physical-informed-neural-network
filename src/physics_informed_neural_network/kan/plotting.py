"""Plotting helpers for the KAN tutorial notebook."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .data import BurgersGridDataset

_C = {
    "ink": "#16213e",
    "ocean": "#0f4c75",
    "teal": "#1b9aaa",
    "sand": "#f4d35e",
    "brick": "#b23a48",
    "forest": "#3a5a40",
    "paper": "#fbf7ef",
}


def apply_plot_style() -> None:
    """Apply a coherent notebook style."""
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "figure.facecolor": _C["paper"],
            "figure.dpi": 120,
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#2f3e46",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.color": "#6c757d",
            "grid.linewidth": 0.45,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "font.size": 10,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def plot_training_history(history) -> Figure:
    """Plot training loss and validation relative error."""
    frame = history.to_frame()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    axes[0].plot(frame["epoch"], frame["train_loss"], color=_C["ocean"], linewidth=2.4)
    axes[0].plot(frame["epoch"], frame["validation_loss"], color=_C["brick"], linewidth=2.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Optimization history")
    axes[0].legend(["Train", "Validation MSE"])
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1].plot(frame["epoch"], frame["validation_relative_l2"], color=_C["forest"], linewidth=2.4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative $L^2$ error")
    axes[1].set_title("Validation relative error")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    return fig


def plot_solution_comparison(dataset: BurgersGridDataset, prediction: np.ndarray, title: str) -> Figure:
    """Compare exact and predicted solution surfaces on one grid."""
    truth = dataset.solution
    error = np.abs(prediction - truth)
    extent = [dataset.x[0], dataset.x[-1], dataset.t[0], dataset.t[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    panels = [
        (truth, "Exact Burgers solution", "RdBu_r"),
        (prediction, "KAN prediction", "RdBu_r"),
        (error, "Absolute error", "magma"),
    ]
    for axis, (field, label, cmap) in zip(axes, panels):
        image = axis.imshow(field, extent=extent, origin="lower", aspect="auto", cmap=cmap)
        axis.set_title(label)
        axis.set_xlabel("$x$")
        fig.colorbar(image, ax=axis, shrink=0.82)
    axes[0].set_ylabel("$t$")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_time_slices(
    dataset: BurgersGridDataset,
    prediction: np.ndarray,
    target_times: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Figure:
    """Plot several temporal slices through the exact and predicted fields."""
    fig, axes = plt.subplots(1, len(target_times), figsize=(4.1 * len(target_times), 4.2), sharey=True)
    if len(target_times) == 1:
        axes = np.array([axes])

    for axis, target_time in zip(axes, target_times):
        index = int(np.argmin(np.abs(dataset.t - target_time)))
        axis.plot(dataset.x, dataset.solution[index], color=_C["ink"], linewidth=2.4, label="Exact")
        axis.plot(dataset.x, prediction[index], color=_C["brick"], linewidth=2.0, linestyle="--", label="KAN")
        axis.set_title(f"$t={dataset.t[index]:.2f}$")
        axis.set_xlabel("$x$")
    axes[0].set_ylabel("$u(x,t)$")
    axes[0].legend()
    fig.tight_layout()
    return fig


def plot_edge_functions(
    samples: np.ndarray,
    responses: dict[int, dict[int, np.ndarray]],
    input_labels: tuple[str, ...] = ("x", "t"),
) -> Figure:
    """Plot selected first-layer edge functions."""
    output_indices = tuple(responses.keys())
    fig, axes = plt.subplots(len(output_indices), len(input_labels), figsize=(6.2 * len(input_labels), 3.2 * len(output_indices)))
    if len(output_indices) == 1:
        axes = np.array([axes])

    for row, output_index in enumerate(output_indices):
        for col, label in enumerate(input_labels):
            axis = axes[row, col]
            axis.plot(samples, responses[output_index][col], color=_C["teal"], linewidth=2.2)
            axis.axhline(0.0, color="#6c757d", linewidth=0.8, linestyle=":")
            axis.set_title(f"Neuron {output_index} edge for {label}")
            axis.set_xlabel(f"Normalized {label}")
            axis.set_ylabel("$\\phi(x)$")

    fig.tight_layout()
    return fig


def plot_residual_distribution(residuals: np.ndarray) -> Figure:
    """Show the Burgers residual distribution of the learned KAN field."""
    residuals = np.asarray(residuals, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(residuals, bins=50, color=_C["sand"], edgecolor="white")
    axes[0].set_title("Residual histogram")
    axes[0].set_xlabel("$r(x,t)$")
    axes[0].set_ylabel("Count")

    axes[1].hist(np.abs(residuals), bins=50, color=_C["brick"], edgecolor="white", log=True)
    axes[1].set_title("Absolute residual histogram")
    axes[1].set_xlabel("$|r(x,t)|$")
    axes[1].set_ylabel("Count (log)")

    fig.tight_layout()
    return fig
