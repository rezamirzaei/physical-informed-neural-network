"""Publication-quality 2-D visualizations for the Darcy-flow FNO experiment.

All figures use a unified professional style with LaTeX-like math rendering,
consistent colour palettes, and carefully designed layouts.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .data_2d import DarcyDataset, DarcySample
from .schemas import NeuralOperatorExperimentSummary


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_colorbar(mappable, ax, label: str = "", fmt: str = "%.3g") -> None:
    """Attach a proportionally-sized colorbar to *ax*."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(mappable, cax=cax, label=label, format=fmt)


def _field_imshow(ax, field: np.ndarray, extent, cmap: str, title: str, label: str = "", vmin=None, vmax=None):
    """Consistent imshow rendering for a 2-D field."""
    norm = None
    if vmin is not None and vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(
        field.T, origin="lower", extent=extent, cmap=cmap, aspect="equal", norm=norm, interpolation="bilinear",
    )
    ax.set_title(title, fontsize=13, fontweight="medium")
    _add_colorbar(im, ax, label=label)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    return im


# ---------------------------------------------------------------------------
# Dataset gallery
# ---------------------------------------------------------------------------

def plot_darcy_dataset_examples(
    dataset: DarcyDataset,
    sample_indices: tuple[int, ...] = (0, 1, 2),
) -> Figure:
    """Show diffusivity, forcing, and solution for several samples as 2-D heatmaps."""
    n_rows = len(sample_indices)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    extent = [dataset.grid_x[0], dataset.grid_x[-1], dataset.grid_y[0], dataset.grid_y[-1]]

    for row, idx in enumerate(sample_indices):
        sample = dataset.sample(idx)

        _field_imshow(axes[row, 0], sample.diffusivity, extent, "YlOrBr",
                      f"Sample {idx} — $a(x,y)$", label="$a$")
        _field_imshow(axes[row, 1], sample.forcing, extent, "Blues",
                      "$f(x,y)$", label="$f$")
        _field_imshow(axes[row, 2], sample.solution, extent, "RdBu_r",
                      "$u(x,y)$", label="$u$")

    fig.suptitle(
        f"Darcy flow — {dataset.name.replace('_', ' ').title()} samples",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Prediction comparison (the key figure)
# ---------------------------------------------------------------------------

def plot_darcy_prediction_comparison(
    sample: DarcySample,
    prediction: np.ndarray,
    title: str = "FNO-2d prediction vs exact solution",
) -> Figure:
    """Side-by-side: input field, exact solution, FNO prediction, and log-error map."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    extent = [sample.grid_x[0], sample.grid_x[-1], sample.grid_y[0], sample.grid_y[-1]]

    # Input diffusivity
    _field_imshow(axes[0], sample.diffusivity, extent, "YlOrBr", "$a(x,y)$", label="$a$")

    # Exact solution
    vmin, vmax = sample.solution.min(), sample.solution.max()
    _field_imshow(axes[1], sample.solution, extent, "RdBu_r", "Exact $u(x,y)$", label="$u$",
                  vmin=vmin, vmax=vmax)

    # FNO prediction (same colour range)
    _field_imshow(axes[2], prediction, extent, "RdBu_r", "FNO prediction", label="$\\hat{u}$",
                  vmin=vmin, vmax=vmax)

    # Absolute error (log scale)
    error = np.abs(prediction - sample.solution)
    error_floor = max(error.max() * 1e-6, 1e-12)
    error_clipped = np.clip(error, error_floor, None)
    im_err = axes[3].imshow(
        error_clipped.T, origin="lower", extent=extent, cmap="inferno",
        norm=LogNorm(vmin=error_floor, vmax=error.max()), aspect="equal", interpolation="bilinear",
    )
    axes[3].set_title("$|u - \\hat{u}|$  (log scale)", fontsize=13, fontweight="medium")
    _add_colorbar(im_err, axes[3], label="error")
    axes[3].set_xlabel("$x$")
    axes[3].set_ylabel("$y$")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Error distribution across test set
# ---------------------------------------------------------------------------

def plot_darcy_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Per-sample relative $L^2$ error distribution",
) -> Figure:
    """Histogram + CDF of per-sample relative L² errors."""
    n_samples = predictions.shape[0]
    errors = np.array([
        np.linalg.norm((predictions[i] - targets[i]).ravel())
        / max(np.linalg.norm(targets[i].ravel()), 1e-12)
        for i in range(n_samples)
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Histogram
    ax1.hist(errors, bins=30, color="#12355b", edgecolor="white", alpha=0.9)
    ax1.axvline(np.median(errors), color="#9b2226", linewidth=2.0, linestyle="--", label=f"Median = {np.median(errors):.4f}")
    ax1.set_xlabel("Relative $L^2$ error")
    ax1.set_ylabel("Count")
    ax1.set_title("Error histogram")
    ax1.legend()

    # CDF
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax2.plot(sorted_errors, cdf, color="#2d6a4f", linewidth=2.5)
    ax2.axhline(0.5, color="#888", linewidth=1, linestyle=":")
    ax2.axvline(np.median(errors), color="#9b2226", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Relative $L^2$ error")
    ax2.set_ylabel("CDF")
    ax2.set_title("Cumulative distribution")
    ax2.set_ylim(0, 1.02)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Resolution transfer bar chart
# ---------------------------------------------------------------------------

def plot_darcy_resolution_metrics(summary: NeuralOperatorExperimentSummary) -> Figure:
    """Grouped bar chart comparing native and refined evaluation metrics."""
    labels, rel_l2, mae = [], [], []
    for key in ("test", "refined_test"):
        ev = summary.evaluations[key]
        labels.append(f"{ev.split.replace('_', ' ').title()}\n({ev.resolution}×{ev.resolution})")
        rel_l2.append(ev.metrics.relative_l2)
        mae.append(ev.metrics.mae)

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars1 = ax.bar(x - width / 2, rel_l2, width, color="#12355b", label="Relative $L^2$", edgecolor="white")
    bars2 = ax.bar(x + width / 2, mae, width, color="#bc6c25", label="MAE", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Metric value")
    ax.set_title("2-D resolution-transfer evaluation", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cross-section comparison (1-D slices through 2-D fields)
# ---------------------------------------------------------------------------

def plot_darcy_cross_sections(
    sample: DarcySample,
    prediction: np.ndarray,
    y_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> Figure:
    """Plot 1-D cross-sections of exact and predicted solutions at fixed y values."""
    n_plots = len(y_fractions)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.2), sharey=True)
    if n_plots == 1:
        axes = [axes]

    ny = len(sample.grid_y)
    for ax, frac in zip(axes, y_fractions):
        j = min(int(frac * (ny - 1)), ny - 1)
        y_val = sample.grid_y[j]
        ax.plot(sample.grid_x, sample.solution[:, j], "k-", linewidth=2.2, label="Exact")
        ax.plot(sample.grid_x, prediction[:, j], color="#9b2226", linewidth=2.0, linestyle="--", label="FNO")
        ax.fill_between(
            sample.grid_x,
            sample.solution[:, j],
            prediction[:, j],
            alpha=0.15,
            color="#9b2226",
        )
        ax.set_title(f"$y = {y_val:.2f}$", fontsize=13)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$u$")
    axes[0].legend()

    fig.suptitle("Cross-section comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig

