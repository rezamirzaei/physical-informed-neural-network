"""Publication-quality visualisations for the Burgers-equation PINN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .schemas import ReferenceSolution, TrainingHistory


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "figure.facecolor": "#f7f5f2",
            "axes.facecolor": "#fffdfa",
            "axes.edgecolor": "#2c3e50",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.color": "#8c9aa8",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "font.size": 11,
        }
    )


# ---------------------------------------------------------------------------
# Reference solution heatmap
# ---------------------------------------------------------------------------

def plot_reference_solution(ref: ReferenceSolution) -> Figure:
    """2-D heatmap of the analytical reference solution u(x, t)."""
    X, T = ref.meshgrid()
    U = ref.u_array()

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(X, T, U, cmap="RdBu_r", shading="gouraud")
    fig.colorbar(pcm, ax=ax, label="u(x, t)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Analytical reference — Burgers equation")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Side-by-side: reference vs PINN prediction
# ---------------------------------------------------------------------------

def plot_comparison(
    x: np.ndarray,
    t: np.ndarray,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
) -> Figure:
    """Side-by-side heatmaps of reference, PINN prediction, and pointwise error."""
    X, T = np.meshgrid(x, t)
    error = np.abs(u_pred - u_ref)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin, vmax = u_ref.min(), u_ref.max()

    pcm0 = axes[0].pcolormesh(X, T, u_ref, cmap="RdBu_r", shading="gouraud", vmin=vmin, vmax=vmax)
    fig.colorbar(pcm0, ax=axes[0])
    axes[0].set_title("Reference u(x, t)")

    pcm1 = axes[1].pcolormesh(X, T, u_pred, cmap="RdBu_r", shading="gouraud", vmin=vmin, vmax=vmax)
    fig.colorbar(pcm1, ax=axes[1])
    axes[1].set_title("PINN prediction")

    pcm2 = axes[2].pcolormesh(X, T, error, cmap="hot_r", shading="gouraud")
    fig.colorbar(pcm2, ax=axes[2])
    axes[2].set_title("Pointwise absolute error")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    fig.suptitle("Burgers equation — PINN vs analytical reference", fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Time-slice comparison
# ---------------------------------------------------------------------------

def plot_time_slices(
    x: np.ndarray,
    t: np.ndarray,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
    time_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Figure:
    """Line plots of u(x) at selected time fractions."""
    n_slices = len(time_fractions)
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4), sharey=True)
    if n_slices == 1:
        axes = [axes]

    for ax, frac in zip(axes, time_fractions):
        idx = min(int(frac * (len(t) - 1)), len(t) - 1)
        ax.plot(x, u_ref[idx], "k-", linewidth=2.2, label="Reference")
        ax.plot(x, u_pred[idx], "r--", linewidth=1.8, label="PINN")
        ax.set_title(f"t = {t[idx]:.3f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend(loc="upper right")

    fig.suptitle("Solution snapshots at selected times", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pointwise error heatmap (standalone)
# ---------------------------------------------------------------------------

def plot_pointwise_error(
    x: np.ndarray,
    t: np.ndarray,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
) -> Figure:
    X, T = np.meshgrid(x, t)
    error = np.abs(u_pred - u_ref)

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(X, T, error, cmap="hot_r", shading="gouraud")
    fig.colorbar(pcm, ax=ax, label="|error|")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Pointwise absolute error")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Loss history
# ---------------------------------------------------------------------------

def plot_loss_history(history: TrainingHistory) -> Figure:
    df = history.to_frame()
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["step"], df["total_loss"], label="Total", linewidth=2.4, color="#12355b")
    ax.plot(df["step"], df["pde_loss"], label="PDE residual", linewidth=2.0, color="#2d6a4f")
    ax.plot(df["step"], df["boundary_loss"], label="Boundary", linewidth=2.0, color="#bc6c25")
    ax.plot(df["step"], df["initial_loss"], label="Initial cond.", linewidth=2.0, color="#9b2226")
    ax.plot(df["step"], df["data_loss"], label="Data", linewidth=1.8, color="#6a040f", linestyle="--")

    ax.set_yscale("log")
    ax.set_title("Training loss history")
    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Loss (log scale)")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PDE residual distribution
# ---------------------------------------------------------------------------

def plot_residual_distribution(residuals: np.ndarray) -> Figure:
    """Histogram of PDE residual magnitudes at interior collocation points."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals.ravel(), bins=80, color="#12355b", edgecolor="white", alpha=0.85)
    ax.set_xlabel("PDE residual")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of PDE residuals on interior points")
    ax.axvline(0.0, color="#9b2226", linewidth=1.5, linestyle="--")
    fig.tight_layout()
    return fig
