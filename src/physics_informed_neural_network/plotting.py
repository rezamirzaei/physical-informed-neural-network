"""Publication-quality visualizations for the Burgers-equation PINN.

Uses a unified professional style with LaTeX-like math rendering,
consistent colour palettes, contour overlays, and carefully designed layouts.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .schemas import ReferenceSolution, TrainingHistory


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


def _add_colorbar(mappable, ax, label: str = "") -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    plt.colorbar(mappable, cax=cax, label=label)


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def apply_plot_style() -> None:
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


# ---------------------------------------------------------------------------
# Reference solution heatmap
# ---------------------------------------------------------------------------

def plot_reference_solution(ref: ReferenceSolution) -> Figure:
    """2-D heatmap of the analytical reference solution $u(x, t)$."""
    X, T = ref.meshgrid()
    U = ref.u_array()

    vabs = max(abs(U.min()), abs(U.max()))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pcm = ax.pcolormesh(X, T, U, cmap="RdBu_r", shading="gouraud", norm=norm)
    # Contour overlay
    levels = np.linspace(-vabs, vabs, 12)
    ax.contour(X, T, U, levels=levels, colors="k", linewidths=0.4, alpha=0.35)
    _add_colorbar(pcm, ax, label="$u(x, t)$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_title("Analytical reference — Burgers equation", fontsize=15, fontweight="bold")
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

    vabs = max(abs(u_ref.min()), abs(u_ref.max()))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    pcm0 = axes[0].pcolormesh(X, T, u_ref, cmap="RdBu_r", shading="gouraud", norm=norm)
    axes[0].contour(X, T, u_ref, levels=10, colors="k", linewidths=0.35, alpha=0.3)
    _add_colorbar(pcm0, axes[0], "$u$")
    axes[0].set_title("Reference $u(x, t)$")

    pcm1 = axes[1].pcolormesh(X, T, u_pred, cmap="RdBu_r", shading="gouraud", norm=norm)
    axes[1].contour(X, T, u_pred, levels=10, colors="k", linewidths=0.35, alpha=0.3)
    _add_colorbar(pcm1, axes[1], "$\\hat{u}$")
    axes[1].set_title("PINN prediction $\\hat{u}(x, t)$")

    pcm2 = axes[2].pcolormesh(X, T, error, cmap="inferno", shading="gouraud")
    _add_colorbar(pcm2, axes[2], "$|\\epsilon|$")
    axes[2].set_title("Pointwise absolute error")

    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")

    fig.suptitle(
        "Burgers equation — PINN vs analytical reference",
        fontsize=16, fontweight="bold", y=1.03,
    )
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
    """Line plots of $u(x)$ at selected time fractions with error bands."""
    n_slices = len(time_fractions)
    fig, axes = plt.subplots(1, n_slices, figsize=(4.2 * n_slices, 4.2), sharey=True)
    if n_slices == 1:
        axes = [axes]

    for ax, frac in zip(axes, time_fractions):
        idx = min(int(frac * (len(t) - 1)), len(t) - 1)
        ax.plot(x, u_ref[idx], color=_C["dark"], linewidth=2.4, label="Reference")
        ax.plot(x, u_pred[idx], color=_C["crimson"], linewidth=2.0, linestyle="--", label="PINN")
        ax.fill_between(x, u_ref[idx], u_pred[idx], alpha=0.12, color=_C["crimson"])
        ax.set_title(f"$t = {t[idx]:.3f}$")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$u$")
    axes[0].legend(loc="upper right")

    fig.suptitle("Solution snapshots at selected times", fontsize=14, fontweight="bold", y=1.03)
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

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pcm = ax.pcolormesh(X, T, error, cmap="inferno", shading="gouraud")
    _add_colorbar(pcm, ax, label="$|u - \\hat{u}|$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_title("Pointwise absolute error", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Loss history
# ---------------------------------------------------------------------------

def plot_loss_history(history: TrainingHistory) -> Figure:
    df = history.to_frame()
    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.plot(df["step"], df["total_loss"], label="Total", linewidth=2.6, color=_C["navy"])
    ax.plot(df["step"], df["pde_loss"], label="PDE residual", linewidth=2.0, color=_C["forest"])
    ax.plot(df["step"], df["boundary_loss"], label="Boundary", linewidth=2.0, color=_C["amber"])
    ax.plot(df["step"], df["initial_loss"], label="Initial cond.", linewidth=2.0, color=_C["crimson"])
    ax.plot(df["step"], df["data_loss"], label="Data", linewidth=1.8, color=_C["rust"], linestyle="--")

    ax.set_yscale("log")
    ax.set_title("Training loss history", fontsize=15, fontweight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss (log scale)")
    ax.legend(ncol=2)

    # Mark phase boundary (Adam → L-BFGS)
    adam_steps = df[df["phase"] == "adam"]["step"]
    if len(adam_steps) > 0 and len(df[df["phase"] == "lbfgs"]) > 0:
        boundary = adam_steps.iloc[-1]
        ax.axvline(boundary, color=_C["teal"], linewidth=1.5, linestyle="--", alpha=0.6)
        ax.text(boundary, ax.get_ylim()[1] * 0.7, " L-BFGS →",
                fontsize=10, color=_C["teal"], va="top")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PDE residual distribution
# ---------------------------------------------------------------------------

def plot_residual_distribution(residuals: np.ndarray) -> Figure:
    """Histogram of PDE residual magnitudes at interior collocation points."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    ax1.hist(residuals.ravel(), bins=80, color=_C["navy"], edgecolor="white", alpha=0.85, density=True)
    ax1.set_xlabel("PDE residual")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution of PDE residuals")
    ax1.axvline(0.0, color=_C["crimson"], linewidth=2, linestyle="--", label="Zero")
    ax1.legend()

    abs_res = np.abs(residuals.ravel())
    sorted_res = np.sort(abs_res)
    cdf = np.arange(1, len(sorted_res) + 1) / len(sorted_res)
    ax2.plot(sorted_res, cdf, color=_C["forest"], linewidth=2.2)
    ax2.set_xlabel("$|\\text{residual}|$")
    ax2.set_ylabel("CDF")
    ax2.set_title("Cumulative distribution of $|r|$")
    ax2.axhline(0.95, color=_C["amber"], linewidth=1, linestyle=":", label="95th percentile")
    p95 = np.percentile(abs_res, 95)
    ax2.axvline(p95, color=_C["amber"], linewidth=1, linestyle=":")
    ax2.text(p95, 0.85, f"  {p95:.2e}", fontsize=10, color=_C["amber"])
    ax2.legend()

    fig.suptitle("PDE residual analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
