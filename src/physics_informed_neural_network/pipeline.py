"""End-to-end experiment pipeline for the Burgers-equation PINN.

Orchestrates: data generation → collocation sampling → model construction →
training → prediction → metric computation → artifact export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from .config import PDEConfig, ProjectConfig
from .data import (
    evaluate_reference_solution,
    generate_reference_solution,
    sample_boundary_collocation,
    sample_initial_collocation,
    sample_interior_collocation,
    sample_observations,
)
from .model import BurgersPINN
from .physics import BurgersResidual
from .plotting import (
    apply_plot_style,
    plot_comparison,
    plot_loss_history,
    plot_pointwise_error,
    plot_reference_solution,
    plot_residual_distribution,
    plot_time_slices,
)
from .schemas import (
    ErrorMetrics,
    ExperimentSummary,
    ObservationSet,
    ReferenceSolution,
    TrainingHistory,
)
from .training import PINNTrainer
from .utils import ensure_directory, select_device, set_global_seed


# ---------------------------------------------------------------------------
# Experiment output container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentArtifacts:
    reference: ReferenceSolution
    observations: ObservationSet
    history: TrainingHistory
    u_pred: np.ndarray
    x_pred: np.ndarray
    t_pred: np.ndarray
    metrics: ErrorMetrics
    summary: ExperimentSummary
    artifact_paths: dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(u_ref: np.ndarray, u_pred: np.ndarray) -> ErrorMetrics:
    diff = u_pred - u_ref
    l2_rel = float(np.linalg.norm(diff) / (np.linalg.norm(u_ref) + 1e-30))
    return ErrorMetrics(
        l2_relative_error=l2_rel,
        mse=float(np.mean(diff ** 2)),
        mae=float(np.mean(np.abs(diff))),
        max_absolute_error=float(np.max(np.abs(diff))),
    )


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def _save_artifacts(
    output_dir: Path,
    pde: PDEConfig,
    reference: ReferenceSolution,
    observations: ObservationSet,
    history: TrainingHistory,
    u_pred: np.ndarray,
    x_pred: np.ndarray,
    t_pred: np.ndarray,
    u_ref_pred: np.ndarray,
    trainer: PINNTrainer,
) -> dict[str, Path]:
    ensure_directory(output_dir)
    paths: dict[str, Path] = {}

    # CSV: training history
    history_path = output_dir / "training_history.csv"
    history.to_frame().to_csv(history_path, index=False)
    paths["training_history"] = history_path

    # CSV: observations
    obs_path = output_dir / "observations.csv"
    pd.DataFrame({"x": observations.x, "t": observations.t, "u": observations.u}).to_csv(obs_path, index=False)
    paths["observations"] = obs_path

    # CSV: predictions on the fine grid
    X, T = np.meshgrid(x_pred, t_pred)
    pred_path = output_dir / "predictions.csv"
    pd.DataFrame({"x": X.ravel(), "t": T.ravel(), "u_pred": u_pred.ravel(), "u_ref": u_ref_pred.ravel()}).to_csv(
        pred_path, index=False
    )
    paths["predictions"] = pred_path

    # Plots
    apply_plot_style()

    fig_ref = plot_reference_solution(reference)
    p = output_dir / "reference_solution.png"
    fig_ref.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig_ref)
    paths["reference_plot"] = p

    fig_cmp = plot_comparison(x_pred, t_pred, u_ref_pred, u_pred)
    p = output_dir / "comparison.png"
    fig_cmp.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig_cmp)
    paths["comparison_plot"] = p

    fig_slices = plot_time_slices(x_pred, t_pred, u_ref_pred, u_pred)
    p = output_dir / "time_slices.png"
    fig_slices.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig_slices)
    paths["time_slices_plot"] = p

    fig_err = plot_pointwise_error(x_pred, t_pred, u_ref_pred, u_pred)
    p = output_dir / "pointwise_error.png"
    fig_err.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig_err)
    paths["error_plot"] = p

    fig_loss = plot_loss_history(history)
    p = output_dir / "loss_history.png"
    fig_loss.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig_loss)
    paths["loss_plot"] = p

    # Residual distribution (computed fresh on a separate set of points)
    interior_pts = sample_interior_collocation(pde, 5000, seed=999)
    interior_t = torch.tensor(interior_pts, dtype=torch.float32, device=next(trainer.model.parameters()).device)
    interior_t.requires_grad_(True)
    with torch.enable_grad():
        residuals = trainer.pde.residual(trainer.model, interior_t).detach().cpu().numpy()
    fig_res = plot_residual_distribution(residuals)
    p = output_dir / "residual_distribution.png"
    fig_res.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig_res)
    paths["residual_plot"] = p

    return paths


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(config: ProjectConfig) -> ExperimentArtifacts:
    """Execute the full Burgers-equation PINN experiment."""
    set_global_seed(config.training.seed)
    device = select_device(config.training.device)
    print(f"Device: {device}")

    # 1. Generate analytical reference solution
    print("Generating analytical reference solution …")
    reference = generate_reference_solution(config.pde, config.data)
    print(f"  Reference grid: {reference.nx} × {reference.nt}  ({reference.nx * reference.nt:,} points)")

    # 2. Sample sparse observations
    observations = sample_observations(
        reference,
        n_points=config.data.n_observed,
        noise_std=config.data.noise_std,
        seed=config.data.seed,
    )
    print(f"  Observations: {observations.n_points}  (noise σ = {observations.noise_std})")

    # 3. Sample collocation points
    interior_pts = sample_interior_collocation(config.pde, config.training.n_collocation, seed=config.training.seed)
    boundary_pts = sample_boundary_collocation(config.pde, config.training.n_boundary, seed=config.training.seed + 1)
    initial_pts = sample_initial_collocation(config.pde, config.training.n_initial, seed=config.training.seed + 2)
    print(
        f"  Collocation — interior: {len(interior_pts)}, boundary: {len(boundary_pts)}, initial: {len(initial_pts)}"
    )

    # 4. Build model + trainer
    model = BurgersPINN(config.network).to(device)
    pde = BurgersResidual(viscosity=config.pde.viscosity)
    trainer = PINNTrainer(
        model=model,
        pde=pde,
        observations=observations,
        interior_pts=interior_pts,
        boundary_pts=boundary_pts,
        initial_pts=initial_pts,
        config=config.training,
        device=device,
    )
    print(f"  Model: {model.architecture_string()}  ({model.count_parameters():,} parameters)")
    print("Training …")

    # 5. Train
    history = trainer.train()

    # 6. Predict on a fine evaluation grid
    x_pred = np.linspace(config.pde.x_min, config.pde.x_max, config.artifacts.prediction_nx)
    t_pred = np.linspace(config.pde.t_min, config.pde.t_max, config.artifacts.prediction_nt)
    u_pred = trainer.predict(x_pred, t_pred)

    # 7. Evaluate the analytical solution on the same grid for comparison
    u_ref_pred = evaluate_reference_solution(x_pred, t_pred, config.pde.viscosity)

    # 8. Metrics
    metrics = _compute_metrics(u_ref_pred, u_pred)
    print(f"\nMetrics:")
    print(f"  L2 relative error : {metrics.l2_relative_error:.6e}")
    print(f"  MSE               : {metrics.mse:.6e}")
    print(f"  MAE               : {metrics.mae:.6e}")
    print(f"  Max |error|       : {metrics.max_absolute_error:.6e}")

    # 9. Build summary
    last_entry = history.entries[-1] if history.entries else None
    final_losses = {
        "total": last_entry.total_loss if last_entry else 0.0,
        "pde": last_entry.pde_loss if last_entry else 0.0,
        "boundary": last_entry.boundary_loss if last_entry else 0.0,
        "initial": last_entry.initial_loss if last_entry else 0.0,
        "data": last_entry.data_loss if last_entry else 0.0,
    }

    summary = ExperimentSummary(
        viscosity=config.pde.viscosity,
        domain_x=(config.pde.x_min, config.pde.x_max),
        domain_t=(config.pde.t_min, config.pde.t_max),
        reference_grid=(config.data.nx, config.data.nt),
        n_observations=config.data.n_observed,
        architecture=model.architecture_string(),
        trainable_parameters=model.count_parameters(),
        device=str(device),
        adam_epochs=config.training.adam_epochs,
        lbfgs_iterations=config.training.lbfgs_iterations,
        metrics=metrics,
        final_losses=final_losses,
    )

    # 10. Save artifacts
    artifact_paths: dict[str, Path] = {}
    if config.artifacts.save_artifacts:
        artifact_paths = _save_artifacts(
            output_dir=config.artifacts.output_dir,
            pde=config.pde,
            reference=reference,
            observations=observations,
            history=history,
            u_pred=u_pred,
            x_pred=x_pred,
            t_pred=t_pred,
            u_ref_pred=u_ref_pred,
            trainer=trainer,
        )
        summary_path = config.artifacts.output_dir / "summary.json"
        artifact_paths["summary"] = summary_path
        summary = summary.model_copy(
            update={"artifact_paths": {k: str(v) for k, v in artifact_paths.items()}}
        )
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    return ExperimentArtifacts(
        reference=reference,
        observations=observations,
        history=history,
        u_pred=u_pred,
        x_pred=x_pred,
        t_pred=t_pred,
        metrics=metrics,
        summary=summary,
        artifact_paths=artifact_paths,
    )
