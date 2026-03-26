"""Generate the Burgers-equation PINN notebook programmatically.

Produces a standard Jupyter notebook using nbformat.

Usage::

    python tools/generate_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    cells: list[nbf.NotebookNode] = []

    # ---- Title ----
    cells.append(
        nbf.v4.new_markdown_cell(
            """# Physics-Informed Neural Network for the Viscous Burgers Equation

This notebook demonstrates a structured PINN implementation for the 1-D viscous Burgers equation:

$$
\\frac{\\partial u}{\\partial t} + u \\frac{\\partial u}{\\partial x} = \\nu \\frac{\\partial^2 u}{\\partial x^2}, \\quad x \\in [-1, 1],\\; t \\in [0, 1]
$$

with initial condition $u(x, 0) = -\\sin(\\pi x)$ and Dirichlet boundaries $u(\\pm 1, t) = 0$.

**Architecture highlights:**
- Random Fourier feature input encoding (Tancik et al. 2020) to overcome spectral bias
- Residual-connected MLP blocks (Wang et al. 2021) for stable deep training
- Adaptive (learnable) per-term loss weighting
- Two-phase optimisation: Adam with cosine LR schedule → L-BFGS fine-tuning
- Latin Hypercube collocation sampling for efficient domain coverage

The analytical reference is computed via the **Cole-Hopf transform**.
"""
        )
    )

    # ---- Setup ----
    cells.append(nbf.v4.new_markdown_cell("## 1 — Setup and Configuration\n\nAll configuration is Pydantic-validated."))

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physics_informed_neural_network.config import ProjectConfig
from physics_informed_neural_network.data import evaluate_reference_solution, generate_reference_solution, sample_observations
from physics_informed_neural_network.pipeline import run_experiment
from physics_informed_neural_network.plotting import (
    apply_plot_style,
    plot_comparison,
    plot_loss_history,
    plot_pointwise_error,
    plot_reference_solution,
    plot_time_slices,
)

apply_plot_style()
pd.set_option("display.float_format", lambda v: f"{v:,.6f}")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """config = ProjectConfig()
config.artifacts.output_dir = PROJECT_ROOT / "artifacts" / "notebook_burgers_pinn"
display(config.model_dump())"""
        )
    )

    # ---- Reference Solution ----
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2 — Analytical Reference Solution (Cole-Hopf Transform)\n\n"
            "High-fidelity reference on a $256 \\times 100$ grid."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """reference = generate_reference_solution(config.pde, config.data)
print(f"Grid: {reference.nx} × {reference.nt} = {reference.nx * reference.nt:,} points")
print(f"ν = {reference.viscosity:.6f}")"""
        )
    )

    cells.append(nbf.v4.new_code_cell("fig_ref = plot_reference_solution(reference)\nfig_ref"))

    # ---- Training ----
    cells.append(nbf.v4.new_markdown_cell("## 3 — Train the PINN"))
    cells.append(nbf.v4.new_code_cell("experiment = run_experiment(config)"))

    # ---- Results ----
    cells.append(nbf.v4.new_markdown_cell("## 4 — Results: PINN vs Reference"))

    cells.append(
        nbf.v4.new_code_cell(
            """u_ref = evaluate_reference_solution(experiment.x_pred, experiment.t_pred, config.pde.viscosity)
fig_cmp = plot_comparison(experiment.x_pred, experiment.t_pred, u_ref, experiment.u_pred)
fig_cmp"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig_slices = plot_time_slices(experiment.x_pred, experiment.t_pred, u_ref, experiment.u_pred)
fig_slices"""
        )
    )

    # ---- Error ----
    cells.append(nbf.v4.new_markdown_cell("## 5 — Error Analysis"))
    cells.append(
        nbf.v4.new_code_cell(
            """metrics_df = pd.DataFrame([experiment.metrics.model_dump()])
display(metrics_df)"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """fig_err = plot_pointwise_error(experiment.x_pred, experiment.t_pred, u_ref, experiment.u_pred)
fig_err"""
        )
    )

    # ---- Loss ----
    cells.append(nbf.v4.new_markdown_cell("## 6 — Training Diagnostics"))
    cells.append(nbf.v4.new_code_cell("fig_loss = plot_loss_history(experiment.history)\nfig_loss"))
    cells.append(nbf.v4.new_code_cell("display(experiment.history.to_frame().tail(5))"))

    # ---- Summary ----
    cells.append(nbf.v4.new_markdown_cell("## 7 — Summary and Artifacts"))
    cells.append(nbf.v4.new_code_cell("display(experiment.summary.model_dump())"))

    cells.append(
        nbf.v4.new_code_cell(
            """artifact_df = pd.DataFrame([
    {"artifact": k, "path": str(v)} for k, v in experiment.artifact_paths.items()
])
display(artifact_df)"""
        )
    )

    notebook.cells = cells
    return notebook


def main() -> None:
    notebook = build_notebook()
    output_path = Path(__file__).resolve().parent.parent / "notebooks" / "burgers_equation_pinn.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(output_path))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
