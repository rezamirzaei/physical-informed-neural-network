"""Generate the KAN tutorial notebook programmatically."""

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

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Kolmogorov-Arnold Networks for the Burgers Equation

This notebook studies **KANs** through a clean, verifiable function-approximation problem:
the exact solution field of the 1-D viscous Burgers equation.

The goal is not just to train another model. The notebook connects:

- the **Kolmogorov-Arnold representation viewpoint**
- the practical **KAN layer** with learned univariate edge functions
- a **verified analytical target** via the Cole-Hopf transform
- a reusable implementation with **Pydantic configs**, **OO trainers**, and **notebook-safe plots**
- quantitative verification through **held-out error metrics** and **PDE residual checks**

Primary reference:

- Liu et al. (2024), *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1 — Theory: Why KANs Are Different

Standard MLPs place nonlinearities on **nodes** and use linear weights on edges.
KANs reverse that pattern: each edge carries a **learned univariate function**, and each neuron mainly sums edge responses.

At a high level, a KAN layer computes

$$
y_j = \\sum_i \\phi_{j,i}(x_i),
$$

where each $\\phi_{j,i}$ is a learnable scalar function. In the paper, these edge functions are represented with spline parameterizations plus a simple base function.

This links back to the **Kolmogorov-Arnold representation theorem**, which states that multivariate continuous functions can be expressed through sums and compositions of univariate functions.

Important caveat:

- the theorem is an **existence result**, not a turnkey training guarantee
- practical KANs are a **model class inspired by that theorem**, not a direct constructive proof
- they do not eliminate optimization difficulty or the curse of dimensionality in arbitrary regimes

So the right way to read KAN is: it provides a structured, interpretable function class with strong inductive bias toward low-dimensional separable structure.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2 — Theory: The Practical KAN Layer Used Here

This repository implements a compact, stable KAN suitable for scientific-regression notebooks:

1. coordinates $(x,t)$ are normalized to $[-1,1]^2$
2. each edge function is

$$
\\phi_{j,i}(z) = w^{(b)}_{j,i} \\, b(z) + w^{(s)}_{j,i} \\, s_{j,i}(z),
$$

where $b$ is a base activation and $s_{j,i}$ is a learned **piecewise-linear spline**
3. hidden layers sum these edge functions to build increasingly rich latent coordinates

This is faithful to the KAN philosophy while keeping the code small, testable, and easy to inspect.

We use **piecewise-linear hat bases** instead of a more elaborate adaptive-grid spline system because this tutorial prioritizes:

- deterministic behavior
- readable implementation
- reliable smoke testing
- straightforward visualization of learned edge functions
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3 — Problem Setup: Exact Burgers Supervision

The target field is the exact solution of

$$
u_t + u u_x = \\nu u_{xx}, \\qquad x \\in [-1,1], \\quad t \\in [0,1],
$$

with

$$
u(x,0) = -\\sin(\\pi x), \\qquad u(-1,t)=u(1,t)=0.
$$

The exact solution is generated with the **Cole-Hopf transform**, so our supervision is analytical rather than numerical. That gives us a strong verification story:

- coarse-grid training points come from a known exact field
- finer-grid evaluation checks interpolation quality
- autograd-based PDE residuals reveal whether the learned surrogate preserves the underlying physics
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physics_informed_neural_network.kan.data import build_dataset_splits
from physics_informed_neural_network.kan.pipeline import run_kan_experiment
from physics_informed_neural_network.kan.plotting import (
    apply_plot_style,
    plot_edge_functions,
    plot_residual_distribution,
    plot_solution_comparison,
    plot_time_slices,
    plot_training_history,
)
from physics_informed_neural_network.kan.presets import build_smoke_test_config, build_tutorial_config

apply_plot_style()
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4 — Pydantic Configuration

The KAN stack is structured as reusable library code:

- `KANExperimentConfig`: top-level experiment contract
- `KANDataConfig`: train/validation/test/evaluation grids
- `PiecewiseLinearKANConfig`: spline-edge architecture
- `KANOptimizationConfig`: optimizer and scheduler settings

The tutorial preset below trains on a coarse grid and evaluates on a much finer grid.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """config = build_tutorial_config(output_dir=PROJECT_ROOT / "artifacts" / "notebook_kan")
display(config.model_dump())"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """splits = build_dataset_splits(config)
dataset_frame = pd.DataFrame(
    [
        {"split": "train", "nx": splits.train.nx, "nt": splits.train.nt, "points": splits.train.n_points},
        {"split": "validation", "nx": splits.validation.nx, "nt": splits.validation.nt, "points": splits.validation.n_points},
        {"split": "test", "nx": splits.test.nx, "nt": splits.test.nt, "points": splits.test.n_points},
        {"split": "evaluation", "nx": splits.evaluation.nx, "nt": splits.evaluation.nt, "points": splits.evaluation.n_points},
    ]
)
display(dataset_frame)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 5 — Why This Is a Useful KAN Experiment

This tutorial is deliberately **not** a PINN training run. Instead, it isolates the approximation question:

- Can a KAN learn the Burgers solution field from coarse samples?
- Does that approximation remain accurate on a finer unseen grid?
- What do the learned first-layer edge functions look like?
- Does the learned surrogate still exhibit a small Burgers residual?

That separation is useful because it lets us evaluate the KAN architecture itself before mixing in physics-informed optimization.
"""
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6 — Train the KAN"))
    cells.append(nbf.v4.new_code_cell("experiment = run_kan_experiment(config)"))

    cells.append(
        nbf.v4.new_code_cell(
            """summary_frame = pd.DataFrame([experiment.summary.model_dump()])
display(summary_frame)"""
        )
    )

    cells.append(nbf.v4.new_code_cell("fig_history = plot_training_history(experiment.history)\nfig_history"))

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 7 — Solution Quality on the Dense Evaluation Grid

The model is trained on the coarse train grid but visualized on the much denser evaluation grid.
This is the right place to inspect whether the learned field is smooth, stable, and physically plausible.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig_solution = plot_solution_comparison(
    experiment.datasets.evaluation,
    experiment.evaluation_prediction,
    title="KAN vs exact Burgers field on the dense evaluation grid",
)
fig_solution"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig_slices = plot_time_slices(experiment.datasets.evaluation, experiment.evaluation_prediction)
fig_slices"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 8 — Verification: PDE Residual of the Learned Surrogate

Even though this model is trained purely by supervised regression, we can still test whether the learned field respects the Burgers equation:

$$
r(x,t) = u_t + u u_x - \\nu u_{xx}.
$$

Small residuals do not make the model a PINN, but they are a useful sanity check that the approximation is not merely pointwise accurate while violating the underlying dynamics everywhere else.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """residual_metrics = pd.DataFrame([experiment.summary.residual_metrics.model_dump()])
display(residual_metrics)"""
        )
    )

    cells.append(nbf.v4.new_code_cell("fig_residual = plot_residual_distribution(experiment.residuals)\nfig_residual"))

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 9 — Interpretability: Learned First-Layer Edge Functions

One of the appealing parts of KANs is that the nonlinear structure sits on explicit 1-D edge functions.
For the first layer, those functions are directly interpretable because the inputs are just normalized $x$ and $t$ coordinates.

We inspect several hidden neurons below. Each row corresponds to a hidden neuron and each column corresponds to one input coordinate.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """sample_axis = np.linspace(-1.0, 1.0, 400, dtype=np.float32)
sample_tensor = torch.tensor(sample_axis)
edge_responses_torch = experiment.model.evaluate_first_layer_edges(sample_tensor, output_indices=(0, 1, 2, 3))
edge_responses = {
    neuron: {input_idx: response.detach().cpu().numpy() for input_idx, response in edges.items()}
    for neuron, edges in edge_responses_torch.items()
}
fig_edges = plot_edge_functions(sample_axis, edge_responses)
fig_edges"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 10 — Interpreting the Learned Structure

Typical patterns to look for:

- **odd/even structure in $x$ edges** because the Burgers initial condition is sinusoidal and spatially structured
- **time-dependent attenuation** because viscosity damps the solution as $t$ increases
- **nonlinear local corrections** added by the spline component on top of the base activation

These plots do not fully explain the whole network, but they make the first compositional layer much more inspectable than a dense matrix of MLP weights.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 11 — Limits and Honest Takeaways

This notebook demonstrates what KANs are good at in a scientific-computing workflow:

- compact low-dimensional regression
- explicit univariate nonlinearities
- good visual interpretability for the first layer
- straightforward integration with exact solvers and residual diagnostics

It also shows what KANs are **not**:

- a theorem-powered guarantee of easy training
- a replacement for every MLP or operator model
- a direct substitute for physics-informed losses when constraints must be enforced during optimization

In this repository, the KAN tutorial complements the existing PINN and neural-operator stacks:

- the **PINN** emphasizes physics-constrained optimization
- the **FNO/neural operator** emphasizes learning maps between functions
- the **KAN** emphasizes interpretable function approximation through learned edge nonlinearities
"""
        )
    )

    notebook.cells = cells
    return notebook


def main() -> None:
    notebook = build_notebook()
    output_path = Path(__file__).resolve().parent.parent / "notebooks" / "kolmogorov_arnold_networks_burgers.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(output_path))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
