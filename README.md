# Physics-Informed Neural Network for the 1-D Burgers Equation

Clean, reproducible PyTorch implementation of a Physics-Informed Neural Network (PINN) for the viscous 1-D Burgers equation. The repository is structured as a small research codebase rather than a one-off notebook: packaged source code, a CLI, tests, CI, and example artifacts are all included.

![Reference vs PINN prediction](artifacts/burgers_pinn/comparison.png)

## Problem

The model solves

$$
u_t + u u_x = \nu u_{xx}, \quad x \in [-1, 1], \ t \in [0, 1]
$$

with:

- Initial condition: $u(x, 0) = -\sin(\pi x)$
- Boundary conditions: $u(-1, t) = u(1, t) = 0$
- Default viscosity: $\nu = 0.01 / \pi$

The reference solution is generated analytically with the Cole-Hopf transform and used both for sparse observations and evaluation.

## What Is In The Repo

- `src/physics_informed_neural_network/`: the single packaged implementation, public API, and CLI
- `src/physics_informed_neural_network/kan/`: reusable KAN stack for exact Burgers regression
- `tests/`: fast automated checks, including an end-to-end smoke test
- `artifacts/burgers_pinn/`: example outputs from a reference run
- `notebooks/burgers_equation_pinn.ipynb`: generated exploratory notebook
- `notebooks/neural_operator_function_spaces.ipynb`: theory-heavy neural-operator tutorial notebook
- `notebooks/kolmogorov_arnold_networks_burgers.ipynb`: theory-heavy KAN tutorial notebook
- `.github/workflows/ci.yml`: GitHub Actions workflow for install + test + notebook generation

## Method

- Random Fourier features for coordinate encoding
- Residual MLP blocks for the backbone
- Multi-term PINN loss with PDE, boundary, initial-condition, and data supervision
- Adam training with optional LR scheduling, followed by L-BFGS refinement
- Latin Hypercube sampling for collocation points
- Pydantic-validated configuration objects for experiment settings

## Neural Operator

This repository now also includes a neural-operator tutorial, not just a PINN example.

A **neural operator** learns a map between function spaces instead of a map between fixed-length vectors. In practical terms, the goal is to learn

$$
G^\dagger : (a(x), f(x)) \mapsto u(x),
$$

where the input is itself one or more functions and the output is another function. That is different from a standard neural network that expects a single discretized vector at one fixed resolution.

The neural-operator tutorial in this repo focuses on a **Fourier neural operator (FNO)**. The idea is:

- lift pointwise input channels into a higher-dimensional latent representation
- apply nonlocal mixing in Fourier space on the low-frequency modes
- combine that spectral operator with a local linear path
- project the latent field back to the target solution field

Why this matters:

- it is an **operator-learning** view of PDEs, not just pointwise regression
- the same learned model can often be evaluated on **different grid resolutions**
- Fourier-space parameterization gives an efficient way to model long-range interactions

The included notebook, `notebooks/neural_operator_function_spaces.ipynb`, makes this concrete using the 1-D variable-coefficient diffusion problem

$$
-\left(a(x)u'(x)\right)' = f(x), \qquad u(0)=u(1)=0,
$$

with randomly generated smooth coefficient and forcing functions. The tutorial is designed to show all the important parts:

- the operator-learning formulation from the neural-operator literature
- the difference between PINNs and neural operators
- an exact 1-D solution formula used to generate verified supervision data
- native-resolution evaluation and zero-shot finer-grid transfer
- spectral error analysis and PDE-residual checks on predictions

The reusable implementation lives under `src/physics_informed_neural_network/neural_operator/`, and the notebook is generated from code so the theory walkthrough stays aligned with the library implementation.

## KAN

This repository also includes a **Kolmogorov-Arnold Network (KAN)** tutorial built around the exact Burgers solution.

A KAN replaces fixed scalar edge weights with **learned univariate functions**. Instead of a dense layer computing

$$
y = W x + b,
$$

the KAN viewpoint is closer to

$$
y_j = \sum_i \phi_{j,i}(x_i),
$$

where each edge function $\phi_{j,i}$ is learned from data. In this repository, those edge functions are implemented as a combination of:

- a simple base activation path
- a learned **piecewise-linear spline**
- explicit coordinate normalization so the spline domain is stable and inspectable

Why this matters:

- it gives a more **interpretable** nonlinear structure than a standard MLP
- it matches the **Kolmogorov-Arnold** perspective of composing multivariate functions from univariate pieces
- it is well suited to **low-dimensional scientific regression**, where inspecting learned response curves is useful

The KAN notebook, `notebooks/kolmogorov_arnold_networks_burgers.ipynb`, is intentionally different from the PINN and neural-operator tutorials.

It does **not** train a physics-informed model. Instead, it isolates the approximation problem:

- train a spline-edge KAN on coarse samples of the exact Burgers field
- evaluate on a finer unseen grid
- inspect first-layer learned edge functions for the normalized $(x,t)$ inputs
- verify the learned surrogate with both regression metrics and Burgers PDE residual checks

That makes the KAN tutorial a good complement to the rest of the repo:

- the **PINN** section is about enforcing physics during optimization
- the **neural operator** section is about learning maps between function spaces
- the **KAN** section is about interpretable scalar function approximation with learned univariate edge nonlinearities

The reusable implementation lives under `src/physics_informed_neural_network/kan/`, and the notebook is generated from code so the theory and the implementation stay aligned.

## Installation

Recommended with `uv`:

```bash
uv sync --extra dev
```

This repository now includes a checked-in `uv.lock` and platform-aware dependency markers.
On Intel macOS, `uv` will resolve to `torch 2.2.2` and `numpy 1.26.x` because newer Torch releases do not ship `macOS x86_64` wheels.

`pip` remains supported:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick Start

Run a fast smoke test:

```bash
burgers-pinn --smoke-test
```

Run the default experiment:

```bash
burgers-pinn
```

Run the installed module directly:

```bash
python -m physics_informed_neural_network --smoke-test
```

You can also use the repository entry point without installation:

```bash
python main.py --smoke-test
```

Run through `uv` without manually activating an environment:

```bash
uv run burgers-pinn --smoke-test
```

Useful flags:

```bash
burgers-pinn --smoke-test --no-artifacts --json
burgers-pinn --device cpu --adam-epochs 5000 --lbfgs-iterations 300
burgers-pinn --output-dir artifacts/custom_run
```

## Testing

With `uv`:

```bash
uv run pytest -q
```

With an activated virtualenv:

```bash
pytest -q
```

The test suite includes:

- configuration validation
- reference-solution and sampler shape checks
- an end-to-end smoke run of the training pipeline
- neural-operator solver, model-shape, and smoke-experiment checks
- KAN basis, model-shape, and smoke-experiment checks

## Example Outputs

The repository includes a sample run in `artifacts/burgers_pinn/`:

- `comparison.png`: analytical solution vs PINN prediction
- `time_slices.png`: selected temporal slices
- `pointwise_error.png`: absolute error heatmap
- `loss_history.png`: training curves
- `summary.json`: experiment metadata and metrics

## Notebook

The notebooks are generated from code so that the narrative view stays consistent with the package implementation:

```bash
uv run python tools/generate_notebook.py
uv run python tools/generate_neural_operator_notebook.py
uv run python tools/generate_kan_notebook.py
```

Equivalent commands inside an activated virtualenv:

```bash
python tools/generate_notebook.py
python tools/generate_neural_operator_notebook.py
python tools/generate_kan_notebook.py
```

## Shareability Checklist

This project is reasonable to publish on GitHub now because it has:

- real source code instead of only a notebook
- a documented installation and execution path
- a smoke-test mode for quick verification
- automated tests and a CI workflow
- a license
- included sample outputs

## References

1. Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
2. Tancik, M., et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS*.
3. Wang, S., Teng, Y., and Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*.
4. Kovachki, N., et al. (2023). Neural operator: Learning maps between function spaces. *Journal of Machine Learning Research*, 24(89), 1-97.
5. Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold networks. *arXiv preprint arXiv:2404.19756*.

## License

MIT. See [LICENSE](LICENSE).
