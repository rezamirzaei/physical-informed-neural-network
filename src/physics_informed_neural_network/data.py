"""Data generation for the 1-D viscous Burgers equation.

The analytical reference is computed via the **Cole-Hopf transform**, which
converts the nonlinear Burgers equation into the linear heat equation whose
solution is a Fourier series that can be evaluated to machine precision.

This module also provides Latin-Hypercube-based collocation-point samplers.
"""

from __future__ import annotations

import numpy as np

from .config import DataConfig, PDEConfig
from .schemas import ObservationSet, ReferenceSolution
from .utils import latin_hypercube_sample


# ---------------------------------------------------------------------------
# Analytical (Cole-Hopf) solution  u(x, t)  for  u_t + u u_x = nu u_xx
# with  u(x, 0) = -sin(pi x),  u(-1, t) = u(1, t) = 0.
# ---------------------------------------------------------------------------

def _cole_hopf_solution(
    x: np.ndarray,
    t: np.ndarray,
    nu: float,
    n_terms: int = 100,
) -> np.ndarray:
    """Evaluate the exact Burgers solution on an (x, t) grid.

    Uses the Cole-Hopf transform with Simpson-rule numerical integration,
    fully vectorised over the spatial dimension for each time step.

    Parameters
    ----------
    x : 1-D array of shape (nx,)
    t : 1-D array of shape (nt,)
    nu : kinematic viscosity.
    n_terms : unused (kept for API compatibility).

    Returns
    -------
    u : array of shape (len(t), len(x))
    """
    from scipy.integrate import simpson

    nx, nt = len(x), len(t)

    # Quadrature grid — 1001 points (odd for Simpson) over a wide window
    ny = 1001
    y = np.linspace(-4.0, 4.0, ny)
    potential = np.cos(np.pi * y) / (2.0 * np.pi * nu)  # (ny,)

    u = np.empty((nt, nx), dtype=np.float64)

    for i in range(nt):
        ti = t[i]
        if ti <= 0.0:
            u[i, :] = -np.sin(np.pi * x)
            continue

        # diff[j, k] = x[k] - y[j],  shape (ny, nx)
        diff = x[np.newaxis, :] - y[:, np.newaxis]

        exponent = -(diff ** 2) / (4.0 * nu * ti) + potential[:, np.newaxis]
        exponent -= exponent.max(axis=0, keepdims=True)  # stabilise

        phi_vals = np.exp(exponent)
        phi = simpson(phi_vals, x=y, axis=0)

        dphi_dx_vals = phi_vals * (-diff / (2.0 * nu * ti))
        dphi_dx = simpson(dphi_dx_vals, x=y, axis=0)

        safe = np.abs(phi) > 1e-30
        u[i, :] = np.where(safe, -2.0 * nu * dphi_dx / np.where(safe, phi, 1.0), 0.0)

    return u


def evaluate_reference_solution(
    x: np.ndarray,
    t: np.ndarray,
    viscosity: float,
    n_terms: int = 100,
) -> np.ndarray:
    """Evaluate the analytical Burgers solution on a custom mesh."""
    return _cole_hopf_solution(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64), viscosity, n_terms)


def generate_reference_solution(pde: PDEConfig, data: DataConfig) -> ReferenceSolution:
    """Build the analytical Burgers solution on a regular (nx × nt) grid."""
    x = np.linspace(pde.x_min, pde.x_max, data.nx)
    t = np.linspace(pde.t_min, pde.t_max, data.nt)

    u = evaluate_reference_solution(x, t, pde.viscosity, n_terms=150)

    return ReferenceSolution(
        x=x.tolist(),
        t=t.tolist(),
        u=u.tolist(),
        viscosity=pde.viscosity,
        nx=data.nx,
        nt=data.nt,
    )


def sample_observations(
    ref: ReferenceSolution,
    n_points: int,
    noise_std: float,
    seed: int,
) -> ObservationSet:
    """Randomly sample *n_points* from the reference solution, optionally with noise."""
    rng = np.random.default_rng(seed)
    x_arr = np.array(ref.x)
    t_arr = np.array(ref.t)
    u_arr = ref.u_array()  # (nt, nx)

    ti = rng.integers(0, ref.nt, size=n_points)
    xi = rng.integers(0, ref.nx, size=n_points)

    x_obs = x_arr[xi]
    t_obs = t_arr[ti]
    u_obs = u_arr[ti, xi]

    if noise_std > 0.0:
        u_obs = u_obs + rng.normal(0.0, noise_std, size=n_points)

    return ObservationSet(
        x=x_obs.tolist(),
        t=t_obs.tolist(),
        u=u_obs.tolist(),
        n_points=n_points,
        noise_std=noise_std,
    )


# ---------------------------------------------------------------------------
# Collocation point samplers (Latin Hypercube)
# ---------------------------------------------------------------------------

def sample_interior_collocation(
    pde: PDEConfig,
    n: int,
    seed: int,
) -> np.ndarray:
    """Return (n, 2) array of [x, t] interior collocation points via LHS."""
    return latin_hypercube_sample(
        n,
        bounds=[(pde.x_min, pde.x_max), (pde.t_min, pde.t_max)],
        seed=seed,
    ).astype(np.float32)


def sample_boundary_collocation(
    pde: PDEConfig,
    n_per_edge: int,
    seed: int,
) -> np.ndarray:
    """Return (2*n_per_edge, 2) array for x = x_min and x = x_max boundaries."""
    rng = np.random.default_rng(seed)
    t_left = rng.uniform(pde.t_min, pde.t_max, size=n_per_edge)
    t_right = rng.uniform(pde.t_min, pde.t_max, size=n_per_edge)
    left = np.column_stack([np.full(n_per_edge, pde.x_min), t_left])
    right = np.column_stack([np.full(n_per_edge, pde.x_max), t_right])
    return np.vstack([left, right]).astype(np.float32)


def sample_initial_collocation(
    pde: PDEConfig,
    n: int,
    seed: int,
) -> np.ndarray:
    """Return (n, 2) array for the initial condition t = t_min."""
    rng = np.random.default_rng(seed)
    x_vals = rng.uniform(pde.x_min, pde.x_max, size=n)
    return np.column_stack([x_vals, np.full(n, pde.t_min)]).astype(np.float32)
