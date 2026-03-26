"""Data generation for the 2-D Darcy-flow benchmark.

Implements:
- Gaussian Random Field (GRF) sampling via the spectral method
- Five-point finite-difference Darcy solver with Dirichlet BCs
- Dataset classes for 2-D operator learning with resolution sub-sampling

The GRF uses the covariance operator  C = τ^{2α} (-Δ + τ²I)^{-α}
on the unit square with periodic extension, sampled via FFT.

Reference: Li et al. (2021) *Fourier Neural Operator for Parametric PDEs*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

from .config import DarcyDatasetConfig, DarcyExperimentConfig, DarcyProblemConfig


# ---------------------------------------------------------------------------
# GRF sampling
# ---------------------------------------------------------------------------

def _grf_sample_2d(
    n: int,
    alpha: float,
    tau: float,
    rng: np.random.Generator,
    n_samples: int = 1,
) -> np.ndarray:
    """Sample from a Gaussian random field on an ``n × n`` periodic grid.

    Returns an array of shape ``(n_samples, n, n)``.
    """
    freq_x = np.fft.fftfreq(n, d=1.0 / n)
    freq_y = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(freq_x, freq_y, indexing="ij")
    # Eigenvalues of -Δ + τ²I on periodic domain
    eigenvalues = (2.0 * np.pi) ** 2 * (kx ** 2 + ky ** 2) + tau ** 2
    # Covariance spectrum: τ^{2α} * λ^{-α}
    amplitude = tau ** (2.0 * alpha) * eigenvalues ** (-alpha)
    amplitude = np.sqrt(amplitude)

    samples = np.empty((n_samples, n, n), dtype=np.float64)
    for i in range(n_samples):
        noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        field_ft = amplitude * noise
        field = np.real(np.fft.ifft2(field_ft)) * n  # normalisation
        samples[i] = field

    return samples


def sample_grf_diffusivity(
    n_samples: int,
    resolution: int,
    problem: DarcyProblemConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate diffusivity fields on an ``resolution × resolution`` grid.

    If ``use_piecewise_constant`` is True, threshold the GRF to produce
    a binary field with values ``a_lo`` and ``a_hi`` (matching the FNO paper setup).
    Otherwise, use ``minimum_diffusion + exp(z)`` to ensure positivity.

    Returns shape ``(n_samples, resolution, resolution)``.
    """
    raw = _grf_sample_2d(resolution, problem.grf_alpha, problem.grf_tau, rng, n_samples)
    if problem.use_piecewise_constant:
        return np.where(raw >= 0.0, problem.a_hi, problem.a_lo)
    else:
        return problem.minimum_diffusion + np.exp(raw)


# ---------------------------------------------------------------------------
# 2-D Darcy solver: -∇·(a ∇u) = f, u|∂Ω = 0
# ---------------------------------------------------------------------------

def solve_darcy_2d(
    diffusivity: np.ndarray,
    forcing: np.ndarray,
    h: float,
) -> np.ndarray:
    """Solve the 2-D Darcy equation on an interior grid using a 5-point FD stencil.

    Parameters
    ----------
    diffusivity : (N, N) coefficient field (including boundary)
    forcing     : (N, N) right-hand side
    h           : grid spacing

    Returns
    -------
    solution    : (N, N) with boundary values set to zero
    """
    N = diffusivity.shape[0]
    n = N - 2  # interior points
    if n < 1:
        return np.zeros_like(forcing)

    # Build the interior system: -∇·(a ∇u) = f
    # Use harmonic average of diffusivity at cell faces for better accuracy
    a = diffusivity
    rhs = forcing[1:-1, 1:-1].ravel()

    # Coefficient arrays for the 5-point stencil on interior (n x n) grid
    # a_e, a_w, a_n, a_s are face-averaged diffusivities
    a_e = 0.5 * (a[1:-1, 1:-1] + a[1:-1, 2:])
    a_w = 0.5 * (a[1:-1, 1:-1] + a[1:-1, :-2])
    a_n = 0.5 * (a[1:-1, 1:-1] + a[2:, 1:-1])
    a_s = 0.5 * (a[1:-1, 1:-1] + a[:-2, 1:-1])

    center = (a_e + a_w + a_n + a_s).ravel() / h ** 2
    east = (-a_e[:, :-1]).ravel() / h ** 2
    west = (-a_w[:, 1:]).ravel() / h ** 2
    north = (-a_n[:-1, :]).ravel() / h ** 2
    south = (-a_s[1:, :]).ravel() / h ** 2

    nn = n * n
    diag_data = [center]
    offsets = [0]

    # East/west neighbours: offset ±1 with row-boundary zeros
    if len(east) > 0:
        diag_data.append(east)
        offsets.append(1)
    if len(west) > 0:
        diag_data.append(west)
        offsets.append(-1)
    # North/south neighbours: offset ±n
    if len(north) > 0:
        diag_data.append(north)
        offsets.append(n)
    if len(south) > 0:
        diag_data.append(south)
        offsets.append(-n)

    # Build the sparse matrix
    from scipy.sparse import lil_matrix

    A = lil_matrix((nn, nn), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = (a_e[i, j] + a_w[i, j] + a_n[i, j] + a_s[i, j]) / h ** 2
            if j + 1 < n:
                A[idx, idx + 1] = -a_e[i, j] / h ** 2
            if j - 1 >= 0:
                A[idx, idx - 1] = -a_w[i, j] / h ** 2
            if i + 1 < n:
                A[idx, idx + n] = -a_n[i, j] / h ** 2
            if i - 1 >= 0:
                A[idx, idx - n] = -a_s[i, j] / h ** 2

    u_interior = spsolve(A.tocsc(), rhs)
    solution = np.zeros((N, N), dtype=np.float64)
    solution[1:-1, 1:-1] = u_interior.reshape(n, n)
    return solution


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DarcySample:
    """Single 2-D function-space input/output pair."""
    grid_x: np.ndarray   # (N,)
    grid_y: np.ndarray   # (N,)
    diffusivity: np.ndarray  # (N, N)
    forcing: np.ndarray      # (N, N)
    solution: np.ndarray     # (N, N)


@dataclass(slots=True)
class DarcyDataset:
    """Dataset of 2-D Darcy samples on a fixed spatial grid."""
    name: str
    grid_x: np.ndarray    # (N,)
    grid_y: np.ndarray    # (N,)
    diffusivity: np.ndarray  # (n_samples, N, N)
    forcing: np.ndarray      # (n_samples, N, N)
    solution: np.ndarray     # (n_samples, N, N)

    @property
    def n_samples(self) -> int:
        return self.diffusivity.shape[0]

    @property
    def resolution(self) -> int:
        return len(self.grid_x)

    def features(self) -> np.ndarray:
        """Return input tensor with shape ``(n_samples, N, N, 4)`` — channels ``[a, f, x, y]``."""
        N = self.resolution
        xx, yy = np.meshgrid(self.grid_x, self.grid_y, indexing="ij")
        coords_x = np.broadcast_to(xx[None, :, :, None], (self.n_samples, N, N, 1))
        coords_y = np.broadcast_to(yy[None, :, :, None], (self.n_samples, N, N, 1))
        return np.concatenate(
            [
                self.diffusivity[..., None],
                self.forcing[..., None],
                coords_x,
                coords_y,
            ],
            axis=-1,
        ).astype(np.float32)

    def targets(self) -> np.ndarray:
        """Return solution tensors ``(n_samples, N, N, 1)``."""
        return self.solution[..., None].astype(np.float32)

    def sample(self, index: int) -> DarcySample:
        return DarcySample(
            grid_x=self.grid_x.copy(),
            grid_y=self.grid_y.copy(),
            diffusivity=self.diffusivity[index].copy(),
            forcing=self.forcing[index].copy(),
            solution=self.solution[index].copy(),
        )


@dataclass(slots=True)
class DarcyDatasetSplits:
    """Canonical splits for 2-D Darcy experiments."""
    train: DarcyDataset
    validation: DarcyDataset
    test: DarcyDataset
    refined_test: DarcyDataset


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _subsample_field(field: np.ndarray, target_res: int) -> np.ndarray:
    """Sub-sample a ``(... , N, N)`` field to ``(... , target_res, target_res)`` via stride."""
    current_res = field.shape[-1]
    if target_res >= current_res:
        return field
    step = (current_res - 1) / (target_res - 1)
    indices = np.round(np.linspace(0, current_res - 1, target_res)).astype(int)
    # Handle arbitrary leading dims
    return field[..., indices[:, None], indices[None, :]]


def _generate_darcy_data(
    n_samples: int,
    fine_resolution: int,
    problem: DarcyProblemConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (diffusivity, forcing, solution) on a fine grid.

    Returns arrays of shape ``(n_samples, fine_resolution, fine_resolution)``.
    """
    h = (problem.domain_max - problem.domain_min) / (fine_resolution - 1)

    diffusivity = sample_grf_diffusivity(n_samples, fine_resolution, problem, rng)
    forcing = np.full((n_samples, fine_resolution, fine_resolution), problem.forcing_constant, dtype=np.float64)

    solution = np.empty_like(diffusivity)
    for i in range(n_samples):
        solution[i] = solve_darcy_2d(diffusivity[i], forcing[i], h)

    return diffusivity, forcing, solution


def build_darcy_dataset(
    name: str,
    fine_diffusivity: np.ndarray,
    fine_forcing: np.ndarray,
    fine_solution: np.ndarray,
    fine_resolution: int,
    target_resolution: int,
    problem: DarcyProblemConfig,
) -> DarcyDataset:
    """Sub-sample fine-grid data to the target resolution and wrap in a DarcyDataset."""
    grid_1d = np.linspace(problem.domain_min, problem.domain_max, target_resolution)
    return DarcyDataset(
        name=name,
        grid_x=grid_1d,
        grid_y=grid_1d,
        diffusivity=_subsample_field(fine_diffusivity, target_resolution),
        forcing=_subsample_field(fine_forcing, target_resolution),
        solution=_subsample_field(fine_solution, target_resolution),
    )


def build_darcy_splits(config: DarcyExperimentConfig) -> DarcyDatasetSplits:
    """Create train/validation/test/refined_test splits for the Darcy-flow benchmark."""
    rng_train = np.random.default_rng(config.data.seed)
    rng_val = np.random.default_rng(config.data.seed + 1)
    rng_test = np.random.default_rng(config.data.seed + 2)

    fine_res = config.data.grf_resolution

    train_a, train_f, train_u = _generate_darcy_data(
        config.data.train_samples, fine_res, config.problem, rng_train
    )
    val_a, val_f, val_u = _generate_darcy_data(
        config.data.validation_samples, fine_res, config.problem, rng_val
    )
    test_a, test_f, test_u = _generate_darcy_data(
        config.data.test_samples, fine_res, config.problem, rng_test
    )

    return DarcyDatasetSplits(
        train=build_darcy_dataset(
            "train", train_a, train_f, train_u, fine_res,
            config.data.train_resolution, config.problem,
        ),
        validation=build_darcy_dataset(
            "validation", val_a, val_f, val_u, fine_res,
            config.data.train_resolution, config.problem,
        ),
        test=build_darcy_dataset(
            "test", test_a, test_f, test_u, fine_res,
            config.data.train_resolution, config.problem,
        ),
        refined_test=build_darcy_dataset(
            "refined_test", test_a, test_f, test_u, fine_res,
            config.data.evaluation_resolution, config.problem,
        ),
    )


