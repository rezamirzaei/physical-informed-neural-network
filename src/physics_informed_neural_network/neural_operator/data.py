"""Data generation and exact solvers for the neural-operator tutorial problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .config import NeuralOperatorExperimentConfig, OperatorProblemConfig


@dataclass(slots=True)
class FieldDraw:
    """Latent random coefficients used to evaluate fields on any grid resolution."""

    diffusion_sin: np.ndarray
    diffusion_cos: np.ndarray
    forcing_sin: np.ndarray
    forcing_cos: np.ndarray
    diffusion_offset: float
    forcing_offset: float


@dataclass(slots=True)
class OperatorSample:
    """Single function-space input/output pair."""

    grid: np.ndarray
    diffusion: np.ndarray
    forcing: np.ndarray
    solution: np.ndarray


@dataclass(slots=True)
class OperatorDataset:
    """Dataset evaluated on a fixed spatial grid."""

    name: str
    grid: np.ndarray
    diffusion: np.ndarray
    forcing: np.ndarray
    solution: np.ndarray
    draws: tuple[FieldDraw, ...]

    def __post_init__(self) -> None:
        n_samples = len(self.draws)
        resolution = len(self.grid)
        expected = (n_samples, resolution)
        if self.diffusion.shape != expected:
            raise ValueError(f"diffusion shape {self.diffusion.shape} does not match expected {expected}")
        if self.forcing.shape != expected:
            raise ValueError(f"forcing shape {self.forcing.shape} does not match expected {expected}")
        if self.solution.shape != expected:
            raise ValueError(f"solution shape {self.solution.shape} does not match expected {expected}")

    @property
    def n_samples(self) -> int:
        return self.diffusion.shape[0]

    @property
    def resolution(self) -> int:
        return self.diffusion.shape[1]

    def features(self) -> np.ndarray:
        """Return the neural-operator input tensor with channels [a, f, x]."""
        coordinates = np.broadcast_to(self.grid[None, :, None], (self.n_samples, self.resolution, 1))
        return np.concatenate(
            [self.diffusion[..., None], self.forcing[..., None], coordinates],
            axis=-1,
        ).astype(np.float32)

    def targets(self) -> np.ndarray:
        """Return solution tensors with explicit channel dimension."""
        return self.solution[..., None].astype(np.float32)

    def sample(self, index: int) -> OperatorSample:
        return OperatorSample(
            grid=self.grid.copy(),
            diffusion=self.diffusion[index].copy(),
            forcing=self.forcing[index].copy(),
            solution=self.solution[index].copy(),
        )


@dataclass(slots=True)
class OperatorDatasetSplits:
    """Canonical train/validation/test splits plus resolution-transfer test data."""

    train: OperatorDataset
    validation: OperatorDataset
    test: OperatorDataset
    refined_test: OperatorDataset


def create_uniform_grid(n_points: int, domain_min: float = 0.0, domain_max: float = 1.0) -> np.ndarray:
    """Return a uniform 1-D grid including both endpoints."""
    return np.linspace(domain_min, domain_max, n_points, dtype=np.float64)


def sample_field_draws(
    n_samples: int,
    problem: OperatorProblemConfig,
    seed: int,
) -> tuple[FieldDraw, ...]:
    """Sample latent Fourier-series coefficients for diffusivity and forcing families."""
    rng = np.random.default_rng(seed)

    diffusion_scale = problem.diffusion_amplitude / np.power(
        np.arange(1, problem.diffusion_modes + 1, dtype=np.float64),
        problem.spectral_decay,
    )
    forcing_scale = problem.forcing_amplitude / np.power(
        np.arange(1, problem.forcing_modes + 1, dtype=np.float64),
        problem.spectral_decay,
    )

    draws: list[FieldDraw] = []
    for _ in range(n_samples):
        draws.append(
            FieldDraw(
                diffusion_sin=rng.normal(scale=diffusion_scale).astype(np.float64),
                diffusion_cos=rng.normal(scale=diffusion_scale).astype(np.float64),
                forcing_sin=rng.normal(scale=forcing_scale).astype(np.float64),
                forcing_cos=rng.normal(scale=forcing_scale).astype(np.float64),
                diffusion_offset=float(problem.diffusion_bias + rng.normal(scale=0.15 * problem.diffusion_amplitude)),
                forcing_offset=float(rng.normal(scale=0.15 * problem.forcing_amplitude)),
            )
        )
    return tuple(draws)


def solve_dirichlet_diffusion_1d(
    grid: np.ndarray,
    diffusion: np.ndarray,
    forcing: np.ndarray,
) -> np.ndarray:
    """Solve ``-(a(x) u'(x))' = f(x)`` with homogeneous Dirichlet boundary conditions.

    The solution is obtained from the exact 1-D flux form:

    ``a(x) u'(x) = C - ∫_0^x f(s) ds``

    with ``C`` chosen so that ``u(1) = 0``.
    """
    forcing_primitive = cumulative_trapezoid(forcing, grid, initial=0.0)
    inv_diffusion = 1.0 / diffusion

    constant = float(np.trapz(forcing_primitive * inv_diffusion, grid) / np.trapz(inv_diffusion, grid))
    integrand = (constant - forcing_primitive) * inv_diffusion
    solution = cumulative_trapezoid(integrand, grid, initial=0.0)

    solution[0] = 0.0
    solution[-1] = 0.0
    return solution.astype(np.float64)


def compute_discrete_diffusion_residual(
    grid: np.ndarray,
    diffusion: np.ndarray,
    solution: np.ndarray,
    forcing: np.ndarray,
) -> np.ndarray:
    """Approximate the PDE residual on interior cells using a finite-volume stencil."""
    spacing = np.diff(grid)
    solution_gradient = np.diff(solution) / spacing
    diffusion_mid = 0.5 * (diffusion[:-1] + diffusion[1:])
    flux = diffusion_mid * solution_gradient
    cell_width = 0.5 * (spacing[:-1] + spacing[1:])
    return -(flux[1:] - flux[:-1]) / cell_width - forcing[1:-1]


def evaluate_field_draw(
    draw: FieldDraw,
    grid: np.ndarray,
    problem: OperatorProblemConfig,
) -> OperatorSample:
    """Evaluate a latent draw onto a concrete grid and solve the corresponding PDE."""
    x_unit = (grid - problem.domain_min) / (problem.domain_max - problem.domain_min)

    diffusion_idx = np.arange(1, problem.diffusion_modes + 1, dtype=np.float64)[:, None]
    diffusion_angles = 2.0 * np.pi * diffusion_idx * x_unit[None, :]
    log_diffusion = (
        draw.diffusion_offset
        + (draw.diffusion_sin[:, None] * np.sin(diffusion_angles)).sum(axis=0)
        + (draw.diffusion_cos[:, None] * np.cos(diffusion_angles)).sum(axis=0)
    )
    diffusion = problem.minimum_diffusion + np.exp(log_diffusion)

    forcing_idx = np.arange(1, problem.forcing_modes + 1, dtype=np.float64)[:, None]
    forcing_angles = 2.0 * np.pi * forcing_idx * x_unit[None, :]
    forcing = (
        draw.forcing_offset
        + (draw.forcing_sin[:, None] * np.sin(forcing_angles)).sum(axis=0)
        + (draw.forcing_cos[:, None] * np.cos(forcing_angles)).sum(axis=0)
    )

    solution = solve_dirichlet_diffusion_1d(grid, diffusion, forcing)
    return OperatorSample(grid=grid.copy(), diffusion=diffusion, forcing=forcing, solution=solution)


def build_operator_dataset(
    name: str,
    draws: tuple[FieldDraw, ...],
    resolution: int,
    problem: OperatorProblemConfig,
) -> OperatorDataset:
    """Evaluate a list of latent draws on a fixed-resolution grid."""
    grid = create_uniform_grid(resolution, problem.domain_min, problem.domain_max)
    samples = [evaluate_field_draw(draw, grid, problem) for draw in draws]

    return OperatorDataset(
        name=name,
        grid=grid,
        diffusion=np.stack([sample.diffusion for sample in samples], axis=0),
        forcing=np.stack([sample.forcing for sample in samples], axis=0),
        solution=np.stack([sample.solution for sample in samples], axis=0),
        draws=draws,
    )


def build_dataset_splits(config: NeuralOperatorExperimentConfig) -> OperatorDatasetSplits:
    """Create canonical dataset splits for native-resolution and transfer-resolution evaluation."""
    train_draws = sample_field_draws(config.data.train_samples, config.problem, seed=config.data.seed)
    validation_draws = sample_field_draws(config.data.validation_samples, config.problem, seed=config.data.seed + 1)
    test_draws = sample_field_draws(config.data.test_samples, config.problem, seed=config.data.seed + 2)

    return OperatorDatasetSplits(
        train=build_operator_dataset("train", train_draws, config.data.train_resolution, config.problem),
        validation=build_operator_dataset("validation", validation_draws, config.data.train_resolution, config.problem),
        test=build_operator_dataset("test", test_draws, config.data.train_resolution, config.problem),
        refined_test=build_operator_dataset(
            "refined_test",
            test_draws,
            config.data.evaluation_resolution,
            config.problem,
        ),
    )
