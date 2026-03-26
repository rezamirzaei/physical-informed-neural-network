"""Exact data generation utilities for KAN regression on the Burgers solution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import PDEConfig
from ..data import evaluate_reference_solution
from .config import KANExperimentConfig


@dataclass(slots=True)
class CoordinateNormalizer:
    """Affine map between physical coordinates and the KAN's normalized input domain."""

    minimum: np.ndarray
    maximum: np.ndarray
    target_min: float = -1.0
    target_max: float = 1.0

    @classmethod
    def from_pde(cls, pde: PDEConfig) -> "CoordinateNormalizer":
        return cls(
            minimum=np.array([pde.x_min, pde.t_min], dtype=np.float64),
            maximum=np.array([pde.x_max, pde.t_max], dtype=np.float64),
        )

    def transform_numpy(self, coordinates: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        scale = (self.target_max - self.target_min) / (self.maximum - self.minimum)
        return self.target_min + (coordinates - self.minimum) * scale

    def transform_tensor(self, coordinates: torch.Tensor) -> torch.Tensor:
        minimum = torch.as_tensor(self.minimum, dtype=coordinates.dtype, device=coordinates.device)
        maximum = torch.as_tensor(self.maximum, dtype=coordinates.dtype, device=coordinates.device)
        scale = (self.target_max - self.target_min) / (maximum - minimum)
        return self.target_min + (coordinates - minimum) * scale


@dataclass(slots=True)
class BurgersGridDataset:
    """One dataset split represented both as a structured grid and as point samples."""

    name: str
    x: np.ndarray
    t: np.ndarray
    solution: np.ndarray
    normalizer: CoordinateNormalizer

    def __post_init__(self) -> None:
        expected = (len(self.t), len(self.x))
        if self.solution.shape != expected:
            raise ValueError(f"solution shape {self.solution.shape} does not match expected {expected}")

    @property
    def nx(self) -> int:
        return int(len(self.x))

    @property
    def nt(self) -> int:
        return int(len(self.t))

    @property
    def n_points(self) -> int:
        return int(self.nx * self.nt)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.t)

    def coordinates(self) -> np.ndarray:
        grid_x, grid_t = self.mesh()
        return np.column_stack([grid_x.ravel(), grid_t.ravel()]).astype(np.float32)

    def normalized_coordinates(self) -> np.ndarray:
        return self.normalizer.transform_numpy(self.coordinates()).astype(np.float32)

    def targets(self) -> np.ndarray:
        return self.solution.reshape(-1, 1).astype(np.float32)

    def reshape_prediction(self, prediction: np.ndarray) -> np.ndarray:
        return np.asarray(prediction, dtype=np.float64).reshape(self.nt, self.nx)


@dataclass(slots=True)
class KANDatasetSplits:
    """Canonical train/validation/test/evaluation datasets for KAN experiments."""

    train: BurgersGridDataset
    validation: BurgersGridDataset
    test: BurgersGridDataset
    evaluation: BurgersGridDataset


def _build_axis(lower: float, upper: float, n_points: int) -> np.ndarray:
    return np.linspace(lower, upper, n_points, dtype=np.float64)


def build_burgers_grid_dataset(
    name: str,
    x: np.ndarray,
    t: np.ndarray,
    viscosity: float,
    normalizer: CoordinateNormalizer,
) -> BurgersGridDataset:
    """Evaluate the exact Burgers solution on a rectangular grid."""
    solution = evaluate_reference_solution(x, t, viscosity)
    return BurgersGridDataset(name=name, x=x, t=t, solution=solution, normalizer=normalizer)


def build_dataset_splits(config: KANExperimentConfig) -> KANDatasetSplits:
    """Create all dataset splits used by the KAN tutorial stack."""
    normalizer = CoordinateNormalizer.from_pde(config.pde)

    return KANDatasetSplits(
        train=build_burgers_grid_dataset(
            "train",
            _build_axis(config.pde.x_min, config.pde.x_max, config.data.train_nx),
            _build_axis(config.pde.t_min, config.pde.t_max, config.data.train_nt),
            config.pde.viscosity,
            normalizer,
        ),
        validation=build_burgers_grid_dataset(
            "validation",
            _build_axis(config.pde.x_min, config.pde.x_max, config.data.validation_nx),
            _build_axis(config.pde.t_min, config.pde.t_max, config.data.validation_nt),
            config.pde.viscosity,
            normalizer,
        ),
        test=build_burgers_grid_dataset(
            "test",
            _build_axis(config.pde.x_min, config.pde.x_max, config.data.test_nx),
            _build_axis(config.pde.t_min, config.pde.t_max, config.data.test_nt),
            config.pde.viscosity,
            normalizer,
        ),
        evaluation=build_burgers_grid_dataset(
            "evaluation",
            _build_axis(config.pde.x_min, config.pde.x_max, config.data.evaluation_nx),
            _build_axis(config.pde.t_min, config.pde.t_max, config.data.evaluation_nt),
            config.pde.viscosity,
            normalizer,
        ),
    )
