"""Pydantic schemas for data, training logs, metrics, and experiment summary."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt, model_validator


# ---------------------------------------------------------------------------
# Reference solution
# ---------------------------------------------------------------------------

class ReferenceSolution(BaseModel):
    """High-fidelity analytical solution of the Burgers equation on a regular grid."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    x: list[float] = Field(description="Spatial coordinates (length nx).")
    t: list[float] = Field(description="Temporal coordinates (length nt).")
    u: list[list[float]] = Field(description="Solution matrix u(t_i, x_j), shape (nt, nx).")
    viscosity: PositiveFloat
    nx: PositiveInt
    nt: PositiveInt

    @model_validator(mode="after")
    def _validate_shapes(self) -> "ReferenceSolution":
        if len(self.x) != self.nx:
            raise ValueError(f"x length {len(self.x)} != nx {self.nx}")
        if len(self.t) != self.nt:
            raise ValueError(f"t length {len(self.t)} != nt {self.nt}")
        if len(self.u) != self.nt:
            raise ValueError(f"u has {len(self.u)} rows but expected nt={self.nt}")
        for i, row in enumerate(self.u):
            if len(row) != self.nx:
                raise ValueError(f"u[{i}] has {len(row)} cols but expected nx={self.nx}")
        return self

    def u_array(self) -> np.ndarray:
        """Return the solution as a NumPy array of shape (nt, nx)."""
        return np.array(self.u, dtype=np.float64)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, T) meshgrids each of shape (nt, nx)."""
        return np.meshgrid(np.array(self.x), np.array(self.t))


# ---------------------------------------------------------------------------
# Observations (sparse noisy measurements)
# ---------------------------------------------------------------------------

class ObservationSet(BaseModel):
    """Sparse observations sampled from the reference solution."""

    model_config = ConfigDict(validate_assignment=True)

    x: list[float]
    t: list[float]
    u: list[float]
    n_points: PositiveInt
    noise_std: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "ObservationSet":
        if not (len(self.x) == len(self.t) == len(self.u) == self.n_points):
            raise ValueError("x, t, u must all have length n_points.")
        return self


# ---------------------------------------------------------------------------
# Training log
# ---------------------------------------------------------------------------

class TrainingLogEntry(BaseModel):
    phase: Literal["adam", "lbfgs"]
    step: int
    total_loss: NonNegativeFloat
    pde_loss: NonNegativeFloat
    boundary_loss: NonNegativeFloat
    initial_loss: NonNegativeFloat
    data_loss: NonNegativeFloat


class TrainingHistory(BaseModel):
    entries: list[TrainingLogEntry] = Field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([entry.model_dump() for entry in self.entries])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class ErrorMetrics(BaseModel):
    """Error metrics comparing PINN prediction against the analytical reference."""

    l2_relative_error: NonNegativeFloat = Field(description="||u_pred - u_ref||_2 / ||u_ref||_2")
    mse: NonNegativeFloat
    mae: NonNegativeFloat
    max_absolute_error: NonNegativeFloat


# ---------------------------------------------------------------------------
# Experiment summary
# ---------------------------------------------------------------------------

class ExperimentSummary(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    pde: str = "Burgers"
    viscosity: PositiveFloat
    domain_x: tuple[float, float]
    domain_t: tuple[float, float]
    reference_grid: tuple[int, int] = Field(description="(nx, nt) of the reference solution.")
    n_observations: PositiveInt
    architecture: str
    trainable_parameters: PositiveInt
    device: str
    adam_epochs: PositiveInt
    lbfgs_iterations: int
    metrics: ErrorMetrics
    final_losses: dict[str, float]
    artifact_paths: dict[str, str] = Field(default_factory=dict)
