"""Pydantic-validated configuration for the Burgers-equation PINN project."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


# ---------------------------------------------------------------------------
# PDE definition
# ---------------------------------------------------------------------------

class PDEConfig(BaseModel):
    """Physical parameters and domain bounds for the 1-D viscous Burgers equation."""

    model_config = ConfigDict(validate_assignment=True)

    viscosity: PositiveFloat = Field(
        default=0.01 / 3.14159265358979,
        description="Kinematic viscosity nu in u_t + u*u_x = nu*u_xx.",
    )
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    @model_validator(mode="after")
    def _check_bounds(self) -> "PDEConfig":
        if self.x_max <= self.x_min:
            raise ValueError("x_max must exceed x_min.")
        if self.t_max <= self.t_min:
            raise ValueError("t_max must exceed t_min.")
        return self


# ---------------------------------------------------------------------------
# Reference-solution generation
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Controls how the analytical reference solution is generated."""

    model_config = ConfigDict(validate_assignment=True)

    nx: PositiveInt = Field(default=256, description="Spatial grid points for the reference solution.")
    nt: PositiveInt = Field(default=100, description="Temporal grid points for the reference solution.")
    n_observed: PositiveInt = Field(default=2000, description="Randomly sampled observation points from the reference.")
    noise_std: float = Field(default=0.0, ge=0.0, description="Gaussian noise added to sampled observations.")
    seed: PositiveInt = 42


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class NetworkConfig(BaseModel):
    """Architecture hyper-parameters for the Burgers PINN."""

    model_config = ConfigDict(validate_assignment=True)

    hidden_dim: PositiveInt = 64
    hidden_layers: PositiveInt = 4
    fourier_features: PositiveInt = Field(default=64, description="Number of random Fourier feature pairs.")
    fourier_scale: PositiveFloat = Field(default=2.0, description="Standard deviation of the Fourier feature matrix.")
    activation: Literal["tanh", "sin", "gelu"] = "tanh"


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class LossWeights(BaseModel):
    """Fixed weights for each loss term."""

    model_config = ConfigDict(validate_assignment=True)

    pde_residual: PositiveFloat = 1.0
    boundary: PositiveFloat = 100.0
    initial_condition: PositiveFloat = 100.0
    data: PositiveFloat = 20.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    """Optimizer, scheduler, and collocation settings."""

    model_config = ConfigDict(validate_assignment=True)

    seed: PositiveInt = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    adam_epochs: PositiveInt = 15000
    adam_learning_rate: PositiveFloat = 1e-3
    lbfgs_iterations: int = Field(default=500, ge=0)

    scheduler: Literal["cosine", "step", "none"] = "cosine"
    warmup_steps: int = Field(default=500, ge=0)

    n_collocation: PositiveInt = Field(default=10_000, description="Interior collocation points.")
    n_boundary: PositiveInt = Field(default=200, description="Boundary collocation points per edge.")
    n_initial: PositiveInt = Field(default=300, description="Initial-condition collocation points.")

    log_every: PositiveInt = 500
    loss: LossWeights = Field(default_factory=LossWeights)

    adaptive_weights: bool = Field(default=False, description="Learn per-term loss weights (softmax-based).")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

class ArtifactConfig(BaseModel):
    """Where and how to save experiment outputs."""

    model_config = ConfigDict(validate_assignment=True)

    output_dir: Path = Path("artifacts/burgers_pinn")
    prediction_nx: PositiveInt = 256
    prediction_nt: PositiveInt = 200
    save_artifacts: bool = True


# ---------------------------------------------------------------------------
# Top-level project configuration
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Root configuration validated by Pydantic on construction."""

    model_config = ConfigDict(validate_assignment=True)

    pde: PDEConfig = Field(default_factory=PDEConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
