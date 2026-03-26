"""Pydantic configuration for the neural-operator tutorial stack."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class OperatorProblemConfig(BaseModel):
    """Defines the parametric 1-D diffusion problem used in the tutorial."""

    model_config = ConfigDict(validate_assignment=True)

    domain_min: float = 0.0
    domain_max: float = 1.0

    diffusion_modes: PositiveInt = 8
    forcing_modes: PositiveInt = 8

    diffusion_bias: float = -0.25
    diffusion_amplitude: PositiveFloat = 0.45
    forcing_amplitude: PositiveFloat = 1.0
    spectral_decay: PositiveFloat = 2.0
    minimum_diffusion: PositiveFloat = 0.15

    @model_validator(mode="after")
    def _validate_domain(self) -> "OperatorProblemConfig":
        if self.domain_max <= self.domain_min:
            raise ValueError("domain_max must exceed domain_min.")
        return self


class OperatorDatasetConfig(BaseModel):
    """Sampling configuration for train/validation/test datasets."""

    model_config = ConfigDict(validate_assignment=True)

    train_samples: PositiveInt = 256
    validation_samples: PositiveInt = 64
    test_samples: PositiveInt = 64

    train_resolution: PositiveInt = 64
    evaluation_resolution: PositiveInt = 128
    seed: PositiveInt = 17


class FourierNeuralOperatorConfig(BaseModel):
    """Architecture hyper-parameters for a 1-D Fourier neural operator."""

    model_config = ConfigDict(validate_assignment=True)

    input_channels: PositiveInt = 3
    width: PositiveInt = 48
    modes: PositiveInt = 16
    layers: PositiveInt = 4
    padding: int = Field(default=8, ge=0)
    activation: Literal["gelu", "relu", "tanh"] = "gelu"


class OperatorOptimizationConfig(BaseModel):
    """Training-loop hyper-parameters."""

    model_config = ConfigDict(validate_assignment=True)

    seed: PositiveInt = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    batch_size: PositiveInt = 32
    epochs: PositiveInt = 120
    learning_rate: PositiveFloat = 3e-3
    weight_decay: float = Field(default=1e-5, ge=0.0)
    scheduler_step: PositiveInt = 40
    scheduler_gamma: float = Field(default=0.7, gt=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    log_every: PositiveInt = 10


class FourierNeuralOperator2dConfig(BaseModel):
    """Architecture hyper-parameters for a 2-D Fourier neural operator (Darcy flow)."""

    model_config = ConfigDict(validate_assignment=True)

    input_channels: PositiveInt = 4  # [a, f, x_coord, y_coord]
    width: PositiveInt = 32
    modes_x: PositiveInt = 12
    modes_y: PositiveInt = 12
    layers: PositiveInt = 4
    padding: int = Field(default=9, ge=0)
    activation: Literal["gelu", "relu", "tanh"] = "gelu"


class DarcyProblemConfig(BaseModel):
    """Defines the 2-D Darcy flow problem: -∇·(a(x,y)∇u) = f on the unit square."""

    model_config = ConfigDict(validate_assignment=True)

    domain_min: float = 0.0
    domain_max: float = 1.0

    grf_alpha: PositiveFloat = Field(default=2.0, description="Smoothness parameter α for GRF covariance.")
    grf_tau: PositiveFloat = Field(default=3.0, description="Inverse length-scale τ for GRF covariance.")
    minimum_diffusion: PositiveFloat = Field(default=0.1, description="Floor on diffusivity.")
    use_piecewise_constant: bool = Field(
        default=True,
        description="If True, threshold the GRF to produce a piecewise-constant field a ∈ {a_lo, a_hi}.",
    )
    a_lo: PositiveFloat = Field(default=3.0, description="Low-value in piecewise-constant field.")
    a_hi: PositiveFloat = Field(default=12.0, description="High-value in piecewise-constant field.")
    forcing_constant: float = Field(default=1.0, description="Uniform forcing f(x,y) = const.")

    @model_validator(mode="after")
    def _validate_domain(self) -> "DarcyProblemConfig":
        if self.domain_max <= self.domain_min:
            raise ValueError("domain_max must exceed domain_min.")
        return self


class DarcyDatasetConfig(BaseModel):
    """Sampling configuration for Darcy flow datasets."""

    model_config = ConfigDict(validate_assignment=True)

    train_samples: PositiveInt = 1000
    validation_samples: PositiveInt = 200
    test_samples: PositiveInt = 200

    grf_resolution: PositiveInt = Field(default=85, description="Fine-grid resolution for GRF generation.")
    train_resolution: PositiveInt = Field(default=29, description="Sub-sampled resolution for training.")
    evaluation_resolution: PositiveInt = Field(default=57, description="Finer sub-sampled resolution for transfer.")
    seed: PositiveInt = 17


class OperatorArtifactConfig(BaseModel):
    """Artifact settings for notebook-oriented experiment runs."""

    model_config = ConfigDict(validate_assignment=True)

    output_dir: Path = Path("artifacts/neural_operator")
    save_artifacts: bool = False


class NeuralOperatorExperimentConfig(BaseModel):
    """Top-level configuration for neural-operator experiments."""

    model_config = ConfigDict(validate_assignment=True)

    problem: OperatorProblemConfig = Field(default_factory=OperatorProblemConfig)
    data: OperatorDatasetConfig = Field(default_factory=OperatorDatasetConfig)
    model: FourierNeuralOperatorConfig = Field(default_factory=FourierNeuralOperatorConfig)
    optimization: OperatorOptimizationConfig = Field(default_factory=OperatorOptimizationConfig)
    artifacts: OperatorArtifactConfig = Field(default_factory=OperatorArtifactConfig)

    @model_validator(mode="after")
    def _validate_resolutions(self) -> "NeuralOperatorExperimentConfig":
        max_modes = min(self.data.train_resolution, self.data.evaluation_resolution) // 2 + 1
        if self.model.modes > max_modes:
            raise ValueError(
                f"model.modes={self.model.modes} exceeds the resolvable Fourier modes for the configured grids."
            )
        return self


class DarcyExperimentConfig(BaseModel):
    """Top-level configuration for the 2-D Darcy-flow FNO experiment."""

    model_config = ConfigDict(validate_assignment=True)

    problem: DarcyProblemConfig = Field(default_factory=DarcyProblemConfig)
    data: DarcyDatasetConfig = Field(default_factory=DarcyDatasetConfig)
    model: FourierNeuralOperator2dConfig = Field(default_factory=FourierNeuralOperator2dConfig)
    optimization: OperatorOptimizationConfig = Field(default_factory=OperatorOptimizationConfig)
    artifacts: OperatorArtifactConfig = Field(default_factory=OperatorArtifactConfig)

    @model_validator(mode="after")
    def _validate_resolutions(self) -> "DarcyExperimentConfig":
        max_modes = min(self.data.train_resolution, self.data.evaluation_resolution) // 2 + 1
        if self.model.modes_x > max_modes or self.model.modes_y > max_modes:
            raise ValueError(
                f"model modes ({self.model.modes_x}, {self.model.modes_y}) exceed resolvable "
                f"Fourier modes for the configured grids (max={max_modes})."
            )
        return self


