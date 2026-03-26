"""Pydantic configuration for the Kolmogorov-Arnold Network tutorial stack."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from ..config import PDEConfig


class KANDataConfig(BaseModel):
    """Resolution settings for coarse-grid training and dense-grid evaluation."""

    model_config = ConfigDict(validate_assignment=True)

    train_nx: PositiveInt = 33
    train_nt: PositiveInt = 25
    validation_nx: PositiveInt = 49
    validation_nt: PositiveInt = 33
    test_nx: PositiveInt = 65
    test_nt: PositiveInt = 41
    evaluation_nx: PositiveInt = 129
    evaluation_nt: PositiveInt = 81
    seed: PositiveInt = 21


class PiecewiseLinearKANConfig(BaseModel):
    """Architecture hyper-parameters for a spline-edge KAN."""

    model_config = ConfigDict(validate_assignment=True)

    input_dim: PositiveInt = 2
    hidden_widths: tuple[int, ...] = Field(default=(24, 24))
    num_knots: PositiveInt = 17
    spline_domain_min: float = -1.0
    spline_domain_max: float = 1.0
    base_activation: Literal["identity", "silu", "tanh", "relu"] = "silu"
    hidden_input_activation: Literal["identity", "tanh"] = "tanh"
    use_bias: bool = True
    spline_scale_init: PositiveFloat = 1.0

    @model_validator(mode="after")
    def _validate_shape(self) -> "PiecewiseLinearKANConfig":
        if self.input_dim != 2:
            raise ValueError("This tutorial implementation expects input_dim=2 for (x, t) coordinates.")
        if not self.hidden_widths:
            raise ValueError("hidden_widths must contain at least one hidden layer width.")
        if any(width <= 0 for width in self.hidden_widths):
            raise ValueError("hidden_widths must contain only positive integers.")
        if self.num_knots < 2:
            raise ValueError("num_knots must be at least 2.")
        if self.spline_domain_max <= self.spline_domain_min:
            raise ValueError("spline_domain_max must exceed spline_domain_min.")
        return self


class KANOptimizationConfig(BaseModel):
    """Training-loop hyper-parameters."""

    model_config = ConfigDict(validate_assignment=True)

    seed: PositiveInt = 7
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    batch_size: PositiveInt = 256
    epochs: PositiveInt = 300
    learning_rate: PositiveFloat = 2e-3
    weight_decay: float = Field(default=1e-6, ge=0.0)
    scheduler_step: PositiveInt = 100
    scheduler_gamma: float = Field(default=0.6, gt=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    log_every: PositiveInt = 25
    patience: int = Field(default=0, ge=0, description="Early-stopping patience (0 = disabled).")
    min_delta: float = Field(default=1e-6, ge=0.0, description="Minimum improvement to reset patience counter.")


class KANArtifactConfig(BaseModel):
    """Artifact settings for notebook-friendly experiment runs."""

    model_config = ConfigDict(validate_assignment=True)

    output_dir: Path = Path("artifacts/kan")
    save_artifacts: bool = False


class KANExperimentConfig(BaseModel):
    """Top-level KAN experiment configuration."""

    model_config = ConfigDict(validate_assignment=True)

    pde: PDEConfig = Field(default_factory=PDEConfig)
    data: KANDataConfig = Field(default_factory=KANDataConfig)
    model: PiecewiseLinearKANConfig = Field(default_factory=PiecewiseLinearKANConfig)
    optimization: KANOptimizationConfig = Field(default_factory=KANOptimizationConfig)
    artifacts: KANArtifactConfig = Field(default_factory=KANArtifactConfig)

    @model_validator(mode="after")
    def _validate_grids(self) -> "KANExperimentConfig":
        if self.data.train_nx > self.data.evaluation_nx or self.data.train_nt > self.data.evaluation_nt:
            raise ValueError("evaluation grid should not be coarser than the training grid in this tutorial setup.")
        return self
