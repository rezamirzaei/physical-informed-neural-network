"""Schemas for KAN metrics, logs, and experiment summaries."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveInt


class KANErrorMetrics(BaseModel):
    """Regression metrics for scalar function approximation."""

    relative_l2: NonNegativeFloat
    mse: NonNegativeFloat
    mae: NonNegativeFloat
    max_absolute_error: NonNegativeFloat


class ResidualMetrics(BaseModel):
    """PDE residual summary for a predicted Burgers field."""

    mean_absolute_residual: NonNegativeFloat
    root_mean_square_residual: NonNegativeFloat
    max_absolute_residual: NonNegativeFloat


class KANTrainingLogEntry(BaseModel):
    """One epoch of supervised KAN optimization."""

    epoch: PositiveInt
    train_loss: NonNegativeFloat
    validation_loss: NonNegativeFloat
    validation_relative_l2: NonNegativeFloat
    learning_rate: NonNegativeFloat


class KANTrainingHistory(BaseModel):
    """Full KAN training history."""

    model_config = ConfigDict(validate_assignment=True)

    entries: list[KANTrainingLogEntry] = Field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([entry.model_dump() for entry in self.entries])


class KANExperimentSummary(BaseModel):
    """High-level summary returned by the KAN experiment pipeline."""

    model_config = ConfigDict(validate_assignment=True)

    architecture: str
    trainable_parameters: PositiveInt
    device: str
    train_points: PositiveInt
    validation_points: PositiveInt
    test_points: PositiveInt
    evaluation_points: PositiveInt
    train_grid: tuple[PositiveInt, PositiveInt]
    evaluation_grid: tuple[PositiveInt, PositiveInt]
    test_metrics: KANErrorMetrics
    evaluation_metrics: KANErrorMetrics
    residual_metrics: ResidualMetrics
