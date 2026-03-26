"""Schemas for neural-operator metrics, logs, and summaries."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveInt


class OperatorErrorMetrics(BaseModel):
    """Prediction-error metrics on a function-valued dataset."""

    relative_l2: NonNegativeFloat
    mse: NonNegativeFloat
    mae: NonNegativeFloat
    max_absolute_error: NonNegativeFloat


class ResolutionEvaluation(BaseModel):
    """Evaluation summary for a particular dataset split and resolution."""

    split: str
    resolution: PositiveInt
    metrics: OperatorErrorMetrics


class OperatorTrainingLogEntry(BaseModel):
    """One epoch of optimization history."""

    epoch: PositiveInt
    train_loss: NonNegativeFloat
    validation_loss: NonNegativeFloat
    validation_relative_l2: NonNegativeFloat
    learning_rate: NonNegativeFloat


class OperatorTrainingHistory(BaseModel):
    """Full training history."""

    model_config = ConfigDict(validate_assignment=True)

    entries: list[OperatorTrainingLogEntry] = Field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([entry.model_dump() for entry in self.entries])


class NeuralOperatorExperimentSummary(BaseModel):
    """High-level summary returned by the tutorial experiment pipeline."""

    model_config = ConfigDict(validate_assignment=True)

    architecture: str
    trainable_parameters: PositiveInt
    device: str
    train_resolution: PositiveInt
    evaluation_resolution: PositiveInt
    train_samples: PositiveInt
    validation_samples: PositiveInt
    test_samples: PositiveInt
    evaluations: dict[str, ResolutionEvaluation]
