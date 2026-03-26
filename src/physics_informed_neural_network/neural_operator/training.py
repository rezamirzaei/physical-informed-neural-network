"""Training utilities for the neural-operator tutorial stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import OperatorOptimizationConfig
from .data import OperatorDataset
from .model import FourierNeuralOperator1d
from .schemas import OperatorErrorMetrics, OperatorTrainingHistory, OperatorTrainingLogEntry


@dataclass(slots=True)
class TensorNormalizer:
    """Channel-wise affine normalizer for tensors of shape ``(batch, n_points, channels)``."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, tensor: torch.Tensor, dims: tuple[int, ...]) -> "TensorNormalizer":
        mean = tensor.mean(dim=dims, keepdim=True)
        std = tensor.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean

    def to(self, device: torch.device) -> "TensorNormalizer":
        return TensorNormalizer(mean=self.mean.to(device), std=self.std.to(device))


class ArrayOperatorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset backed by dense function-grid arrays."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def compute_error_metrics(prediction: np.ndarray, target: np.ndarray) -> OperatorErrorMetrics:
    """Compute aggregate regression metrics on function-valued outputs."""
    diff = prediction - target
    denom = float(np.linalg.norm(target.ravel()) + 1e-12)
    return OperatorErrorMetrics(
        relative_l2=float(np.linalg.norm(diff.ravel()) / denom),
        mse=float(np.mean(diff ** 2)),
        mae=float(np.mean(np.abs(diff))),
        max_absolute_error=float(np.max(np.abs(diff))),
    )


class NeuralOperatorTrainer:
    """Owns optimization, normalization, prediction, and evaluation logic."""

    def __init__(
        self,
        model: FourierNeuralOperator1d,
        config: OperatorOptimizationConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.history = OperatorTrainingHistory()
        self.loss_fn = torch.nn.MSELoss()

        self.input_normalizer: TensorNormalizer | None = None
        self.target_normalizer: TensorNormalizer | None = None

    def fit(self, train_dataset: OperatorDataset, validation_dataset: OperatorDataset) -> OperatorTrainingHistory:
        train_features = torch.tensor(train_dataset.features(), dtype=torch.float32)
        train_targets = torch.tensor(train_dataset.targets(), dtype=torch.float32)

        self.input_normalizer = TensorNormalizer.fit(train_features, dims=(0, 1)).to(self.device)
        self.target_normalizer = TensorNormalizer.fit(train_targets, dims=(0, 1)).to(self.device)

        loader = DataLoader(
            ArrayOperatorDataset(train_dataset.features(), train_dataset.targets()),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.scheduler_step,
            gamma=self.config.scheduler_gamma,
        )

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running_loss = 0.0
            seen_samples = 0

            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                prediction_normalized = self.model(self.input_normalizer.transform(batch_features))
                loss = self.loss_fn(prediction_normalized, self.target_normalizer.transform(batch_targets))
                loss.backward()

                if self.config.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                optimizer.step()

                batch_size = batch_features.shape[0]
                running_loss += float(loss.detach().cpu()) * batch_size
                seen_samples += batch_size

            validation_prediction = self.predict_dataset(validation_dataset)
            validation_metrics = compute_error_metrics(validation_prediction, validation_dataset.solution)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            self.history.entries.append(
                OperatorTrainingLogEntry(
                    epoch=epoch,
                    train_loss=running_loss / max(seen_samples, 1),
                    validation_loss=validation_metrics.mse,
                    validation_relative_l2=validation_metrics.relative_l2,
                    learning_rate=learning_rate,
                )
            )

            if epoch == 1 or epoch % self.config.log_every == 0 or epoch == self.config.epochs:
                print(
                    f"  [FNO {epoch:>4d}/{self.config.epochs}] "
                    f"train_loss={running_loss / max(seen_samples, 1):.6f} "
                    f"val_rel_l2={validation_metrics.relative_l2:.6f} "
                    f"val_mse={validation_metrics.mse:.6f}"
                )

            scheduler.step()

        return self.history

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        if self.input_normalizer is None or self.target_normalizer is None:
            raise RuntimeError("Trainer must be fit before calling predict_features().")

        loader = DataLoader(
            torch.tensor(features, dtype=torch.float32),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        predictions: list[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for batch_features in loader:
                batch_features = batch_features.to(self.device)
                batch_prediction = self.model(self.input_normalizer.transform(batch_features))
                batch_prediction = self.target_normalizer.inverse(batch_prediction)
                predictions.append(batch_prediction.cpu())

        return torch.cat(predictions, dim=0).numpy()

    def predict_dataset(self, dataset: OperatorDataset) -> np.ndarray:
        return self.predict_features(dataset.features()).squeeze(-1)

    def evaluate_dataset(self, dataset: OperatorDataset) -> OperatorErrorMetrics:
        return compute_error_metrics(self.predict_dataset(dataset), dataset.solution)
