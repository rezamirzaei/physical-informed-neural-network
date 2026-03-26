"""Training utilities for KAN regression on the Burgers equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config import PDEConfig
from ..physics import BurgersResidual
from .config import KANOptimizationConfig
from .data import BurgersGridDataset, CoordinateNormalizer
from .model import KolmogorovArnoldNetwork
from .schemas import KANErrorMetrics, KANTrainingHistory, KANTrainingLogEntry, ResidualMetrics


@dataclass(slots=True)
class TensorNormalizer:
    """Affine normalizer for tensors with a feature axis on the last dimension."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, tensor: torch.Tensor, dim: int = 0) -> "TensorNormalizer":
        mean = tensor.mean(dim=dim, keepdim=True)
        std = tensor.std(dim=dim, keepdim=True, unbiased=False).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean

    def to(self, device: torch.device) -> "TensorNormalizer":
        return TensorNormalizer(mean=self.mean.to(device), std=self.std.to(device))


class ArrayRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset backed by dense arrays."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class PhysicalCoordinateKAN(nn.Module):
    """Wrap a normalized-coordinate KAN so autograd can act on physical coordinates."""

    def __init__(
        self,
        model: KolmogorovArnoldNetwork,
        coordinate_normalizer: CoordinateNormalizer,
        target_normalizer: TensorNormalizer,
    ) -> None:
        super().__init__()
        self.model = model
        self.coordinate_normalizer = coordinate_normalizer
        self.target_normalizer = target_normalizer

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        normalized_coordinates = self.coordinate_normalizer.transform_tensor(coordinates)
        normalized_prediction = self.model(normalized_coordinates)
        return self.target_normalizer.inverse(normalized_prediction)


def compute_error_metrics(prediction: np.ndarray, target: np.ndarray) -> KANErrorMetrics:
    """Compute regression metrics on scalar outputs."""
    diff = prediction - target
    denom = float(np.linalg.norm(target.ravel()) + 1e-12)
    return KANErrorMetrics(
        relative_l2=float(np.linalg.norm(diff.ravel()) / denom),
        mse=float(np.mean(diff ** 2)),
        mae=float(np.mean(np.abs(diff))),
        max_absolute_error=float(np.max(np.abs(diff))),
    )


def compute_residual_metrics(residuals: np.ndarray) -> ResidualMetrics:
    """Summarize PDE residual values."""
    residuals = np.asarray(residuals, dtype=np.float64)
    return ResidualMetrics(
        mean_absolute_residual=float(np.mean(np.abs(residuals))),
        root_mean_square_residual=float(np.sqrt(np.mean(residuals ** 2))),
        max_absolute_residual=float(np.max(np.abs(residuals))),
    )


class KANTrainer:
    """Owns optimization, target normalization, and evaluation logic."""

    def __init__(
        self,
        model: KolmogorovArnoldNetwork,
        config: KANOptimizationConfig,
        device: torch.device,
        coordinate_normalizer: CoordinateNormalizer,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.coordinate_normalizer = coordinate_normalizer
        self.history = KANTrainingHistory()
        self.loss_fn = nn.MSELoss()
        self.target_normalizer: TensorNormalizer | None = None

    def fit(self, train_dataset: BurgersGridDataset, validation_dataset: BurgersGridDataset) -> KANTrainingHistory:
        train_features = train_dataset.normalized_coordinates()
        train_targets = train_dataset.targets()
        validation_features = validation_dataset.normalized_coordinates()
        validation_targets = validation_dataset.targets()

        self.target_normalizer = TensorNormalizer.fit(torch.tensor(train_targets, dtype=torch.float32)).to(self.device)

        loader = DataLoader(
            ArrayRegressionDataset(train_features, train_targets),
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

        # Early-stopping state
        best_val_loss = float("inf")
        best_state_dict: dict | None = None
        patience_counter = 0
        use_early_stopping = self.config.patience > 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running_loss = 0.0
            seen_samples = 0

            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(batch_features)
                loss = self.loss_fn(prediction, self.target_normalizer.transform(batch_targets))
                loss.backward()

                if self.config.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                optimizer.step()

                batch_size = batch_features.shape[0]
                running_loss += float(loss.detach().cpu()) * batch_size
                seen_samples += batch_size

            validation_prediction = self.predict_coordinates(validation_features)
            validation_metrics = compute_error_metrics(validation_prediction, validation_targets.squeeze(-1))
            learning_rate = float(optimizer.param_groups[0]["lr"])
            self.history.entries.append(
                KANTrainingLogEntry(
                    epoch=epoch,
                    train_loss=running_loss / max(seen_samples, 1),
                    validation_loss=validation_metrics.mse,
                    validation_relative_l2=validation_metrics.relative_l2,
                    learning_rate=learning_rate,
                )
            )

            if epoch == 1 or epoch % self.config.log_every == 0 or epoch == self.config.epochs:
                print(
                    f"  [KAN {epoch:>4d}/{self.config.epochs}] "
                    f"train_loss={running_loss / max(seen_samples, 1):.6f} "
                    f"val_rel_l2={validation_metrics.relative_l2:.6f} "
                    f"val_mse={validation_metrics.mse:.6f}"
                )

            # Early-stopping check
            if use_early_stopping:
                if validation_metrics.mse < best_val_loss - self.config.min_delta:
                    best_val_loss = validation_metrics.mse
                    best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        print(f"  [KAN] Early stopping at epoch {epoch} (patience={self.config.patience})")
                        break

            scheduler.step()

        # Restore best model if early stopping was active
        if use_early_stopping and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return self.history

    def predict_coordinates(self, normalized_coordinates: np.ndarray) -> np.ndarray:
        if self.target_normalizer is None:
            raise RuntimeError("Trainer must be fit before calling predict_coordinates().")

        loader = DataLoader(
            torch.tensor(normalized_coordinates, dtype=torch.float32),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        predictions: list[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for batch_features in loader:
                batch_features = batch_features.to(self.device)
                batch_prediction = self.model(batch_features)
                batch_prediction = self.target_normalizer.inverse(batch_prediction)
                predictions.append(batch_prediction.cpu())

        return torch.cat(predictions, dim=0).numpy().squeeze(-1)

    def predict_dataset(self, dataset: BurgersGridDataset) -> np.ndarray:
        return self.predict_coordinates(dataset.normalized_coordinates())

    def evaluate_dataset(self, dataset: BurgersGridDataset) -> KANErrorMetrics:
        return compute_error_metrics(self.predict_dataset(dataset), dataset.solution.ravel())

    def compute_pde_residuals(self, pde: PDEConfig, coordinates: np.ndarray) -> np.ndarray:
        if self.target_normalizer is None:
            raise RuntimeError("Trainer must be fit before calling compute_pde_residuals().")

        wrapped_model = PhysicalCoordinateKAN(
            self.model,
            coordinate_normalizer=self.coordinate_normalizer,
            target_normalizer=self.target_normalizer,
        ).to(self.device)
        collocation = torch.tensor(coordinates, dtype=torch.float32, device=self.device, requires_grad=True)

        with torch.enable_grad():
            residuals = BurgersResidual(viscosity=pde.viscosity).residual(wrapped_model, collocation)
        return residuals.detach().cpu().numpy().squeeze(-1)
