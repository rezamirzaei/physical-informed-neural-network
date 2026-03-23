"""Training loop for the Burgers-equation PINN.

Supports:
- Two-phase optimisation (Adam → L-BFGS)
- Cosine-annealing or step-decay LR scheduling with optional warm-up
- Adaptive (learnable) per-term loss weights
- PDE residual, boundary, initial-condition, and sparse-data loss terms
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from physical_informed_neural_network.config import TrainingConfig
from physical_informed_neural_network.model import AdaptiveLossWeights, BurgersPINN
from physical_informed_neural_network.physics import BurgersResidual
from physical_informed_neural_network.schemas import ObservationSet, TrainingHistory, TrainingLogEntry


# ---------------------------------------------------------------------------
# Snapshot of a single forward pass
# ---------------------------------------------------------------------------

@dataclass
class LossSnapshot:
    total: torch.Tensor
    pde: torch.Tensor
    boundary: torch.Tensor
    initial: torch.Tensor
    data: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total.detach().cpu()),
            "pde_loss": float(self.pde.detach().cpu()),
            "boundary_loss": float(self.boundary.detach().cpu()),
            "initial_loss": float(self.initial.detach().cpu()),
            "data_loss": float(self.data.detach().cpu()),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PINNTrainer:
    """Orchestrates training of a :class:`BurgersPINN`."""

    def __init__(
        self,
        model: BurgersPINN,
        pde: BurgersResidual,
        observations: ObservationSet,
        interior_pts: np.ndarray,
        boundary_pts: np.ndarray,
        initial_pts: np.ndarray,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.pde = pde
        self.config = config
        self.device = device
        self.mse = nn.MSELoss()
        self.history = TrainingHistory()

        # Collocation tensors
        self.interior = self._to_tensor(interior_pts, requires_grad=True)
        self.boundary = self._to_tensor(boundary_pts, requires_grad=False)
        self.initial = self._to_tensor(initial_pts, requires_grad=True)

        # Observation tensors
        obs_xt = np.column_stack([observations.x, observations.t]).astype(np.float32)
        self.obs_xt = self._to_tensor(obs_xt, requires_grad=False)
        self.obs_u = torch.tensor(
            observations.u, dtype=torch.float32, device=device
        ).unsqueeze(1)

        # Adaptive weights (optional)
        self.adaptive: AdaptiveLossWeights | None = None
        if config.adaptive_weights:
            w = config.loss
            self.adaptive = AdaptiveLossWeights(
                initial_pde=w.pde_residual,
                initial_bc=w.boundary,
                initial_ic=w.initial_condition,
                initial_data=w.data,
            ).to(device)

    # -- helpers --

    def _to_tensor(self, arr: np.ndarray, requires_grad: bool) -> torch.Tensor:
        t = torch.tensor(arr, dtype=torch.float32, device=self.device)
        if requires_grad:
            t = t.requires_grad_(True)
        return t

    # -- loss computation --

    def compute_losses(self) -> LossSnapshot:
        # PDE residual on interior
        residual = self.pde.residual(self.model, self.interior)
        pde_loss = self.mse(residual, torch.zeros_like(residual))

        # Boundary loss
        u_bc = self.model(self.boundary)
        u_bc_exact = self.pde.boundary_value(self.boundary)
        bc_loss = self.mse(u_bc, u_bc_exact)

        # Initial-condition loss
        u_ic = self.model(self.initial)
        x_ic = self.initial[:, 0:1]
        u_ic_exact = self.pde.initial_condition(x_ic)
        ic_loss = self.mse(u_ic, u_ic_exact)

        # Sparse data loss
        u_data = self.model(self.obs_xt)
        data_loss = self.mse(u_data, self.obs_u)

        # Weighted total
        if self.adaptive is not None:
            w = self.adaptive()
            total = (
                w["pde_residual"] * pde_loss
                + w["boundary"] * bc_loss
                + w["initial_condition"] * ic_loss
                + w["data"] * data_loss
            )
        else:
            w_cfg = self.config.loss
            total = (
                w_cfg.pde_residual * pde_loss
                + w_cfg.boundary * bc_loss
                + w_cfg.initial_condition * ic_loss
                + w_cfg.data * data_loss
            )

        return LossSnapshot(total=total, pde=pde_loss, boundary=bc_loss, initial=ic_loss, data=data_loss)

    # -- logging --

    def _record(self, phase: str, step: int, snap: LossSnapshot) -> None:
        d = snap.detached()
        self.history.entries.append(
            TrainingLogEntry(
                phase=phase,
                step=step,
                total_loss=d["total_loss"],
                pde_loss=d["pde_loss"],
                boundary_loss=d["boundary_loss"],
                initial_loss=d["initial_loss"],
                data_loss=d["data_loss"],
            )
        )

    # -- scheduler --

    @staticmethod
    def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        kind: str,
        total_epochs: int,
        warmup: int,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if kind == "none":
            return None
        if kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup, 1))
        if kind == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(total_epochs // 3, 1), gamma=0.5)
        return None

    # -- main training entry point --

    def train(self) -> TrainingHistory:
        # Collect all trainable parameters (model + adaptive weights)
        params: list[dict] = [{"params": self.model.parameters()}]
        if self.adaptive is not None:
            params.append({"params": self.adaptive.parameters(), "lr": self.config.adam_learning_rate * 5})

        optimizer = torch.optim.Adam(params, lr=self.config.adam_learning_rate)
        scheduler = self._build_scheduler(
            optimizer, self.config.scheduler, self.config.adam_epochs, self.config.warmup_steps
        )

        # ---- Adam phase ----
        for epoch in range(1, self.config.adam_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            snap = self.compute_losses()
            snap.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Warm-up: linearly ramp LR
            if epoch <= self.config.warmup_steps:
                frac = epoch / max(self.config.warmup_steps, 1)
                for pg in optimizer.param_groups:
                    pg["lr"] = self.config.adam_learning_rate * frac
            elif scheduler is not None:
                scheduler.step()

            if epoch == 1 or epoch % self.config.log_every == 0 or epoch == self.config.adam_epochs:
                self._record("adam", epoch, snap)
                if epoch % self.config.log_every == 0:
                    d = snap.detached()
                    print(
                        f"  [Adam {epoch:>6d}/{self.config.adam_epochs}]  "
                        f"total={d['total_loss']:.6f}  pde={d['pde_loss']:.6f}  "
                        f"bc={d['boundary_loss']:.6f}  ic={d['initial_loss']:.6f}  "
                        f"data={d['data_loss']:.6f}"
                    )

        # ---- L-BFGS phase ----
        if self.config.lbfgs_iterations > 0:
            # Only optimise the model (not adaptive weights) with L-BFGS
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=self.config.lbfgs_iterations,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            latest: LossSnapshot | None = None
            step_counter = [0]

            def closure() -> torch.Tensor:
                nonlocal latest
                lbfgs.zero_grad(set_to_none=True)
                latest = self.compute_losses()
                latest.total.backward()
                step_counter[0] += 1
                return latest.total

            lbfgs.step(closure)
            if latest is not None:
                self._record("lbfgs", self.config.adam_epochs + step_counter[0], latest)
                d = latest.detached()
                print(
                    f"  [L-BFGS  done ]  total={d['total_loss']:.6f}  "
                    f"pde={d['pde_loss']:.6f}  bc={d['boundary_loss']:.6f}  "
                    f"ic={d['initial_loss']:.6f}  data={d['data_loss']:.6f}"
                )

        return self.history

    # -- prediction --

    @torch.no_grad()
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict u on a meshgrid defined by 1-D arrays *x* and *t*.

        Returns
        -------
        u : np.ndarray of shape (len(t), len(x))
        """
        X, T = np.meshgrid(x, t)
        xt = np.column_stack([X.ravel(), T.ravel()]).astype(np.float32)
        xt_tensor = torch.tensor(xt, dtype=torch.float32, device=self.device)
        u_flat = self.model(xt_tensor).cpu().numpy().ravel()
        return u_flat.reshape(len(t), len(x))

    @torch.no_grad()
    def predict_points(self, xt: np.ndarray) -> np.ndarray:
        """Predict u at arbitrary (x, t) points of shape (N, 2)."""
        tensor = torch.tensor(xt.astype(np.float32), device=self.device)
        return self.model(tensor).cpu().numpy().ravel()
