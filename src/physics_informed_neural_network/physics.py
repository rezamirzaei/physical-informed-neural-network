"""Physics residuals for the 1-D viscous Burgers equation.

Separates the PDE definition from the network and training logic so that
new equations can be added without touching the trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class PDEResidual(ABC):
    """Abstract base for computing PDE residuals via automatic differentiation."""

    @abstractmethod
    def residual(
        self,
        model: torch.nn.Module,
        collocation: torch.Tensor,
    ) -> torch.Tensor:
        """Return a tensor of residual values at the collocation points."""

    @abstractmethod
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Return the exact u(x, 0) at the given spatial locations."""

    @abstractmethod
    def boundary_value(self, xt: torch.Tensor) -> torch.Tensor:
        """Return the exact u on the boundary at the given (x, t) pairs."""


class BurgersResidual(PDEResidual):
    """Residual of  u_t + u·u_x − ν·u_xx = 0."""

    def __init__(self, viscosity: float) -> None:
        self.nu = viscosity

    # ---- PDE residual ----

    def residual(
        self,
        model: torch.nn.Module,
        collocation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute  r = u_t + u·u_x − ν·u_xx  at *collocation* points.

        Parameters
        ----------
        model : callable  (x, t) → u  where input is (N, 2) and output is (N, 1).
        collocation : Tensor of shape (N, 2) with requires_grad=True.

        Returns
        -------
        Tensor of shape (N, 1).
        """
        collocation = collocation.requires_grad_(True)
        u = model(collocation)  # (N, 1)

        # First-order derivatives
        grad_u = torch.autograd.grad(
            u, collocation, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]  # (N, 2)
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        # Second-order derivative u_xx
        grad_ux = torch.autograd.grad(
            u_x, collocation, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        u_xx = grad_ux[:, 0:1]

        return u_t + u * u_x - self.nu * u_xx

    # ---- Initial condition:  u(x, 0) = −sin(π x) ----

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.sin(torch.pi * x)

    # ---- Boundary condition:  u(−1, t) = u(1, t) = 0 ----

    def boundary_value(self, xt: torch.Tensor) -> torch.Tensor:
        return torch.zeros(xt.shape[0], 1, device=xt.device, dtype=xt.dtype)

