"""Neural-network architecture for the Burgers-equation PINN.

Key components
--------------
- **FourierFeatureEmbedding** — random Fourier features (Tancik et al. 2020)
  to overcome spectral bias in coordinate-based networks.
- **ResidualBlock** — Linear → activation → Linear + skip connection
  (Wang et al. 2021) for deeper, more stable training.
- **BurgersPINN** — the full network mapping (x, t) → u(x, t).
- **AdaptiveLossWeights** — learnable log-scale weights for multi-term losses.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from physical_informed_neural_network.config import NetworkConfig


# ---------------------------------------------------------------------------
# Fourier Feature Embedding (Tancik et al., NeurIPS 2020)
# ---------------------------------------------------------------------------

class FourierFeatureEmbedding(nn.Module):
    """Map low-dimensional inputs to a high-dimensional sinusoidal space.

    Given input ``z`` of dimension *d*, the output is::

        [sin(2π B z), cos(2π B z)]

    where *B* is a fixed random matrix of shape ``(n_features, d)``.
    """

    def __init__(self, input_dim: int, n_features: int, scale: float) -> None:
        super().__init__()
        B = torch.randn(n_features, input_dim) * scale
        self.register_buffer("B", B)

    @property
    def output_dim(self) -> int:
        return self.B.shape[0] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * math.pi * x @ self.B.T  # (N, n_features)
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


# ---------------------------------------------------------------------------
# Residual Block (Wang et al., 2021 — Understanding and Mitigating Gradient
# Flow Pathologies in PINNs)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two-layer block with a skip connection: ``x + act(W2 · act(W1 · x + b1) + b2)``."""

    def __init__(self, dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.linear2(self.activation(self.linear1(x))))


# ---------------------------------------------------------------------------
# Full PINN model
# ---------------------------------------------------------------------------

def _make_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "sin":
        # Sine activation (SIREN-style)
        class Sine(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sin(x)
        return Sine()
    raise ValueError(f"Unknown activation: {name}")


class BurgersPINN(nn.Module):
    """Physics-informed neural network for the 1-D viscous Burgers equation.

    Architecture::

        (x, t) → FourierFeatures → Linear → [ResidualBlock × N] → Linear → u
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        # Input encoding
        self.fourier = FourierFeatureEmbedding(
            input_dim=2,
            n_features=config.fourier_features,
            scale=config.fourier_scale,
        )

        activation = _make_activation(config.activation)
        in_dim = self.fourier.output_dim

        # Projection from Fourier space → hidden dim
        self.input_proj = nn.Sequential(nn.Linear(in_dim, config.hidden_dim), activation)

        # Stack of residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(config.hidden_dim, activation) for _ in range(config.hidden_layers)]
        )

        # Output head → scalar u(x, t)
        self.output_head = nn.Linear(config.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """Forward pass: (N, 2) → (N, 1)."""
        h = self.fourier(xt)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        return self.output_head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_string(self) -> str:
        c = self.config
        return (
            f"BurgersPINN(fourier={c.fourier_features}x{c.fourier_scale}, "
            f"hidden={c.hidden_dim}x{c.hidden_layers}res, "
            f"act={c.activation})"
        )


# ---------------------------------------------------------------------------
# Adaptive loss weights (learnable, softmax-normalised)
# ---------------------------------------------------------------------------

class AdaptiveLossWeights(nn.Module):
    """Learnable per-term loss weights stored as raw logits.

    Effective weights are obtained via ``softmax(logits) * n_terms`` so they
    always sum to the number of terms and remain positive.
    """

    TERM_NAMES = ("pde_residual", "boundary", "initial_condition", "data")

    def __init__(self, initial_pde: float, initial_bc: float, initial_ic: float, initial_data: float) -> None:
        super().__init__()
        self.logits = nn.Parameter(
            torch.tensor(
                [math.log(initial_pde), math.log(initial_bc), math.log(initial_ic), math.log(initial_data)],
                dtype=torch.float32,
            )
        )

    def forward(self) -> dict[str, torch.Tensor]:
        weights = F.softmax(self.logits, dim=0) * len(self.TERM_NAMES)
        return {name: weights[i] for i, name in enumerate(self.TERM_NAMES)}

    def snapshot(self) -> dict[str, float]:
        with torch.no_grad():
            w = self.forward()
            return {k: float(v.cpu()) for k, v in w.items()}
