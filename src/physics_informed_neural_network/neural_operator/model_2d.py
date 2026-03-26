"""2-D Fourier neural-operator building blocks for Darcy-flow operator learning.

Implements the 2-D spectral convolution layer and the full FNO-2d architecture
following Kovachki et al. (2023) and Li et al. (2021).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import FourierNeuralOperator2dConfig


def _apply_activation(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name == "gelu":
        return F.gelu(tensor)
    if name == "relu":
        return F.relu(tensor)
    if name == "tanh":
        return torch.tanh(tensor)
    raise ValueError(f"Unsupported activation: {name}")


class SpectralConv2d(nn.Module):
    """Low-mode 2-D Fourier convolution that supports variable input resolutions.

    Parameterizes a kernel in Fourier space by retaining only the first
    ``(modes_x, modes_y)`` modes along each axis.  Both positive and negative
    frequency bands along the first spatial axis are handled.
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / max(in_channels * out_channels, 1)
        # Two sets of weights: one for the positive kx band and one for the negative kx band.
        self.weight_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape ``(batch, channels, Nx, Ny)``."""
        batch_size = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        _, _, freq_x, freq_y = x_ft.shape

        out_ft = torch.zeros(batch_size, self.out_channels, freq_x, freq_y, dtype=torch.cfloat, device=x.device)

        mx = min(self.modes_x, freq_x)
        my = min(self.modes_y, freq_y)

        # Positive kx band
        out_ft[:, :, :mx, :my] = torch.einsum(
            "bixm,iomn->boxn",
            x_ft[:, :, :mx, :my],
            self.weight_pos[:, :, :mx, :my],
        )

        # Negative kx band (the high indices in the DFT correspond to negative freqs)
        if mx <= freq_x:
            out_ft[:, :, -mx:, :my] = torch.einsum(
                "bixm,iomn->boxn",
                x_ft[:, :, -mx:, :my],
                self.weight_neg[:, :, :mx, :my],
            )

        return torch.fft.irfft2(out_ft, s=(size_x, size_y), dim=(-2, -1))


class FourierResidualBlock2d(nn.Module):
    """Spectral operator block with a local 1×1 convolution skip path."""

    def __init__(self, width: int, modes_x: int, modes_y: int, activation: str) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_activation(self.activation, self.spectral(x) + self.local(x))


class FourierNeuralOperator2d(nn.Module):
    """2-D Fourier neural operator with coordinate augmentation for Darcy-type problems.

    Input shape: ``(batch, Nx, Ny, in_channels)``  – channels-last, as produced by the
    dataset ``features()`` method.

    Output shape: ``(batch, Nx, Ny, 1)``
    """

    def __init__(self, config: FourierNeuralOperator2dConfig) -> None:
        super().__init__()
        self.config = config

        self.lift = nn.Linear(config.input_channels, config.width)
        self.blocks = nn.ModuleList(
            [
                FourierResidualBlock2d(config.width, config.modes_x, config.modes_y, config.activation)
                for _ in range(config.layers)
            ]
        )
        self.projection_in = nn.Linear(config.width, config.width)
        self.projection_out = nn.Linear(config.width, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, Nx, Ny, channels)`` → ``(batch, Nx, Ny, 1)``."""
        if inputs.ndim != 4:
            raise ValueError(
                f"Expected input shape (batch, Nx, Ny, channels), received {tuple(inputs.shape)}"
            )

        # Lift: pointwise (batch, Nx, Ny, C) → (batch, Nx, Ny, width)
        hidden = self.lift(inputs)
        # To channels-first: (batch, width, Nx, Ny)
        hidden = hidden.permute(0, 3, 1, 2)

        if self.config.padding > 0:
            hidden = F.pad(hidden, (0, self.config.padding, 0, self.config.padding))

        for block in self.blocks:
            hidden = block(hidden)

        if self.config.padding > 0:
            hidden = hidden[..., : -self.config.padding, : -self.config.padding]

        # Back to channels-last
        hidden = hidden.permute(0, 2, 3, 1)
        hidden = _apply_activation(self.config.activation, self.projection_in(hidden))
        return self.projection_out(hidden)

    def architecture_string(self) -> str:
        cfg = self.config
        return (
            f"FourierNeuralOperator2d(width={cfg.width}, "
            f"modes=({cfg.modes_x},{cfg.modes_y}), "
            f"layers={cfg.layers}, padding={cfg.padding}, act={cfg.activation})"
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

