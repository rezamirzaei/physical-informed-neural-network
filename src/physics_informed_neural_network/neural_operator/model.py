"""Fourier neural-operator building blocks for 1-D operator learning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import FourierNeuralOperatorConfig


def _apply_activation(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name == "gelu":
        return F.gelu(tensor)
    if name == "relu":
        return F.relu(tensor)
    if name == "tanh":
        return torch.tanh(tensor)
    raise ValueError(f"Unsupported activation: {name}")


class SpectralConv1d(nn.Module):
    """Low-mode Fourier convolution that supports variable input resolutions."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, signal_length = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )
        active_modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :active_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :active_modes],
            self.weight[:, :, :active_modes],
        )
        return torch.fft.irfft(out_ft, n=signal_length, dim=-1)


class FourierResidualBlock1d(nn.Module):
    """Spectral operator block with a local linear skip path."""

    def __init__(self, width: int, modes: int, activation: str) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_activation(self.activation, self.spectral(x) + self.local(x))


class FourierNeuralOperator1d(nn.Module):
    """Simple 1-D FNO model with coordinate augmentation."""

    def __init__(self, config: FourierNeuralOperatorConfig) -> None:
        super().__init__()
        self.config = config

        self.lift = nn.Linear(config.input_channels, config.width)
        self.blocks = nn.ModuleList(
            [FourierResidualBlock1d(config.width, config.modes, config.activation) for _ in range(config.layers)]
        )
        self.projection_in = nn.Linear(config.width, config.width)
        self.projection_out = nn.Linear(config.width, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map a batch of function samples with shape ``(batch, n_points, channels)`` to solutions."""
        if inputs.ndim != 3:
            raise ValueError(f"Expected input shape (batch, n_points, channels), received {tuple(inputs.shape)}")

        hidden = self.lift(inputs).permute(0, 2, 1)
        if self.config.padding > 0:
            hidden = F.pad(hidden, (0, self.config.padding))

        for block in self.blocks:
            hidden = block(hidden)

        if self.config.padding > 0:
            hidden = hidden[..., :-self.config.padding]
        hidden = hidden.permute(0, 2, 1)
        hidden = _apply_activation(self.config.activation, self.projection_in(hidden))
        return self.projection_out(hidden)

    def architecture_string(self) -> str:
        cfg = self.config
        return (
            f"FourierNeuralOperator1d(width={cfg.width}, modes={cfg.modes}, "
            f"layers={cfg.layers}, padding={cfg.padding}, act={cfg.activation})"
        )

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
