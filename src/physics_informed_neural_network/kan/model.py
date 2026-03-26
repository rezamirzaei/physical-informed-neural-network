"""Spline-edge Kolmogorov-Arnold Network building blocks."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import PiecewiseLinearKANConfig


def _apply_activation(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name == "identity":
        return tensor
    if name == "silu":
        return F.silu(tensor)
    if name == "tanh":
        return torch.tanh(tensor)
    if name == "relu":
        return F.relu(tensor)
    raise ValueError(f"Unsupported activation: {name}")


class PiecewiseLinearBasis(nn.Module):
    """Uniform hat-function basis on a bounded 1-D interval."""

    def __init__(self, num_knots: int, domain_min: float = -1.0, domain_max: float = 1.0) -> None:
        super().__init__()
        if num_knots < 2:
            raise ValueError("num_knots must be at least 2.")
        if domain_max <= domain_min:
            raise ValueError("domain_max must exceed domain_min.")

        self.num_knots = num_knots
        self.domain_min = float(domain_min)
        self.domain_max = float(domain_max)
        self.step = (self.domain_max - self.domain_min) / (self.num_knots - 1)

        self.register_buffer(
            "knot_positions",
            torch.linspace(self.domain_min, self.domain_max, self.num_knots, dtype=torch.float32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return basis values with shape ``(..., num_knots)``."""
        clamped = inputs.clamp(self.domain_min, self.domain_max)
        scaled = (clamped - self.domain_min) / self.step
        left_index = torch.floor(scaled).long().clamp(min=0, max=self.num_knots - 1)
        right_index = (left_index + 1).clamp(max=self.num_knots - 1)
        fraction = (scaled - left_index.to(dtype=inputs.dtype)).unsqueeze(-1)

        left_basis = F.one_hot(left_index, num_classes=self.num_knots).to(dtype=inputs.dtype)
        right_basis = F.one_hot(right_index, num_classes=self.num_knots).to(dtype=inputs.dtype)
        return (1.0 - fraction) * left_basis + fraction * right_basis


class KANLayer(nn.Module):
    """One KAN layer with learned univariate edge functions."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: PiecewiseLinearKANConfig,
        input_activation: str,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_activation = config.base_activation
        self.input_activation = input_activation
        self.use_bias = config.use_bias
        self.basis = PiecewiseLinearBasis(
            num_knots=config.num_knots,
            domain_min=config.spline_domain_min,
            domain_max=config.spline_domain_max,
        )

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_coefficients = nn.Parameter(torch.empty(out_features, in_features, config.num_knots))
        self.spline_scale = nn.Parameter(torch.full((out_features, in_features), config.spline_scale_init))
        self.bias = nn.Parameter(torch.zeros(out_features)) if config.use_bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(self.in_features, 1))
        nn.init.uniform_(self.base_weight, -bound, bound)
        nn.init.normal_(self.spline_coefficients, mean=0.0, std=0.03)
        with torch.no_grad():
            self.spline_coefficients.sub_(self.spline_coefficients.mean(dim=-1, keepdim=True))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        return _apply_activation(self.input_activation, inputs)

    def edge_contributions(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return per-edge contributions with shape ``(batch, out_features, in_features)``."""
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(
                f"Expected inputs of shape (batch, {self.in_features}), received {tuple(inputs.shape)}"
            )

        processed = self._preprocess(inputs)
        basis_values = self.basis(processed)
        spline_component = torch.einsum("bik,oik->boi", basis_values, self.spline_coefficients)
        spline_component = spline_component * self.spline_scale.unsqueeze(0)

        base_inputs = _apply_activation(self.base_activation, processed)
        base_component = torch.einsum("bi,oi->boi", base_inputs, self.base_weight)
        return base_component + spline_component

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.edge_contributions(inputs).sum(dim=-1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def evaluate_edge_function(
        self,
        input_index: int,
        output_index: int,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one learned univariate edge function on a 1-D sample grid."""
        if input_index < 0 or input_index >= self.in_features:
            raise IndexError(f"input_index {input_index} is out of bounds for {self.in_features} inputs.")
        if output_index < 0 or output_index >= self.out_features:
            raise IndexError(f"output_index {output_index} is out of bounds for {self.out_features} outputs.")
        if samples.ndim != 1:
            raise ValueError(f"Expected 1-D samples, received shape {tuple(samples.shape)}")

        processed = self._preprocess(samples.unsqueeze(-1)).squeeze(-1)
        basis_values = self.basis(processed.unsqueeze(-1)).squeeze(1)

        spline = torch.einsum("bk,k->b", basis_values, self.spline_coefficients[output_index, input_index])
        spline = spline * self.spline_scale[output_index, input_index]
        base = self.base_weight[output_index, input_index] * _apply_activation(self.base_activation, processed)
        return base + spline


class KolmogorovArnoldNetwork(nn.Module):
    """A compact KAN for scalar-valued function regression."""

    def __init__(self, config: PiecewiseLinearKANConfig) -> None:
        super().__init__()
        self.config = config

        widths = (config.input_dim, *config.hidden_widths, 1)
        layers: list[KANLayer] = []
        for layer_index, (in_features, out_features) in enumerate(zip(widths[:-1], widths[1:])):
            input_activation = "identity" if layer_index == 0 else config.hidden_input_activation
            layers.append(KANLayer(in_features, out_features, config=config, input_activation=input_activation))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected inputs of shape (batch, {self.config.input_dim}), received {tuple(inputs.shape)}"
            )

        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    def architecture_string(self) -> str:
        hidden = ",".join(str(width) for width in self.config.hidden_widths)
        return (
            f"KolmogorovArnoldNetwork(input_dim={self.config.input_dim}, "
            f"hidden=[{hidden}], knots={self.config.num_knots}, "
            f"base_act={self.config.base_activation}, hidden_input_act={self.config.hidden_input_activation})"
        )

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def evaluate_first_layer_edges(
        self,
        samples: torch.Tensor,
        output_indices: tuple[int, ...] = (0, 1, 2, 3),
    ) -> dict[int, dict[int, torch.Tensor]]:
        """Evaluate selected first-layer edge functions for notebook visualizations."""
        first_layer = self.layers[0]
        responses: dict[int, dict[int, torch.Tensor]] = {}
        for output_index in output_indices:
            if output_index >= first_layer.out_features:
                continue
            responses[output_index] = {
                input_index: first_layer.evaluate_edge_function(input_index, output_index, samples)
                for input_index in range(first_layer.in_features)
            }
        return responses
