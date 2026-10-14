from __future__ import annotations

import math

import torch
from torch import nn


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.reshape(-1, 1)
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=values.device, dtype=values.dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        phase = values * frequencies[None, :]
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)


class CSIEncoder(nn.Module):
    output_dim = 128

    def __init__(self, input_dim: int = 312, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.output_dim), nn.GELU(), nn.LayerNorm(self.output_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[1] != 312:
            raise ValueError("condition must be [B,312]")
        return self.network(condition)


class FiLMBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, condition_dim: int = 128):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim))
        self.modulation = nn.Linear(condition_dim, 2 * out_dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        values = self.block(values)
        scale, shift = self.modulation(condition).chunk(2, dim=1)
        return values * (1.0 + scale) + shift


class ConditionedField(nn.Module):
    """A 2-D coordinate field with the project's FiLM MLP scale."""

    def __init__(self, *, two_times: bool = False, embedding_dim: int = 128):
        super().__init__()
        self.condition_encoder = CSIEncoder(312, embedding_dim)
        self.state_embedding = nn.Sequential(nn.Linear(2, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim))
        self.time_embedding = ScalarTimeEmbedding(embedding_dim)
        self.two_times = bool(two_times)
        if self.two_times:
            self.reference_embedding = ScalarTimeEmbedding(embedding_dim)
        d1, d2 = embedding_dim // 2, embedding_dim // 4
        self.down1 = FiLMBlock(embedding_dim, d1, embedding_dim)
        self.down2 = FiLMBlock(d1, d2, embedding_dim)
        self.middle = FiLMBlock(d2, d2, embedding_dim)
        self.up1 = FiLMBlock(2 * d2, d1, embedding_dim)
        self.up2 = FiLMBlock(2 * d1, embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, 2)

    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return self.condition_encoder(condition)

    def forward_from_condition(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition_embedding: torch.Tensor,
        reference_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != 2:
            raise ValueError("state must be [B,2]")
        if time.reshape(-1).shape[0] != len(state):
            raise ValueError("time/state batch mismatch")
        base = self.state_embedding(state) + self.time_embedding(time)
        if self.two_times:
            if reference_time is None:
                raise ValueError("MeanFlow requires reference time")
            base = base + self.reference_embedding(reference_time)
        x1 = self.down1(base, condition_embedding)
        x2 = self.down2(x1, condition_embedding)
        mid = self.middle(x2, condition_embedding)
        up1 = self.up1(torch.cat((mid, x2), dim=1), condition_embedding)
        up2 = self.up2(torch.cat((up1, x1), dim=1), condition_embedding)
        return self.output(up2)

    def forward(self, state: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.forward_from_condition(state, time, self.encode_condition(condition))


class ConsistencyModel(nn.Module):
    def __init__(self, *, sigma_min: float = 0.002, sigma_data: float = 1.0):
        super().__init__()
        self.field = ConditionedField(two_times=False)
        self.sigma_min = float(sigma_min)
        self.sigma_data = float(sigma_data)

    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return self.field.encode_condition(condition)

    def forward_from_condition(self, state: torch.Tensor, sigma: torch.Tensor, condition_embedding: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1).to(dtype=state.dtype)
        shifted_sigma = (sigma - self.sigma_min).clamp_min(0.0)
        denominator = torch.sqrt(sigma.square() + self.sigma_data**2)
        c_skip = self.sigma_data**2 / (shifted_sigma.square() + self.sigma_data**2)
        c_out = shifted_sigma * self.sigma_data / denominator
        c_in = 1.0 / denominator
        c_noise = torch.log(sigma.clamp_min(1e-5)) / 4.0
        raw = self.field.forward_from_condition(state * c_in[:, None], c_noise, condition_embedding)
        return c_skip[:, None] * state + c_out[:, None] * raw

    def forward(self, state: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.forward_from_condition(state, sigma, self.encode_condition(condition))


class MeanFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.field = ConditionedField(two_times=True)

    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return self.field.encode_condition(condition)

    def forward_from_condition(
        self, state: torch.Tensor, reference_time: torch.Tensor, time: torch.Tensor, condition_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.field.forward_from_condition(state, time, condition_embedding, reference_time)

    def forward(self, state: torch.Tensor, reference_time: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.forward_from_condition(state, reference_time, time, self.encode_condition(condition))


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
