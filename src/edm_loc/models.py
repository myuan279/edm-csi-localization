from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn

from .contracts import DatasetBatch


class ConditionEncoder(Protocol):
    output_dim: int

    def __call__(
        self,
        csi: torch.Tensor,
        subcarrier_mask: torch.Tensor,
        port_mask: torch.Tensor,
    ) -> torch.Tensor: ...


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class SharedPerPortBranch(nn.Module):
    """Parallel local convolutions shared across all antenna ports."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, channels, kernel_size=5, padding=2)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # values [B,P,F], mask [B,F]
        batch, ports, subcarriers = values.shape
        # Zero before convolution so padded values cannot bleed through a
        # kernel boundary into the last valid subcarrier.
        masked_values = values * mask[:, None, :].to(values.dtype)
        flat = masked_values.reshape(batch * ports, 1, subcarriers)
        features = torch.cat((self.conv3(flat), self.conv5(flat)), dim=1)
        features = self.activation(features)
        expanded_mask = mask[:, None, :].expand(batch, ports, subcarriers)
        expanded_mask = expanded_mask.reshape(batch * ports, 1, subcarriers)
        features = features * expanded_mask.to(features.dtype)
        denominator = expanded_mask.sum(dim=2).clamp_min(1).to(features.dtype)
        pooled = features.sum(dim=2) / denominator
        return pooled.reshape(batch, ports, -1)


class StructureAwareCSIEncoder(nn.Module):
    """Amplitude/phase separated, subcarrier-local, port-aware CSI encoder."""

    def __init__(
        self,
        output_dim: int = 128,
        conv_channels: int = 24,
        port_dim: int = 96,
        attention_heads: int = 4,
        max_ports: int = 32,
    ):
        super().__init__()
        if port_dim % attention_heads:
            raise ValueError("port_dim must be divisible by attention_heads")
        self.output_dim = int(output_dim)
        self.max_ports = int(max_ports)
        self.amplitude = SharedPerPortBranch(conv_channels)
        self.phase = SharedPerPortBranch(conv_channels)
        self.port_projection = nn.Sequential(
            nn.Linear(4 * conv_channels, port_dim), nn.GELU(), nn.LayerNorm(port_dim)
        )
        self.port_embedding = nn.Embedding(max_ports, port_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, port_dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.attention = nn.MultiheadAttention(
            port_dim, attention_heads, batch_first=True, dropout=0.0
        )
        self.output = nn.Sequential(
            nn.Linear(port_dim, output_dim), nn.GELU(), nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        csi: torch.Tensor,
        subcarrier_mask: torch.Tensor,
        port_mask: torch.Tensor,
    ) -> torch.Tensor:
        if csi.ndim != 4 or csi.shape[1] != 2:
            raise ValueError("csi must be [B,2,F,P]")
        batch, _, subcarriers, ports = csi.shape
        if tuple(subcarrier_mask.shape) != (batch, subcarriers):
            raise ValueError("subcarrier mask shape mismatch")
        if tuple(port_mask.shape) != (batch, ports):
            raise ValueError("port mask shape mismatch")
        if ports > self.max_ports:
            raise ValueError(f"port count {ports} exceeds max_ports={self.max_ports}")
        if not torch.all(subcarrier_mask.any(dim=1)) or not torch.all(port_mask.any(dim=1)):
            raise ValueError("every row needs a valid subcarrier and port")
        # [B,F,P] -> [B,P,F]
        amplitude = self.amplitude(csi[:, 0].transpose(1, 2), subcarrier_mask)
        phase = self.phase(csi[:, 1].transpose(1, 2), subcarrier_mask)
        ports_encoded = self.port_projection(torch.cat((amplitude, phase), dim=2))
        positions = torch.arange(ports, device=csi.device)
        ports_encoded = ports_encoded + self.port_embedding(positions)[None, :, :]
        ports_encoded = ports_encoded * port_mask[:, :, None].to(ports_encoded.dtype)
        query = self.pool_query.expand(batch, -1, -1)
        pooled, _ = self.attention(
            query, ports_encoded, ports_encoded, key_padding_mask=~port_mask, need_weights=False
        )
        return self.output(pooled[:, 0])


class FlatCSIEncoder(nn.Module):
    """Masked flat control with the same 128-D output contract."""

    def __init__(self, subcarriers: int, ports: int, hidden_dim: int, output_dim: int = 128):
        super().__init__()
        self.subcarriers = int(subcarriers)
        self.ports = int(ports)
        self.output_dim = int(output_dim)
        flat_dim = 2 * self.subcarriers * self.ports
        self.network = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        csi: torch.Tensor,
        subcarrier_mask: torch.Tensor,
        port_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (2, self.subcarriers, self.ports)
        if tuple(csi.shape[1:]) != expected:
            raise ValueError(f"flat control expects [B,{expected[0]},{expected[1]},{expected[2]}]")
        mask = subcarrier_mask[:, None, :, None] & port_mask[:, None, None, :]
        return self.network((csi * mask.to(csi.dtype)).flatten(1))


def matched_flat_encoder(
    structure_encoder: nn.Module,
    *,
    subcarriers: int,
    ports: int,
    output_dim: int = 128,
    tolerance: float = 0.05,
) -> tuple[FlatCSIEncoder, dict[str, float | int]]:
    target = trainable_parameter_count(structure_encoder)
    candidates: list[tuple[int, int, FlatCSIEncoder]] = []
    for width in range(1, 4097):
        encoder = FlatCSIEncoder(subcarriers, ports, width, output_dim)
        count = trainable_parameter_count(encoder)
        candidates.append((abs(count - target), count, encoder))
        if count > target and width > 2:
            break
    _, count, selected = min(candidates, key=lambda item: (item[0], item[1]))
    relative = abs(count - target) / float(target)
    if relative > tolerance:
        raise ValueError(
            f"cannot parameter-match flat encoder: structure={target}, flat={count}, delta={relative:.3%}"
        )
    return selected, {
        "structure_encoder_parameters": target,
        "flat_encoder_parameters": count,
        "relative_difference": relative,
        "flat_hidden_dim": selected.network[0].out_features,
    }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        if dimension % 2:
            raise ValueError("time embedding dimension must be even")
        self.dimension = dimension

    def forward(self, timestep: torch.Tensor, num_steps: int) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32)
            * -(math.log(10_000.0) / max(half - 1, 1))
        )
        phase = timestep.float()[:, None] / float(max(num_steps - 1, 1))
        phase = phase * frequency[None, :]
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)


class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()
        self.normalization = nn.LayerNorm(out_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(self.activation(self.linear(values)))


class ZeroAffine(nn.Module):
    """Independent zero-initialized scale/shift injection."""

    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.projection = nn.Linear(condition_dim, 2 * feature_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.projection(condition).chunk(2, dim=1)
        return values * (1.0 + scale) + shift


class ConditionalDenoiser(nn.Module):
    """Shared DDPM backbone with untied down/middle/up conditioning."""

    def __init__(
        self,
        condition_encoder: nn.Module,
        *,
        embedding_dim: int = 128,
        coordinate_dim: int = 2,
        num_steps: int = 1000,
    ):
        super().__init__()
        self.condition_encoder = condition_encoder
        self.embedding_dim = int(embedding_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.num_steps = int(num_steps)
        if getattr(condition_encoder, "output_dim", None) != embedding_dim:
            raise ValueError("condition encoder output must equal embedding_dim")
        d1, d2 = embedding_dim // 2, embedding_dim // 4
        self.sample_embedding = FeedForwardBlock(coordinate_dim, embedding_dim)
        self.time_embedding = SinusoidalTimeEmbedding(embedding_dim)
        self.down1 = FeedForwardBlock(embedding_dim, d1)
        self.down2 = FeedForwardBlock(d1, d2)
        self.middle = FeedForwardBlock(d2, d2)
        self.up1 = FeedForwardBlock(2 * d2, d1)
        self.up2 = FeedForwardBlock(2 * d1, embedding_dim)
        self.output = nn.Linear(embedding_dim, coordinate_dim)
        self.mod_down1 = ZeroAffine(embedding_dim, d1)
        self.mod_down2 = ZeroAffine(embedding_dim, d2)
        self.mod_middle = ZeroAffine(embedding_dim, d2)
        self.mod_up1 = ZeroAffine(embedding_dim, d1)
        self.mod_up2 = ZeroAffine(embedding_dim, embedding_dim)

    def encode_condition(self, batch: DatasetBatch) -> torch.Tensor:
        return self.condition_encoder(batch.csi, batch.subcarrier_mask, batch.port_mask)

    def forward_from_condition(
        self, noisy_coordinate: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        base = self.sample_embedding(noisy_coordinate) + self.time_embedding(timestep, self.num_steps)
        down1 = self.mod_down1(self.down1(base), condition)
        down2 = self.mod_down2(self.down2(down1), condition)
        middle = self.mod_middle(self.middle(down2), condition)
        up1 = self.mod_up1(self.up1(torch.cat((middle, down2), dim=1)), condition)
        up2 = self.mod_up2(self.up2(torch.cat((up1, down1), dim=1)), condition)
        return self.output(up2)

    def forward(
        self, noisy_coordinate: torch.Tensor, timestep: torch.Tensor, batch: DatasetBatch
    ) -> torch.Tensor:
        return self.forward_from_condition(noisy_coordinate, timestep, self.encode_condition(batch))


def paired_model_parameter_audit(
    structure_model: nn.Module, flat_model: nn.Module, tolerance: float = 0.05
) -> dict[str, float | int | bool]:
    structure = trainable_parameter_count(structure_model)
    flat = trainable_parameter_count(flat_model)
    relative = abs(structure - flat) / float(structure)
    return {
        "structure_parameters": structure,
        "flat_parameters": flat,
        "relative_difference": relative,
        "passes_5_percent_gate": relative <= tolerance,
    }
