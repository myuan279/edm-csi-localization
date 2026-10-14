"""Frozen safe312 conditional DDPM components for Step 2.5.

The training path is float32.  Evaluation uses an independent float64 copy and
stateless, index-addressed noise so that sampling is independent of batching,
data order, model-training RNG state, and the current global PyTorch RNG state.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


CONDITION_DIM = 312
POSITION_DIM = 2
EMBEDDING_DIM = 128
DIFFUSION_STEPS = 1000
BETA_START = 1.0e-4
BETA_END = 2.0e-2
POSTERIOR_SAMPLES = 20


class FeedForward(nn.Module):
    """Linear -> GELU -> LayerNorm block used by the frozen backbone."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc(values)


class SinusoidalTimestepEmbedding(nn.Module):
    """Parameter-free sinusoidal embedding in the caller's floating dtype."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("The timestep embedding dimension must be even")
        self.dim = int(dim)

    def forward(self, normalized_timestep: torch.Tensor) -> torch.Tensor:
        if normalized_timestep.ndim == 1:
            normalized_timestep = normalized_timestep.unsqueeze(1)
        if not normalized_timestep.is_floating_point():
            raise TypeError("Normalized timesteps must have a floating dtype")
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(
                half,
                device=normalized_timestep.device,
                dtype=normalized_timestep.dtype,
            )
            * -(math.log(1.0e4) / half)
        ).unsqueeze(0)
        arguments = normalized_timestep * frequencies
        return torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)


class FullyConnectedUNet(nn.Module):
    """Six-layer fully connected U-Net with concatenation skip connections."""

    def __init__(self, in_dim: int = EMBEDDING_DIM, out_dim: int = POSITION_DIM) -> None:
        super().__init__()
        half = in_dim // 2
        quarter = in_dim // 4
        self.down1 = FeedForward(in_dim, half)
        self.down2 = FeedForward(half, quarter)
        self.middle = FeedForward(quarter, quarter)
        self.up1 = FeedForward(2 * quarter, half)
        self.up2 = FeedForward(2 * half, in_dim)
        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(values)
        skip2 = self.down2(skip1)
        middle = self.middle(skip2)
        values = self.up1(torch.cat((middle, skip2), dim=1))
        values = self.up2(torch.cat((values, skip1), dim=1))
        return self.output(values)


class Safe312DenoiseNet(nn.Module):
    """Frozen epsilon-prediction network for p(position | full312 CSI)."""

    def __init__(
        self,
        condition_dim: int = CONDITION_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        position_dim: int = POSITION_DIM,
        diffusion_steps: int = DIFFUSION_STEPS,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.embedding_dim = int(embedding_dim)
        self.position_dim = int(position_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.sample_embedding = FeedForward(position_dim, embedding_dim)
        self.timestep_embedding = SinusoidalTimestepEmbedding(embedding_dim)
        self.condition_embedding = FeedForward(condition_dim, embedding_dim)
        self.backbone = FullyConnectedUNet(embedding_dim, position_dim)

    def embed_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return self.condition_embedding(condition)

    def forward_with_condition_embedding(
        self,
        noisy_position: torch.Tensor,
        timestep: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> torch.Tensor:
        normalized_timestep = timestep.to(dtype=noisy_position.dtype).unsqueeze(1)
        normalized_timestep = normalized_timestep / float(self.diffusion_steps)
        time_embedding = self.timestep_embedding(normalized_timestep)
        return self.forward_with_precomputed_embeddings(
            noisy_position, time_embedding, condition_embedding
        )

    def forward_with_precomputed_embeddings(
        self,
        noisy_position: torch.Tensor,
        time_embedding: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluation fast path; a single time row is broadcast across a batch."""

        combined = self.sample_embedding(noisy_position) + time_embedding
        combined = combined + condition_embedding
        return self.backbone(combined)

    def forward(
        self,
        noisy_position: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_condition_embedding(
            noisy_position, timestep, self.embed_condition(condition)
        )


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_linear_schedule(
    device: torch.device,
    dtype: torch.dtype,
    diffusion_steps: int = DIFFUSION_STEPS,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
) -> DiffusionSchedule:
    betas = torch.linspace(
        beta_start, beta_end, diffusion_steps, device=device, dtype=dtype
    )
    alphas = 1.0 - betas
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=torch.cumprod(alphas, dim=0),
    )


def epsilon_prediction_loss(
    model: Safe312DenoiseNet,
    normalized_position: torch.Tensor,
    condition: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Standard epsilon-MSE objective with float32 training randomness."""

    batch_size = normalized_position.shape[0]
    timestep = torch.randint(
        0,
        model.diffusion_steps,
        (batch_size,),
        device=normalized_position.device,
        dtype=torch.long,
    )
    epsilon = torch.randn_like(normalized_position)
    alpha_bar = schedule.alpha_bars.index_select(0, timestep).unsqueeze(1)
    noisy_position = alpha_bar.sqrt() * normalized_position
    noisy_position = noisy_position + (1.0 - alpha_bar).sqrt() * epsilon
    prediction = model(noisy_position, timestep, condition)
    return torch.mean(torch.square(prediction - epsilon))


# A compact, explicitly frozen integer generator.  The modulus is below 2^31,
# so all products below stay within signed int64.  Box-Muller converts the two
# independently salted uniforms to the two position coordinates.
_PRIME = 2_147_483_647
_SNAPSHOT_COEFFICIENT = 1_000_003
_SAMPLE_COEFFICIENT = 97_409
_TIMESTEP_COEFFICIENT = 10_009
_DOMAIN_COEFFICIENT = 1_009
_LCG_MULTIPLIER = 48_271
_STREAM_SALTS = (1_048_583, 1_299_709)


def _stateless_uniform(
    snapshot_id: torch.Tensor,
    sample_id: torch.Tensor,
    timestep: int,
    base_seed: int,
    domain: int,
    stream: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if stream not in (0, 1):
        raise ValueError("Only streams 0 and 1 are defined")
    key = (
        snapshot_id.to(torch.int64) * _SNAPSHOT_COEFFICIENT
        + sample_id.to(torch.int64) * _SAMPLE_COEFFICIENT
        + int(timestep) * _TIMESTEP_COEFFICIENT
        + int(domain) * _DOMAIN_COEFFICIENT
        + int(base_seed) % _PRIME
        + _STREAM_SALTS[stream]
    ).remainder(_PRIME)
    key = (key * key + 1_234_567 + _STREAM_SALTS[stream]).remainder(_PRIME)
    key = (key * _LCG_MULTIPLIER + 1 + 2 * stream).remainder(_PRIME)
    key = (key * key + 7_654_321 + 17 * stream).remainder(_PRIME)
    return (key.to(dtype=dtype) + 0.5) / float(_PRIME)


def stateless_position_noise(
    snapshot_id: torch.Tensor,
    sample_id: torch.Tensor,
    timestep: int,
    base_seed: int,
    domain: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return deterministic N(0,I_2) values addressed by immutable IDs."""

    uniform1 = _stateless_uniform(
        snapshot_id, sample_id, timestep, base_seed, domain, 0, dtype
    )
    uniform2 = _stateless_uniform(
        snapshot_id, sample_id, timestep, base_seed, domain, 1, dtype
    )
    radius = torch.sqrt(-2.0 * torch.log(uniform1))
    angle = (2.0 * math.pi) * uniform2
    return torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)


def build_float64_evaluator(
    training_model: Safe312DenoiseNet,
    device: torch.device,
) -> Safe312DenoiseNet:
    """Deep-copy the training model without consuming or modifying RNG state."""

    evaluator = copy.deepcopy(training_model).to(device=device, dtype=torch.float64)
    evaluator.eval()
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)
    return evaluator


def ddpm_reverse_step(
    position: torch.Tensor,
    predicted_epsilon: torch.Tensor,
    timestep: int,
    schedule: DiffusionSchedule,
    reverse_noise: torch.Tensor | None,
) -> torch.Tensor:
    """Frozen reverse update with variance beta_t and no noise at t=0."""

    beta = schedule.betas[timestep]
    alpha = schedule.alphas[timestep]
    alpha_bar = schedule.alpha_bars[timestep]
    reverse_mean = torch.rsqrt(alpha) * (
        position - beta * torch.rsqrt(1.0 - alpha_bar) * predicted_epsilon
    )
    if timestep == 0:
        return reverse_mean
    if reverse_noise is None:
        raise ValueError("reverse_noise is required for every timestep above zero")
    return reverse_mean + torch.sqrt(beta) * reverse_noise


class IndexedDDPMSampler:
    """T=1000, S=20 DDPM sampler with common random numbers."""

    def __init__(
        self,
        evaluator: Safe312DenoiseNet,
        base_seed: int,
        sample_count: int = POSTERIOR_SAMPLES,
        beta_start: float = BETA_START,
        beta_end: float = BETA_END,
    ) -> None:
        if next(evaluator.parameters()).dtype != torch.float64:
            raise TypeError("The evaluation model must be float64")
        if evaluator.training:
            raise RuntimeError("The evaluation model must be in eval mode")
        if sample_count != POSTERIOR_SAMPLES:
            raise ValueError("The frozen Step 2.5 sample count is 20")
        self.evaluator = evaluator
        self.device = next(evaluator.parameters()).device
        self.base_seed = int(base_seed)
        self.sample_count = int(sample_count)
        self.schedule = make_linear_schedule(
            self.device,
            torch.float64,
            evaluator.diffusion_steps,
            beta_start,
            beta_end,
        )
        normalized_timesteps = torch.arange(
            evaluator.diffusion_steps, device=self.device, dtype=torch.float64
        ).unsqueeze(1) / float(evaluator.diffusion_steps)
        with torch.inference_mode():
            self.time_embedding_table = evaluator.timestep_embedding(
                normalized_timesteps
            )

    def sample_many(
        self,
        condition: torch.Tensor,
        snapshot_id: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Return normalized samples with shape (N,20,2) in float64."""

        condition = condition.to(self.device, dtype=torch.float64)
        snapshot_id = torch.as_tensor(snapshot_id, device=self.device, dtype=torch.int64)
        if condition.ndim != 2 or condition.shape[1] != CONDITION_DIM:
            raise ValueError("Expected condition shape (N,312)")
        if snapshot_id.shape != (condition.shape[0],):
            raise ValueError("snapshot_id must contain one ID per condition row")
        sample_ids = torch.arange(
            self.sample_count, device=self.device, dtype=torch.int64
        )
        repeated_snapshot = snapshot_id.repeat_interleave(self.sample_count)
        repeated_sample = sample_ids.repeat(condition.shape[0])
        condition_embedding = self.evaluator.embed_condition(condition)
        repeated_condition_embedding = condition_embedding.repeat_interleave(
            self.sample_count, dim=0
        )
        position = stateless_position_noise(
            repeated_snapshot,
            repeated_sample,
            self.evaluator.diffusion_steps,
            self.base_seed,
            domain=0,
            dtype=torch.float64,
        )
        with torch.inference_mode():
            for timestep in range(self.evaluator.diffusion_steps - 1, -1, -1):
                predicted_epsilon = self.evaluator.forward_with_precomputed_embeddings(
                    position,
                    self.time_embedding_table[timestep : timestep + 1],
                    repeated_condition_embedding,
                )
                if timestep == 0:
                    position = ddpm_reverse_step(
                        position, predicted_epsilon, timestep, self.schedule, None
                    )
                else:
                    reverse_noise = stateless_position_noise(
                        repeated_snapshot,
                        repeated_sample,
                        timestep,
                        self.base_seed,
                        domain=1,
                        dtype=torch.float64,
                    )
                    position = ddpm_reverse_step(
                        position, predicted_epsilon, timestep, self.schedule,
                        reverse_noise,
                    )
        samples = position.reshape(
            condition.shape[0], self.sample_count, self.evaluator.position_dim
        )
        if not torch.isfinite(samples).all():
            raise RuntimeError("DDPM sampling produced NaN or Inf")
        return samples

    def sample_many_reference(
        self,
        condition: torch.Tensor,
        snapshot_id: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Reference path: sample one snapshot at a time, preserving all 20 IDs."""

        condition = torch.as_tensor(condition)
        snapshot_id = torch.as_tensor(snapshot_id, dtype=torch.int64)
        outputs = []
        for row in range(condition.shape[0]):
            outputs.append(
                self.sample_many(condition[row : row + 1], snapshot_id[row : row + 1])
            )
        return torch.cat(outputs, dim=0)


def posterior_summary(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return S=20 sample mean and unbiased 2x2 covariance in float64."""

    if samples.dtype != torch.float64 or samples.ndim != 3:
        raise TypeError("Expected float64 samples with shape (N,S,2)")
    mean = samples.mean(dim=1)
    centered = samples - mean.unsqueeze(1)
    covariance = centered.transpose(1, 2) @ centered
    covariance = covariance / float(samples.shape[1] - 1)
    return mean, covariance


def architecture_inventory(model: Safe312DenoiseNet) -> dict[str, Any]:
    return {
        "condition_dim": model.condition_dim,
        "position_dim": model.position_dim,
        "embedding_dim": model.embedding_dim,
        "diffusion_steps": model.diffusion_steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
    }
