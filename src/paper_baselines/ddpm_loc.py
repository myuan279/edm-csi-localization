"""DDPM-Loc denoiser adapted to the project's 312D R1-FRI CSI contract.

The active forward graph follows the public DDPM-Loc implementation:
sample embedding + sinusoidal timestep embedding + condition embedding,
followed by the six-layer fully-connected U-Net.  The condition embedding is
the requested 312 -> 142 -> 128 MLP so that the CSI encoder matches the EDM
flat encoder.  The upstream repository also declares an unused ``MLP`` member
in ``DenoiseNet.forward``; it is intentionally not included because it is not
part of the executed network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ddpm import (
    DIFFUSION_STEPS,
    EMBEDDING_DIM,
    POSITION_DIM,
    FeedForward,
    FullyConnectedUNet,
    SinusoidalTimestepEmbedding,
)


CONDITION_DIM = 312
CONDITION_HIDDEN_DIM = 142
DDPM_LOC_PARAMETER_COUNT = 96_892


class DDPMLocConditionEncoder(nn.Module):
    """Requested two-layer CSI condition encoder: 312 -> 142 -> 128."""

    def __init__(
        self,
        condition_dim: int = CONDITION_DIM,
        hidden_dim: int = CONDITION_HIDDEN_DIM,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape [B,{self.condition_dim}], got {tuple(condition.shape)}"
            )
        return self.network(condition)


class DDPMLocDenoiseNet(nn.Module):
    """DDPM-Loc active denoising network with the project's 312D CSI input."""

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
        self.condition_embedding = DDPMLocConditionEncoder(
            condition_dim=condition_dim,
            hidden_dim=CONDITION_HIDDEN_DIM,
            embedding_dim=embedding_dim,
        )
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


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def architecture_inventory(model: DDPMLocDenoiseNet) -> dict[str, int | str]:
    count = parameter_count(model)
    if count != DDPM_LOC_PARAMETER_COUNT:
        raise RuntimeError(
            f"DDPM-Loc parameter count drift: expected {DDPM_LOC_PARAMETER_COUNT}, got {count}"
        )
    return {
        "implementation": "ddpm_loc_baseline.DDPMLocDenoiseNet",
        "condition_dim": model.condition_dim,
        "condition_hidden_dim": CONDITION_HIDDEN_DIM,
        "embedding_dim": model.embedding_dim,
        "position_dim": model.position_dim,
        "diffusion_steps": model.diffusion_steps,
        "parameter_count": count,
    }
