from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch

from .contracts import DatasetBatch


TargetMode = Literal["absolute", "mean_centered", "diagonal_whitened", "full_whitened"]


class Preconditioner(Protocol):
    def predict(self, batch: DatasetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frozen FullCov mean [B,2] and lower Cholesky [B,2,2] in normalized space."""


@dataclass(frozen=True)
class TargetTransform:
    mode: TargetMode
    position_mean_cm: np.ndarray
    position_std_cm: np.ndarray
    minimum_cholesky_diagonal: float = 1e-6

    def __post_init__(self) -> None:
        mean = np.asarray(self.position_mean_cm, dtype=np.float64)
        std = np.asarray(self.position_std_cm, dtype=np.float64)
        if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0):
            raise ValueError("position normalization must contain two positive scales")
        if self.mode not in {"absolute", "mean_centered", "diagonal_whitened", "full_whitened"}:
            raise ValueError(f"unknown target mode {self.mode}")

    def _normalizer(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.position_mean_cm, dtype=reference.dtype, device=reference.device)
        std = torch.as_tensor(self.position_std_cm, dtype=reference.dtype, device=reference.device)
        return mean, std

    def normalize(self, coordinates_cm: torch.Tensor) -> torch.Tensor:
        mean, std = self._normalizer(coordinates_cm)
        return (coordinates_cm - mean) / std

    def denormalize(self, coordinates_normalized: torch.Tensor) -> torch.Tensor:
        mean, std = self._normalizer(coordinates_normalized)
        return coordinates_normalized * std + mean

    def _validate_preconditioner(
        self,
        target: torch.Tensor,
        mean: torch.Tensor | None,
        cholesky: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mean is None or cholesky is None:
            raise ValueError(f"{self.mode} requires a frozen FullCov preconditioner")
        if mean.shape != target.shape or cholesky.shape != (len(target), 2, 2):
            raise ValueError("FullCov preconditioner shape mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(cholesky).all():
            raise FloatingPointError("non-finite FullCov preconditioner")
        if not torch.allclose(cholesky, torch.tril(cholesky), atol=1e-7, rtol=0.0):
            raise ValueError("FullCov factor must be lower triangular")
        if torch.any(torch.diagonal(cholesky, dim1=1, dim2=2) <= self.minimum_cholesky_diagonal):
            raise ValueError("FullCov factor is not strictly SPD")
        return mean.detach(), cholesky.detach()

    def transform(
        self,
        coordinates_cm: torch.Tensor,
        mean_normalized: torch.Tensor | None = None,
        cholesky_normalized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.normalize(coordinates_cm)
        if self.mode == "absolute":
            return normalized
        mean, cholesky = self._validate_preconditioner(
            normalized, mean_normalized, cholesky_normalized
        )
        residual = normalized - mean
        if self.mode == "mean_centered":
            return residual
        if self.mode == "diagonal_whitened":
            marginal_std = torch.sqrt(torch.sum(cholesky.square(), dim=2))
            return residual / marginal_std
        return torch.linalg.solve_triangular(
            cholesky, residual.unsqueeze(2), upper=False
        ).squeeze(2)

    def inverse(
        self,
        target: torch.Tensor,
        mean_normalized: torch.Tensor | None = None,
        cholesky_normalized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "absolute":
            normalized = target
        else:
            mean, cholesky = self._validate_preconditioner(
                target, mean_normalized, cholesky_normalized
            )
            if self.mode == "mean_centered":
                residual = target
            elif self.mode == "diagonal_whitened":
                marginal_std = torch.sqrt(torch.sum(cholesky.square(), dim=2))
                residual = target * marginal_std
            else:
                residual = torch.bmm(cholesky, target.unsqueeze(2)).squeeze(2)
            normalized = residual + mean
        return self.denormalize(normalized)


class FrozenModulePreconditioner:
    """Adapter for a FullCov module without allowing DDPM gradients into it."""

    def __init__(self, module: torch.nn.Module, predictor):
        self.module = module.eval()
        self.predictor = predictor
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def predict(self, batch: DatasetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        mean, cholesky = self.predictor(self.module, batch)
        return mean.detach(), cholesky.detach()
