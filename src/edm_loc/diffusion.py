from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch
from torch import nn

from probabilistic_rng import gaussian_base_noise

from .contracts import DatasetBatch, DistributionOutput
from .targets import Preconditioner, TargetTransform


ScheduleName = Literal["linear", "cosine"]
LossName = Literal["epsilon_mse", "min_snr_5"]


def make_beta_schedule(
    num_steps: int,
    schedule: ScheduleName,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64).float()
    if schedule != "cosine":
        raise ValueError(f"unknown noise schedule {schedule}")
    # Improved DDPM cosine alpha-bar discretization (s=0.008).
    time = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64) / num_steps
    alpha_bar = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clamp(1e-8, 0.999).float()


def min_snr_epsilon_weight(alpha_hat: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    snr = alpha_hat / (1.0 - alpha_hat).clamp_min(torch.finfo(alpha_hat.dtype).eps)
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp_min(
        torch.finfo(alpha_hat.dtype).eps
    )


@dataclass(frozen=True)
class DiffusionConfig:
    num_steps: int = 1000
    schedule: ScheduleName = "linear"
    loss: LossName = "epsilon_mse"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    min_snr_gamma: float = 5.0
    ema_decay: float = 0.9999

    def validate(self) -> None:
        if self.num_steps not in {50, 100, 200, 1000}:
            raise ValueError("registered chains are T=50/100/200/1000")
        if self.schedule not in {"linear", "cosine"}:
            raise ValueError("noise schedule must be linear or cosine")
        if self.loss not in {"epsilon_mse", "min_snr_5"}:
            raise ValueError("loss must be epsilon_mse or min_snr_5")
        if self.loss == "min_snr_5" and self.min_snr_gamma != 5.0:
            raise ValueError("the protocol fixes Min-SNR gamma=5")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")


class ExponentialMovingAverage:
    def __init__(self, module: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, value in module.state_dict().items():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        state = module.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}


class StructureAwareDDPM(nn.Module):
    """DDPM-v2 with explicit target transform and canonical UID sampling."""

    def __init__(
        self,
        denoiser: nn.Module,
        config: DiffusionConfig,
        target_transform: TargetTransform,
        preconditioner: Preconditioner | None = None,
    ):
        super().__init__()
        config.validate()
        self.denoiser = denoiser
        self.config = config
        self.target_transform = target_transform
        self.preconditioner = preconditioner
        if target_transform.mode != "absolute" and preconditioner is None:
            raise ValueError(f"{target_transform.mode} requires a frozen FullCov preconditioner")
        beta = make_beta_schedule(
            config.num_steps,
            config.schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        alpha = 1.0 - beta
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", torch.cumprod(alpha, dim=0))

    def _precondition(self, batch: DatasetBatch) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.preconditioner is None:
            return None, None
        mean, cholesky = self.preconditioner.predict(batch)
        return mean.to(batch.csi.device), cholesky.to(batch.csi.device)

    def q_sample(
        self, target: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_hat = self.alpha_hat[timestep].unsqueeze(1)
        return alpha_hat.sqrt() * target + (1.0 - alpha_hat).sqrt() * noise

    def training_loss(
        self,
        batch: DatasetBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch.validate()
        device_batch = batch.to(self.beta.device)
        count = len(device_batch.csi)
        timestep = torch.randint(
            0,
            self.config.num_steps,
            (count,),
            generator=generator,
            device=self.beta.device,
        )
        noise = torch.randn(
            device_batch.coordinates_cm.shape,
            generator=generator,
            device=self.beta.device,
            dtype=device_batch.coordinates_cm.dtype,
        )
        mean, cholesky = self._precondition(device_batch)
        target = self.target_transform.transform(
            device_batch.coordinates_cm, mean, cholesky
        )
        noisy = self.q_sample(target, timestep, noise)
        prediction = self.denoiser(noisy, timestep, device_batch)
        loss = (prediction - noise).square().mean(dim=1)
        if self.config.loss == "min_snr_5":
            loss = loss * min_snr_epsilon_weight(
                self.alpha_hat[timestep], self.config.min_snr_gamma
            )
        return loss.mean()

    @staticmethod
    def _canonical_noise(
        uids: np.ndarray,
        *,
        seed: int,
        dataset_id: str,
        model_id: str,
        role: str,
        sample_count: int,
    ) -> np.ndarray:
        return gaussian_base_noise(
            uids,
            training_seed=int(seed),
            stream_namespace=f"structure_aware_csi_ddpm_v2/{dataset_id}/{model_id}/{role}",
            paired_noise_group="structure_aware_csi_ddpm_v2_independent",
            protocol_rng_seed=20260801,
            sample_count=sample_count,
            noise_role=role,
        )

    @torch.no_grad()
    def sample(
        self,
        batch: DatasetBatch,
        *,
        training_seed: int,
        model_id: str,
        sample_count: int = 20,
    ) -> DistributionOutput:
        if not 2 <= sample_count <= 20:
            raise ValueError("posterior sample count must be in 2..20")
        batch.validate()
        device_batch = batch.to(self.beta.device)
        uids = batch.global_uids.detach().cpu().numpy().astype(np.int64)
        condition = self.denoiser.encode_condition(device_batch)
        count, condition_dim = condition.shape
        expanded_condition = condition[:, None, :].expand(count, sample_count, condition_dim)
        expanded_condition = expanded_condition.reshape(count * sample_count, condition_dim)
        initial = self._canonical_noise(
            uids,
            seed=training_seed,
            dataset_id=batch.dataset_id,
            model_id=model_id,
            role="initial",
            sample_count=sample_count,
        )
        current = torch.from_numpy(initial.reshape(-1, 2)).to(
            device=self.beta.device, dtype=device_batch.csi.dtype
        )
        for timestep_value in reversed(range(self.config.num_steps)):
            timestep = torch.full(
                (len(current),), timestep_value, dtype=torch.long, device=self.beta.device
            )
            predicted_noise = self.denoiser.forward_from_condition(
                current, timestep, expanded_condition
            )
            beta = self.beta[timestep_value]
            alpha = self.alpha[timestep_value]
            alpha_hat = self.alpha_hat[timestep_value]
            mean_step = (current - beta / torch.sqrt(1.0 - alpha_hat) * predicted_noise) / torch.sqrt(alpha)
            if timestep_value > 0:
                reverse = self._canonical_noise(
                    uids,
                    seed=training_seed,
                    dataset_id=batch.dataset_id,
                    model_id=model_id,
                    role=f"reverse_step_{timestep_value}",
                    sample_count=sample_count,
                )
                noise = torch.from_numpy(reverse.reshape(-1, 2)).to(
                    device=self.beta.device, dtype=current.dtype
                )
                current = mean_step + torch.sqrt(beta) * noise
            else:
                current = mean_step

        pre_mean, pre_cholesky = self._precondition(device_batch)
        if pre_mean is not None:
            pre_mean = pre_mean[:, None, :].expand(count, sample_count, 2).reshape(-1, 2)
            pre_cholesky = pre_cholesky[:, None, :, :].expand(
                count, sample_count, 2, 2
            ).reshape(-1, 2, 2)
        samples_cm = self.target_transform.inverse(
            current, pre_mean, pre_cholesky
        ).reshape(count, sample_count, 2)
        return DistributionOutput.from_samples(
            samples_cm.detach().cpu().numpy().astype(np.float32), uids
        )
