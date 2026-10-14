from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from probabilistic_rng import gaussian_base_noise

from .contracts import DatasetBatch, DistributionOutput
from .models import FeedForwardBlock, FlatCSIEncoder, ZeroAffine, trainable_parameter_count


PROTOCOL_ID = "opencsi_edm_conditional_v1"
LOG_PROB_STATUS = "unavailable_for_registered_edm_v1"
ALLOWED_SUBSETS = frozenset({"development_train_300k", "selection_validation_rp10"})
FORBIDDEN_SUBSETS = frozenset(
    {"validation", "calibration", "calibration_v2", "test", "external", "external_dataset"}
)

# RUN_013 A1 uses the matched flat CSI encoder selected by factories.py.
A1_FLAT_HIDDEN_DIM = 142
A1_TRAINABLE_PARAMETERS = 179_452


@dataclass(frozen=True)
class EDMConfig:
    sigma_data: float = 1.0
    log_sigma_mean: float = -1.2
    log_sigma_std: float = 1.2
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    noise_levels: int = 32
    rho: float = 7.0
    churn: float = 0.0
    posterior_samples: int = 20

    def validate(self) -> None:
        expected = {
            "sigma_data": 1.0,
            "log_sigma_mean": -1.2,
            "log_sigma_std": 1.2,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "noise_levels": 32,
            "rho": 7.0,
            "churn": 0.0,
            "posterior_samples": 20,
        }
        actual = {
            "sigma_data": float(self.sigma_data),
            "log_sigma_mean": float(self.log_sigma_mean),
            "log_sigma_std": float(self.log_sigma_std),
            "sigma_min": float(self.sigma_min),
            "sigma_max": float(self.sigma_max),
            "noise_levels": int(self.noise_levels),
            "rho": float(self.rho),
            "churn": float(self.churn),
            "posterior_samples": int(self.posterior_samples),
        }
        if actual != expected:
            raise ValueError(f"EDM v1 configuration drift: {actual}")

    @property
    def expected_heun_nfe(self) -> int:
        # Two evaluations for every nonterminal Heun update and one Euler
        # evaluation for the final sigma_min -> 0 update.
        return 2 * int(self.noise_levels) - 1


def registered_edm_config() -> EDMConfig:
    config = EDMConfig()
    config.validate()
    return config


def assert_edm_subset_allowed(subset: str) -> None:
    value = str(subset)
    if value not in ALLOWED_SUBSETS:
        raise PermissionError(
            f"EDM v1 may open only Development subsets {sorted(ALLOWED_SUBSETS)}; got {value!r}"
        )


def _sigma_column(sigma: torch.Tensor, *, rows: int | None = None) -> torch.Tensor:
    value = torch.as_tensor(sigma)
    if value.ndim == 0:
        value = value.reshape(1)
    if value.ndim != 1:
        raise ValueError("sigma must be scalar or one-dimensional")
    if rows is not None and len(value) not in {1, int(rows)}:
        raise ValueError(f"sigma must contain one or {rows} values")
    if not torch.isfinite(value).all() or torch.any(value <= 0):
        raise ValueError("sigma must be finite and strictly positive")
    return value[:, None]


def edm_preconditioning(
    sigma: torch.Tensor, sigma_data: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if float(sigma_data) <= 0.0:
        raise ValueError("sigma_data must be positive")
    sigma_value = _sigma_column(sigma)
    data = torch.as_tensor(float(sigma_data), device=sigma_value.device, dtype=sigma_value.dtype)
    denominator = sigma_value.square() + data.square()
    c_skip = data.square() / denominator
    c_out = sigma_value * data / torch.sqrt(denominator)
    c_in = torch.rsqrt(denominator)
    c_noise = torch.log(sigma_value) / 4.0
    return c_skip, c_out, c_in, c_noise


def edm_loss_weight(sigma: torch.Tensor, sigma_data: float = 1.0) -> torch.Tensor:
    sigma_value = _sigma_column(sigma)
    data = torch.as_tensor(float(sigma_data), device=sigma_value.device, dtype=sigma_value.dtype)
    return (sigma_value.square() + data.square()) / (sigma_value * data).square()


def sample_training_sigma(
    count: int,
    *,
    generator: torch.Generator | None,
    device: torch.device | str,
    dtype: torch.dtype,
    config: EDMConfig | None = None,
) -> torch.Tensor:
    settings = config or registered_edm_config()
    settings.validate()
    if int(count) < 1:
        raise ValueError("sigma sample count must be positive")
    normal = torch.randn(
        (int(count),), generator=generator, device=device, dtype=dtype
    )
    sigma = torch.exp(settings.log_sigma_mean + settings.log_sigma_std * normal)
    return sigma.clamp(min=settings.sigma_min, max=settings.sigma_max)


def karras_schedule(
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    config: EDMConfig | None = None,
) -> torch.Tensor:
    settings = config or registered_edm_config()
    settings.validate()
    ramp = torch.linspace(0.0, 1.0, settings.noise_levels, device=device, dtype=torch.float64)
    high = settings.sigma_max ** (1.0 / settings.rho)
    low = settings.sigma_min ** (1.0 / settings.rho)
    nonzero = (high + ramp * (low - high)).pow(settings.rho)
    schedule = torch.cat((nonzero, torch.zeros(1, device=device, dtype=torch.float64)))
    if len(schedule) != settings.noise_levels + 1:
        raise RuntimeError("Karras schedule length drift")
    if not torch.all(schedule[:-2] > schedule[1:-1]):
        raise RuntimeError("Karras schedule is not strictly decreasing")
    if float(schedule[0]) != settings.sigma_max or not torch.isclose(
        schedule[-2], torch.tensor(settings.sigma_min, dtype=schedule.dtype, device=schedule.device)
    ):
        raise RuntimeError("Karras schedule boundary drift")
    return schedule.to(dtype=dtype)


class ContinuousNoiseEmbedding(nn.Module):
    """Parameter-free continuous embedding for EDM c_noise=log(sigma)/4."""

    def __init__(self, dimension: int):
        super().__init__()
        if int(dimension) % 2:
            raise ValueError("noise embedding dimension must be even")
        self.dimension = int(dimension)

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(c_noise)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError("c_noise must be [B] or [B,1]")
        half = self.dimension // 2
        frequency = torch.exp(
            torch.arange(half, device=values.device, dtype=torch.float32)
            * -(math.log(10_000.0) / max(half - 1, 1))
        ).to(dtype=values.dtype)
        phase = values[:, None] * frequency[None, :]
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)


class EDMConditionalDenoiser(nn.Module):
    """RUN_013 A1 flat encoder, FiLM injections, and shared-scale backbone."""

    def __init__(
        self,
        condition_encoder: nn.Module,
        *,
        embedding_dim: int = 128,
        coordinate_dim: int = 2,
    ):
        super().__init__()
        if getattr(condition_encoder, "output_dim", None) != int(embedding_dim):
            raise ValueError("condition encoder output must equal embedding_dim")
        self.condition_encoder = condition_encoder
        self.embedding_dim = int(embedding_dim)
        self.coordinate_dim = int(coordinate_dim)
        d1, d2 = self.embedding_dim // 2, self.embedding_dim // 4
        self.sample_embedding = FeedForwardBlock(self.coordinate_dim, self.embedding_dim)
        self.noise_embedding = ContinuousNoiseEmbedding(self.embedding_dim)
        self.down1 = FeedForwardBlock(self.embedding_dim, d1)
        self.down2 = FeedForwardBlock(d1, d2)
        self.middle = FeedForwardBlock(d2, d2)
        self.up1 = FeedForwardBlock(2 * d2, d1)
        self.up2 = FeedForwardBlock(2 * d1, self.embedding_dim)
        self.output = nn.Linear(self.embedding_dim, self.coordinate_dim)
        self.mod_down1 = ZeroAffine(self.embedding_dim, d1)
        self.mod_down2 = ZeroAffine(self.embedding_dim, d2)
        self.mod_middle = ZeroAffine(self.embedding_dim, d2)
        self.mod_up1 = ZeroAffine(self.embedding_dim, d1)
        self.mod_up2 = ZeroAffine(self.embedding_dim, self.embedding_dim)

    def encode_condition(self, batch: DatasetBatch) -> torch.Tensor:
        return self.condition_encoder(batch.csi, batch.subcarrier_mask, batch.port_mask)

    def forward_from_condition(
        self,
        scaled_noisy_coordinate: torch.Tensor,
        c_noise: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        base = self.sample_embedding(scaled_noisy_coordinate) + self.noise_embedding(c_noise)
        down1 = self.mod_down1(self.down1(base), condition)
        down2 = self.mod_down2(self.down2(down1), condition)
        middle = self.mod_middle(self.middle(down2), condition)
        up1 = self.mod_up1(self.up1(torch.cat((middle, down2), dim=1)), condition)
        up2 = self.mod_up2(self.up2(torch.cat((up1, down1), dim=1)), condition)
        return self.output(up2)

    def forward(
        self, scaled_noisy_coordinate: torch.Tensor, c_noise: torch.Tensor, batch: DatasetBatch
    ) -> torch.Tensor:
        return self.forward_from_condition(
            scaled_noisy_coordinate, c_noise, self.encode_condition(batch)
        )


def seed_everything(seed: int) -> dict[str, Any]:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    return {
        "seed": value,
        "random_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_all_seeded": True,
    }


def build_registered_denoiser(seed: int) -> EDMConditionalDenoiser:
    seed_everything(int(seed))
    encoder = FlatCSIEncoder(
        subcarriers=39,
        ports=4,
        hidden_dim=A1_FLAT_HIDDEN_DIM,
        output_dim=128,
    )
    return EDMConditionalDenoiser(encoder, embedding_dim=128, coordinate_dim=2)


def parameter_gate(
    module: nn.Module, *, reference_parameters: int = A1_TRAINABLE_PARAMETERS
) -> dict[str, Any]:
    candidate = int(trainable_parameter_count(module))
    limit = int(math.floor(float(reference_parameters) * 1.05))
    return {
        "reference_model": "RUN_013_A1",
        "reference_parameters": int(reference_parameters),
        "candidate_parameters": candidate,
        "maximum_parameters": limit,
        "candidate_over_reference_ratio": candidate / float(reference_parameters),
        "passes_105_percent_gate": candidate <= limit,
    }


@dataclass
class PreparedEDMSampling:
    device_batch: DatasetBatch
    global_uids: np.ndarray
    expanded_condition: torch.Tensor
    initial_state: torch.Tensor
    sample_count: int


class EDMConditionalDiffusion(nn.Module):
    def __init__(self, denoiser: EDMConditionalDenoiser, target_transform: Any, config: EDMConfig | None = None):
        super().__init__()
        self.config = config or registered_edm_config()
        self.config.validate()
        self.denoiser = denoiser
        self.target_transform = target_transform
        if getattr(target_transform, "mode", None) != "absolute":
            raise ValueError("EDM v1 requires the absolute OpenCSI train-normalized target")
        self.register_buffer("device_anchor", torch.tensor(0.0, dtype=torch.float32))

    def denoise_from_condition(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        sigma_vector = torch.as_tensor(sigma, device=noisy.device, dtype=noisy.dtype)
        sigma_column = _sigma_column(sigma_vector, rows=len(noisy))
        if len(sigma_column) == 1 and len(noisy) != 1:
            sigma_column = sigma_column.expand(len(noisy), 1)
        c_skip, c_out, c_in, c_noise = edm_preconditioning(
            sigma_column[:, 0], self.config.sigma_data
        )
        prediction = self.denoiser.forward_from_condition(
            c_in * noisy, c_noise[:, 0], condition
        )
        return c_skip * noisy + c_out * prediction

    def denoise(self, noisy: torch.Tensor, sigma: torch.Tensor, batch: DatasetBatch) -> torch.Tensor:
        device_batch = batch.to(noisy.device)
        return self.denoise_from_condition(
            noisy, sigma, self.denoiser.encode_condition(device_batch)
        )

    def training_loss(
        self,
        batch: DatasetBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch.validate()
        device_batch = batch.to(self.device_anchor.device)
        target = self.target_transform.transform(device_batch.coordinates_cm, None, None)
        sigma = sample_training_sigma(
            len(target),
            generator=generator,
            device=target.device,
            dtype=target.dtype,
            config=self.config,
        )
        noise = torch.randn(
            target.shape,
            generator=generator,
            device=target.device,
            dtype=target.dtype,
        )
        noisy = target + sigma[:, None] * noise
        denoised = self.denoise(noisy, sigma, device_batch)
        weight = edm_loss_weight(sigma, self.config.sigma_data)
        loss = weight * (denoised - target).square()
        if not torch.isfinite(loss).all():
            raise FloatingPointError("EDM training loss produced NaN or Inf")
        return loss.mean()

    @staticmethod
    def _canonical_initial_noise(
        uids: np.ndarray,
        *,
        seed: int,
        dataset_id: str,
        model_id: str,
        sample_count: int,
    ) -> np.ndarray:
        return gaussian_base_noise(
            uids,
            training_seed=int(seed),
            stream_namespace=(
                f"{PROTOCOL_ID}/{dataset_id}/{model_id}/edm_heun_churn0_initial"
            ),
            paired_noise_group="opencsi_edm_conditional_v1_independent",
            protocol_rng_seed=20260810,
            sample_count=int(sample_count),
            noise_role="initial",
        )

    @torch.no_grad()
    def prepare_sampling(
        self,
        batch: DatasetBatch,
        *,
        training_seed: int,
        model_id: str,
        sample_count: int = 20,
    ) -> PreparedEDMSampling:
        if int(sample_count) != self.config.posterior_samples:
            raise ValueError("registered EDM v1 sampling requires S=20")
        batch.validate()
        device_batch = batch.to(self.device_anchor.device)
        uids = batch.global_uids.detach().cpu().numpy().astype(np.int64)
        condition = self.denoiser.encode_condition(device_batch)
        count, condition_dim = condition.shape
        expanded_condition = condition[:, None, :].expand(
            count, sample_count, condition_dim
        ).reshape(count * sample_count, condition_dim)
        initial = self._canonical_initial_noise(
            uids,
            seed=training_seed,
            dataset_id=batch.dataset_id,
            model_id=model_id,
            sample_count=sample_count,
        )
        current = torch.from_numpy(initial.reshape(-1, 2)).to(
            device=self.device_anchor.device, dtype=device_batch.csi.dtype
        )
        current = current * float(self.config.sigma_max)
        return PreparedEDMSampling(
            device_batch=device_batch,
            global_uids=uids,
            expanded_condition=expanded_condition,
            initial_state=current,
            sample_count=int(sample_count),
        )

    @torch.no_grad()
    def sampling_core(self, prepared: PreparedEDMSampling) -> tuple[torch.Tensor, dict[str, int]]:
        schedule = karras_schedule(
            device=self.device_anchor.device,
            dtype=prepared.initial_state.dtype,
            config=self.config,
        )
        current = prepared.initial_state.clone()
        nfe = 0
        solver_steps = 0
        for index in range(self.config.noise_levels):
            sigma = schedule[index]
            sigma_next = schedule[index + 1]
            sigma_batch = torch.full(
                (len(current),), float(sigma), device=current.device, dtype=current.dtype
            )
            denoised = self.denoise_from_condition(
                current, sigma_batch, prepared.expanded_condition
            )
            nfe += 1
            derivative = (current - denoised) / sigma
            delta = sigma_next - sigma
            euler = current + delta * derivative
            if float(sigma_next) != 0.0:
                next_batch = torch.full(
                    (len(current),),
                    float(sigma_next),
                    device=current.device,
                    dtype=current.dtype,
                )
                denoised_next = self.denoise_from_condition(
                    euler, next_batch, prepared.expanded_condition
                )
                nfe += 1
                derivative_next = (euler - denoised_next) / sigma_next
                current = current + delta * (derivative + derivative_next) / 2.0
            else:
                current = euler
            solver_steps += 1
        trace = {
            "noise_levels": int(self.config.noise_levels),
            "solver_steps": int(solver_steps),
            "nfe": int(nfe),
        }
        if solver_steps != self.config.noise_levels or nfe != self.config.expected_heun_nfe:
            raise RuntimeError(f"EDM Heun NFE drift: {trace}")
        if not torch.isfinite(current).all():
            raise FloatingPointError("EDM Heun sampler produced NaN or Inf")
        return current, trace

    @torch.no_grad()
    def finalize_sampling(
        self, prepared: PreparedEDMSampling, normalized: torch.Tensor
    ) -> DistributionOutput:
        count = len(prepared.global_uids)
        samples_cm = self.target_transform.inverse(
            normalized, None, None
        ).reshape(count, prepared.sample_count, 2)
        return DistributionOutput.from_samples(
            samples_cm.detach().cpu().numpy().astype(np.float32), prepared.global_uids
        )

    @torch.no_grad()
    def sample_with_trace(
        self,
        batch: DatasetBatch,
        *,
        training_seed: int,
        model_id: str,
        sample_count: int = 20,
    ) -> tuple[DistributionOutput, dict[str, int]]:
        prepared = self.prepare_sampling(
            batch,
            training_seed=training_seed,
            model_id=model_id,
            sample_count=sample_count,
        )
        normalized, trace = self.sampling_core(prepared)
        return self.finalize_sampling(prepared, normalized), trace

    @torch.no_grad()
    def sample(
        self,
        batch: DatasetBatch,
        *,
        training_seed: int,
        model_id: str,
        sample_count: int = 20,
    ) -> DistributionOutput:
        output, _ = self.sample_with_trace(
            batch,
            training_seed=training_seed,
            model_id=model_id,
            sample_count=sample_count,
        )
        return output


def checkpoint_key(metrics: Mapping[str, float], epoch: int) -> tuple[float, float, int]:
    required = ("energy_score_u20_cm", "mean_cm")
    if any(name not in metrics for name in required):
        raise KeyError("checkpoint metrics must include ES_U20 and Mean")
    key = (float(metrics[required[0]]), float(metrics[required[1]]), int(epoch))
    if not all(math.isfinite(value) for value in key[:2]) or key[2] < 1:
        raise ValueError("checkpoint key must be finite with positive epoch")
    return key


def macro_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if len(rows) != 3:
        raise ValueError("EDM v1 comparison requires exactly three seeds")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("metric rows must have identical ordered keys")
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def development_gate(
    candidate_rows: Sequence[Mapping[str, float]],
    safe312_rows: Sequence[Mapping[str, float]],
    *,
    integrity_passed: bool,
) -> dict[str, Any]:
    if len(candidate_rows) != 3 or len(safe312_rows) != 3:
        raise ValueError("EDM v1 gate requires exactly three paired seeds")
    candidate = macro_metrics(candidate_rows)
    control = macro_metrics(safe312_rows)
    improvement = {
        name: (control[name] - candidate[name]) / control[name]
        for name in ("mean_cm", "rp_macro_mean_cm", "energy_score_u20_cm")
    }
    mean_directions = [
        float(candidate_row["mean_cm"]) < float(control_row["mean_cm"])
        for candidate_row, control_row in zip(candidate_rows, safe312_rows, strict=True)
    ]
    es_directions = [
        float(candidate_row["energy_score_u20_cm"])
        < float(control_row["energy_score_u20_cm"])
        for candidate_row, control_row in zip(candidate_rows, safe312_rows, strict=True)
    ]
    checks = {
        "mean_improvement_at_least_2_percent": improvement["mean_cm"] >= 0.02,
        "rp_macro_mean_improvement_at_least_2_percent": improvement["rp_macro_mean_cm"] >= 0.02,
        "energy_score_u20_improvement_at_least_1_percent": improvement["energy_score_u20_cm"] >= 0.01,
        "mean_direction_favorable_3_of_3": all(mean_directions),
        "energy_score_direction_favorable_3_of_3": all(es_directions),
        "macro_p90_not_worse": candidate["p90_cm"] <= control["p90_cm"],
        "macro_p99_not_worse": candidate["p99_cm"] <= control["p99_cm"],
        "integrity_uid_rng_access_hashes_passed": bool(integrity_passed),
    }
    return {
        "candidate_macro": candidate,
        "safe312_macro": control,
        "relative_improvement": improvement,
        "mean_seed_directions": mean_directions,
        "energy_score_seed_directions": es_directions,
        "checks": checks,
        "eligible_to_request_next_stage_authorization": all(checks.values()),
        "validation_authorized": False,
    }
