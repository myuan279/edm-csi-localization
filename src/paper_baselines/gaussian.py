"""Source protocol: heteroscedastic_gaussian_mlp_v1.

Derived method: a single-forward full-covariance Gaussian MLP for normalized
two-dimensional coordinates.  This module is intentionally independent of the
frozen point-estimate baselines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


HIDDEN_DIMENSIONS = (128, 64, 32, 32, 64, 128)
SIGMA_FLOOR = 1e-4
POSTERIOR_SAMPLES = 20
METHOD_ID = "FullCov-HGMLP"


class FullCovarianceGaussianMLP(nn.Module):
    """MLP-02 backbone with a five-value (mean plus Cholesky) output head."""

    def __init__(self, in_dim: int = 312, sigma_floor: float = SIGMA_FLOOR):
        super().__init__()
        dims = (in_dim, *HIDDEN_DIMENSIONS)
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(left, right), nn.GELU(), nn.LayerNorm(right)))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(HIDDEN_DIMENSIONS[-1], 5)
        self.sigma_floor = float(sigma_floor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def distribution_parameters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return output_to_distribution(self(x), self.sigma_floor)


class HomoscedasticGaussianMLP(nn.Module):
    """MLP-02 mean network plus one learned global full 2D covariance."""

    def __init__(self, in_dim: int = 312, sigma_floor: float = SIGMA_FLOOR):
        super().__init__()
        dims = (in_dim, *HIDDEN_DIMENSIONS)
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(left, right), nn.GELU(), nn.LayerNorm(right)))
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(HIDDEN_DIMENSIONS[-1], 2)
        # softplus(inv_softplus(1-epsilon)) + epsilon == 1 at initialization.
        raw_identity = math.log(math.expm1(1.0 - float(sigma_floor)))
        self.global_cholesky_raw = nn.Parameter(torch.tensor([raw_identity, 0.0, raw_identity], dtype=torch.float32))
        self.sigma_floor = float(sigma_floor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.backbone(x))

    def distribution_parameters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self(x)
        raw = self.global_cholesky_raw.to(dtype=mu.dtype).unsqueeze(0).expand(mu.shape[0], -1)
        return mean_and_cholesky(mu, raw, self.sigma_floor)


def output_to_distribution(output: torch.Tensor, sigma_floor: float = SIGMA_FLOOR) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean, lower Cholesky factor, and covariance; diagonals are positive."""
    if output.ndim != 2 or output.shape[1] != 5:
        raise ValueError("expected [batch, 5] Gaussian output")
    return mean_and_cholesky(output[:, :2], output[:, 2:], sigma_floor)


def mean_and_cholesky(mu: torch.Tensor, raw_cholesky: torch.Tensor, sigma_floor: float = SIGMA_FLOOR) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a full covariance from a mean and [raw_l11, l21, raw_l22]."""
    if mu.ndim != 2 or mu.shape[1] != 2 or raw_cholesky.shape != mu.shape[:1] + (3,):
        raise ValueError("expected mean [batch,2] and Cholesky raw values [batch,3]")
    l11 = torch.nn.functional.softplus(raw_cholesky[:, 0]) + float(sigma_floor)
    l21 = raw_cholesky[:, 1]
    l22 = torch.nn.functional.softplus(raw_cholesky[:, 2]) + float(sigma_floor)
    zeros = torch.zeros_like(l11)
    chol = torch.stack((torch.stack((l11, zeros), dim=1), torch.stack((l21, l22), dim=1)), dim=1)
    covariance = chol @ chol.transpose(-1, -2)
    return mu, chol, covariance


def gaussian_nll_terms(mu: torch.Tensor, chol: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample NLL, Mahalanobis, and log-determinant without an inverse."""
    if not (torch.isfinite(mu).all() and torch.isfinite(chol).all() and torch.isfinite(target).all()):
        raise FloatingPointError("non-finite Gaussian NLL input")
    diagonal = torch.diagonal(chol, dim1=-2, dim2=-1)
    if torch.any(diagonal <= 0) or torch.any(diagonal > 1e6):
        raise FloatingPointError("invalid Cholesky diagonal")
    delta = (target - mu).unsqueeze(-1)
    solved = torch.linalg.solve_triangular(chol, delta, upper=False)
    mahalanobis = torch.sum(torch.square(solved.squeeze(-1)), dim=1)
    logdet = 2.0 * torch.sum(torch.log(diagonal), dim=1)
    nll = 0.5 * (mahalanobis + logdet + 2.0 * math.log(2.0 * math.pi))
    if not torch.isfinite(nll).all():
        raise FloatingPointError("non-finite Gaussian NLL")
    return nll, mahalanobis, logdet


def gaussian_nll(mu: torch.Tensor, chol: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return gaussian_nll_terms(mu, chol, target)[0].mean()


def parameter_count(model: nn.Module) -> int:
    return sum(item.numel() for item in model.parameters())


def backbone_parameter_count(model: FullCovarianceGaussianMLP) -> int:
    return sum(item.numel() for item in model.backbone.parameters())


def _stage_code(stage: str) -> int:
    # Fixed FNV-1a code makes the stage a required, reproducible key component.
    value = 2166136261
    for byte in stage.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0x7FFFFFFF
    return value


def indexed_standard_normal(snapshot_ids: torch.Tensor, sample_ids: torch.Tensor, *, method: str, seed: int, stage: str, dtype: torch.dtype) -> torch.Tensor:
    """Stateless N(0, I_2), addressed by method, seed, snapshot, sample, stage."""
    if snapshot_ids.ndim != 1 or sample_ids.ndim != 1 or snapshot_ids.shape != sample_ids.shape:
        raise ValueError("snapshot_ids and sample_ids must be same-length vectors")
    prime = 2_147_483_647
    method_code = _stage_code(method)
    stage_code = _stage_code(stage)
    key = (snapshot_ids.to(torch.int64) * 1_000_003 + sample_ids.to(torch.int64) * 1_299_709 + int(seed) * 97_531 + method_code * 17 + stage_code).remainder(prime)
    def uniform(stream: int) -> torch.Tensor:
        value = (key + (stream + 1) * 104_729).remainder(prime)
        value = (value * value + 1_234_567 + stream * 31).remainder(prime)
        value = (value * 48_271 + 17 + stream).remainder(prime)
        return (value.to(dtype=dtype) + 0.5) / float(prime)
    one, two = uniform(0), uniform(1)
    radius = torch.sqrt(-2.0 * torch.log(one))
    angle = 2.0 * math.pi * two
    return torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)


def gaussian_samples(mu: torch.Tensor, chol: torch.Tensor, snapshot_ids: torch.Tensor, *, method: str, seed: int, stage: str, sample_count: int = POSTERIOR_SAMPLES) -> torch.Tensor:
    """Return [batch, S, 2] indexed samples without using global RNG state."""
    if sample_count != POSTERIOR_SAMPLES:
        raise ValueError("the source protocol freezes S=20")
    parts = []
    for sample_id in range(sample_count):
        ids = torch.full_like(snapshot_ids, sample_id)
        eps = indexed_standard_normal(snapshot_ids, ids, method=method, seed=seed, stage=stage, dtype=mu.dtype)
        parts.append(mu + torch.bmm(chol, eps.unsqueeze(-1)).squeeze(-1))
    return torch.stack(parts, dim=1)


def normalized_to_cm(mu_normalized: np.ndarray, covariance_normalized: np.ndarray, pos_mean_cm: np.ndarray, pos_std_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply y_cm = A y_norm + b and Sigma_cm = A Sigma_norm A.T explicitly."""
    mean = np.asarray(pos_mean_cm, dtype=np.float64)
    scale = np.asarray(pos_std_cm, dtype=np.float64)
    if mean.shape != (2,) or scale.shape != (2,) or np.any(scale <= 0):
        raise ValueError("position normalization must be two positive scales")
    matrix = np.diag(scale)
    mu_cm = np.asarray(mu_normalized, dtype=np.float64) @ matrix.T + mean
    covariance_cm2 = matrix @ np.asarray(covariance_normalized, dtype=np.float64) @ matrix.T
    return mu_cm, covariance_cm2


def energy_score_s20(samples_cm: np.ndarray, truth_cm: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples_cm, dtype=np.float64)
    truth = np.asarray(truth_cm, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1:] != (POSTERIOR_SAMPLES, 2):
        raise ValueError("expected [N, 20, 2] samples")
    # Chunking bounds the full-Validation working set. The arithmetic is still
    # float64 and each snapshot uses the exact locked S^2 estimator.
    result = np.empty(len(samples), dtype=np.float64)
    for start in range(0, len(samples), 4096):
        stop = min(start + 4096, len(samples))
        current = samples[start:stop]
        first = np.linalg.norm(current - truth[start:stop, None, :], axis=2).mean(axis=1)
        pair = np.linalg.norm(current[:, :, None, :] - current[:, None, :, :], axis=3)
        # The locked S=20 definition includes the zero diagonal and uses S^2.
        second = pair.sum(axis=(1, 2)) / float(2 * POSTERIOR_SAMPLES * POSTERIOR_SAMPLES)
        result[start:stop] = first - second
    return result


# Compatibility alias.  The locked protocol records the formula above, not the
# older unbiased denominator used by historical diffusion experiments.
unbiased_energy_score = energy_score_s20


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, start, count = np.unique(values[order], return_index=True, return_counts=True)
    del unique
    for first, size in zip(start, count):
        if size > 1:
            ranks[order[first:first + size]] = first + (size - 1) / 2.0
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a, b = _rankdata(left), _rankdata(right)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _binary_auroc(score: np.ndarray, label: np.ndarray) -> float | None:
    positive = int(np.sum(label))
    negative = len(label) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = _rankdata(score) + 1.0
    return float((ranks[label].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


@dataclass(frozen=True)
class MetricArtifacts:
    point: dict[str, Any]
    probabilistic: dict[str, Any]
    risk_rows: list[dict[str, float]]


def metric_artifacts(mu_cm: np.ndarray, covariance_cm2: np.ndarray, samples_cm: np.ndarray, truth_cm: np.ndarray, rp_ids: np.ndarray) -> MetricArtifacts:
    """Float64 formal metrics from persisted float32 artifacts only."""
    mu, cov, samples, truth = (np.asarray(value, dtype=np.float64) for value in (mu_cm, covariance_cm2, samples_cm, truth_cm))
    ids = np.asarray(rp_ids)
    if not (np.isfinite(mu).all() and np.isfinite(cov).all() and np.isfinite(samples).all()):
        raise FloatingPointError("non-finite persisted artifact")
    error = np.linalg.norm(mu - truth, axis=1)
    rp_mean = [float(error[ids == current].mean()) for current in np.unique(ids)]
    point = {"mean_cm": float(error.mean()), "median_cm": float(np.median(error)), "rmse_cm": float(np.sqrt(np.mean(np.square(error)))), "p90_cm": float(np.percentile(error, 90)), "p95_cm": float(np.percentile(error, 95)), "p99_cm": float(np.percentile(error, 99)), "maximum_cm": float(error.max()), "rp_macro_mean_cm": float(np.mean(rp_mean))}
    chol = np.linalg.cholesky(cov)
    solved = np.linalg.solve(chol, (truth - mu)[..., None]).squeeze(-1)
    mahal = np.sum(np.square(solved), axis=1)
    logdet = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    nll = 0.5 * (mahal + logdet + 2.0 * math.log(2.0 * math.pi))
    trace = np.trace(cov, axis1=1, axis2=2)
    determinant = np.linalg.det(cov)
    if np.any(determinant <= 0) or not np.isfinite(nll).all():
        raise FloatingPointError("non-SPD covariance artifact")
    coverage: dict[str, float] = {}
    areas: dict[str, float] = {}
    for nominal in (0.50, 0.80, 0.90, 0.95):
        threshold = -2.0 * math.log(1.0 - nominal)  # chi-square df=2 quantile
        coverage[f"raw_uncalibrated_coverage_{int(nominal * 100)}"] = float(np.mean(mahal <= threshold))
        areas[f"ellipse_area_cm2_{int(nominal * 100)}"] = float(np.mean(math.pi * threshold * np.sqrt(determinant)))
    order = np.argsort(trace, kind="mergesort")
    ordered_error = error[order]
    prefix_count = np.arange(1, len(error) + 1, dtype=np.float64)
    prefix_risk = np.cumsum(ordered_error, dtype=np.float64) / prefix_count
    prefix_coverage = prefix_count / float(len(error))
    risk_rows = [
        {"coverage": float(coverage), "risk_mean_cm": float(risk)}
        for coverage, risk in zip(prefix_coverage, prefix_risk)
    ]
    sharpness = np.sqrt(trace)
    # 100 cm is frozen before Development data access by the protocol lock.
    probabilistic: dict[str, Any] = {"energy_score": float(energy_score_s20(samples, truth).mean()), "gaussian_nll": float(nll.mean()), "mean_covariance_trace_cm2": float(trace.mean()), "mean_covariance_determinant_cm4": float(determinant.mean()), "sharpness_sqrt_trace_cm_mean": float(sharpness.mean()), "sharpness_sqrt_trace_cm_median": float(np.median(sharpness)), "uncertainty_error_spearman": _spearman(sharpness, error), "high_error_threshold_cm": 100.0, "high_error_auroc": _binary_auroc(sharpness, error >= 100.0), "mahalanobis": {"mean": float(mahal.mean()), "median": float(np.median(mahal)), "p90": float(np.percentile(mahal, 90)), "p95": float(np.percentile(mahal, 95)), "p99": float(np.percentile(mahal, 99)), "minimum_eigenvalue": float(np.linalg.eigvalsh(cov).min())}, "risk_coverage_prefix_count": len(risk_rows), "aurc": float(np.trapz(prefix_risk, prefix_coverage))}
    probabilistic.update(coverage)
    probabilistic.update(areas)
    return MetricArtifacts(point=point, probabilistic=probabilistic, risk_rows=risk_rows)
