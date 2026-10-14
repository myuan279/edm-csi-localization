"""Numerical primitives for probabilistic localization quality metrics.

All formal artifact recomputations are performed in float64.  The functions in
this module are independent from the frozen FullCov evaluator so that the audit
does not merely call the implementation it is intended to check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


POSTERIOR_SAMPLES = 20
CHI2_DF2_THRESHOLDS = {
    50: 1.3862943611198906,
    80: 3.2188758248682006,
    90: 4.605170185988092,
    95: 5.991464547107979,
}
SUMMARY_PERCENTILES = (1.0, 5.0, 50.0, 95.0, 99.0)


def _as_float64(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _validate_gaussian_arrays(
    mu: np.ndarray, chol: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = _as_float64(mu)
    factor = _as_float64(chol)
    target = _as_float64(truth)
    if mean.ndim != 2 or mean.shape[1] != 2:
        raise ValueError("mu must have shape [N,2]")
    if factor.shape != (len(mean), 2, 2) or target.shape != mean.shape:
        raise ValueError("chol must be [N,2,2] and truth must match mu")
    if not (
        np.isfinite(mean).all()
        and np.isfinite(factor).all()
        and np.isfinite(target).all()
    ):
        raise FloatingPointError("non-finite Gaussian input")
    if np.any(factor[:, 0, 1] != 0.0):
        raise ValueError("Cholesky factor must be lower triangular")
    diagonal = np.diagonal(factor, axis1=1, axis2=2)
    if np.any(diagonal <= 0.0):
        raise FloatingPointError("Cholesky diagonal is not strictly positive")
    return mean, factor, target


def gaussian_nll_decomposition(
    mu: np.ndarray, chol: np.ndarray, truth: np.ndarray
) -> dict[str, np.ndarray]:
    """Decompose the two-dimensional Gaussian NLL without a matrix inverse."""
    mean, factor, target = _validate_gaussian_arrays(mu, chol, truth)
    residual = target - mean
    z0 = residual[:, 0] / factor[:, 0, 0]
    z1 = (residual[:, 1] - factor[:, 1, 0] * z0) / factor[:, 1, 1]
    mahalanobis = np.square(z0) + np.square(z1)
    half_mahalanobis = 0.5 * mahalanobis
    logdet_l = np.log(factor[:, 0, 0]) + np.log(factor[:, 1, 1])
    log_2pi = np.full(len(mean), math.log(2.0 * math.pi), dtype=np.float64)
    nll = half_mahalanobis + logdet_l + log_2pi
    if not np.isfinite(nll).all():
        raise FloatingPointError("non-finite decomposed NLL")
    return {
        "half_mahalanobis": half_mahalanobis,
        "mahalanobis": mahalanobis,
        "logdet_l": logdet_l,
        "log_2pi": log_2pi,
        "nll": nll,
    }


def covariance_to_cholesky(covariance: np.ndarray) -> np.ndarray:
    cov = _as_float64(covariance)
    if cov.ndim != 3 or cov.shape[1:] != (2, 2):
        raise ValueError("covariance must have shape [N,2,2]")
    if not np.isfinite(cov).all():
        raise FloatingPointError("non-finite covariance")
    return np.linalg.cholesky(cov)


def reference_nll_from_covariance(
    mu: np.ndarray, covariance: np.ndarray, truth: np.ndarray
) -> dict[str, np.ndarray]:
    """Independent covariance-form reference using solve and slogdet."""
    mean = _as_float64(mu)
    cov = _as_float64(covariance)
    target = _as_float64(truth)
    if cov.shape != (len(mean), 2, 2) or target.shape != mean.shape:
        raise ValueError("Gaussian reference array shape mismatch")
    residual = target - mean
    solved = np.linalg.solve(cov, residual[..., None]).squeeze(-1)
    mahalanobis = np.einsum("ni,ni->n", residual, solved)
    sign, logdet = np.linalg.slogdet(cov)
    if np.any(sign <= 0):
        raise FloatingPointError("reference covariance is not SPD")
    nll = 0.5 * (
        mahalanobis + logdet + 2.0 * math.log(2.0 * math.pi)
    )
    return {"mahalanobis": mahalanobis, "logdet_sigma": logdet, "nll": nll}


def affine_gaussian_audit(
    mu_normalized: np.ndarray,
    chol_normalized: np.ndarray,
    truth_normalized: np.ndarray,
    scale: np.ndarray,
    translation: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Check NLL and Mahalanobis identities under y_cm=A y_norm+d."""
    mean, factor, target = _validate_gaussian_arrays(
        mu_normalized, chol_normalized, truth_normalized
    )
    scale64 = _as_float64(scale)
    offset64 = _as_float64(translation)
    if scale64.shape != (2,) or offset64.shape != (2,) or np.any(scale64 <= 0):
        raise ValueError("affine scale/translation must be positive [2] vectors")
    matrix = np.diag(scale64)
    mu_cm = mean @ matrix.T + offset64
    truth_cm = target @ matrix.T + offset64
    chol_cm = matrix[None, :, :] @ factor
    normalized = gaussian_nll_decomposition(mean, factor, target)
    physical = gaussian_nll_decomposition(mu_cm, chol_cm, truth_cm)
    expected_shift = float(np.log(scale64).sum())
    return {
        "nll_normalized": normalized["nll"],
        "nll_cm": physical["nll"],
        "nll_shift_error": physical["nll"]
        - normalized["nll"]
        - expected_shift,
        "mahalanobis_error": physical["mahalanobis"]
        - normalized["mahalanobis"],
        "log_abs_det_a": expected_shift,
        "mu_cm": mu_cm,
        "truth_cm": truth_cm,
        "chol_cm": chol_cm,
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    array = _as_float64(values).reshape(-1)
    if len(array) == 0 or not np.isfinite(array).all():
        raise FloatingPointError("summary requires finite non-empty values")
    result = {"min": float(array.min())}
    for percentile in SUMMARY_PERCENTILES:
        key = "median" if percentile == 50.0 else f"p{int(percentile)}"
        result[key] = float(np.percentile(array, percentile))
    result["max"] = float(array.max())
    return result


def absolute_error_summary(values: np.ndarray) -> dict[str, float]:
    return summarize(np.abs(_as_float64(values)))


def covariance_spectrum_audit(covariance: np.ndarray) -> dict[str, Any]:
    cov = _as_float64(covariance)
    if cov.ndim != 3 or cov.shape[1:] != (2, 2):
        raise ValueError("covariance must be [N,2,2]")
    finite_rows = np.isfinite(cov).all(axis=(1, 2))
    safe = cov[finite_rows]
    result: dict[str, Any] = {
        "sample_count": int(len(cov)),
        "nonfinite_count": int((~finite_rows).sum()),
        "numeric_overflow_count": int(np.isinf(cov).any(axis=(1, 2)).sum()),
    }
    if len(safe) == 0:
        raise FloatingPointError("no finite covariance rows")
    eigenvalues = np.linalg.eigvalsh(safe)
    minimum = eigenvalues[:, 0]
    maximum = eigenvalues[:, 1]
    determinant = np.linalg.det(safe)
    trace = np.trace(safe, axis1=1, axis2=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = maximum / minimum
        correlation = safe[:, 0, 1] / np.sqrt(
            safe[:, 0, 0] * safe[:, 1, 1]
        )
    invalid_derived = ~(
        np.isfinite(condition)
        & np.isfinite(correlation)
        & np.isfinite(determinant)
        & np.isfinite(trace)
    )
    result.update(
        {
            "non_spd_count": int((minimum <= 0.0).sum()),
            "invalid_derived_count": int(invalid_derived.sum()),
            "minimum_eigenvalue": summarize(minimum),
            "maximum_eigenvalue": summarize(maximum),
            "condition_number": summarize(condition[np.isfinite(condition)]),
            "trace": summarize(trace),
            "determinant": summarize(determinant),
            "correlation": summarize(correlation[np.isfinite(correlation)]),
        }
    )
    return result


def sigma_floor_audit(
    cholesky_normalized: np.ndarray, epsilon: float = 1e-4
) -> dict[str, Any]:
    factor = _as_float64(cholesky_normalized)
    if factor.ndim != 3 or factor.shape[1:] != (2, 2):
        raise ValueError("Cholesky array must be [N,2,2]")
    diagonal = np.diagonal(factor, axis1=1, axis2=2)
    thresholds = {
        "epsilon_plus_1e-6": float(epsilon + 1e-6),
        "1.01_epsilon": float(1.01 * epsilon),
        "1.1_epsilon": float(1.1 * epsilon),
        "2_epsilon": float(2.0 * epsilon),
    }
    output: dict[str, Any] = {
        "epsilon": float(epsilon),
        "strictly_below_epsilon_count": int((diagonal < epsilon).sum()),
        "diagonals": {},
    }
    for column, name in enumerate(("L11", "L22")):
        values = diagonal[:, column]
        records: dict[str, Any] = {"summary": summarize(values)}
        for label, threshold in thresholds.items():
            count = int(np.sum(values <= threshold))
            records[label] = {
                "threshold": threshold,
                "count": count,
                "ratio": float(count / len(values)),
            }
        output["diagonals"][name] = records
    return output


def coverage_from_mahalanobis(
    mahalanobis: np.ndarray,
) -> dict[str, float]:
    values = _as_float64(mahalanobis)
    return {
        f"coverage_{nominal}": float(np.mean(values <= threshold))
        for nominal, threshold in CHI2_DF2_THRESHOLDS.items()
    }


def energy_scores_u_v(
    samples_cm: np.ndarray, truth_cm: np.ndarray, chunk_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    samples = _as_float64(samples_cm)
    truth = _as_float64(truth_cm)
    if (
        samples.ndim != 3
        or samples.shape[1:] != (POSTERIOR_SAMPLES, 2)
        or truth.shape != (len(samples), 2)
    ):
        raise ValueError("expected samples [N,20,2] and truth [N,2]")
    u = np.empty(len(samples), dtype=np.float64)
    v = np.empty(len(samples), dtype=np.float64)
    s = POSTERIOR_SAMPLES
    for start in range(0, len(samples), chunk_size):
        stop = min(start + chunk_size, len(samples))
        current = samples[start:stop]
        first = np.linalg.norm(
            current - truth[start:stop, None, :], axis=2
        ).mean(axis=1)
        pair = np.linalg.norm(
            current[:, :, None, :] - current[:, None, :, :], axis=3
        )
        off_diagonal_sum = pair.sum(axis=(1, 2))
        v[start:stop] = first - off_diagonal_sum / float(2 * s * s)
        u[start:stop] = first - off_diagonal_sum / float(2 * s * (s - 1))
    return u, v


def sample_mean_covariance(
    samples_cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = _as_float64(samples_cm)
    if samples.ndim != 3 or samples.shape[1:] != (POSTERIOR_SAMPLES, 2):
        raise ValueError("samples must be [N,20,2]")
    mean = samples.mean(axis=1)
    centered = samples - mean[:, None, :]
    covariance = np.einsum("nsi,nsj->nij", centered, centered)
    covariance /= float(POSTERIOR_SAMPLES - 1)
    return mean, covariance


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = _as_float64(values)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    rank_left = _rankdata(left)
    rank_right = _rankdata(right)
    if np.std(rank_left) == 0.0 or np.std(rank_right) == 0.0:
        return None
    return float(np.corrcoef(rank_left, rank_right)[0, 1])


def binary_auroc(score: np.ndarray, label: np.ndarray) -> float | None:
    values = _as_float64(score)
    labels = np.asarray(label, dtype=bool)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return None
    ranks = _rankdata(values) + 1.0
    return float(
        (
            ranks[labels].sum()
            - positive * (positive + 1) / 2.0
        )
        / (positive * negative)
    )


def _point_metrics(error: np.ndarray, rp_ids: np.ndarray) -> dict[str, float]:
    values = _as_float64(error)
    ids = np.asarray(rp_ids)
    rp_means = [values[ids == rp].mean() for rp in np.unique(ids)]
    return {
        "mean_cm": float(values.mean()),
        "median_cm": float(np.median(values)),
        "rmse_cm": float(np.sqrt(np.mean(np.square(values)))),
        "p90_cm": float(np.percentile(values, 90)),
        "p95_cm": float(np.percentile(values, 95)),
        "p99_cm": float(np.percentile(values, 99)),
        "maximum_cm": float(values.max()),
        "rp_macro_mean_cm": float(np.mean(rp_means)),
    }


@dataclass(frozen=True)
class UnifiedMetrics:
    point: dict[str, Any]
    probabilistic: dict[str, Any]
    arrays: dict[str, np.ndarray]
    risk_curve: dict[str, np.ndarray] | None


def unified_metrics(
    *,
    mu_cm: np.ndarray,
    truth_cm: np.ndarray,
    rp_ids: np.ndarray,
    snapshot_ids: np.ndarray,
    covariance_cm2: np.ndarray | None,
    samples_cm: np.ndarray | None,
    model_kind: str,
) -> UnifiedMetrics:
    """Compute the frozen cross-model metric definition."""
    mean = _as_float64(mu_cm)
    truth = _as_float64(truth_cm)
    ids = np.asarray(rp_ids)
    snapshots = np.asarray(snapshot_ids, dtype=np.int64)
    if (
        mean.shape != truth.shape
        or mean.ndim != 2
        or mean.shape[1] != 2
        or ids.shape != (len(mean),)
        or snapshots.shape != (len(mean),)
    ):
        raise ValueError("unified metric alignment shape mismatch")
    if len(np.unique(snapshots)) != len(snapshots):
        raise ValueError("duplicate global snapshot IDs")
    error = np.linalg.norm(mean - truth, axis=1)
    point = _point_metrics(error, ids)
    if model_kind == "mlp_degenerate":
        zeros = np.zeros(len(mean), dtype=np.float64)
        probabilistic = {
            "forecast_type": "degenerate deterministic forecast",
            "energy_score_u20": float(error.mean()),
            "energy_score_v20": float(error.mean()),
            "uncertainty_measure": "constant zero",
            "uncertainty_error_spearman": None,
            "high_error_auroc": 0.5,
            "high_error_auroc_interpretation": (
                "constant-score convention; not discrimination ability"
            ),
            "aurc": None,
        }
        return UnifiedMetrics(
            point,
            probabilistic,
            {
                "error_cm": error,
                "energy_score_u20": error.copy(),
                "energy_score_v20": error.copy(),
                "uncertainty_trace_cm2": zeros,
            },
            None,
        )
    if covariance_cm2 is None or samples_cm is None:
        raise ValueError("probabilistic forecast requires covariance and samples")
    covariance = _as_float64(covariance_cm2)
    if covariance.shape != (len(mean), 2, 2):
        raise ValueError("covariance alignment mismatch")
    energy_u, energy_v = energy_scores_u_v(samples_cm, truth)
    trace = np.trace(covariance, axis1=1, axis2=2)
    determinant = np.linalg.det(covariance)
    if (
        np.any(np.linalg.eigvalsh(covariance)[:, 0] <= 0)
        or not np.isfinite(trace).all()
        or not np.isfinite(determinant).all()
    ):
        raise FloatingPointError("unified covariance is not finite SPD")
    chol = np.linalg.cholesky(covariance)
    nll_terms = gaussian_nll_decomposition(mean, chol, truth)
    mahalanobis = nll_terms["mahalanobis"]
    order = np.lexsort((snapshots, trace))
    ordered_error = error[order]
    counts = np.arange(1, len(error) + 1, dtype=np.float64)
    risk = np.cumsum(ordered_error, dtype=np.float64) / counts
    retained = counts / float(len(error))
    coverage = coverage_from_mahalanobis(mahalanobis)
    areas = {
        f"ellipse_area_cm2_{nominal}": float(
            np.mean(math.pi * threshold * np.sqrt(determinant))
        )
        for nominal, threshold in CHI2_DF2_THRESHOLDS.items()
    }
    coverage_prefix = (
        "raw_gaussian_coverage"
        if model_kind in {"fullcov", "homoscedastic"}
        else "raw_gaussian_moment_ellipse_coverage"
    )
    coverage_named = {
        f"{coverage_prefix}_{nominal}": coverage[f"coverage_{nominal}"]
        for nominal in CHI2_DF2_THRESHOLDS
    }
    probabilistic = {
        "forecast_type": model_kind,
        "energy_score_u20": float(energy_u.mean()),
        "energy_score_v20": float(energy_v.mean()),
        "gaussian_nll": (
            float(nll_terms["nll"].mean())
            if model_kind in {"fullcov", "homoscedastic"}
            else None
        ),
        "uncertainty_measure": "trace of predictive covariance",
        "uncertainty_error_spearman": spearman(trace, error),
        "high_error_definition": "Euclidean point error >= 100 cm",
        "high_error_auroc": binary_auroc(trace, error >= 100.0),
        "aurc": float(np.trapz(risk, retained)),
        "mean_covariance_trace_cm2": float(trace.mean()),
        "mean_covariance_determinant_cm4": float(determinant.mean()),
        "risk_tie_break": "global snapshot ID ascending",
        "risk_prefix_count": int(len(risk)),
        "coverage_interpretation": (
            "raw uncalibrated Gaussian Validation coverage"
            if model_kind in {"fullcov", "homoscedastic"}
            else (
                "descriptive raw Gaussian-moment ellipse coverage from "
                "S=20 first and second moments; not a DDPM credible region"
            )
        ),
    }
    probabilistic.update(coverage_named)
    probabilistic.update(areas)
    arrays = {
        "error_cm": error,
        "energy_score_u20": energy_u,
        "energy_score_v20": energy_v,
        "uncertainty_trace_cm2": trace,
        "mahalanobis": mahalanobis,
        "gaussian_nll": nll_terms["nll"],
    }
    return UnifiedMetrics(
        point,
        probabilistic,
        arrays,
        {
            "coverage": retained,
            "risk_mean_cm": risk,
            "ordered_snapshot_id": snapshots[order],
        },
    )


def validate_alignment(
    *,
    expected_source_indices: np.ndarray,
    expected_snapshot_ids: np.ndarray,
    expected_rp_ids: np.ndarray,
    expected_truth_cm: np.ndarray,
    fullcov_snapshot_ids: np.ndarray,
    fullcov_rp_ids: np.ndarray,
    ddpm_source_indices: np.ndarray,
    ddpm_rp_ids: np.ndarray,
    ddpm_truth_cm: np.ndarray,
    mlp_source_indices: np.ndarray,
    mlp_rp_ids: np.ndarray,
    mlp_truth_cm: np.ndarray,
) -> dict[str, bool | int]:
    expected_source = np.asarray(expected_source_indices, dtype=np.int64)
    expected_snapshot = np.asarray(expected_snapshot_ids, dtype=np.int64)
    expected_rp = np.asarray(expected_rp_ids, dtype=np.int64)
    expected_truth = _as_float64(expected_truth_cm)
    checks: dict[str, bool | int] = {
        "snapshot_count": int(len(expected_source)),
        "expected_source_unique": len(np.unique(expected_source))
        == len(expected_source),
        "expected_global_snapshot_unique": len(np.unique(expected_snapshot))
        == len(expected_snapshot),
        "fullcov_global_snapshot_exact": np.array_equal(
            np.asarray(fullcov_snapshot_ids, dtype=np.int64), expected_snapshot
        ),
        "fullcov_rp_exact": np.array_equal(
            np.asarray(fullcov_rp_ids, dtype=np.int64), expected_rp
        ),
        "ddpm_source_exact": np.array_equal(
            np.asarray(ddpm_source_indices, dtype=np.int64), expected_source
        ),
        "ddpm_rp_exact": np.array_equal(
            np.asarray(ddpm_rp_ids, dtype=np.int64), expected_rp
        ),
        "ddpm_truth_exact_float32": np.array_equal(
            np.asarray(ddpm_truth_cm, dtype=np.float32),
            expected_truth.astype(np.float32),
        ),
        "mlp_source_exact": np.array_equal(
            np.asarray(mlp_source_indices, dtype=np.int64), expected_source
        ),
        "mlp_rp_exact": np.array_equal(
            np.asarray(mlp_rp_ids, dtype=np.int64), expected_rp
        ),
        "mlp_truth_exact_float32": np.array_equal(
            np.asarray(mlp_truth_cm, dtype=np.float32),
            expected_truth.astype(np.float32),
        ),
    }
    if len(expected_source) != 145_452 or len(np.unique(expected_rp)) != 398:
        raise ValueError("frozen Validation cardinality mismatch")
    if not all(bool(value) for key, value in checks.items() if key != "snapshot_count"):
        failed = [key for key, value in checks.items() if key != "snapshot_count" and not value]
        raise ValueError(f"cross-model alignment failed: {failed}")
    return checks


def rp_index(rp_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(np.asarray(rp_ids), return_inverse=True)
    if len(unique) != 398:
        raise ValueError(f"expected 398 Validation RPs, found {len(unique)}")
    return unique.astype(np.int64), inverse.astype(np.int64)


def generate_rp_bootstrap_draws(
    replicates: int = 10_000, seed: int = 20_260_728, rp_count: int = 398
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(
        0, rp_count, size=(replicates, rp_count), dtype=np.int16
    )


def draws_to_counts(draws: np.ndarray, rp_count: int = 398) -> np.ndarray:
    values = np.asarray(draws)
    if values.ndim != 2 or values.shape[1] != rp_count:
        raise ValueError("RP draws must be [replicates,398]")
    counts = np.zeros((len(values), rp_count), dtype=np.int16)
    rows = np.repeat(np.arange(len(values), dtype=np.int64), rp_count)
    np.add.at(counts, (rows, values.reshape(-1)), 1)
    return counts


def _bootstrap_kernel():
    """Create the numba kernel lazily so unit-test import remains lightweight."""
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def kernel(
        error: np.ndarray,
        uncertainty: np.ndarray,
        energy_u: np.ndarray,
        energy_v: np.ndarray,
        rp_inverse: np.ndarray,
        snapshot_ids: np.ndarray,
        counts: np.ndarray,
        ranking_applicable: bool,
    ) -> np.ndarray:
        n = len(error)
        replicates = counts.shape[0]
        output = np.empty((replicates, 9), dtype=np.float64)
        error_order = np.argsort(error)
        # lexsort equivalent: stable snapshot ordering followed by stable score.
        snapshot_order = np.argsort(snapshot_ids)
        uncertainty_order = snapshot_order[
            np.argsort(uncertainty[snapshot_order], kind="mergesort")
        ]
        rp_sum = np.zeros(398, dtype=np.float64)
        rp_n = np.zeros(398, dtype=np.int64)
        for i in range(n):
            rp = rp_inverse[i]
            rp_sum[rp] += error[i]
            rp_n[rp] += 1
        rp_mean = rp_sum / rp_n
        high_error = error >= 100.0
        for b in prange(replicates):
            weights = np.empty(n, dtype=np.int16)
            total = 0
            sum_error = 0.0
            sum_u = 0.0
            sum_v = 0.0
            rp_macro = 0.0
            for rp in range(398):
                rp_macro += counts[b, rp] * rp_mean[rp]
            for i in range(n):
                weight = counts[b, rp_inverse[i]]
                weights[i] = weight
                total += weight
                sum_error += weight * error[i]
                sum_u += weight * energy_u[i]
                sum_v += weight * energy_v[i]
            output[b, 0] = sum_error / total
            output[b, 3] = rp_macro / 398.0
            output[b, 4] = sum_u / total
            output[b, 5] = sum_v / total
            # Exact NumPy-linear percentile positions on the replicated sample.
            target90 = 0.90 * (total - 1)
            target99 = 0.99 * (total - 1)
            low90 = int(math.floor(target90))
            high90 = int(math.ceil(target90))
            low99 = int(math.floor(target99))
            high99 = int(math.ceil(target99))
            value_low90 = 0.0
            value_high90 = 0.0
            value_low99 = 0.0
            value_high99 = 0.0
            cumulative = 0
            for j in range(n):
                i = error_order[j]
                weight = weights[i]
                next_cumulative = cumulative + weight
                if cumulative <= low90 < next_cumulative:
                    value_low90 = error[i]
                if cumulative <= high90 < next_cumulative:
                    value_high90 = error[i]
                if cumulative <= low99 < next_cumulative:
                    value_low99 = error[i]
                if cumulative <= high99 < next_cumulative:
                    value_high99 = error[i]
                cumulative = next_cumulative
            output[b, 1] = value_low90 + (target90 - low90) * (
                value_high90 - value_low90
            )
            output[b, 2] = value_low99 + (target99 - low99) * (
                value_high99 - value_low99
            )
            if not ranking_applicable:
                output[b, 6] = np.nan
                output[b, 7] = 0.5
                output[b, 8] = np.nan
                continue
            # Weighted midranks of error, exact for duplicated RP clusters.
            error_rank = np.empty(n, dtype=np.float64)
            cumulative_weight = 0
            start = 0
            while start < n:
                stop = start + 1
                value = error[error_order[start]]
                while stop < n and error[error_order[stop]] == value:
                    stop += 1
                group_weight = 0
                for j in range(start, stop):
                    group_weight += weights[error_order[j]]
                midrank = cumulative_weight + 0.5 * (group_weight - 1)
                for j in range(start, stop):
                    error_rank[error_order[j]] = midrank
                cumulative_weight += group_weight
                start = stop
            sum_x = 0.0
            sum_y = 0.0
            sum_xx = 0.0
            sum_yy = 0.0
            sum_xy = 0.0
            positive = 0
            rank_sum_positive = 0.0
            cumulative_weight = 0
            start = 0
            while start < n:
                stop = start + 1
                value = uncertainty[uncertainty_order[start]]
                while (
                    stop < n
                    and uncertainty[uncertainty_order[stop]] == value
                ):
                    stop += 1
                group_weight = 0
                for j in range(start, stop):
                    group_weight += weights[uncertainty_order[j]]
                midrank = cumulative_weight + 0.5 * (group_weight - 1)
                for j in range(start, stop):
                    i = uncertainty_order[j]
                    weight = weights[i]
                    yrank = error_rank[i]
                    sum_x += weight * midrank
                    sum_y += weight * yrank
                    sum_xx += weight * midrank * midrank
                    sum_yy += weight * yrank * yrank
                    sum_xy += weight * midrank * yrank
                    if high_error[i]:
                        positive += weight
                        rank_sum_positive += weight * midrank
                cumulative_weight += group_weight
                start = stop
            covariance = sum_xy - sum_x * sum_y / total
            variance_x = sum_xx - sum_x * sum_x / total
            variance_y = sum_yy - sum_y * sum_y / total
            if variance_x > 0.0 and variance_y > 0.0:
                output[b, 6] = covariance / math.sqrt(
                    variance_x * variance_y
                )
            else:
                output[b, 6] = np.nan
            negative = total - positive
            if positive > 0 and negative > 0:
                output[b, 7] = (
                    rank_sum_positive - positive * (positive - 1) / 2.0
                ) / (positive * negative)
            else:
                output[b, 7] = np.nan
            # Exact all-prefix trapezoidal AURC on the replicated risk order.
            prefix_count = 0
            prefix_error = 0.0
            previous_coverage = 0.0
            previous_risk = 0.0
            area = 0.0
            first = True
            for j in range(n):
                i = uncertainty_order[j]
                weight = weights[i]
                for _ in range(weight):
                    prefix_count += 1
                    prefix_error += error[i]
                    coverage = prefix_count / total
                    risk = prefix_error / prefix_count
                    if first:
                        first = False
                    else:
                        area += (
                            0.5
                            * (risk + previous_risk)
                            * (coverage - previous_coverage)
                        )
                    previous_coverage = coverage
                    previous_risk = risk
            output[b, 8] = area
        return output

    return kernel


BOOTSTRAP_METRIC_NAMES = (
    "mean_cm",
    "p90_cm",
    "p99_cm",
    "rp_macro_mean_cm",
    "energy_score_u20",
    "energy_score_v20",
    "uncertainty_error_spearman",
    "high_error_auroc",
    "aurc",
)


def rp_cluster_bootstrap_metrics(
    *,
    error_cm: np.ndarray,
    uncertainty_trace_cm2: np.ndarray,
    energy_score_u20: np.ndarray,
    energy_score_v20: np.ndarray,
    rp_inverse: np.ndarray,
    snapshot_ids: np.ndarray,
    counts: np.ndarray,
    ranking_applicable: bool,
) -> np.ndarray:
    arrays = [
        _as_float64(error_cm),
        _as_float64(uncertainty_trace_cm2),
        _as_float64(energy_score_u20),
        _as_float64(energy_score_v20),
    ]
    n = len(arrays[0])
    if any(len(item) != n for item in arrays):
        raise ValueError("bootstrap array length mismatch")
    inverse = np.asarray(rp_inverse, dtype=np.int64)
    snapshots = np.asarray(snapshot_ids, dtype=np.int64)
    count_array = np.asarray(counts, dtype=np.int16)
    if (
        inverse.shape != (n,)
        or snapshots.shape != (n,)
        or count_array.ndim != 2
        or count_array.shape[1] != 398
    ):
        raise ValueError("bootstrap shape mismatch")
    kernel = _bootstrap_kernel()
    return kernel(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        inverse,
        snapshots,
        count_array,
        bool(ranking_applicable),
    )


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    array = _as_float64(values)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {"lower": float("nan"), "upper": float("nan")}
    return {
        "lower": float(np.percentile(finite, 2.5)),
        "upper": float(np.percentile(finite, 97.5)),
    }
