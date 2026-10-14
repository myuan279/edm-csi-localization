from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from probabilistic_metrics import energy_scores_u_v, spearman

from .contracts import DistributionOutput


CHI2_DF2 = {
    50: 1.3862943611198906,
    80: 3.2188758248682006,
    90: 4.605170185988092,
    95: 5.991464547107979,
}


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


def _aurc(error: np.ndarray, risk: np.ndarray, uids: np.ndarray) -> float:
    order = np.lexsort((uids, risk))
    prefix = np.cumsum(error[order], dtype=np.float64) / np.arange(1, len(error) + 1)
    coverage = np.arange(1, len(error) + 1, dtype=np.float64) / len(error)
    return float(np.trapz(prefix, coverage))


def evaluate_distribution(
    output: DistributionOutput,
    truth_cm: np.ndarray,
    group_ids: Sequence[Any] | np.ndarray,
    *,
    open_csi_100cm_auroc: bool = False,
) -> EvaluationResult:
    """Recompute all formal metrics in float64 from persisted float32 arrays."""
    samples = np.asarray(output.posterior_samples_cm, dtype=np.float64)
    truth = np.asarray(truth_cm, dtype=np.float64)
    groups = np.asarray(group_ids)
    uids = np.asarray(output.global_uids, dtype=np.int64)
    if samples.shape[0] != len(truth) or truth.shape != (len(samples), 2):
        raise ValueError("truth/sample shape mismatch")
    if groups.shape != (len(samples),) or len(np.unique(uids)) != len(uids):
        raise ValueError("group/UID alignment failure")
    if not np.isfinite(samples).all() or not np.isfinite(truth).all():
        raise FloatingPointError("formal artifacts contain non-finite values")

    mean = samples.mean(axis=1)
    centered = samples - mean[:, None, :]
    covariance = np.einsum("nsi,nsj->nij", centered, centered)
    covariance /= float(samples.shape[1] - 1)
    # A tiny declared numerical floor makes moment ellipses invertible without
    # pretending that DDPM supplies analytic covariance.
    covariance += np.eye(2, dtype=np.float64)[None, :, :] * 1e-9
    error = np.linalg.norm(mean - truth, axis=1)
    energy_u, energy_v = energy_scores_u_v(samples, truth)
    risk = np.trace(covariance, axis1=1, axis2=2)
    group_means = np.asarray([error[groups == group].mean() for group in np.unique(groups)])
    point = {
        "mean_cm": float(error.mean()),
        "median_cm": float(np.median(error)),
        "rmse_cm": float(np.sqrt(np.mean(np.square(error)))),
        "p90_cm": float(np.percentile(error, 90)),
        "p99_cm": float(np.percentile(error, 99)),
        "rp_or_block_macro_mean_cm": float(group_means.mean()),
    }
    probabilistic: dict[str, Any] = {
        "energy_score_u20_cm": float(energy_u.mean()),
        "energy_score_v20_cm": float(energy_v.mean()),
        "uncertainty_error_spearman": spearman(risk, error),
        "aurc": _aurc(error, risk, uids),
    }
    if open_csi_100cm_auroc:
        from probabilistic_metrics import binary_auroc

        probabilistic["error_over_100cm_auroc"] = binary_auroc(risk, error >= 100.0)
    chol = np.linalg.cholesky(covariance)
    solved = np.linalg.solve(chol, (truth - mean)[..., None]).squeeze(-1)
    mahalanobis = np.square(solved).sum(axis=1)
    determinant = np.linalg.det(covariance)
    regions = {}
    for nominal, threshold in CHI2_DF2.items():
        regions[f"raw_coverage_{nominal}"] = float(np.mean(mahalanobis <= threshold))
        regions[f"raw_mean_area_cm2_{nominal}"] = float(
            np.mean(math.pi * threshold * np.sqrt(determinant))
        )
    return EvaluationResult(
        metrics={"point": point, "probabilistic": probabilistic, "regions": regions},
        arrays={
            "error_cm": error,
            "energy_score_u20_cm": energy_u,
            "energy_score_v20_cm": energy_v,
            "risk_trace_cm2": risk,
            "mahalanobis_squared": mahalanobis,
            "mean_cm": mean,
            "covariance_cm2": covariance,
        },
    )


def generate_cluster_draws(
    group_count: int, *, replicates: int = 10_000, seed: int = 20260801
) -> np.ndarray:
    if group_count < 2 or replicates < 1:
        raise ValueError("cluster bootstrap requires at least two groups and one replicate")
    dtype = np.int16 if group_count <= np.iinfo(np.int16).max else np.int32
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, group_count, size=(replicates, group_count), dtype=dtype)


def draws_to_counts(draws: np.ndarray, group_count: int) -> np.ndarray:
    source = np.asarray(draws)
    if source.ndim != 2 or source.shape[1] != group_count:
        raise ValueError("draw matrix shape mismatch")
    counts = np.zeros((len(source), group_count), dtype=np.int16)
    rows = np.repeat(np.arange(len(source)), group_count)
    np.add.at(counts, (rows, source.reshape(-1)), 1)
    return counts


def paired_cluster_bootstrap(
    candidate: EvaluationResult,
    control: EvaluationResult,
    group_ids: Sequence[Any] | np.ndarray,
    global_uids: Sequence[int] | np.ndarray,
    *,
    draws: np.ndarray,
) -> dict[str, Any]:
    """Paired cluster bootstrap using one shared draw matrix for every metric."""
    groups = np.asarray(group_ids)
    uids = np.asarray(global_uids, dtype=np.int64)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(groups) != len(uids) or len(np.unique(uids)) != len(uids):
        raise ValueError("paired bootstrap alignment failure")
    counts = draws_to_counts(draws, len(unique))
    candidate_arrays = candidate.arrays
    control_arrays = control.arrays
    for name in ("error_cm", "energy_score_u20_cm"):
        if len(candidate_arrays[name]) != len(groups) or len(control_arrays[name]) != len(groups):
            raise ValueError("candidate/control row count mismatch")
    samples = _bootstrap_differences(
        np.asarray(candidate_arrays["error_cm"], dtype=np.float64),
        np.asarray(control_arrays["error_cm"], dtype=np.float64),
        np.asarray(candidate_arrays["energy_score_u20_cm"], dtype=np.float64),
        np.asarray(control_arrays["energy_score_u20_cm"], dtype=np.float64),
        inverse.astype(np.int64),
        counts,
    )
    names = ("mean_cm", "p90_cm", "p99_cm", "rp_macro_mean_cm", "energy_score_u20_cm")
    result: dict[str, Any] = {"replicates": int(len(draws)), "group_count": int(len(unique))}
    for column, name in enumerate(names):
        values = samples[:, column]
        result[name] = {
            "candidate_minus_control": float(np.mean(values)),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        }
    return result


def _bootstrap_differences(
    candidate_error: np.ndarray,
    control_error: np.ndarray,
    candidate_energy: np.ndarray,
    control_energy: np.ndarray,
    group_inverse: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    try:
        from numba import njit, prange
    except ImportError as exc:
        raise RuntimeError("formal 10,000-replicate bootstrap requires numba") from exc

    @njit(parallel=True, cache=True)
    def kernel(c_err, b_err, c_es, b_es, inverse, cluster_counts):
        n = len(c_err)
        group_count = cluster_counts.shape[1]
        output = np.empty((cluster_counts.shape[0], 5), dtype=np.float64)
        c_order = np.argsort(c_err)
        b_order = np.argsort(b_err)
        c_group_mean = np.zeros(group_count, dtype=np.float64)
        b_group_mean = np.zeros(group_count, dtype=np.float64)
        group_n = np.zeros(group_count, dtype=np.int64)
        for row in range(n):
            group = inverse[row]
            c_group_mean[group] += c_err[row]
            b_group_mean[group] += b_err[row]
            group_n[group] += 1
        for group in range(group_count):
            c_group_mean[group] /= group_n[group]
            b_group_mean[group] /= group_n[group]
        for replicate in prange(cluster_counts.shape[0]):
            weights = np.empty(n, dtype=np.int16)
            total = 0
            c_sum = 0.0
            b_sum = 0.0
            c_es_sum = 0.0
            b_es_sum = 0.0
            macro = 0.0
            for group in range(group_count):
                macro += cluster_counts[replicate, group] * (
                    c_group_mean[group] - b_group_mean[group]
                )
            for row in range(n):
                weight = cluster_counts[replicate, inverse[row]]
                weights[row] = weight
                total += weight
                c_sum += weight * c_err[row]
                b_sum += weight * b_err[row]
                c_es_sum += weight * c_es[row]
                b_es_sum += weight * b_es[row]
            output[replicate, 0] = (c_sum - b_sum) / total
            output[replicate, 3] = macro / group_count
            output[replicate, 4] = (c_es_sum - b_es_sum) / total
            for metric_index in range(2):
                quantile = 0.90 if metric_index == 0 else 0.99
                position = quantile * (total - 1)
                lower = int(math.floor(position))
                upper = int(math.ceil(position))
                c_low = c_high = b_low = b_high = 0.0
                cumulative = 0
                for sorted_row in range(n):
                    row = c_order[sorted_row]
                    next_value = cumulative + weights[row]
                    if cumulative <= lower < next_value:
                        c_low = c_err[row]
                    if cumulative <= upper < next_value:
                        c_high = c_err[row]
                    cumulative = next_value
                cumulative = 0
                for sorted_row in range(n):
                    row = b_order[sorted_row]
                    next_value = cumulative + weights[row]
                    if cumulative <= lower < next_value:
                        b_low = b_err[row]
                    if cumulative <= upper < next_value:
                        b_high = b_err[row]
                    cumulative = next_value
                c_value = c_low + (position - lower) * (c_high - c_low)
                b_value = b_low + (position - lower) * (b_high - b_low)
                output[replicate, 1 + metric_index] = c_value - b_value
        return output

    return kernel(
        candidate_error,
        control_error,
        candidate_energy,
        control_energy,
        group_inverse,
        counts,
    )


def development_gate(
    per_seed: Sequence[Mapping[str, float]],
    *,
    integrity_passed: bool,
) -> dict[str, Any]:
    if len(per_seed) != 3:
        raise ValueError("Development gate requires exactly three paired seeds")
    mean_delta_pct = np.asarray([row["mean_delta_percent"] for row in per_seed], dtype=np.float64)
    result = {
        "mean_average_improves_at_least_2_percent": float(mean_delta_pct.mean()) <= -2.0,
        "at_least_2_of_3_mean_directions_favorable": int(np.sum(mean_delta_pct < 0.0)) >= 2,
        "rp_macro_mean_improves": float(np.mean([row["rp_macro_delta_cm"] for row in per_seed])) < 0.0,
        "energy_u20_improves_at_least_1_percent": float(
            np.mean([row["energy_u20_delta_percent"] for row in per_seed])
        ) <= -1.0,
        "p90_not_worse_than_3_percent": max(row["p90_delta_percent"] for row in per_seed) <= 3.0,
        "p99_not_worse": max(row["p99_delta_cm"] for row in per_seed) <= 0.0,
        "integrity_passed": bool(integrity_passed),
    }
    result["promote"] = all(result.values())
    return result


def validation_gate(
    *,
    seed_rows: Sequence[Mapping[str, float]],
    dataset_rows: Sequence[Mapping[str, float]],
    bootstrap: Mapping[str, Any],
    opencsi_candidate_p99_cm: float,
    integrity_passed: bool,
) -> dict[str, Any]:
    """Frozen five-seed / three-dataset promotion gate."""
    if len(seed_rows) != 5 or len(dataset_rows) != 3:
        raise ValueError("Validation gate requires five seeds and three datasets")
    mean_improved_datasets = sum(row["mean_delta_cm"] < 0.0 for row in dataset_rows)
    energy_improved_datasets = sum(row["energy_u20_delta_cm"] < 0.0 for row in dataset_rows)
    result = {
        "matched_flat_mean_improves_at_least_5_percent": float(
            np.mean([row["mean_delta_percent"] for row in seed_rows])
        ) <= -5.0,
        "rp_macro_mean_improves": float(
            np.mean([row["rp_macro_delta_cm"] for row in seed_rows])
        ) < 0.0,
        "mean_bootstrap_ci_upper_below_zero": float(bootstrap["mean_cm"]["ci95"][1]) < 0.0,
        "energy_bootstrap_ci_upper_below_zero": float(
            bootstrap["energy_score_u20_cm"]["ci95"][1]
        ) < 0.0,
        "at_least_4_of_5_seed_directions_favorable": sum(
            row["mean_delta_cm"] < 0.0 for row in seed_rows
        ) >= 4,
        "p90_not_worse_than_3_percent": max(
            row["p90_delta_percent"] for row in seed_rows
        ) <= 3.0,
        "p99_not_worse": max(row["p99_delta_cm"] for row in seed_rows) <= 0.0,
        "opencsi_historical_p99_ceiling": float(opencsi_candidate_p99_cm) <= 391.966,
        "mean_improves_on_at_least_2_of_3_datasets": mean_improved_datasets >= 2,
        "energy_improves_on_at_least_2_of_3_datasets": energy_improved_datasets >= 2,
        "no_dataset_p99_degrades_over_3_percent": max(
            row["p99_delta_percent"] for row in dataset_rows
        ) <= 3.0,
        "integrity_passed": bool(integrity_passed),
    }
    result["promote"] = all(result.values())
    return result


def short_chain_gate(
    *,
    delta_mean_cm: float,
    delta_rp_macro_cm: float,
    delta_energy_u20_cm: float,
    delta_p90_cm: float,
    delta_p99_cm: float,
    measured_speedup: float,
) -> dict[str, bool]:
    result = {
        "mean_noninferiority_0_5cm": delta_mean_cm <= 0.5,
        "rp_macro_noninferiority_0_5cm": delta_rp_macro_cm <= 0.5,
        "energy_noninferiority_0_5cm": delta_energy_u20_cm <= 0.5,
        "p90_not_worse": delta_p90_cm <= 0.0,
        "p99_not_worse": delta_p99_cm <= 0.0,
        "measured_speedup_at_least_8x": measured_speedup >= 8.0,
    }
    result["promote"] = all(result.values())
    return result
