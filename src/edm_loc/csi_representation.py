"""Scientific core for the EDM4 CSI phase-representation Development protocol.

This module is deliberately data-agnostic.  It contains only deterministic
transforms, phase-quality strata, RP-macro metrics, and the preregistered
promotion gate.  Importing it never opens an OpenCSI artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "edm4_csi_phase_representation_v1"
SEEDS = (2031, 2032, 2033)
RAW_SHAPE = (2, 39, 4)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816

CONTROL_ID = "R0_RAW"
ROUND_1 = ("R1_FREQ_RI", "R2_IFFT39_RI")
ROUND_2 = ("R3_LOG_RI", "R4_PORT0_REL_RI", "R5_DELTAK_RI")
REPRESENTATION_IDS = (CONTROL_ID,) + ROUND_1 + ROUND_2


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Map phase to [-pi, pi] without relying on an unwrap convention."""

    value = np.asarray(phase, dtype=np.float64)
    return np.arctan2(np.sin(value), np.cos(value))


def phase_branch_number(phase: np.ndarray) -> np.ndarray:
    """Integer n in phase = wrap(phase) + 2*pi*n, up to float tolerance."""

    value = np.asarray(phase, dtype=np.float64)
    return np.rint((value - wrap_phase(value)) / (2.0 * np.pi)).astype(np.int64)


def _validate_raw(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw)
    require(value.shape[-3:] == RAW_SHAPE, f"raw CSI must end in {RAW_SHAPE}; got {value.shape}")
    require(np.isfinite(value).all(), "raw CSI contains NaN or Inf")
    return value.astype(np.float64, copy=False)


def representation_magnitude(raw: np.ndarray, representation_id: str) -> np.ndarray:
    value = _validate_raw(raw)
    amplitude = np.maximum(value[..., 0, :, :], 0.0)
    if representation_id in {"R3_LOG_RI", "R4_PORT0_REL_RI", "R5_DELTAK_RI"}:
        return np.log1p(amplitude)
    if representation_id in {"R1_FREQ_RI", "R2_IFFT39_RI"}:
        return amplitude
    raise ValueError(f"no amplitude-only rule for {representation_id!r}")


def transform_csi(raw: np.ndarray, representation_id: str) -> np.ndarray:
    """Transform raw [amplitude, phase] before any featurewise z-score."""

    value = _validate_raw(raw)
    require(representation_id in REPRESENTATION_IDS, f"unknown representation: {representation_id}")
    if representation_id == CONTROL_ID:
        return value.astype(np.float32)

    amplitude = np.maximum(value[..., 0, :, :], 0.0)
    phase = value[..., 1, :, :]
    if representation_id == "R1_FREQ_RI":
        magnitude = amplitude
        effective_phase = phase
    elif representation_id == "R2_IFFT39_RI":
        frequency = amplitude * np.exp(1j * phase)
        delay = np.fft.ifft(frequency, n=39, axis=-2, norm="ortho")
        return np.stack((delay.real, delay.imag), axis=-3).astype(np.float32)
    elif representation_id == "R3_LOG_RI":
        magnitude = np.log1p(amplitude)
        effective_phase = phase
    elif representation_id == "R4_PORT0_REL_RI":
        magnitude = np.log1p(amplitude)
        effective_phase = np.empty_like(phase)
        effective_phase[..., 0] = wrap_phase(phase[..., 0])
        effective_phase[..., 1:] = wrap_phase(phase[..., 1:] - phase[..., [0]])
    elif representation_id == "R5_DELTAK_RI":
        magnitude = np.log1p(amplitude)
        effective_phase = np.empty_like(phase)
        effective_phase[..., 0, :] = wrap_phase(phase[..., 0, :])
        effective_phase[..., 1:, :] = wrap_phase(phase[..., 1:, :] - phase[..., :-1, :])
    else:  # pragma: no cover - guarded above
        raise AssertionError(representation_id)
    return np.stack(
        (magnitude * np.cos(effective_phase), magnitude * np.sin(effective_phase)),
        axis=-3,
    ).astype(np.float32)


def amplitude_only_transform(raw: np.ndarray, selected_representation_id: str) -> np.ndarray:
    """Matched ablation: selected magnitude in channel 0 and a zero phase channel."""

    magnitude = representation_magnitude(raw, selected_representation_id)
    return np.stack((magnitude, np.zeros_like(magnitude)), axis=-3).astype(np.float32)


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray
    sample_count: int
    source_subset: str = "development_train_300k"

    def validate(self) -> None:
        require(self.mean.shape == RAW_SHAPE, "feature mean shape drift")
        require(self.std.shape == RAW_SHAPE, "feature std shape drift")
        require(self.sample_count > 0, "feature stats need positive sample count")
        require(np.isfinite(self.mean).all() and np.isfinite(self.std).all(), "non-finite stats")
        require(np.all(self.std > 0.0), "feature std must be positive")
        require(self.source_subset == "development_train_300k", "stats source is not Train-only")


def fit_feature_stats(chunks: Iterable[np.ndarray]) -> FeatureStats:
    """Fit per-feature mean/std from Development Train chunks only."""

    total = np.zeros(RAW_SHAPE, dtype=np.float64)
    total_square = np.zeros(RAW_SHAPE, dtype=np.float64)
    count = 0
    for chunk in chunks:
        value = np.asarray(chunk, dtype=np.float64)
        require(value.ndim == 4 and value.shape[1:] == RAW_SHAPE, "stats chunk shape mismatch")
        require(np.isfinite(value).all(), "stats chunk contains NaN or Inf")
        total += value.sum(axis=0, dtype=np.float64)
        total_square += np.square(value).sum(axis=0, dtype=np.float64)
        count += len(value)
    require(count > 1, "at least two Train rows are required")
    mean = total / float(count)
    variance = np.maximum(total_square / float(count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    # A constant feature cannot carry information; scale it by one without
    # changing its centered value.
    std = np.where(std > 1e-12, std, 1.0)
    result = FeatureStats(mean.astype(np.float32), std.astype(np.float32), count)
    result.validate()
    return result


def normalize_features(features: np.ndarray, stats: FeatureStats) -> np.ndarray:
    stats.validate()
    value = np.asarray(features, dtype=np.float32)
    require(value.shape[-3:] == RAW_SHAPE, "normalized feature shape mismatch")
    output = (value - stats.mean) / stats.std
    require(np.isfinite(output).all(), "normalization produced NaN or Inf")
    return output.astype(np.float32)


def circular_resultant(phase: np.ndarray, *, axis: int | tuple[int, ...]) -> np.ndarray:
    value = np.asarray(phase, dtype=np.float64)
    return np.abs(np.mean(np.exp(1j * value), axis=axis))


def phase_quality_score(raw: np.ndarray) -> np.ndarray:
    """Truth/error-blind row score used to freeze phase-quality quartiles.

    Equal weight is assigned to port-0-relative coherence and adjacent-
    subcarrier-difference coherence.  Higher means more internally coherent.
    """

    phase = _validate_raw(raw)[..., 1, :, :]
    relative = wrap_phase(phase[..., 1:] - phase[..., [0]])
    delta = wrap_phase(phase[..., 1:, :] - phase[..., :-1, :])
    relative_r = circular_resultant(relative, axis=(-2, -1))
    delta_r = circular_resultant(delta, axis=(-2, -1))
    return (0.5 * relative_r + 0.5 * delta_r).astype(np.float64)


def fit_phase_quality_quartiles(train_raw: np.ndarray) -> np.ndarray:
    score = phase_quality_score(train_raw)
    thresholds = np.quantile(score, (0.25, 0.50, 0.75), method="linear")
    require(np.isfinite(thresholds).all(), "non-finite phase-quality thresholds")
    return np.asarray(thresholds, dtype=np.float64)


def assign_phase_quality_quartile(raw: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    edges = np.asarray(thresholds, dtype=np.float64)
    require(edges.shape == (3,) and np.all(edges[:-1] <= edges[1:]), "invalid quartiles")
    return np.searchsorted(edges, phase_quality_score(raw), side="right").astype(np.int8)


def phase_audit_arrays(
    raw: np.ndarray,
    *,
    rp_ids: Sequence[int] | np.ndarray | None = None,
    snapshot_ids: Sequence[int] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return sufficient arrays for the preregistered no-training audit."""

    phase = _validate_raw(raw)[..., 1, :, :]
    wrapped = wrap_phase(phase)
    relative = wrap_phase(phase[..., 1:] - phase[..., [0]])
    delta_k = wrap_phase(phase[..., 1:, :] - phase[..., :-1, :])
    result = {
        "wrapped_phase": wrapped.astype(np.float32),
        "branch_number": phase_branch_number(phase),
        "circular_concentration_by_subcarrier_port": circular_resultant(wrapped, axis=0).astype(np.float32),
        "port0_relative_phase": relative.astype(np.float32),
        "adjacent_subcarrier_phase": delta_k.astype(np.float32),
        "phase_quality_score": phase_quality_score(raw).astype(np.float32),
    }
    if rp_ids is not None or snapshot_ids is not None:
        require(rp_ids is not None and snapshot_ids is not None, "RP and snapshot IDs must be paired")
        rps = np.asarray(rp_ids, dtype=np.int64)
        snapshots = np.asarray(snapshot_ids, dtype=np.int64)
        require(rps.shape == snapshots.shape == (len(phase),), "phase audit identity shape mismatch")
        order = np.lexsort((snapshots, rps))
        consecutive = (rps[order][1:] == rps[order][:-1]) & (
            snapshots[order][1:] == snapshots[order][:-1] + 1
        )
        result["adjacent_snapshot_phase"] = wrap_phase(
            phase[order][1:][consecutive] - phase[order][:-1][consecutive]
        ).astype(np.float32)
    return result


def point_errors(mean_cm: np.ndarray, truth_cm: np.ndarray) -> np.ndarray:
    predicted = np.asarray(mean_cm, dtype=np.float64)
    truth = np.asarray(truth_cm, dtype=np.float64)
    require(predicted.shape == truth.shape and predicted.ndim == 2 and predicted.shape[1] == 2, "point/truth shape mismatch")
    require(np.isfinite(predicted).all() and np.isfinite(truth).all(), "non-finite point/truth")
    return np.linalg.norm(predicted - truth, axis=1)


def error_report(error_cm: np.ndarray, rp_ids: Sequence[int] | np.ndarray) -> dict[str, float]:
    error = np.asarray(error_cm, dtype=np.float64)
    groups = np.asarray(rp_ids)
    require(error.ndim == 1 and groups.shape == error.shape and len(error) > 0, "error/RP shape mismatch")
    unique = np.unique(groups)
    rp_mean = np.asarray([error[groups == group].mean() for group in unique])
    rp_p90 = np.asarray([np.percentile(error[groups == group], 90) for group in unique])
    return {
        "rp_macro_mean_cm": float(rp_mean.mean()),
        "rp_macro_p90_cm": float(rp_p90.mean()),
        "p50_cm": float(np.percentile(error, 50)),
        "p75_cm": float(np.percentile(error, 75)),
        "p90_cm": float(np.percentile(error, 90)),
        "p99_cm": float(np.percentile(error, 99)),
        "error_over_200cm_rate": float(np.mean(error > 200.0)),
    }


def _per_rp_primary(error: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    means = np.asarray([error[groups == group].mean() for group in unique], dtype=np.float64)
    p90 = np.asarray([np.percentile(error[groups == group], 90) for group in unique], dtype=np.float64)
    return unique, means, p90


def paired_rp_bootstrap(
    candidate_errors_by_seed: Sequence[np.ndarray],
    control_errors_by_seed: Sequence[np.ndarray],
    rp_ids: Sequence[int] | np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired RP-cluster bootstrap of the two RP-macro co-primary metrics."""

    require(len(candidate_errors_by_seed) == len(control_errors_by_seed) == 3, "bootstrap requires three seeds")
    groups = np.asarray(rp_ids)
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    expected_groups: np.ndarray | None = None
    for candidate, control in zip(candidate_errors_by_seed, control_errors_by_seed, strict=True):
        c_error = np.asarray(candidate, dtype=np.float64)
        b_error = np.asarray(control, dtype=np.float64)
        require(c_error.shape == b_error.shape == groups.shape, "paired error alignment failure")
        unique, c_mean, c_p90 = _per_rp_primary(c_error, groups)
        b_unique, b_mean, b_p90 = _per_rp_primary(b_error, groups)
        require(np.array_equal(unique, b_unique), "candidate/control RP mismatch")
        if expected_groups is None:
            expected_groups = unique
        require(np.array_equal(unique, expected_groups), "seed RP mismatch")
        rows.append((c_mean - b_mean, c_p90 - b_p90))
    differences = np.stack([np.stack(item, axis=1) for item in rows], axis=0).mean(axis=0)
    group_count = len(differences)
    require(group_count >= 2 and replicates == 10_000, "formal bootstrap requires 10,000 replicates and >=2 RPs")
    draws = np.random.default_rng(int(seed)).integers(0, group_count, size=(replicates, group_count))
    sampled = differences[draws].mean(axis=1)
    result: dict[str, Any] = {
        "replicates": int(replicates),
        "seed": int(seed),
        "cluster_unit": "reference_point",
        "group_count": int(group_count),
    }
    for column, name in enumerate(("rp_macro_mean_cm", "rp_macro_p90_cm")):
        values = sampled[:, column]
        result[name] = {
            "candidate_minus_control": float(differences[:, column].mean()),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        }
    return result


def checkpoint_selection_key(
    candidate_metrics: Mapping[str, float], control_metrics: Mapping[str, float], epoch: int
) -> tuple[float, float, int]:
    """Frozen checkpoint key: co-primary normalized J, then ES_U20, then epoch."""

    mean_ratio = float(candidate_metrics["rp_macro_mean_cm"]) / float(control_metrics["rp_macro_mean_cm"])
    p90_ratio = float(candidate_metrics["rp_macro_p90_cm"]) / float(control_metrics["rp_macro_p90_cm"])
    key = (0.5 * mean_ratio + 0.5 * p90_ratio, float(candidate_metrics["energy_score_u20_cm"]), int(epoch))
    require(all(math.isfinite(value) for value in key[:2]) and key[2] > 0, "invalid checkpoint key")
    return key


def candidate_gate(
    candidate_rows: Sequence[Mapping[str, float]],
    control_rows: Sequence[Mapping[str, float]],
    bootstrap: Mapping[str, Any],
    *,
    integrity: Mapping[str, bool],
) -> dict[str, Any]:
    require(len(candidate_rows) == len(control_rows) == 3, "gate requires exactly three paired seeds")
    required_integrity = (
        "data_scope",
        "hashes",
        "parameter_count",
        "nfe_63",
        "posterior_samples_20",
        "single_model",
        "uid_alignment",
        "train_only_normalization",
    )
    require(set(required_integrity).issubset(integrity), "integrity fields missing")
    candidate_mean = float(np.mean([row["rp_macro_mean_cm"] for row in candidate_rows]))
    control_mean = float(np.mean([row["rp_macro_mean_cm"] for row in control_rows]))
    candidate_p90 = float(np.mean([row["rp_macro_p90_cm"] for row in candidate_rows]))
    control_p90 = float(np.mean([row["rp_macro_p90_cm"] for row in control_rows]))
    checks = {
        "rp_macro_mean_point_lower": candidate_mean < control_mean,
        "rp_macro_p90_point_lower": candidate_p90 < control_p90,
        "rp_macro_mean_ci_upper_below_zero": float(bootstrap["rp_macro_mean_cm"]["ci95"][1]) < 0.0,
        "rp_macro_p90_ci_upper_below_zero": float(bootstrap["rp_macro_p90_cm"]["ci95"][1]) < 0.0,
        "at_least_2_of_3_mean_directions_favorable": sum(
            float(candidate["rp_macro_mean_cm"]) < float(control["rp_macro_mean_cm"])
            for candidate, control in zip(candidate_rows, control_rows, strict=True)
        ) >= 2,
        "at_least_2_of_3_p90_directions_favorable": sum(
            float(candidate["rp_macro_p90_cm"]) < float(control["rp_macro_p90_cm"])
            for candidate, control in zip(candidate_rows, control_rows, strict=True)
        ) >= 2,
        "integrity_passed": all(bool(integrity[name]) for name in required_integrity),
    }
    promote = all(checks.values())
    ci_only = (
        not promote
        and checks["rp_macro_mean_point_lower"]
        and checks["rp_macro_p90_point_lower"]
        and checks["at_least_2_of_3_mean_directions_favorable"]
        and checks["at_least_2_of_3_p90_directions_favorable"]
        and checks["integrity_passed"]
        and not (
            checks["rp_macro_mean_ci_upper_below_zero"]
            and checks["rp_macro_p90_ci_upper_below_zero"]
        )
    )
    status = (
        "DEVELOPMENT_POSITIVE_REPRESENTATION_FROZEN"
        if promote
        else "CI_ONLY_INCONCLUSIVE_NO_PROMOTION_NO_RETRY"
        if ci_only
        else "DEVELOPMENT_NEGATIVE_NO_PROMOTION_NO_RETRY"
    )
    return {
        "checks": checks,
        "promote": promote,
        "status": status,
        "aggregate": {
            "candidate_rp_macro_mean_cm": candidate_mean,
            "control_rp_macro_mean_cm": control_mean,
            "candidate_rp_macro_p90_cm": candidate_p90,
            "control_rp_macro_p90_cm": control_p90,
            "J": 0.5 * candidate_mean / control_mean + 0.5 * candidate_p90 / control_p90,
        },
        "p99_used_as_gate": False,
    }


def choose_round_winner(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    passing = [row for row in rows if bool(row["gate"]["promote"])]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            float(row["gate"]["aggregate"]["J"]),
            float(row["gate"]["aggregate"]["candidate_rp_macro_p90_cm"]),
            int(row["trainable_parameters"]),
        ),
    )


def lowest_score_representation(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    require(bool(rows), "no representation rows supplied")
    return min(
        rows,
        key=lambda row: (
            float(row["gate"]["aggregate"]["J"]),
            float(row["gate"]["aggregate"]["candidate_rp_macro_p90_cm"]),
            int(row["trainable_parameters"]),
        ),
    )


def validate_round_order(completed: Sequence[str], requested: str) -> None:
    done = set(completed)
    require(requested in REPRESENTATION_IDS[1:], "R0 is an existing control, not a new run")
    if requested in ROUND_2:
        require(set(ROUND_1).issubset(done), "Round 2 is closed until both Round 1 candidates terminate")
    require(requested not in done, "formal candidate may run only once")
