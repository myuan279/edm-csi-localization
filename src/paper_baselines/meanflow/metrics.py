from __future__ import annotations

import numpy as np


def energy_score_u(samples_cm: np.ndarray, truth_cm: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples_cm, dtype=np.float64)
    truth = np.asarray(truth_cm, dtype=np.float64)
    first = np.linalg.norm(samples - truth[:, None, :], axis=2).mean(axis=1)
    pairwise = np.linalg.norm(samples[:, :, None, :] - samples[:, None, :, :], axis=3)
    s = samples.shape[1]
    return first - pairwise.sum(axis=(1, 2)) / (2.0 * s * (s - 1))


def metrics_from_samples(samples_cm: np.ndarray, truth_cm: np.ndarray, rp_ids: np.ndarray, uids: np.ndarray) -> dict:
    samples = np.asarray(samples_cm, dtype=np.float64)
    truth = np.asarray(truth_cm, dtype=np.float64)
    rp_ids = np.asarray(rp_ids)
    uids = np.asarray(uids, dtype=np.int64)
    if samples.ndim != 3 or samples.shape[2] != 2 or samples.shape[0] != len(truth):
        raise ValueError("samples must be [N,S,2] and aligned with truth")
    if len(np.unique(uids)) != len(uids):
        raise ValueError("UIDs must be unique")
    mean = samples.mean(axis=1)
    error = np.linalg.norm(mean - truth, axis=1)
    groups = np.unique(rp_ids)
    rp_mean = np.asarray([error[rp_ids == group].mean() for group in groups])
    rp_p90 = np.asarray([np.quantile(error[rp_ids == group], 0.90, method="linear") for group in groups])
    result = {
        "mean_cm": float(error.mean()),
        "p90_cm": float(np.quantile(error, 0.90, method="linear")),
        "p99_cm": float(np.quantile(error, 0.99, method="linear")),
        "rp_macro_mean_cm": float(rp_mean.mean()),
        "rp_macro_p90_cm": float(rp_p90.mean()),
    }
    if samples.shape[1] >= 2:
        result["energy_score_u20_cm"] = float(energy_score_u(samples, truth).mean())
        centered = samples - mean[:, None, :]
        result["sample_covariance_mean_trace_cm2"] = float(np.trace(np.einsum("nsi,nsj->nij", centered, centered) / (samples.shape[1] - 1), axis1=1, axis2=2).mean())
    return result
