"""Portable NPZ input for new EDM-Loc training and evaluation runs.

The four arrays are raw OpenCSI amplitude/phase, physical positions in cm,
RP IDs and globally unique snapshot IDs. Split selection happens before this
loader; the training CLI checks Train/Development UID disjointness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .contracts import DatasetBatch
from .csi_representation import transform_csi


@dataclass(frozen=True)
class Arrays:
    raw_ap: np.ndarray
    coordinates_cm: np.ndarray
    rp_ids: np.ndarray
    global_uids: np.ndarray


def load_arrays(path: Path) -> Arrays:
    with np.load(path, allow_pickle=False) as data:
        required = {"raw_ap", "coordinates_cm", "rp_ids", "global_uids"}
        if not required.issubset(data.files):
            raise ValueError(f"{path}: missing {sorted(required - set(data.files))}")
        raw = np.asarray(data["raw_ap"], dtype=np.float32)
        positions = np.asarray(data["coordinates_cm"], dtype=np.float32)
        rp = np.asarray(data["rp_ids"], dtype=np.int64)
        uids = np.asarray(data["global_uids"], dtype=np.int64)
    n = len(raw)
    if raw.shape != (n, 2, 39, 4) or positions.shape != (n, 2) or rp.shape != (n,) or uids.shape != (n,):
        raise ValueError("NPZ shapes must be [N,2,39,4], [N,2], [N], [N]")
    if n == 0 or len(np.unique(uids)) != n or not np.isfinite(raw).all() or not np.isfinite(positions).all():
        raise ValueError("NPZ must be nonempty, finite and have unique global_uids")
    return Arrays(raw, positions, rp, uids)


def fit_train_stats(train: Arrays) -> dict[str, list[float]]:
    ri = transform_csi(train.raw_ap, "R1_FREQ_RI").astype(np.float64)
    feature_mean = ri.mean(axis=0)
    feature_std = ri.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    position_mean = train.coordinates_cm.astype(np.float64).mean(axis=0)
    position_std = train.coordinates_cm.astype(np.float64).std(axis=0)
    position_std[position_std == 0] = 1.0
    return {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "position_mean_cm": position_mean.tolist(),
        "position_std_cm": position_std.tolist(),
    }


def normalized_ri(arrays: Arrays, stats: dict) -> np.ndarray:
    ri = transform_csi(arrays.raw_ap, "R1_FREQ_RI").astype(np.float32)
    mean = np.asarray(stats["feature_mean"], dtype=np.float32)
    std = np.asarray(stats["feature_std"], dtype=np.float32)
    if mean.shape != (2, 39, 4) or std.shape != (2, 39, 4) or np.any(std <= 0):
        raise ValueError("invalid Train-only RI statistics")
    return (ri - mean) / std


def make_batch(arrays: Arrays, normalized: np.ndarray, indices: np.ndarray, dataset_id: str) -> DatasetBatch:
    selected = np.asarray(indices, dtype=np.int64)
    count = len(selected)
    return DatasetBatch(
        csi=torch.from_numpy(normalized[selected].copy()),
        coordinates_cm=torch.from_numpy(arrays.coordinates_cm[selected].copy()),
        rp_ids=torch.from_numpy(arrays.rp_ids[selected].copy()),
        global_uids=torch.from_numpy(arrays.global_uids[selected].copy()),
        subcarrier_mask=torch.ones(count, 39, dtype=torch.bool),
        port_mask=torch.ones(count, 4, dtype=torch.bool),
        dataset_id=dataset_id,
    )


def sample_metrics(samples: np.ndarray, truth: np.ndarray, rp_ids: np.ndarray) -> dict[str, float]:
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    rp_ids = np.asarray(rp_ids)
    if samples.ndim != 3 or samples.shape[1:] != (20, 2) or truth.shape != (len(samples), 2):
        raise ValueError("expected [N,20,2] samples and [N,2] truth")
    error = np.linalg.norm(samples.mean(axis=1) - truth, axis=1)
    rp_mean, rp_p90 = [], []
    for rp in np.unique(rp_ids):
        group = error[rp_ids == rp]
        rp_mean.append(group.mean())
        rp_p90.append(np.percentile(group, 90))
    direct = np.linalg.norm(samples - truth[:, None, :], axis=2).mean(axis=1)
    pairwise = np.linalg.norm(samples[:, :, None, :] - samples[:, None, :, :], axis=3).sum(axis=(1, 2))
    es = direct - pairwise / (2 * 20 * 19)
    return {
        "mean_cm": float(np.mean(error)),
        "rp_macro_mean_cm": float(np.mean(rp_mean)),
        "rp_macro_p90_cm": float(np.mean(rp_p90)),
        "energy_score_u20_cm": float(np.mean(es)),
    }
