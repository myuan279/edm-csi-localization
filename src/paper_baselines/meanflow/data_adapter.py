from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from opencsi_data import OpenCSIArrayBundle

SNAPSHOT_COUNT = 1_458_078
RAW_SHAPE = (2, 39, 4)
FEATURE_DIM = 312
TRAIN_ROWS = 300_000
PANEL_ROWS = 3_980
ALLOWED_SUBSETS = frozenset({"development_train_300k", "selection_validation_rp10"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def int64_sha256(values: Sequence[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def resolve_path(value: str | Path, *, base_dir: str | Path) -> Path:
    text = os.path.expandvars(str(value))
    path = Path(text)
    return path if path.is_absolute() else Path(base_dir) / path


def _resolve_project_path(value: str | Path, *, config_dir: str | Path) -> Path:
    """Resolve config-relative paths, then the project root for template configs."""

    path = resolve_path(value, base_dir=config_dir)
    if path.exists() or Path(value).is_absolute():
        return path
    start = Path(config_dir).resolve()
    for parent in (start, *start.parents):
        candidate = resolve_path(value, base_dir=parent)
        if candidate.exists():
            return candidate
    # Preserve a useful missing-path error while keeping the resolution rule
    # deterministic for a server-bound config whose assets are not mounted.
    return resolve_path(value, base_dir=start.parents[-1])


def ri_transform(raw_full312: np.ndarray) -> np.ndarray:
    """Convert official ``(A, phase)`` full312 rows to ``(Re, Im)`` rows."""

    raw = np.asarray(raw_full312, dtype=np.float32)
    if raw.ndim == 2 and raw.shape[1] == FEATURE_DIM:
        raw = raw.reshape((-1,) + RAW_SHAPE)
    if raw.ndim != 4 or tuple(raw.shape[1:]) != RAW_SHAPE:
        raise ValueError("expected full312 raw rows with shape [N,2,39,4]")
    if not np.isfinite(raw).all():
        raise ValueError("raw CSI contains non-finite values")
    amplitude, phase = raw[:, 0], raw[:, 1]
    return np.stack((amplitude * np.cos(phase), amplitude * np.sin(phase)), axis=1).reshape(
        len(raw), FEATURE_DIM
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class NormalizationStats:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    fit_uid_sha256: str
    fit_rows: int

    def validate(self) -> None:
        if self.x_mean.shape != (FEATURE_DIM,) or self.x_std.shape != (FEATURE_DIM,):
            raise ValueError("CSI normalization must contain 312 features")
        if self.y_mean.shape != (2,) or self.y_std.shape != (2,):
            raise ValueError("position normalization must be two-dimensional")
        if not np.isfinite(self.x_mean).all() or not np.isfinite(self.y_mean).all():
            raise ValueError("normalization means are non-finite")
        if not np.all(self.x_std > 0) or not np.all(self.y_std > 0):
            raise ValueError("normalization scales must be positive")
        if self.fit_rows != TRAIN_ROWS or len(self.fit_uid_sha256) != 64:
            raise ValueError("normalization is not bound to the fixed 300K Train panel")


@dataclass(frozen=True)
class ResolvedData:
    manifest: dict[str, Any]
    h5_path: Path
    cache_dir: Path
    normalization_path: Path
    manifest_path: Path
    normalization: NormalizationStats


def load_normalization(path: str | Path) -> NormalizationStats:
    with np.load(path, allow_pickle=False) as source:
        fit_hash = str(np.asarray(source["fit_uid_sha256"]).item())
        stats = NormalizationStats(
            x_mean=np.asarray(source["x_mean"], dtype=np.float64).reshape(FEATURE_DIM),
            x_std=np.asarray(source["x_std"], dtype=np.float64).reshape(FEATURE_DIM),
            y_mean=np.asarray(source["y_mean"], dtype=np.float64).reshape(2),
            y_std=np.asarray(source["y_std"], dtype=np.float64).reshape(2),
            fit_uid_sha256=fit_hash,
            fit_rows=int(np.asarray(source["fit_rows"]).item()),
        )
    stats.validate()
    return stats


def load_resolved_data(config: dict[str, Any], *, config_dir: str | Path) -> ResolvedData:
    data = config.get("data", {})
    required = ("manifest", "h5", "cache_dir", "normalization")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"resolved config is missing data bindings: {missing}")
    manifest_path = _resolve_project_path(data["manifest"], config_dir=config_dir)
    h5_path = _resolve_project_path(data["h5"], config_dir=config_dir)
    cache_dir = _resolve_project_path(data["cache_dir"], config_dir=config_dir)
    normalization_path = _resolve_project_path(data["normalization"], config_dir=config_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset", {}).get("input_shape") != [2, 39, 4]:
        raise ValueError("manifest is not the audited full312 layout")
    if int(manifest["dataset"]["snapshot_count"]) != SNAPSHOT_COUNT:
        raise ValueError("OpenCSI snapshot count drift")
    normalization = load_normalization(normalization_path)
    train = np.asarray(manifest["subsets"]["development_train_300k"]["source_snapshot_indices"], dtype=np.int64)
    if len(train) != TRAIN_ROWS or int64_sha256(train) != manifest["subsets"]["development_train_300k"]["source_snapshot_indices_sha256"]:
        raise ValueError("development_train_300k UID manifest drift")
    if normalization.fit_uid_sha256 != manifest["subsets"]["development_train_300k"]["source_snapshot_indices_sha256"]:
        raise ValueError("normalization is not fitted on development_train_300k")
    return ResolvedData(manifest, h5_path, cache_dir, normalization_path, manifest_path, normalization)


def _cache_paths(h5_path: Path, cache_dir: Path) -> tuple[Path, Path]:
    stat = h5_path.stat()
    signature = hashlib.sha1(
        f"{h5_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|full312-v1".encode()
    ).hexdigest()[:16]
    return cache_dir / f"opencsi_full312_{signature}.npy", cache_dir / f"opencsi_full312_{signature}.json"


def load_permitted_bundle(data: ResolvedData, subset: str) -> tuple[OpenCSIArrayBundle, np.ndarray]:
    if subset not in ALLOWED_SUBSETS:
        raise ValueError(f"subset is outside the Development whitelist: {subset}")
    entry = data.manifest["subsets"][subset]
    indices = np.asarray(entry["source_snapshot_indices"], dtype=np.int64)
    expected_rows = TRAIN_ROWS if subset == "development_train_300k" else PANEL_ROWS
    if len(indices) != expected_rows or int64_sha256(indices) != entry["source_snapshot_indices_sha256"]:
        raise ValueError(f"{subset} UID manifest drift")
    cache_path, metadata_path = _cache_paths(data.h5_path, data.cache_dir)
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("audited full312 cache is required; adapter will not rebuild it")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_meta = {
        "shape": [SNAPSHOT_COUNT, FEATURE_DIM],
        "dtype": "float32",
        "representation": "full312",
        "source": str(data.h5_path.resolve()),
    }
    if metadata != expected_meta:
        raise ValueError("full312 cache metadata drift")
    features = np.load(cache_path, mmap_mode="r")
    if features.shape != (SNAPSHOT_COUNT, FEATURE_DIM) or features.dtype != np.float32:
        raise ValueError("full312 cache shape/dtype drift")

    import tables

    positions = np.full((SNAPSHOT_COUNT, 2), np.nan, dtype=np.float32)
    fingerprint_ids = np.full(SNAPSHOT_COUNT, -1, dtype=np.int64)
    snapshot_ids = np.full(SNAPSHOT_COUNT, -1, dtype=np.int64)
    with tables.open_file(str(data.h5_path), mode="r") as handle:
        for start in range(0, len(indices), 50_000):
            current = indices[start : start + 50_000]
            identifiers = handle.root.X.table.read_coordinates(current * 39, field="values_block_2")
            labels = handle.root.y.table.read_coordinates(current, field="values_block_0")
            fingerprint_ids[current] = np.asarray(identifiers[:, 0], dtype=np.int64)
            snapshot_ids[current] = np.asarray(identifiers[:, 1], dtype=np.int64)
            positions[current] = np.asarray(labels, dtype=np.float32)
    if np.any(fingerprint_ids[indices] < 0) or not np.isfinite(positions[indices]).all():
        raise ValueError("permitted rows failed to load")
    bundle = OpenCSIArrayBundle(
        features=features,
        positions=positions,
        fingerprint_ids=fingerprint_ids,
        snapshot_ids=snapshot_ids,
        source_indices=np.arange(SNAPSHOT_COUNT, dtype=np.int64),
        representation="full312",
        input_shape=RAW_SHAPE,
        h5_path=str(data.h5_path),
    )
    return bundle, indices


class RINormalizedDataset(Dataset):
    def __init__(self, bundle: OpenCSIArrayBundle, indices: np.ndarray, stats: NormalizationStats):
        self.bundle, self.indices, self.stats = bundle, np.asarray(indices, dtype=np.int64), stats
        stats.validate()
        if bundle.input_shape != RAW_SHAPE or bundle.representation != "full312":
            raise ValueError("RI adapter requires full312 source rows")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        uid = int(self.indices[item])
        raw = np.asarray(self.bundle.features[uid], dtype=np.float32)[None, :]
        ri = ri_transform(raw)[0]
        csi = (ri.astype(np.float64) - self.stats.x_mean) / self.stats.x_std
        truth = np.asarray(self.bundle.positions[uid], dtype=np.float64)
        pos = (truth - self.stats.y_mean) / self.stats.y_std
        return {
            "condition": torch.from_numpy(np.ascontiguousarray(csi, dtype=np.float32)),
            "position": torch.from_numpy(np.ascontiguousarray(pos, dtype=np.float32)),
            "truth_cm": torch.from_numpy(np.ascontiguousarray(truth, dtype=np.float32)),
            "global_uid": uid,
            "rp_id": int(self.bundle.fingerprint_ids[uid]),
        }


def write_data_access_receipt(
    path: str | Path,
    *,
    subset: str,
    indices: np.ndarray,
    data: ResolvedData,
    stage: str = "Development_supplemental",
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    expected_hashes = expected_hashes or {}
    receipt = {
        "protocol_id": "ri_acceleration_onestep_v1",
        "stage": stage,
        "subset": subset,
        "rows": int(len(indices)),
        "uid_sha256": int64_sha256(indices),
        "allowed_subsets": sorted(ALLOWED_SUBSETS),
        "h5_sha256": expected_hashes.get("h5_sha256") or sha256_file(data.h5_path),
        "manifest_sha256": expected_hashes.get("manifest_sha256") or sha256_file(data.manifest_path),
        "normalization_sha256": expected_hashes.get("normalization_sha256") or sha256_file(data.normalization_path),
        "hash_binding": "preflight_verified_config" if expected_hashes else "computed_at_run",
        "normalization_fit_rows": data.normalization.fit_rows,
        "normalization_fit_uid_sha256": data.normalization.fit_uid_sha256,
        "protected_access": {"validation": 0, "calibration": 0, "test": 0},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    return receipt
