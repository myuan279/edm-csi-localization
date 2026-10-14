"""OpenCSI loading and leakage-safe preprocessing.

The distributed ``data.h5`` already contains the official OpenCSI
preprocessing output: every CSI snapshot is represented by 39 TAP rows and
each row contains amplitude/phase values for four LTE antenna ports.  The
official tensor is therefore ``(2, 39, 4)`` (312 scalar values), not an
8-dimensional CSI vector.

This module deliberately separates four operations:

1. read snapshot-level data from HDF5;
2. assign train/validation/test indices;
3. fit clipping and normalization statistics on *training indices only*;
4. apply those fixed statistics lazily in ``Dataset.__getitem__``.

``legacy8`` exists only to reproduce the former, unsafe representation.  It
averages over the 39 subcarriers and calls ``np.unwrap(..., axis=0)`` across
ordered snapshots.  Do not use it for scientific results.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
from typing import Dict, Iterable, Literal, Sequence, Tuple

import numpy as np
import tables
import torch
from torch.utils.data import DataLoader, Dataset, Subset


FeatureRepresentation = Literal["full312", "summary8", "legacy8"]
SplitMode = Literal["random", "spatial"]

N_PORTS = 4
N_SUBCARRIERS = 39
N_CHANNELS = 2
FULL_FEATURE_DIM = N_CHANNELS * N_SUBCARRIERS * N_PORTS
SUMMARY_FEATURE_DIM = N_CHANNELS * N_PORTS

RAW_FEATURE_COLUMNS = [
    "CE_0_AMPLITUDE",
    "CE_0_PHASE",
    "CE_1_AMPLITUDE",
    "CE_1_PHASE",
    "CE_2_AMPLITUDE",
    "CE_2_PHASE",
    "CE_3_AMPLITUDE",
    "CE_3_PHASE",
]
AMP_RAW_INDEX = np.asarray([0, 2, 4, 6])
PHASE_RAW_INDEX = np.asarray([1, 3, 5, 7])

ROOM_WIDTH_CM, ROOM_HEIGHT_CM = 500.0, 350.0


@dataclass
class OpenCSIArrayBundle:
    """Raw snapshot-aligned arrays before train-only preprocessing."""

    features: np.ndarray
    positions: np.ndarray
    fingerprint_ids: np.ndarray
    snapshot_ids: np.ndarray
    source_indices: np.ndarray
    representation: FeatureRepresentation
    input_shape: Tuple[int, ...]
    h5_path: str


@dataclass
class PreprocessingStats:
    """Parameters fitted exclusively from the training subset."""

    csi_mean: np.ndarray
    csi_std: np.ndarray
    pos_mean: np.ndarray
    pos_std: np.ndarray
    clip_low: np.ndarray | None
    clip_high: np.ndarray | None
    train_sample_count: int


def _expected_hdf_columns() -> list[str]:
    return ["FINGERPRINT_ID", "SNAPSHOT_ID", "TAP_ID", *RAW_FEATURE_COLUMNS]


def _validate_hdf_layout(h5: tables.File) -> int:
    if not hasattr(h5.root, "X") or not hasattr(h5.root.X, "table"):
        raise ValueError("HDF5 file is missing /X/table")
    if not hasattr(h5.root, "y") or not hasattr(h5.root.y, "table"):
        raise ValueError("HDF5 file is missing /y/table")

    x_table = h5.root.X.table
    y_table = h5.root.y.table
    n_snapshots = int(y_table.nrows)
    if n_snapshots == 0:
        raise ValueError("OpenCSI HDF5 contains no snapshots")
    if int(x_table.nrows) != n_snapshots * N_SUBCARRIERS:
        raise ValueError(
            f"Expected {N_SUBCARRIERS} TAP rows per snapshot, got "
            f"X={x_table.nrows:,}, y={n_snapshots:,}"
        )

    columns = list(h5.root.X._v_attrs.non_index_axes[0][1])
    if columns != _expected_hdf_columns():
        raise ValueError(f"Unexpected /X column order: {columns}")
    return n_snapshots


def _select_source_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = rng.choice(n, size=max_samples, replace=False)
    selected.sort()
    return selected.astype(np.int64, copy=False)


def _to_official_full312(raw: np.ndarray) -> np.ndarray:
    """Convert ``(N, 39, 8 interleaved)`` to ``(N, 2*39*4)``.

    Flattening order is channel (amplitude, phase), subcarrier, antenna port.
    Reshaping a returned row to ``(2, 39, 4)`` restores the CSI-image described
    in the OpenCSI paper.
    """

    amp = raw[:, :, AMP_RAW_INDEX]
    phase = raw[:, :, PHASE_RAW_INDEX]
    return np.stack((amp, phase), axis=1).reshape(len(raw), FULL_FEATURE_DIM)


def _to_summary8(raw: np.ndarray, legacy_unwrap: bool) -> np.ndarray:
    amp = raw[:, :, AMP_RAW_INDEX].mean(axis=1)
    phase = raw[:, :, PHASE_RAW_INDEX].mean(axis=1)
    if legacy_unwrap:
        # Intentionally unsafe and order-dependent. Kept only for the P0 audit.
        phase = np.unwrap(phase, axis=0)
    return np.concatenate((amp, phase), axis=1).astype(np.float32, copy=False)


def _read_selected_snapshots(
    h5: tables.File,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read arbitrary snapshots without loading the 56.9M-row frame."""

    tap_offsets = np.arange(N_SUBCARRIERS, dtype=np.int64)
    row_coordinates = (
        source_indices[:, None] * N_SUBCARRIERS + tap_offsets[None, :]
    ).reshape(-1)
    x_table = h5.root.X.table
    raw = x_table.read_coordinates(
        row_coordinates, field="values_block_1"
    ).reshape(len(source_indices), N_SUBCARRIERS, len(RAW_FEATURE_COLUMNS))
    identifiers = x_table.read_coordinates(
        source_indices * N_SUBCARRIERS, field="values_block_2"
    )
    positions = h5.root.y.table.read_coordinates(
        source_indices, field="values_block_0"
    )
    return raw.astype(np.float32, copy=False), identifiers, positions


def _cache_paths(h5_path: Path, cache_dir: Path) -> tuple[Path, Path]:
    stat = h5_path.stat()
    signature = sha1(
        f"{h5_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|full312-v1".encode()
    ).hexdigest()[:16]
    return (
        cache_dir / f"opencsi_full312_{signature}.npy",
        cache_dir / f"opencsi_full312_{signature}.json",
    )


def _load_or_create_full_cache(
    h5: tables.File,
    h5_path: Path,
    n_snapshots: int,
    cache_dir: Path,
    chunk_snapshots: int = 20_000,
) -> np.memmap:
    """Create a reusable ~1.8GB memory-mapped official representation."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, metadata_path = _cache_paths(h5_path, cache_dir)
    expected = {
        "shape": [n_snapshots, FULL_FEATURE_DIM],
        "dtype": "float32",
        "representation": "full312",
        "source": str(h5_path.resolve()),
    }
    if array_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == expected:
                return np.load(array_path, mmap_mode="r")
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    mmap = np.lib.format.open_memmap(
        array_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_snapshots, FULL_FEATURE_DIM),
    )
    x_table = h5.root.X.table
    for start in range(0, n_snapshots, chunk_snapshots):
        stop = min(start + chunk_snapshots, n_snapshots)
        raw = x_table.read(
            start * N_SUBCARRIERS,
            stop * N_SUBCARRIERS,
            field="values_block_1",
        ).reshape(stop - start, N_SUBCARRIERS, len(RAW_FEATURE_COLUMNS))
        mmap[start:stop] = _to_official_full312(raw)
    mmap.flush()
    metadata_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return np.load(array_path, mmap_mode="r")


def _read_all_summary(
    h5: tables.File,
    n_snapshots: int,
    legacy_unwrap: bool,
    chunk_snapshots: int = 50_000,
) -> np.ndarray:
    features = np.empty((n_snapshots, SUMMARY_FEATURE_DIM), dtype=np.float32)
    x_table = h5.root.X.table
    for start in range(0, n_snapshots, chunk_snapshots):
        stop = min(start + chunk_snapshots, n_snapshots)
        raw = x_table.read(
            start * N_SUBCARRIERS,
            stop * N_SUBCARRIERS,
            field="values_block_1",
        ).reshape(stop - start, N_SUBCARRIERS, len(RAW_FEATURE_COLUMNS))
        # Legacy unwrap must see the complete ordered phase sequence, so only
        # average inside chunks here and unwrap once after concatenation.
        features[start:stop] = _to_summary8(raw, legacy_unwrap=False)
    if legacy_unwrap:
        features[:, N_PORTS:] = np.unwrap(features[:, N_PORTS:], axis=0)
    return features


def load_opencsi_arrays(
    h5_path: str,
    representation: FeatureRepresentation = "full312",
    max_samples: int = 0,
    seed: int = 42,
    row_order_seed: int | None = None,
    cache_dir: str = ".cache/opencsi",
) -> OpenCSIArrayBundle:
    """Load aligned OpenCSI snapshots in one of three representations.

    ``row_order_seed`` is an audit-only option.  It permutes complete snapshots
    and their labels before representation construction.  Correct
    representations remain value-invariant; ``legacy8`` does not.
    """

    if representation not in {"full312", "summary8", "legacy8"}:
        raise ValueError(f"Unknown representation: {representation}")

    path = Path(h5_path)
    with tables.open_file(path, mode="r") as h5:
        n_snapshots = _validate_hdf_layout(h5)
        source_indices = _select_source_indices(n_snapshots, max_samples, seed)
        selected_all = len(source_indices) == n_snapshots

        if selected_all and representation == "full312" and row_order_seed is None:
            features = _load_or_create_full_cache(
                h5, path, n_snapshots, Path(cache_dir)
            )
            identifiers = h5.root.X.table.cols.values_block_2[::N_SUBCARRIERS]
            positions = h5.root.y.table.cols.values_block_0[:]
        elif selected_all and representation in {"summary8", "legacy8"}:
            features = _read_all_summary(
                h5, n_snapshots, legacy_unwrap=False
            )
            identifiers = h5.root.X.table.cols.values_block_2[::N_SUBCARRIERS]
            positions = h5.root.y.table.cols.values_block_0[:]
        else:
            raw, identifiers, positions = _read_selected_snapshots(h5, source_indices)
            features = raw  # converted after the optional row permutation

    if row_order_seed is not None:
        rng = np.random.default_rng(row_order_seed)
        order = rng.permutation(len(source_indices))
        source_indices = source_indices[order]
        identifiers = identifiers[order]
        positions = positions[order]
        features = features[order]

    if features.ndim == 3:
        if representation == "full312":
            features = _to_official_full312(features)
        else:
            features = _to_summary8(
                features, legacy_unwrap=representation == "legacy8"
            )
    elif representation == "legacy8":
        # Full-data summary was intentionally constructed without unwrap above
        # so an audit row permutation, if requested, affects the unsafe step.
        features = np.asarray(features).copy()
        features[:, N_PORTS:] = np.unwrap(features[:, N_PORTS:], axis=0)

    input_shape = (
        (N_CHANNELS, N_SUBCARRIERS, N_PORTS)
        if representation == "full312"
        else (SUMMARY_FEATURE_DIM,)
    )
    return OpenCSIArrayBundle(
        features=features,
        positions=np.asarray(positions, dtype=np.float32),
        fingerprint_ids=np.asarray(identifiers[:, 0]),
        snapshot_ids=np.asarray(identifiers[:, 1]),
        source_indices=np.asarray(source_indices),
        representation=representation,
        input_shape=input_shape,
        h5_path=str(path),
    )


def make_split_indices(
    fingerprint_ids: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_mode: SplitMode = "spatial",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic sample indices.

    ``spatial`` is an RP-disjoint split.  ``random`` is a snapshot split and is
    retained only as a leakage/control experiment.
    """

    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0):
        raise ValueError("Invalid train/validation ratios")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = np.random.default_rng(seed)
    n = len(fingerprint_ids)
    if split_mode == "random":
        order = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            np.sort(order[:n_train]),
            np.sort(order[n_train:n_train + n_val]),
            np.sort(order[n_train + n_val:]),
        )
    if split_mode != "spatial":
        raise ValueError(f"Unknown split mode: {split_mode}")

    all_rp = np.unique(fingerprint_ids).copy()
    rng.shuffle(all_rp)
    n_train_rp = int(len(all_rp) * train_ratio)
    n_val_rp = int(len(all_rp) * val_ratio)
    train_rp = all_rp[:n_train_rp]
    val_rp = all_rp[n_train_rp:n_train_rp + n_val_rp]
    test_rp = all_rp[n_train_rp + n_val_rp:]
    idx = np.arange(n, dtype=np.int64)
    return (
        idx[np.isin(fingerprint_ids, train_rp)],
        idx[np.isin(fingerprint_ids, val_rp)],
        idx[np.isin(fingerprint_ids, test_rp)],
    )


def _iter_index_chunks(indices: np.ndarray, size: int = 50_000) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def _feature_moments(
    features: np.ndarray,
    indices: np.ndarray,
    clip_low: np.ndarray | None = None,
    clip_high: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(features.shape[1], dtype=np.float64)
    total_sq = np.zeros(features.shape[1], dtype=np.float64)
    count = 0
    for chunk_indices in _iter_index_chunks(indices):
        values = np.asarray(features[chunk_indices], dtype=np.float64)
        if clip_low is not None and clip_high is not None:
            values = np.clip(values, clip_low, clip_high)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def fit_preprocessing(
    bundle: OpenCSIArrayBundle,
    train_indices: Sequence[int],
    normalize: bool = True,
    outlier_clip_std: float | None = None,
) -> PreprocessingStats:
    """Fit preprocessing using training samples and no other rows."""

    train_indices = np.asarray(train_indices, dtype=np.int64)
    if len(train_indices) == 0:
        raise ValueError("Training subset is empty")
    if outlier_clip_std is None:
        outlier_clip_std = 0.0 if bundle.representation == "full312" else 3.0

    raw_mean, raw_std = _feature_moments(bundle.features, train_indices)
    if outlier_clip_std > 0:
        clip_low = raw_mean - outlier_clip_std * raw_std
        clip_high = raw_mean + outlier_clip_std * raw_std
        csi_mean, csi_std = _feature_moments(
            bundle.features, train_indices, clip_low, clip_high
        )
    else:
        clip_low = clip_high = None
        csi_mean, csi_std = raw_mean, raw_std

    pos_train = np.asarray(bundle.positions[train_indices], dtype=np.float64)
    pos_mean = pos_train.mean(axis=0).astype(np.float32)
    pos_std = pos_train.std(axis=0).astype(np.float32)

    if not normalize:
        csi_mean = np.zeros_like(csi_mean)
        csi_std = np.ones_like(csi_std)
        pos_mean = np.zeros_like(pos_mean)
        pos_std = np.ones_like(pos_std)
    else:
        csi_std = np.maximum(csi_std, 1e-8)
        pos_std = np.maximum(pos_std, 1e-8)

    return PreprocessingStats(
        csi_mean=csi_mean,
        csi_std=csi_std,
        pos_mean=pos_mean,
        pos_std=pos_std,
        clip_low=clip_low,
        clip_high=clip_high,
        train_sample_count=len(train_indices),
    )


class _NormalizedAccessor:
    def __init__(self, dataset: "OpenCSIDataset", kind: Literal["csi", "pos"]):
        self.dataset = dataset
        self.kind = kind

    @property
    def shape(self) -> tuple[int, int]:
        width = self.dataset.feature_dim if self.kind == "csi" else 2
        return len(self.dataset), width

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        if self.kind == "csi":
            return self.dataset.get_csi(index)
        return self.dataset.get_pos(index)


class OpenCSIDataset(Dataset):
    """Leakage-safe normalized view over an ``OpenCSIArrayBundle``."""

    def __init__(self, bundle: OpenCSIArrayBundle, stats: PreprocessingStats):
        super().__init__()
        self.bundle = bundle
        self.stats = stats
        self.raw_csi = bundle.features
        self.raw_pos = bundle.positions
        self.fingerprint_ids = bundle.fingerprint_ids
        self.snapshot_ids = bundle.snapshot_ids
        self.source_indices = bundle.source_indices
        self.representation = bundle.representation
        self.input_shape = bundle.input_shape
        self.feature_dim = bundle.features.shape[1]

        # Backward-compatible attributes used by train/evaluate code.
        self.csi_mean = stats.csi_mean
        self.csi_std = stats.csi_std
        self.pos_mean = stats.pos_mean
        self.pos_std = stats.pos_std
        self.csi = _NormalizedAccessor(self, "csi")
        self.pos = _NormalizedAccessor(self, "pos")

    def __len__(self) -> int:
        return len(self.raw_csi)

    def get_csi(self, index) -> torch.Tensor:
        values = np.asarray(self.raw_csi[index], dtype=np.float32)
        if self.stats.clip_low is not None and self.stats.clip_high is not None:
            values = np.clip(values, self.stats.clip_low, self.stats.clip_high)
        values = (values - self.stats.csi_mean) / self.stats.csi_std
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))

    def get_pos(self, index) -> torch.Tensor:
        values = np.asarray(self.raw_pos[index], dtype=np.float32)
        values = (values - self.stats.pos_mean) / self.stats.pos_std
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_csi(idx), self.get_pos(idx)

    def denormalize_pos(self, pos_norm: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.pos_mean, dtype=torch.float32, device=pos_norm.device)
        std = torch.as_tensor(self.pos_std, dtype=torch.float32, device=pos_norm.device)
        return pos_norm * std + mean

    def error_cm(self, pred_norm: torch.Tensor, true_norm: torch.Tensor) -> Dict[str, float]:
        pred_cm = self.denormalize_pos(pred_norm.detach().cpu())
        true_cm = self.denormalize_pos(true_norm.detach().cpu())
        distances = (pred_cm - true_cm).norm(dim=1)
        return {
            "mean_cm": distances.mean().item(),
            "median_cm": distances.median().item(),
            "p90_cm": distances.quantile(0.90).item(),
        }


def build_dataloaders(
    h5_path: str,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_mode: SplitMode = "spatial",
    representation: FeatureRepresentation = "full312",
    num_workers: int = 0,
    seed: int = 42,
    max_samples: int = 0,
    normalize: bool = True,
    outlier_clip_std: float | None = None,
    row_order_seed: int | None = None,
    cache_dir: str = ".cache/opencsi",
) -> Tuple[DataLoader, DataLoader, DataLoader, OpenCSIDataset]:
    bundle = load_opencsi_arrays(
        h5_path=h5_path,
        representation=representation,
        max_samples=max_samples,
        seed=seed,
        row_order_seed=row_order_seed,
        cache_dir=cache_dir,
    )
    train_idx, val_idx, test_idx = make_split_indices(
        bundle.fingerprint_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_mode=split_mode,
        seed=seed,
    )
    stats = fit_preprocessing(
        bundle,
        train_idx,
        normalize=normalize,
        outlier_clip_std=outlier_clip_std,
    )
    dataset = OpenCSIDataset(bundle, stats)

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        train_ds, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    n_rp = np.unique(bundle.fingerprint_ids).size
    print(
        f"[OpenCSI] N={len(dataset):,} RPs={n_rp:,} "
        f"representation={representation} CSI dim={dataset.feature_dim}"
    )
    label = "RP-disjoint" if split_mode == "spatial" else "snapshot-random CONTROL"
    print(
        f"[{label}] train={len(train_idx):,} val={len(val_idx):,} "
        f"test={len(test_idx):,}; preprocessing fit on train only"
    )
    return train_loader, val_loader, test_loader, dataset
