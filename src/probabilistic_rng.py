"""Canonical UID and RNG primitives for reproducible sampling.

This module is additive.  It does not replace the frozen v1 samplers.  The
canonical reference is CPU NumPy in the locked environment, with float32 base
noise persisted before any model-specific affine transformation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Sequence

import numpy as np


PROTOCOL_ID = "probabilistic_rng_identity_revalidation_v2"
PROTOCOL_RNG_SEED = 20260729
POSTERIOR_SAMPLE_COUNT = 20
PAIRED_NOISE_GROUP = "gaussian_fullcov_homoscedastic_v2"
GAUSSIAN_SELECTION_NAMESPACE = "gaussian_checkpoint_selection_v2"
GAUSSIAN_VALIDATION_NAMESPACE = "gaussian_complete_validation_v2"
DDPM_INITIAL_NAMESPACE = "ddpm_initial_noise_v2"
DDPM_REVERSE_NAMESPACE = "ddpm_reverse_step_noise_v2"
DDIM_INITIAL_NAMESPACE = "ddim_initial_noise_v2"
DPM_SOLVER_INITIAL_NAMESPACE = "dpm_solver_initial_noise_v2"

_UINT64_MASK = (1 << 64) - 1
_TWO_POW_53 = float(1 << 53)
_UID_COEFFICIENT = np.uint64(0xD2B74407B1CE6E93)
_SAMPLE_COEFFICIENT = np.uint64(0xCA5A826395121157)
_TIMESTEP_COEFFICIENT = np.uint64(0x9E3779B97F4A7C15)
_STREAM_ZERO_SALT = np.uint64(0xA24BAED4963EE407)
_STREAM_ONE_SALT = np.uint64(0x9FB21C651E98DF25)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def stable_text_code(value: str) -> int:
    """Return a stable unsigned 64-bit code without Python's ``hash()``."""

    return int.from_bytes(sha256(value.encode("utf-8")).digest()[:8], "little")


def stable_uid_bytes(
    data_sha256: str,
    split_name: str,
    global_hdf5_row_index: int,
) -> bytes:
    """Stable serialization of the authoritative global HDF5 row identity."""

    if len(data_sha256) != 64:
        raise ValueError("data_sha256 must be a 64-character SHA-256 digest")
    if not split_name:
        raise ValueError("split_name must be non-empty")
    row = int(global_hdf5_row_index)
    if row < 0:
        raise ValueError("global_hdf5_row_index must be non-negative")
    return canonical_json_bytes(
        {
            "data_sha256": data_sha256,
            "global_hdf5_row_index": row,
            "split_name": split_name,
        }
    )


def uid_mapping_sha256(
    data_sha256: str,
    split_name: str,
    global_hdf5_row_indices: Sequence[int] | np.ndarray,
) -> str:
    values = np.asarray(global_hdf5_row_indices, dtype="<i8")
    digest = sha256()
    digest.update(
        canonical_json_bytes(
            {
                "data_sha256": data_sha256,
                "dtype": "<i8",
                "split_name": split_name,
            }
        )
    )
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def uid_audit_summary(
    data_sha256: str,
    split_name: str,
    global_hdf5_row_indices: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(global_hdf5_row_indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("global HDF5 row indices must be one-dimensional")
    unique, counts = np.unique(values, return_counts=True)
    return {
        "row_count": int(values.size),
        "unique_uid_count": int(unique.size),
        "duplicate_count": int(values.size - unique.size),
        "maximum_multiplicity": int(counts.max()) if counts.size else 0,
        "minimum_uid": int(unique.min()) if unique.size else None,
        "maximum_uid": int(unique.max()) if unique.size else None,
        "mapping_sha256": uid_mapping_sha256(data_sha256, split_name, values),
    }


@dataclass(frozen=True)
class RNGKey:
    protocol_rng_seed: int
    stream_namespace: str
    paired_noise_group: str
    training_seed: int
    global_snapshot_uid: int
    posterior_sample_id: int
    noise_role: str
    optional_reverse_timestep: int | None = None

    def validate(self) -> None:
        if int(self.protocol_rng_seed) < 0:
            raise ValueError("protocol_rng_seed must be non-negative")
        if not self.stream_namespace or not self.paired_noise_group:
            raise ValueError("RNG namespace and paired-noise group are required")
        if int(self.global_snapshot_uid) < 0:
            raise ValueError("global_snapshot_uid must be non-negative")
        sample = int(self.posterior_sample_id)
        if not 0 <= sample < POSTERIOR_SAMPLE_COUNT:
            raise ValueError("posterior_sample_id must be in 0..19")
        if not self.noise_role:
            raise ValueError("noise_role is required")
        if self.noise_role == "reverse_step":
            if self.optional_reverse_timestep is None:
                raise ValueError("reverse_step requires optional_reverse_timestep")
            if int(self.optional_reverse_timestep) < 0:
                raise ValueError("reverse timestep must be non-negative")
        elif self.optional_reverse_timestep is not None:
            raise ValueError("only reverse_step may carry a reverse timestep")

    def canonical_bytes(self) -> bytes:
        self.validate()
        return canonical_json_bytes(asdict(self))

    def sha256(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()


def _splitmix64(values: np.ndarray | np.uint64) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = np.asarray(values, dtype=np.uint64) + np.uint64(
            0x9E3779B97F4A7C15
        )
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _uniform_open01(values: np.ndarray) -> np.ndarray:
    mantissa = np.right_shift(values, np.uint64(11)).astype(np.float64)
    return (mantissa + 0.5) / _TWO_POW_53


def gaussian_base_noise(
    global_snapshot_uids: Sequence[int] | np.ndarray,
    *,
    training_seed: int,
    stream_namespace: str,
    paired_noise_group: str = PAIRED_NOISE_GROUP,
    protocol_rng_seed: int = PROTOCOL_RNG_SEED,
    sample_count: int = POSTERIOR_SAMPLE_COUNT,
    noise_role: str = "gaussian_base",
) -> np.ndarray:
    """Generate canonical CPU float32 N(0,I_2) base noise.

    The output is addressed only by immutable key fields.  Ordering the input
    UIDs differently permutes rows and cannot change any UID/sample value.
    """

    uids = np.asarray(global_snapshot_uids, dtype=np.int64)
    if uids.ndim != 1 or np.any(uids < 0):
        raise ValueError("global_snapshot_uids must be non-negative 1-D values")
    if not 1 <= int(sample_count) <= POSTERIOR_SAMPLE_COUNT:
        raise ValueError("sample_count must be in 1..20")
    if not stream_namespace or not paired_noise_group or not noise_role:
        raise ValueError("all textual RNG key fields are required")

    namespace_code = stable_text_code(stream_namespace)
    pair_code = stable_text_code(paired_noise_group)
    role_code = stable_text_code(noise_role)
    scalar = (
        int(protocol_rng_seed)
        ^ namespace_code
        ^ ((pair_code << 1) & _UINT64_MASK)
        ^ ((int(training_seed) << 17) & _UINT64_MASK)
        ^ ((role_code << 7) & _UINT64_MASK)
    )
    uid_grid = uids.astype(np.uint64)[:, None]
    sample_grid = np.arange(int(sample_count), dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        state = (
            np.uint64(scalar)
            + uid_grid * _UID_COEFFICIENT
            + sample_grid * _SAMPLE_COEFFICIENT
        )
    base = _splitmix64(state)
    uniform_one = _uniform_open01(_splitmix64(base ^ _STREAM_ZERO_SALT))
    uniform_two = _uniform_open01(_splitmix64(base ^ _STREAM_ONE_SALT))
    radius = np.sqrt(-2.0 * np.log(uniform_one))
    angle = 2.0 * math.pi * uniform_two
    panel = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=2)
    return panel.astype(np.float32)


def ddpm_key(
    *,
    training_seed: int,
    global_snapshot_uid: int,
    posterior_sample_id: int,
    noise_role: str,
    reverse_timestep: int | None = None,
    protocol_rng_seed: int = PROTOCOL_RNG_SEED,
) -> RNGKey:
    if noise_role == "initial":
        namespace = DDPM_INITIAL_NAMESPACE
        timestep = None
    elif noise_role == "reverse_step":
        namespace = DDPM_REVERSE_NAMESPACE
        timestep = reverse_timestep
    else:
        raise ValueError("DDPM noise_role must be initial or reverse_step")
    return RNGKey(
        protocol_rng_seed=protocol_rng_seed,
        stream_namespace=namespace,
        paired_noise_group="ddpm_independent_v2",
        training_seed=training_seed,
        global_snapshot_uid=global_snapshot_uid,
        posterior_sample_id=posterior_sample_id,
        noise_role=noise_role,
        optional_reverse_timestep=timestep,
    )


def audit_gaussian_key_space(
    global_snapshot_uids: Sequence[int] | np.ndarray,
    *,
    training_seeds: Iterable[int],
    stream_namespace: str,
    sample_count: int = POSTERIOR_SAMPLE_COUNT,
) -> dict[str, Any]:
    uids = np.asarray(global_snapshot_uids, dtype=np.int64)
    seeds = tuple(int(item) for item in training_seeds)
    if uids.ndim != 1 or len(np.unique(uids)) != len(uids):
        raise ValueError("UIDs must be one-dimensional and unique")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    if not 1 <= int(sample_count) <= POSTERIOR_SAMPLE_COUNT:
        raise ValueError("invalid sample count")
    key_count = int(len(uids) * len(seeds) * int(sample_count))
    structural = {
        "protocol_rng_seed": PROTOCOL_RNG_SEED,
        "stream_namespace": stream_namespace,
        "paired_noise_group": PAIRED_NOISE_GROUP,
        "training_seeds": list(seeds),
        "uid_mapping_sha256": sha256(
            np.asarray(uids, dtype="<i8").tobytes(order="C")
        ).hexdigest(),
        "posterior_sample_ids": list(range(int(sample_count))),
        "noise_role": "gaussian_base",
    }
    return {
        "key_count": key_count,
        "unique_key_count": key_count,
        "duplicate_count": 0,
        "proof": (
            "Cartesian product of unique training_seed, global_snapshot_uid, "
            "and posterior_sample_id under fixed required namespace fields"
        ),
        "structural_mapping_sha256": canonical_json_sha256(structural),
    }


def energy_score_v20(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (20, 2):
        raise ValueError("samples must have shape [N,20,2]")
    if target.shape != (len(values), 2):
        raise ValueError("truth must have shape [N,2]")
    first = np.linalg.norm(values - target[:, None, :], axis=2).mean(axis=1)
    pairwise = np.linalg.norm(
        values[:, :, None, :] - values[:, None, :, :], axis=3
    )
    return first - pairwise.sum(axis=(1, 2)) / float(2 * 20 * 20)


def energy_score_u20(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (20, 2):
        raise ValueError("samples must have shape [N,20,2]")
    if target.shape != (len(values), 2):
        raise ValueError("truth must have shape [N,2]")
    first = np.linalg.norm(values - target[:, None, :], axis=2).mean(axis=1)
    pairwise = np.linalg.norm(
        values[:, :, None, :] - values[:, None, :, :], axis=3
    )
    return first - pairwise.sum(axis=(1, 2)) / float(2 * 20 * 19)


def reconstruct_selection_state(
    rows: Sequence[dict[str, Any]],
    *,
    patience: int = 10,
    minimum_stop_epoch: int = 10,
) -> dict[str, Any]:
    """Reproduce the v1 lexicographic selection and patience state machine."""

    if patience <= 0 or minimum_stop_epoch <= 0:
        raise ValueError("patience and minimum_stop_epoch must be positive")
    best_key: tuple[float, float, int] | None = None
    best_epoch: int | None = None
    stale = 0
    trajectory: list[dict[str, Any]] = []
    stop_epoch: int | None = None
    previous_epoch = 0
    for source in rows:
        epoch = int(source["epoch"])
        if epoch <= previous_epoch:
            raise ValueError("evaluation epochs must be strictly increasing")
        previous_epoch = epoch
        key = (
            float(source["selection_energy_score"]),
            float(source["selection_gaussian_nll_cm"]),
            epoch,
        )
        improved = best_key is None or key < best_key
        if improved:
            best_key = key
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        trajectory.append(
            {
                "epoch": epoch,
                "improved": improved,
                "best_epoch": best_epoch,
                "non_improving_epochs": stale,
            }
        )
        if epoch >= minimum_stop_epoch and stale >= patience:
            stop_epoch = epoch
            break
    if best_key is None or best_epoch is None:
        raise ValueError("selection rows are empty")
    return {
        "best_epoch": best_epoch,
        "best_key": {
            "energy_score": best_key[0],
            "gaussian_nll_cm": best_key[1],
            "epoch": best_key[2],
        },
        "theoretical_stop_epoch": stop_epoch,
        "patience": patience,
        "minimum_stop_epoch": minimum_stop_epoch,
        "trajectory": trajectory,
    }


def preregistered_branch(gates: dict[str, bool]) -> dict[str, Any]:
    required = (
        "fullcov_best_and_stop_invariant",
        "homoscedastic_best_and_stop_invariant",
        "candidate_artifacts_complete",
        "canonical_uid_valid",
        "canonical_rng_valid",
        "ddpm_identity_valid_or_isolated",
    )
    missing = [name for name in required if name not in gates]
    if missing:
        raise ValueError(f"fail closed: missing gates {missing}")
    failed = [name for name in required if not bool(gates[name])]
    return {
        "branch": "A" if not failed else "B",
        "failed_gates": failed,
        "ready_for_corrected_evaluation": not failed,
        "rule": "any failed or unidentified gate routes to B",
    }
