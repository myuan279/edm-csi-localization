from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExperimentSpec:
    protocol_id: str
    dataset_id: str
    model_id: str
    stage: str
    seed: int
    training_budget: Mapping[str, Any]
    selection_metric: str
    promotion_gates: Mapping[str, Any]
    allowed_subsets: tuple[str, ...]
    output_dir: str
    parent_protocol_id: str | None = None
    config_sha256: str | None = None

    def validate(self) -> None:
        if not self.protocol_id or not self.dataset_id or not self.model_id:
            raise ValueError("protocol_id, dataset_id, and model_id are required")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if not self.selection_metric:
            raise ValueError("selection_metric is required")
        if not self.allowed_subsets:
            raise ValueError("allowed_subsets must be non-empty")
        if len(set(self.allowed_subsets)) != len(self.allowed_subsets):
            raise ValueError("allowed_subsets contains duplicates")
        if Path(self.output_dir).name in {"", ".", ".."}:
            raise ValueError("output_dir must name a run-specific directory")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class DatasetBatch:
    """A native CSI batch. CSI uses [B, amplitude/phase, subcarrier, port]."""

    csi: torch.Tensor
    coordinates_cm: torch.Tensor
    rp_ids: torch.Tensor
    global_uids: torch.Tensor
    subcarrier_mask: torch.Tensor
    port_mask: torch.Tensor
    dataset_id: str
    layout: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.csi.ndim != 4 or self.csi.shape[1] != 2:
            raise ValueError("csi must have shape [B,2,F,P]")
        batch, _, subcarriers, ports = self.csi.shape
        expected = {
            "coordinates_cm": (batch, 2),
            "rp_ids": (batch,),
            "global_uids": (batch,),
            "subcarrier_mask": (batch, subcarriers),
            "port_mask": (batch, ports),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.subcarrier_mask.dtype != torch.bool or self.port_mask.dtype != torch.bool:
            raise TypeError("masks must be boolean")
        if torch.any(~self.subcarrier_mask.any(dim=1)) or torch.any(~self.port_mask.any(dim=1)):
            raise ValueError("every row needs at least one subcarrier and one port")
        if torch.unique(self.global_uids).numel() != batch:
            raise ValueError("global_uids must be unique within a batch")
        if not self.dataset_id:
            raise ValueError("dataset_id is required")

    def to(self, device: torch.device | str) -> "DatasetBatch":
        return DatasetBatch(
            csi=self.csi.to(device),
            coordinates_cm=self.coordinates_cm.to(device),
            rp_ids=self.rp_ids.to(device),
            global_uids=self.global_uids.to(device),
            subcarrier_mask=self.subcarrier_mask.to(device),
            port_mask=self.port_mask.to(device),
            dataset_id=self.dataset_id,
            layout=self.layout,
        )


@dataclass(frozen=True)
class DistributionOutput:
    posterior_samples_cm: np.ndarray
    mean_cm: np.ndarray
    covariance_cm2: np.ndarray
    risk_score: np.ndarray
    global_uids: np.ndarray

    @classmethod
    def from_samples(
        cls,
        samples_cm: np.ndarray,
        global_uids: Sequence[int] | np.ndarray,
    ) -> "DistributionOutput":
        samples = np.asarray(samples_cm, dtype=np.float32)
        uids = np.asarray(global_uids, dtype=np.int64)
        if samples.ndim != 3 or samples.shape[2] != 2 or len(samples) != len(uids):
            raise ValueError("samples must be [N,S,2] and align with global_uids")
        if samples.shape[1] < 2:
            raise ValueError("at least two posterior samples are required")
        if len(np.unique(uids)) != len(uids):
            raise ValueError("global_uids must be unique")
        work = samples.astype(np.float64)
        mean = work.mean(axis=1)
        centered = work - mean[:, None, :]
        covariance = np.einsum("nsi,nsj->nij", centered, centered)
        covariance /= float(samples.shape[1] - 1)
        risk = np.trace(covariance, axis1=1, axis2=2)
        return cls(
            posterior_samples_cm=samples,
            mean_cm=mean.astype(np.float32),
            covariance_cm2=covariance.astype(np.float32),
            risk_score=risk.astype(np.float32),
            global_uids=uids,
        )


@dataclass
class RunRecord:
    protocol_id: str
    run_id: str
    stage: str
    command: tuple[str, ...]
    environment: Mapping[str, Any]
    status: str = "running"
    started_at_utc: str = field(default_factory=utc_now)
    finished_at_utc: str | None = None
    failure_reason: str | None = None
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    access_ledger: dict[str, int] = field(default_factory=dict)

    def finish(self, artifact_hashes: Mapping[str, str]) -> None:
        if self.status != "running":
            raise RuntimeError("a run may be finished only once")
        self.status = "completed"
        self.finished_at_utc = utc_now()
        self.artifact_hashes = dict(artifact_hashes)

    def fail(self, reason: str) -> None:
        if self.status != "running":
            raise RuntimeError("a run may be failed only once")
        if not reason:
            raise ValueError("failure reason is required")
        self.status = "failed"
        self.failure_reason = reason
        self.finished_at_utc = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
