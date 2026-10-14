from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import PROTOCOL_ID, SEEDS, TRAINING_UPDATES
from .data_adapter import RINormalizedDataset, load_permitted_bundle, load_resolved_data, write_data_access_receipt
from .models import ConditionedField, ConsistencyModel, MeanFlowModel, parameter_count
from .objectives.ct import consistency_loss
from .objectives.ict import ict_loss
from .objectives.meanflow import meanflow_loss


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(method: str) -> torch.nn.Module:
    if method in {"cm_ct", "cm_cd", "ict"}:
        return ConsistencyModel()
    if method == "meanflow":
        return MeanFlowModel()
    if method == "cfm":
        return ConditionedField()
    raise ValueError(f"unregistered training method: {method}")


@torch.no_grad()
def update_target(target: torch.nn.Module, online: torch.nn.Module, decay: float) -> None:
    target_state = target.state_dict()
    online_state = online.state_dict()
    for name, value in target_state.items():
        source = online_state[name].detach().to(device=value.device, dtype=value.dtype)
        if torch.is_floating_point(value):
            value.lerp_(source, 1.0 - decay)
        else:
            value.copy_(source)


def train_loss(method: str, model: torch.nn.Module, target: torch.nn.Module | None, batch: dict[str, Any], generator: torch.Generator) -> torch.Tensor:
    condition = batch["condition"]
    position = batch["position"]
    if method == "cm_ct":
        assert target is not None
        return consistency_loss(model, target, position, condition, generator=generator)
    if method == "ict":
        assert target is not None
        return ict_loss(model, target, position, condition, generator=generator, delta=1.0)
    if method == "meanflow":
        return meanflow_loss(model, position, condition, generator=generator)
    if method == "cfm":
        noise = torch.randn(position.shape, device=position.device, dtype=position.dtype, generator=generator)
        time_value = torch.rand((len(position),), device=position.device, dtype=position.dtype, generator=generator)
        state = (1.0 - time_value[:, None]) * position + time_value[:, None] * noise
        return (model(state, time_value, condition) - (noise - position)).square().sum(dim=1).mean()
    raise ValueError(method)


def resolved_config(config: dict[str, Any], *, method: str, seed: int, run_id: str, config_path: Path) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["protocol_id"] = PROTOCOL_ID
    value["method"] = method
    value["seed"] = int(seed)
    value["run_id"] = run_id
    value["config_source"] = str(config_path.resolve())
    value["resolved_at_utc"] = utc_now()
    value.setdefault("training", {})["updates"] = int(value.get("training", {}).get("updates", TRAINING_UPDATES))
    return value


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    method = args.method or config.get("method")
    if method is None:
        raise ValueError("--method or config.method is required")
    if int(args.seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if config.get("binding_status") not in {"SERVER_BOUND", "LOCAL_ENGINEERING"} and not args.allow_unbound:
        raise RuntimeError("resolved config is TEMPLATE_UNBOUND; bind server paths and source commits before training")
    if method == "cm_cd":
        raise RuntimeError("CM-CD requires an explicitly bound EDM-RI teacher checkpoint; this branch is not silently treated as CT")
    run_id = args.run_id or f"{method.upper()}_RI_S{args.seed}_001"
    run_dir = Path(args.output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = resolved_config(config, method=method, seed=args.seed, run_id=run_id, config_path=config_path)
    write_json(run_dir / "resolved_config.json", resolved)
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        data = load_resolved_data(config, config_dir=config_path.parent)
        bundle, indices = load_permitted_bundle(data, "development_train_300k")
        write_data_access_receipt(run_dir / "data_access_receipt.json", subset="development_train_300k", indices=indices, data=data, expected_hashes=config.get("authority", {}))
        dataset = RINormalizedDataset(bundle, indices, data.normalization)
        loader = DataLoader(
            dataset,
            batch_size=int(config["training"].get("effective_batch", 1024)),
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = make_model(method).to(device=device, dtype=torch.float32)
        target = None
        if method in {"cm_ct", "ict"}:
            target = copy.deepcopy(model).eval()
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"].get("lr", 3e-4)),
            betas=tuple(config["training"].get("betas", [0.9, 0.999])),
            weight_decay=float(config["training"].get("weight_decay", 1e-4)),
        )
        random_generator = torch.Generator(device=device).manual_seed(args.seed ^ 0x5EED)
        log_path = run_dir / "train.jsonl"
        updates = int(args.updates or config["training"].get("updates", TRAINING_UPDATES))
        if updates <= 0:
            raise ValueError("updates must be positive")
        decay = float(config.get("method_config", {}).get("ema_decay", 0.9999))
        update_index = 0
        epoch = 0
        model.train()
        with log_path.open("x", encoding="utf-8") as log:
            while update_index < updates:
                epoch += 1
                for batch in loader:
                    if update_index >= updates:
                        break
                    batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    loss = train_loss(method, model, target, batch, random_generator)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
                    optimizer.step()
                    if target is not None:
                        update_target(target, model, decay)
                    update_index += 1
                    if update_index == 1 or update_index % 100 == 0 or update_index == updates:
                        log.write(json.dumps({"timestamp_utc": utc_now(), "epoch": epoch, "update": update_index, "loss": float(loss.detach().cpu()), "batch_size": len(batch["position"])}) + "\n")
                        log.flush()
        checkpoint = {
            "protocol_id": PROTOCOL_ID,
            "method": method,
            "seed": int(args.seed),
            "updates": update_index,
            "model_state": model.state_dict(),
            "target_state": target.state_dict() if target is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        torch.save(checkpoint, run_dir / "checkpoint.pt")
        write_json(run_dir / "checkpoint_manifest.json", {"checkpoint": "checkpoint.pt", "method": method, "seed": args.seed, "updates": update_index, "parameter_count": parameter_count(model), "target_ema": target is not None, "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt")})
        elapsed_training_seconds = time.perf_counter() - started
        write_json(run_dir / "environment.json", {"python": os.sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device": str(device), "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "elapsed_training_seconds": elapsed_training_seconds, "updates_per_second": update_index / elapsed_training_seconds, "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None, "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
        write_json(run_dir / "source_manifest.json", {"config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(data.manifest_path), "normalization_sha256": sha256_file(data.normalization_path), "h5_sha256": config.get("authority", {}).get("h5_sha256"), "sampler_sources": config.get("sampler_sources", {}), "protocol_id": PROTOCOL_ID})
        write_json(run_dir / "COMPLETED.json", {"status": "COMPLETED", "run_id": run_id, "method": method, "seed": args.seed, "updates": update_index, "evidence_stage": "POSTHOC_DEVELOPMENT_SUPPLEMENT", "protected_access": {"validation": 0, "calibration": 0, "test": 0}})
        return 0
    except Exception as exc:
        write_json(run_dir / "FAILED.json", {"status": "FAILED", "run_id": run_id, "method": method, "seed": args.seed, "error_type": type(exc).__name__, "error": str(exc), "finished_at_utc": utc_now()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one RI one-step baseline run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("cm_ct", "cm_cd", "ict", "meanflow", "cfm"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", default="outputs/paper_baselines/meanflow/runs")
    parser.add_argument("--device")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--allow-unbound", action="store_true", help="engineering-only override; never use for a formal server run")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
