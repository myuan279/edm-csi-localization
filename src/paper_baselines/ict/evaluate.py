from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from probabilistic_rng import gaussian_base_noise

from . import MODEL_SCHEMA_VERSION, PROTOCOL_ID
from .data_adapter import RINormalizedDataset, int64_sha256, load_permitted_bundle, load_resolved_data
from .metrics import metrics_from_samples
from .models import ConditionedField, ConsistencyModel, MeanFlowModel


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write(chr(10))


def _method_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    return dict(config.get("methods", {}).get(method, config.get("method_config", {})))


def _build_model(method: str, config: dict[str, Any]) -> torch.nn.Module:
    cfg = _method_config(config, method)
    if method in {"cm_ct", "cm_cd", "ict"}:
        return ConsistencyModel(
            sigma_min=float(cfg.get("sigma_min", 0.002)),
            sigma_data=float(cfg.get("sigma_data", 1.0)),
        )
    if method == "meanflow":
        return MeanFlowModel()
    if method == "cfm":
        return ConditionedField()
    raise ValueError(f"unregistered checkpoint method: {method}")


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    *,
    config: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    method = checkpoint["method"]
    model = _build_model(method, config)
    inference_state = checkpoint["model_state"]
    inference_weights = "online"
    if method in {"cm_ct", "ict", "meanflow"} and checkpoint.get("inference_ema_state") is not None:
        inference_state = checkpoint["inference_ema_state"]
        inference_weights = "inference_ema"
    elif method == "cm_ct" and checkpoint.get("target_state") is not None:
        inference_state = checkpoint["target_state"]
        inference_weights = "training_target_ema_legacy"
    model.load_state_dict(inference_state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    checkpoint["inference_weights"] = inference_weights
    checkpoint["checkpoint_path"] = str(checkpoint_path)
    return model, checkpoint


def load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    return _load_model_from_checkpoint(run_dir / "checkpoint.pt", device, config=config)


def sample_consistency(
    model: ConsistencyModel,
    condition: torch.Tensor,
    uids: np.ndarray,
    *,
    seed: int,
    method_id: str,
    sample_count: int,
    nfe: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    schedules = {
        1: (80.0,),
        2: (80.0, 1.0),
        4: (80.0, 4.0, 1.0, 0.25),
    }
    if nfe not in schedules:
        raise ValueError("consistency evaluation supports only NFE=1/2/4")
    tau = schedules[nfe]
    noise = gaussian_base_noise(
        uids,
        training_seed=seed,
        stream_namespace=f"ri_onestep_v2/{method_id}/consistency",
        paired_noise_group="ri_onestep_v2",
        protocol_rng_seed=20260910,
        sample_count=sample_count,
        noise_role="initial",
    )
    initial_noise = torch.from_numpy(noise.reshape(-1, 2)).to(device=condition.device, dtype=torch.float32)
    state = tau[0] * initial_noise
    expanded = condition[:, None, :].expand(len(condition), sample_count, 312).reshape(-1, 312)
    calls = []
    with torch.inference_mode():
        for step, sigma_value in enumerate(tau):
            sigma = torch.full((len(state),), sigma_value, device=state.device, dtype=state.dtype)
            denoised = model(state, sigma, expanded)
            calls.append({"step": step, "sigma": sigma_value, "batch_size": len(state)})
            if step + 1 < len(tau):
                next_sigma = tau[step + 1]
                stage_noise = gaussian_base_noise(
                    uids,
                    training_seed=seed,
                    stream_namespace=f"ri_onestep_v2/{method_id}/consistency",
                    paired_noise_group="ri_onestep_v2",
                    protocol_rng_seed=20260910,
                    sample_count=sample_count,
                    noise_role=f"stage_{step + 1}",
                )
                stage_noise_tensor = torch.from_numpy(stage_noise.reshape(-1, 2)).to(
                    device=state.device,
                    dtype=state.dtype,
                )
                state = denoised + max(next_sigma**2 - model.sigma_min**2, 0.0) ** 0.5 * stage_noise_tensor
            else:
                state = denoised
    return state.reshape(len(condition), sample_count, 2).cpu().numpy(), {
        "method": method_id,
        "requested_nfe_per_path": nfe,
        "actual_forward_calls": len(calls),
        "time_convention": "sigma_high_to_low",
        "calls": calls,
    }


def sample_meanflow(
    model: MeanFlowModel,
    condition: torch.Tensor,
    uids: np.ndarray,
    *,
    seed: int,
    method_id: str,
    sample_count: int,
    nfe: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if nfe not in (1, 2, 4):
        raise ValueError("MeanFlow evaluation supports only NFE=1/2/4")
    noise = gaussian_base_noise(
        uids,
        training_seed=seed,
        stream_namespace=f"ri_onestep_v2/{method_id}/meanflow",
        paired_noise_group="ri_onestep_v2",
        protocol_rng_seed=20260910,
        sample_count=sample_count,
        noise_role="initial",
    )
    state = torch.from_numpy(noise.reshape(-1, 2)).to(device=condition.device, dtype=torch.float32)
    expanded = condition[:, None, :].expand(len(condition), sample_count, 312).reshape(-1, 312)
    grid = np.linspace(1.0, 0.0, nfe + 1, dtype=np.float32)
    calls = []
    with torch.inference_mode():
        for step in range(nfe):
            current_time = float(grid[step])
            next_reference = float(grid[step + 1])
            t = torch.full((len(state),), current_time, device=state.device)
            r = torch.full((len(state),), next_reference, device=state.device)
            velocity = model(state, r, t, expanded)
            state = state - (current_time - next_reference) * velocity
            calls.append(
                {
                    "step": step,
                    "reference_time": next_reference,
                    "time": current_time,
                    "batch_size": len(state),
                }
            )
    return state.reshape(len(condition), sample_count, 2).cpu().numpy(), {
        "method": method_id,
        "requested_nfe_per_path": nfe,
        "actual_forward_calls": len(calls),
        "time_convention": "local_interval_r_next_to_t",
        "calls": calls,
    }


def sample_cfm(
    model: ConditionedField,
    condition: torch.Tensor,
    uids: np.ndarray,
    *,
    seed: int,
    method_id: str,
    sample_count: int,
    nfe: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if nfe < 1:
        raise ValueError("CFM NFE must be positive")
    noise = gaussian_base_noise(
        uids,
        training_seed=seed,
        stream_namespace=f"ri_onestep_v2/{method_id}/cfm",
        paired_noise_group="ri_onestep_v2",
        protocol_rng_seed=20260910,
        sample_count=sample_count,
        noise_role="initial",
    )
    state = torch.from_numpy(noise.reshape(-1, 2)).to(device=condition.device, dtype=torch.float32)
    expanded = condition[:, None, :].expand(len(condition), sample_count, 312).reshape(-1, 312)
    grid = np.linspace(1.0, 0.0, nfe + 1, dtype=np.float32)
    calls = []
    with torch.inference_mode():
        for step in range(nfe):
            current_time = float(grid[step])
            next_time = float(grid[step + 1])
            t = torch.full((len(state),), current_time, device=state.device)
            state = state + (next_time - current_time) * model(state, t, expanded)
            calls.append({"step": step, "time": current_time, "next_time": next_time, "batch_size": len(state)})
    return state.reshape(len(condition), sample_count, 2).cpu().numpy(), {
        "method": method_id,
        "requested_nfe_per_path": nfe,
        "actual_forward_calls": len(calls),
        "time_convention": "noise_t1_to_data_t0",
        "calls": calls,
    }


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        config_path = run_dir / "resolved_config.json"
        checkpoint_path = run_dir / "checkpoint.pt"
        return run_dir, config_path, checkpoint_path, json.loads(config_path.read_text(encoding="utf-8"))
    if not args.config or not args.checkpoint:
        raise ValueError("provide --run-dir or both --config and --checkpoint")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    method = config.get("method", "cfm")
    run_id = args.run_id or f"{method.upper()}_RI_V2_EVAL_{Path(args.checkpoint).stem}"
    run_dir = Path(args.output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    return run_dir, config_path, Path(args.checkpoint).resolve(), config


def run(args: argparse.Namespace) -> int:
    run_dir, config_path, checkpoint_path, config = _resolve_paths(args)
    if config.get("binding_status") not in {"SERVER_BOUND", "LOCAL_ENGINEERING"} and not args.allow_unbound:
        raise RuntimeError("evaluation requires a bound resolved config")
    samples = int(args.samples)
    if samples < 1 or samples > 20:
        raise ValueError("samples must be in 1..20")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = _load_model_from_checkpoint(checkpoint_path, device, config=config)
    data = load_resolved_data(config, config_dir=config_path.parent)
    bundle, indices = load_permitted_bundle(data, args.subset)
    dataset = RINormalizedDataset(bundle, indices, data.normalization)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    all_samples, all_truth, all_rp, all_uids = [], [], [], []
    traces = []
    started = time.perf_counter()
    for batch in loader:
        condition = batch["condition"].to(device=device, dtype=torch.float32, non_blocking=True)
        uids = np.asarray(batch["global_uid"], dtype=np.int64)
        if isinstance(model, ConsistencyModel):
            generated, trace = sample_consistency(
                model, condition, uids, seed=int(checkpoint["seed"]), method_id=checkpoint["method"],
                sample_count=samples, nfe=int(args.nfe),
            )
        elif isinstance(model, MeanFlowModel):
            generated, trace = sample_meanflow(
                model, condition, uids, seed=int(checkpoint["seed"]), method_id=checkpoint["method"],
                sample_count=samples, nfe=int(args.nfe),
            )
        else:
            generated, trace = sample_cfm(
                model, condition, uids, seed=int(checkpoint["seed"]), method_id=checkpoint["method"],
                sample_count=samples, nfe=int(args.nfe),
            )
        generated_cm = generated * data.normalization.y_std[None, None, :] + data.normalization.y_mean[None, None, :]
        all_samples.append(generated_cm.astype(np.float32))
        all_truth.append(batch["truth_cm"].numpy().astype(np.float32))
        all_rp.append(np.asarray(batch["rp_id"], dtype=np.int64))
        all_uids.append(uids)
        traces.append(trace)
    samples_cm = np.concatenate(all_samples, axis=0)
    truth = np.concatenate(all_truth, axis=0)
    rp = np.concatenate(all_rp, axis=0)
    uids = np.concatenate(all_uids, axis=0)
    metrics = metrics_from_samples(samples_cm, truth, rp, uids)
    out_dir = run_dir / f"evaluate_{args.subset}_S{samples}_NFE{args.nfe}"
    out_dir.mkdir(exist_ok=False)
    np.savez_compressed(out_dir / "samples.npz", global_uids=uids, rp_ids=rp, truth_cm=truth, samples_cm=samples_cm)
    write_json(
        out_dir / "metrics.json",
        {
            "protocol_id": PROTOCOL_ID,
            "model_schema_version": checkpoint.get("model_schema_version", "legacy_v1_checkpoint"),
            "method": checkpoint["method"],
            "seed": checkpoint["seed"],
            "inference_weights": checkpoint.get("inference_weights", "online"),
            "subset": args.subset,
            "samples": samples,
            "nfe_per_path": int(args.nfe),
            "sample_equivalent_evaluations": samples * int(args.nfe),
            "metrics": metrics,
            "evidence_stage": "POSTHOC_DEVELOPMENT_SUPPLEMENT",
            "protected_access": {"validation": 0, "calibration": 0, "test": 0},
        },
    )
    write_json(
        out_dir / "nfe_trace.json",
        {
            "method": checkpoint["method"],
            "requested_nfe_per_path": int(args.nfe),
            "actual_forward_calls_per_batch": [trace["actual_forward_calls"] for trace in traces],
            "total_batches": len(traces),
            "uid_sha256": int64_sha256(uids),
            "elapsed_seconds": time.perf_counter() - started,
            "trace_example": traces[0] if traces else None,
        },
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RI one-step baseline (v2)")
    parser.add_argument("--run-dir")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", default="outputs/paper_baselines/ict/runs")
    parser.add_argument("--subset", choices=("development_train_300k", "selection_validation_rp10"), default="selection_validation_rp10")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device")
    parser.add_argument("--allow-unbound", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
