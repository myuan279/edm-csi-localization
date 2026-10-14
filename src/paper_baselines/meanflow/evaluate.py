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

from .data_adapter import RINormalizedDataset, load_permitted_bundle, load_resolved_data, int64_sha256
from .metrics import metrics_from_samples
from .models import ConditionedField, ConsistencyModel, MeanFlowModel


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    method = checkpoint["method"]
    model: torch.nn.Module
    if method in {"cm_ct", "cm_cd", "ict"}:
        model = ConsistencyModel()
    elif method == "meanflow":
        model = MeanFlowModel()
    elif method == "cfm":
        model = ConditionedField()
    else:
        raise ValueError(f"unregistered checkpoint method: {method}")
    inference_state = checkpoint["model_state"]
    inference_weights = "online"
    if method in {"cm_ct", "ict"} and checkpoint.get("target_state") is not None:
        inference_state = checkpoint["target_state"]
        inference_weights = "target_ema"
    model.load_state_dict(inference_state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    checkpoint["inference_weights"] = inference_weights
    return model, checkpoint


def sample_consistency(model: ConsistencyModel, condition: torch.Tensor, uids: np.ndarray, *, seed: int, method_id: str, sample_count: int, nfe: int) -> tuple[np.ndarray, dict[str, Any]]:
    if nfe not in (1, 2, 4):
        raise ValueError("consistency evaluation supports only NFE=1/2/4")
    noise = gaussian_base_noise(uids, training_seed=seed, stream_namespace=f"ri_onestep_v1/{method_id}/consistency", paired_noise_group="ri_onestep_v1", protocol_rng_seed=20260910, sample_count=sample_count, noise_role="initial")
    state = torch.from_numpy(noise.reshape(-1, 2) * 80.0).to(device=condition.device, dtype=torch.float32)
    expanded = condition[:, None, :].expand(len(condition), sample_count, 312).reshape(-1, 312)
    sigma_values = np.geomspace(80.0, 0.002, nfe).astype(np.float32)
    calls = []
    with torch.inference_mode():
        for step, sigma_value in enumerate(sigma_values):
            sigma = torch.full((len(state),), float(sigma_value), device=state.device, dtype=state.dtype)
            denoised = model(state, sigma, expanded)
            calls.append({"step": step, "sigma": float(sigma_value), "batch_size": len(state)})
            if step + 1 < nfe:
                next_sigma = float(sigma_values[step + 1])
                state = denoised + next_sigma * torch.from_numpy(noise.reshape(-1, 2)).to(state.device)
            else:
                state = denoised
    return state.reshape(len(condition), sample_count, 2).cpu().numpy(), {"method": method_id, "requested_nfe_per_path": nfe, "actual_forward_calls": len(calls), "calls": calls}


def sample_meanflow(model: MeanFlowModel, condition: torch.Tensor, uids: np.ndarray, *, seed: int, method_id: str, sample_count: int, nfe: int) -> tuple[np.ndarray, dict[str, Any]]:
    if nfe not in (1, 2, 4):
        raise ValueError("MeanFlow evaluation supports only NFE=1/2/4")
    noise = gaussian_base_noise(uids, training_seed=seed, stream_namespace=f"ri_onestep_v1/{method_id}/meanflow", paired_noise_group="ri_onestep_v1", protocol_rng_seed=20260910, sample_count=sample_count, noise_role="initial")
    state = torch.from_numpy(noise.reshape(-1, 2)).to(device=condition.device, dtype=torch.float32)
    expanded = condition[:, None, :].expand(len(condition), sample_count, 312).reshape(-1, 312)
    calls = []
    with torch.inference_mode():
        if nfe == 1:
            reference = torch.zeros(len(state), device=state.device)
            time_value = torch.ones(len(state), device=state.device)
            velocity = model(state, reference, time_value, expanded)
            state = state - velocity
            calls.append({"step": 0, "time": 1.0, "batch_size": len(state)})
        else:
            dt = 1.0 / nfe
            for step in range(nfe):
                current_time = 1.0 - step * dt
                reference = torch.zeros(len(state), device=state.device)
                time_value = torch.full((len(state),), current_time, device=state.device)
                velocity = model(state, reference, time_value, expanded)
                state = state - dt * velocity
                calls.append({"step": step, "time": current_time, "batch_size": len(state)})
    return state.reshape(len(condition), sample_count, 2).cpu().numpy(), {"method": method_id, "requested_nfe_per_path": nfe, "actual_forward_calls": len(calls), "calls": calls}


def sample_cfm(model: ConditionedField, condition: torch.Tensor, uids: np.ndarray, *, seed: int, method_id: str, sample_count: int, nfe: int) -> tuple[np.ndarray, dict[str, Any]]:
    if nfe < 1:
        raise ValueError("CFM NFE must be positive")
    noise = gaussian_base_noise(uids, training_seed=seed, stream_namespace=f"ri_onestep_v1/{method_id}/cfm", paired_noise_group="ri_onestep_v1", protocol_rng_seed=20260910, sample_count=sample_count, noise_role="initial")
    state = torch.from_numpy(noise.reshape(-1, 2)).to(device=condition.device, dtype=torch.float32)
    expanded = condition[:, None, :].expand(len(condition), sample_count, 312).reshape(-1, 312)
    dt = 1.0 / nfe
    calls = []
    with torch.inference_mode():
        for step in range(nfe):
            t = torch.full((len(state),), step * dt, device=state.device)
            state = state + dt * model(state, t, expanded)
            calls.append({"step": step, "time": step * dt, "batch_size": len(state)})
    return state.reshape(len(condition), sample_count, 2).cpu().numpy(), {"method": method_id, "requested_nfe_per_path": nfe, "actual_forward_calls": len(calls), "calls": calls}


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("binding_status") not in {"SERVER_BOUND", "LOCAL_ENGINEERING"} and not args.allow_unbound:
        raise RuntimeError("evaluation requires a bound resolved config")
    samples = int(args.samples)
    if samples < 1 or samples > 20:
        raise ValueError("samples must be in 1..20")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_model(run_dir, device)
    data = load_resolved_data(config, config_dir=config_path.parent)
    bundle, indices = load_permitted_bundle(data, args.subset)
    dataset = RINormalizedDataset(bundle, indices, data.normalization)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    all_samples, all_truth, all_rp, all_uids = [], [], [], []
    traces = []
    started = time.perf_counter()
    for batch in loader:
        condition = batch["condition"].to(device=device, dtype=torch.float32, non_blocking=True)
        uids = np.asarray(batch["global_uid"], dtype=np.int64)
        if isinstance(model, ConsistencyModel):
            generated, trace = sample_consistency(model, condition, uids, seed=int(checkpoint["seed"]), method_id=checkpoint["method"], sample_count=samples, nfe=int(args.nfe))
        elif isinstance(model, MeanFlowModel):
            generated, trace = sample_meanflow(model, condition, uids, seed=int(checkpoint["seed"]), method_id=checkpoint["method"], sample_count=samples, nfe=int(args.nfe))
        else:
            generated, trace = sample_cfm(model, condition, uids, seed=int(checkpoint["seed"]), method_id=checkpoint["method"], sample_count=samples, nfe=int(args.nfe))
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
    write_json(out_dir / "metrics.json", {"protocol_id": "ri_acceleration_onestep_v1", "method": checkpoint["method"], "seed": checkpoint["seed"], "inference_weights": checkpoint.get("inference_weights", "online"), "subset": args.subset, "samples": samples, "nfe_per_path": int(args.nfe), "sample_equivalent_evaluations": samples * int(args.nfe), "metrics": metrics, "evidence_stage": "POSTHOC_DEVELOPMENT_SUPPLEMENT", "protected_access": {"validation": 0, "calibration": 0, "test": 0}})
    write_json(out_dir / "nfe_trace.json", {"method": checkpoint["method"], "requested_nfe_per_path": int(args.nfe), "actual_forward_calls_per_batch": [trace["actual_forward_calls"] for trace in traces], "total_batches": len(traces), "uid_sha256": int64_sha256(uids), "elapsed_seconds": time.perf_counter() - started})
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RI one-step baseline")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subset", choices=("development_train_300k", "selection_validation_rp10"), default="selection_validation_rp10")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device")
    parser.add_argument("--allow-unbound", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
