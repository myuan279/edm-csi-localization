"""Evaluate a public-format EDM-Loc checkpoint on an explicit NPZ input.

Calling this on Test is the user's own evaluation decision. This script does
not update the original project's protected-stage ledger or paper evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edm_loc.conditional_diffusion import EDMConditionalDiffusion, build_registered_denoiser, registered_edm_config
from edm_loc.targets import TargetTransform
from edm_loc.npz_data import load_arrays, make_batch, normalized_ri, sample_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not {"selected_model", "seed", "normalization"}.issubset(checkpoint):
        parser.error("checkpoint does not use the public EDM-Loc format")
    arrays = load_arrays(args.input_npz)
    stats = checkpoint["normalization"]
    ri = normalized_ri(arrays, stats)
    target = TargetTransform("absolute", np.asarray(stats["position_mean_cm"]), np.asarray(stats["position_std_cm"]))
    model = EDMConditionalDiffusion(build_registered_denoiser(int(checkpoint["seed"])), target, registered_edm_config()).to(args.device)
    model.load_state_dict(checkpoint["selected_model"])
    model.eval()
    samples = []
    with torch.no_grad():
        for start in range(0, len(arrays.global_uids), args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, len(arrays.global_uids)))
            batch = make_batch(arrays, ri, indices, "opencsi-explicit-evaluation")
            result, trace = model.sample_with_trace(batch, training_seed=int(checkpoint["seed"]), model_id="EDM-Loc", sample_count=20)
            if trace["nfe"] != 63:
                raise RuntimeError(f"unexpected NFE: {trace}")
            samples.append(result.posterior_samples_cm)
    posterior = np.concatenate(samples)
    metrics = sample_metrics(posterior, arrays.coordinates_cm, arrays.rp_ids)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "predictions.npz", posterior_samples_cm=posterior, truth_cm=arrays.coordinates_cm, rp_ids=arrays.rp_ids, global_uids=arrays.global_uids)
    (args.output_dir / "metrics.json").write_text(json.dumps({"metrics": metrics, "actual_nfe": 63, "samples_per_input": 20, "status": "new_public_evaluation_not_historical_formal_result"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
