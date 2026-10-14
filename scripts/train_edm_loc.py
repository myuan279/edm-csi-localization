"""Portable EDM-Loc retraining on an explicitly supplied Train/Development split.

This is a new public entry point. It does not recreate the historical
authorization ledger or claim to reproduce the paper's frozen checkpoints.
It never opens Calibration or Test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edm_loc.diffusion import ExponentialMovingAverage
from edm_loc.conditional_diffusion import EDMConditionalDiffusion, build_registered_denoiser, registered_edm_config
from edm_loc.targets import TargetTransform
from edm_loc.npz_data import fit_train_stats, load_arrays, make_batch, normalized_ri, sample_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_development(model: EDMConditionalDiffusion, arrays, normalized, seed: int, batch_size: int) -> dict[str, float]:
    model.eval()
    samples = []
    for start in range(0, len(arrays.global_uids), batch_size):
        indices = np.arange(start, min(start + batch_size, len(arrays.global_uids)))
        batch = make_batch(arrays, normalized, indices, "opencsi-development")
        output, trace = model.sample_with_trace(batch, training_seed=seed, model_id="EDM-Loc", sample_count=20)
        if trace["nfe"] != 63:
            raise RuntimeError(f"unexpected NFE: {trace}")
        samples.append(output.posterior_samples_cm)
    return sample_metrics(np.concatenate(samples), arrays.coordinates_cm, arrays.rp_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True, type=Path)
    parser.add_argument("--development-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, choices=(2031, 2032, 2033), required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--selection-every", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.selection_every < 1:
        parser.error("epochs, batch-size and selection-every must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.output_dir.exists():
        parser.error("output-dir already exists; select a new run directory")
    train = load_arrays(args.train_npz)
    development = load_arrays(args.development_npz)
    if np.intersect1d(train.global_uids, development.global_uids).size or np.intersect1d(train.rp_ids, development.rp_ids).size:
        parser.error("Train and Development must have disjoint snapshot UIDs and RPs")
    stats = fit_train_stats(train)
    train_ri = normalized_ri(train, stats)
    dev_ri = normalized_ri(development, stats)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    target = TargetTransform("absolute", np.asarray(stats["position_mean_cm"]), np.asarray(stats["position_std_cm"]))
    model = EDMConditionalDiffusion(build_registered_denoiser(args.seed), target, registered_edm_config()).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ema = ExponentialMovingAverage(model, decay=0.9999)
    noise_rng = torch.Generator(device=args.device).manual_seed(args.seed)
    shuffle_rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "new_public_retraining_not_historical_formal_run",
        "seed": args.seed,
        "train_npz_sha256": sha256(args.train_npz),
        "development_npz_sha256": sha256(args.development_npz),
        "train_rows": len(train.global_uids),
        "development_rows": len(development.global_uids),
        "train_rps": len(np.unique(train.rp_ids)),
        "development_rps": len(np.unique(development.rp_ids)),
        "normalization": stats,
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "selection_every": args.selection_every, "learning_rate": 3e-4, "weight_decay": 1e-4, "ema_decay": 0.9999},
    }
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    best_key = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = shuffle_rng.permutation(len(train.global_uids))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = make_batch(train, train_ri, order[start:start + args.batch_size], "opencsi-train")
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(batch, generator=noise_rng)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            total += float(loss.detach().cpu()) * len(batch.csi)
        record = {"epoch": epoch, "train_loss": total / len(order)}
        if epoch % args.selection_every == 0 or epoch == args.epochs:
            raw_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            ema.copy_to(model)
            scores = evaluate_development(model, development, dev_ri, args.seed, args.eval_batch_size)
            key = (scores["energy_score_u20_cm"], scores["mean_cm"], epoch)
            record["development"] = scores
            record["selected"] = best_key is None or key < best_key
            if record["selected"]:
                best_key = key
                selected = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                torch.save({"selected_model": selected, "seed": args.seed, "epoch": epoch, "normalization": stats, "development": scores}, args.output_dir / "best_checkpoint.pt")
            model.load_state_dict(raw_state)
        history.append(record)
        print(json.dumps(record), flush=True)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
