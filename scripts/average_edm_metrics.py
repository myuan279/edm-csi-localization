"""Average metrics from three separately evaluated EDM-Loc seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


EXPECTED_SEEDS = {2031, 2032, 2033}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", nargs=3, type=Path, help="one metrics.json file per seed")
    parser.add_argument("--output", type=Path, help="optional path for the averaged JSON report")
    args = parser.parse_args()

    try:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics_json]
        seeds = [int(report["seed"]) for report in reports]
        hashes = {report["input_npz_sha256"] for report in reports}
        metric_names = set(reports[0]["metrics"])
        if set(seeds) != EXPECTED_SEEDS or len(seeds) != 3:
            raise ValueError("expected one evaluation each for seeds 2031, 2032, and 2033")
        if len(hashes) != 1:
            raise ValueError("all three evaluations must use the same input NPZ file")
        if any(report["actual_nfe"] != 63 or report["samples_per_input"] != 20 for report in reports):
            raise ValueError("all evaluations must use 63 NFE and 20 samples per input")
        if any(set(report["metrics"]) != metric_names for report in reports):
            raise ValueError("the metric names differ across evaluations")
        if any(report["status"] != "new_public_evaluation_not_historical_formal_result" for report in reports):
            raise ValueError("unexpected evaluation status")
        averaged = {}
        for name in sorted(metric_names):
            values = [float(report["metrics"][name]) for report in reports]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite metric: {name}")
            averaged[name] = mean(values)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    summary = {
        "seeds": sorted(seeds),
        "input_npz_sha256": next(iter(hashes)),
        "actual_nfe": 63,
        "samples_per_input": 20,
        "metrics": averaged,
        "status": "average_of_three_new_public_evaluations",
    }
    output = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
