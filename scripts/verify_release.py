"""Read-only integrity and static checks for a clean public checkout."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with (ROOT / "SOURCE_MANIFEST.tsv").open(encoding="utf-8", newline="") as handle:
        entries = list(csv.DictReader(handle, delimiter="\t"))
    assert entries, "empty source manifest"
    listed = set()
    for row in entries:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"manifest mismatch: {relative}")
        listed.add(relative.as_posix())
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and "outputs" not in path.parts
        and path.name != "SOURCE_MANIFEST.tsv" and path.suffix not in {".pyc"}
    }
    if listed != actual:
        raise ValueError(f"manifest file-set mismatch: missing={sorted(actual-listed)}, extra={sorted(listed-actual)}")
    expected_figures = {
        "figure_1_edm_loc_overview.png",
        "figure_2_sampling_budget.png",
        "figure_3_calibrated_regions.png",
    }
    actual_figures = {path.name for path in (ROOT / "figures").iterdir() if path.is_file()}
    if actual_figures != expected_figures or (ROOT / "figures/data").exists():
        raise ValueError("figures directory must contain only the three manuscript PNG exports")
    for name in expected_figures:
        if (ROOT / "figures" / name).read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"invalid PNG: {name}")
    if (ROOT / "configs/paper_protocol.json").exists():
        raise ValueError("paper protocol summary must not be in the release")
    print(f"PASS: {len(entries)} files, AST/JSON, three manuscript PNG exports")


if __name__ == "__main__":
    main()
