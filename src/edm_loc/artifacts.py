from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes, *, overwrite: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and target.exists():
        raise FileExistsError(target)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    atomic_write_bytes(path, payload, overwrite=overwrite)
    return sha256_bytes(payload)


def atomic_save_npz(path: str | Path, *, overwrite: bool = True, **arrays: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and target.exists():
        raise FileExistsError(target)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npz", dir=target.parent)
    os.close(descriptor)
    try:
        np.savez(temporary, **arrays)
        with open(temporary, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return sha256_file(target)


def require_empty_output_dir(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        if not target.is_dir():
            raise NotADirectoryError(target)
        if next(target.iterdir(), None) is not None:
            raise FileExistsError(f"refusing non-empty output directory: {target}")
    else:
        target.mkdir(parents=True)
    return target


def artifact_manifest(paths: Iterable[str | Path], *, root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for source in sorted((Path(item).resolve() for item in paths), key=str):
        try:
            relative = source.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"artifact outside root: {source}") from exc
        rows.append(
            {"path": relative.as_posix(), "bytes": source.stat().st_size, "sha256": sha256_file(source)}
        )
    return {"schema_version": 1, "root": str(base), "artifacts": rows}


def environment_record() -> dict[str, Any]:
    import numpy
    import torch

    pip = subprocess.run(
        [os.sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.encode("utf-8")
    cuda = torch.version.cuda
    cudnn = torch.backends.cudnn.version()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "python_executable": os.sys.executable,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pytorch": torch.__version__,
        "cuda": cuda,
        "cudnn": cudnn,
        "gpu": gpu,
        "gpu_count": torch.cuda.device_count(),
        "pip_freeze_sha256": sha256_bytes(pip),
    }
