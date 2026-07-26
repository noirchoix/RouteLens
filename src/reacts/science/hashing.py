from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from reacts.storage.tabular import iter_dataset


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


def hash_paths(paths: Iterable[Path], root: Path | None = None) -> str:
    root = root.resolve() if root else None
    digest = hashlib.sha256()
    for path in sorted((Path(p).resolve() for p in paths), key=str):
        name = str(path.relative_to(root)) if root and path.is_relative_to(root) else path.name
        digest.update(name.replace("\\", "/").encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def hash_dataset_columns(root: Path, dataset_name: str, columns: list[str]) -> str:
    """Hash selected columns deterministically without loading the full corpus."""
    digest = hashlib.sha256()
    for chunk in iter_dataset(root, dataset_name, columns=columns):
        for row in chunk[columns].itertuples(index=False, name=None):
            digest.update(canonical_json_hash(row).encode("ascii"))
    return digest.hexdigest()


def portable_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return f"external://{resolved.name}"
