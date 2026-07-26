from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REQUIRED_MEMBERS = {
    "steps": "uspto_llm_multistep_only/multistep_csv/step_table.csv",
    "routes": "uspto_llm_multistep_only/multistep_csv/route_summary.csv",
    "cleaned_routes": "uspto_llm_multistep_only/multistep_csv/cleaned_full.csv",
    "run_summary": "uspto_llm_multistep_only/run_summary.json",
    "manifest": "uspto_llm_multistep_only/artifact_manifest.json",
    "qc": "uspto_llm_multistep_only/qc/multistep_csv_qc.json",
    "condition_qc": "uspto_llm_multistep_only/qc/multistep_csv_condition_qc.json",
}


@dataclass(frozen=True)
class SourceInventory:
    source_path: str
    source_kind: str
    checksum_sha256: str
    members: dict[str, str]
    metadata: dict


class ArtifactSource:
    """Resolves the legacy artifact from a ZIP or an extracted directory.

    Large unchanged source files are never copied into the REACTS release. The
    resolver materializes only a requested member into a caller-owned staging
    directory.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Source artifact not found: {self.path}")
        self.is_zip = self.path.is_file() and self.path.suffix.lower() == ".zip"
        if not self.is_zip and not self.path.is_dir():
            raise ValueError(f"Expected ZIP or directory, got {self.path}")

    def checksum(self) -> str:
        h = hashlib.sha256()
        if self.is_zip:
            with self.path.open("rb") as fh:
                for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                    h.update(block)
        else:
            for p in sorted(self.path.rglob("*")):
                if p.is_file():
                    h.update(str(p.relative_to(self.path)).encode())
                    h.update(str(p.stat().st_size).encode())
        return h.hexdigest()

    def _directory_member(self, member: str) -> Path:
        direct = self.path / member
        if direct.exists():
            return direct
        stripped = member.split("/", 1)[-1]
        alternate = self.path / stripped
        if alternate.exists():
            return alternate
        raise FileNotFoundError(member)

    def read_json(self, logical_name: str) -> dict:
        member = REQUIRED_MEMBERS[logical_name]
        if self.is_zip:
            with zipfile.ZipFile(self.path) as zf:
                return json.loads(zf.read(member).decode("utf-8"))
        return json.loads(self._directory_member(member).read_text(encoding="utf-8"))

    @contextmanager
    def materialize(self, logical_name: str, staging_dir: Path) -> Iterator[Path]:
        member = REQUIRED_MEMBERS[logical_name]
        staging_dir.mkdir(parents=True, exist_ok=True)
        target = staging_dir / Path(member).name
        if self.is_zip:
            with zipfile.ZipFile(self.path) as zf, zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            try:
                yield target
            finally:
                target.unlink(missing_ok=True)
        else:
            yield self._directory_member(member)

    def inventory(self) -> SourceInventory:
        present: dict[str, str] = {}
        if self.is_zip:
            with zipfile.ZipFile(self.path) as zf:
                names = set(zf.namelist())
                for logical, member in REQUIRED_MEMBERS.items():
                    if member in names:
                        present[logical] = member
        else:
            for logical, member in REQUIRED_MEMBERS.items():
                try:
                    present[logical] = str(self._directory_member(member))
                except FileNotFoundError:
                    pass
        metadata = {}
        for name in ("run_summary", "manifest", "qc", "condition_qc"):
            if name in present:
                metadata[name] = self.read_json(name)
        return SourceInventory(
            source_path=str(self.path.resolve()),
            source_kind="zip" if self.is_zip else "directory",
            checksum_sha256=self.checksum(),
            members=present,
            metadata=metadata,
        )
