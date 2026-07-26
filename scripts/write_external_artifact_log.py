from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "external_artifacts.json"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(name: str, expected_path: str, role: str, known_sha256: str, environment_hint: str | None = None) -> dict[str, Any]:
    repo_path = ROOT / expected_path
    candidate = repo_path
    if environment_hint and os.getenv(environment_hint):
        candidate = Path(os.environ[environment_hint]).expanduser().resolve()
    present = candidate.exists()
    item: dict[str, Any] = {
        "name": name,
        "role": role,
        "expected_path": expected_path,
        "environment_override": environment_hint,
        "included_in_release": False,
        "reason": "User-owned source or analysis artifact is unchanged and intentionally omitted from the distributable.",
        "known_sha256": known_sha256,
        "present_at_generation": present,
    }
    if present:
        item.update({"resolved_path": str(candidate), "size_bytes": candidate.stat().st_size, "sha256": checksum(candidate)})
    return item


payload = {
    "contract": "Copy only the runtime source archive into the expected path, or set REACTS_SOURCE_ARTIFACT. The EDA notebook and legacy producer code are provenance references and are not runtime dependencies.",
    "artifacts": [
        record(
            "USPTO-LLM multistep artifact",
            "data/source_artifacts/uspto_llm_multistep_only.zip",
            "Runtime input used to rebuild the canonical dataset.",
            "835e202b32d7f00d2573c1116e65bc7bd763c312d5f3956b2c680a524aa15b8e",
            "REACTS_SOURCE_ARTIFACT",
        ),
        record(
            "Patched EDA notebook",
            "reference_inputs/uspto_llm_sharper_eda_patched.ipynb",
            "Analysis provenance used to reconstruct source-data behavior and destructive cleaning effects.",
            "0e04cf327d66ddc43abd874d3f0018ce1f4b0349679cee27c507550f346cb9d0",
        ),
        record(
            "Legacy reaction_curation producer",
            "reference_inputs/reaction_curation.zip",
            "Producer-code provenance used to identify duplicate-column, ID, condition parsing, QC, and split defects.",
            "f97b5117671addbb6504a32ca5a22a8d48c59be4540d84f2502fb985b981d10c",
        ),
    ],
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(OUTPUT)
