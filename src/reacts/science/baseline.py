from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reacts.contracts import ModelStage
from reacts.ml.registry import Registry
from reacts.science.hashing import hash_paths, portable_path, sha256_file
from reacts.science.promotion import BASELINE_STAGE_OVERRIDES


def freeze_product_one_baseline(
    *,
    registry: Registry,
    canonical_dir: Path,
    model_dir: Path,
    index_dir: Path,
    baseline_dir: Path,
    release_id: str = "v1.0.0-baseline",
) -> dict[str, Any]:
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    existing_snapshot = registry.release_snapshot(release_id)
    if existing_snapshot:
        manifest = registry.resolve_artifact_path(existing_snapshot["manifest_path"])
        return json.loads(manifest.read_text(encoding="utf-8"))

    model_records = registry.list_models()
    staged_models: list[dict[str, Any]] = []
    for record in model_records:
        task = record["task"]
        stage = BASELINE_STAGE_OVERRIDES.get(task, ModelStage.BASELINE)
        registry.set_stage(record["model_id"], stage)
        artifact = registry.resolve_artifact_path(record["artifact_path"])
        staged_models.append(
            {
                "model_id": record["model_id"],
                "task": task,
                "version": record["version"],
                "stage": stage.value,
                "artifact_path": portable_path(artifact, registry.project_root),
                "artifact_sha256": record.get("artifact_sha256") or (sha256_file(artifact) if artifact.exists() else None),
            }
        )

    canonical_manifest = Path(canonical_dir) / "dataset_manifest.json"
    index_manifest = Path(index_dir) / "index_manifest.json"
    copied: dict[str, str] = {}
    for source, name in [
        (canonical_manifest, "dataset_manifest.v1.json"),
        (index_manifest, "index_manifest.v1.json"),
    ]:
        if source.exists():
            target = baseline_dir / name
            shutil.copy2(source, target)
            copied[name] = sha256_file(target)

    model_files = [path for path in Path(model_dir).rglob("*.joblib") if path.is_file()]
    manifest = {
        "release_id": release_id,
        "release_type": "immutable_product_one_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "source_artifacts_unchanged": True,
            "canonical_v1_is_read_only": True,
            "models_reclassified_by_task": True,
            "not_final_scientific_release": True,
        },
        "canonical": {
            "path": portable_path(Path(canonical_dir), registry.project_root),
            "manifest_sha256": sha256_file(canonical_manifest) if canonical_manifest.exists() else None,
        },
        "retrieval": {
            "path": portable_path(Path(index_dir), registry.project_root),
            "manifest_sha256": sha256_file(index_manifest) if index_manifest.exists() else None,
        },
        "models": staged_models,
        "model_tree_sha256": hash_paths(model_files, root=registry.project_root) if model_files else None,
        "copied_manifests": copied,
    }
    path = baseline_dir / "baseline_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    registry.create_release_snapshot(release_id, "baseline", path, manifest, locked=True)
    return manifest
