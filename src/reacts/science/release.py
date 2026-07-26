from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reacts.contracts import ModelStage
from reacts.ml.registry import Registry
from reacts.science.hashing import hash_paths, portable_path, sha256_file
from reacts.settings import Settings


RELEASABLE_STAGES = {
    ModelStage.PRODUCTION.value,
    ModelStage.STAGING.value,
    ModelStage.SCREENING.value,
    ModelStage.BASELINE.value,
}


def lock_product_two_release(
    settings: Settings,
    registry: Registry,
    acceptance_report: dict[str, Any],
    *,
    release_id: str = "v2.0.0",
) -> dict[str, Any]:
    if not acceptance_report.get("strict_pass"):
        raise RuntimeError("Product Two cannot be locked until strict scientific acceptance passes.")
    release_dir = settings.releases_dir / release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = settings.canonical_v2_dir / "dataset_manifest.json"
    index_manifest = settings.index_v2_dir / "index_manifest.json"
    route_index_manifest = settings.index_v2_dir / "routes" / "route_index_manifest.json"
    split_manifest = settings.canonical_v2_dir / "split_manifest.json"
    model_registry = settings.model_dir / "model_registry.json"
    acceptance_path = settings.reports_dir / "product_two_scientific_acceptance.json"
    models: list[dict[str, Any]] = []
    for record in registry.list_models(
        runtime_only=True, dataset_version="uspto_multistep_contextual_v2"
    ):
        if record["stage"] not in RELEASABLE_STAGES:
            continue
        models.append(
            {
                "model_id": record["model_id"],
                "task": record["task"],
                "version": record["version"],
                "stage": record["stage"],
                "artifact_path": record["artifact_path"],
                "artifact_sha256": record.get("artifact_sha256"),
                "dataset_sha256": record.get("dataset_sha256"),
                "feature_sha256": record.get("feature_sha256"),
                "split_sha256": record.get("split_sha256"),
                "lifecycle_state": record.get("lifecycle_state"),
                "runtime_load_required": record.get("runtime_load_required"),
                "training_environment": record.get("training_environment"),
            }
        )
    if not models:
        raise RuntimeError("No qualified Product Two models are available for release locking.")
    artifact_paths = [
        registry.resolve_artifact_path(record["artifact_path"])
        for record in models
        if registry.resolve_artifact_path(record["artifact_path"]).exists()
    ]
    manifest = {
        "release_id": release_id,
        "release_type": "product_two_scientific_release",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_release": "v1.0.0-baseline",
        "dataset": {
            "version": "uspto_multistep_contextual_v2",
            "manifest_path": portable_path(dataset_manifest, settings.project_root),
            "manifest_sha256": sha256_file(dataset_manifest),
        },
        "split_governance": {
            "manifest_path": portable_path(split_manifest, settings.project_root),
            "manifest_sha256": sha256_file(split_manifest),
        },
        "retrieval": {
            "manifest_path": portable_path(index_manifest, settings.project_root),
            "manifest_sha256": sha256_file(index_manifest),
            "route_manifest_path": portable_path(route_index_manifest, settings.project_root),
            "route_manifest_sha256": sha256_file(route_index_manifest),
        },
        "model_registry": {
            "path": portable_path(model_registry, settings.project_root),
            "sha256": sha256_file(model_registry),
        },
        "acceptance": {
            "report_path": portable_path(acceptance_path, settings.project_root),
            "report_sha256": sha256_file(acceptance_path),
            "strict_pass": True,
        },
        "models": models,
        "model_artifact_tree_sha256": hash_paths(artifact_paths, root=settings.project_root),
        "contract": {
            "immutable": True,
            "qualified_models_only": True,
            "task_specific_release_stages": True,
            "evidence_grounded_inference": True,
            "experimental_models_excluded": True,
            "connected_component_split_locked": True,
            "runtime_lifecycle_enforced": True,
            "training_environment_locked": True,
        },
    }
    path = release_dir / "release_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    snapshot = registry.create_release_snapshot(release_id, "scientific_release", path, manifest, locked=True)
    return {**manifest, "snapshot": snapshot}
