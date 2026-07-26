from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reacts.ml.registry import Registry  # noqa: E402
from reacts.settings import Settings  # noqa: E402
from reacts.storage.tabular import dataset_rows  # noqa: E402

CONDITION_TASKS = {"primary_solvent", "time_bucket", "temperature_bucket", "agent_presence"}


def main() -> int:
    settings = Settings(project_root=ROOT).resolve()
    manifest_path = settings.canonical_dir / "dataset_manifest.json"
    errors: list[str] = []
    if not manifest_path.exists():
        errors.append("canonical manifest is missing")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for dataset in ["steps", "routes"]:
        try:
            counts[dataset] = dataset_rows(settings.canonical_dir, dataset)
        except FileNotFoundError:
            errors.append(f"canonical dataset missing: {dataset}")

    registry = Registry(settings.registry_db)
    models = registry.list_models()
    production_models = [m for m in models if m["stage"] == "production"]
    model_checks: list[dict[str, object]] = []
    for model in production_models:
        artifact = registry.resolve_artifact_path(model["artifact_path"])
        check: dict[str, object] = {
            "task": model["task"],
            "artifact_path": model["artifact_path"],
            "artifact_exists": artifact.exists(),
            "portable_path": not Path(model["artifact_path"]).is_absolute(),
        }
        if not artifact.exists():
            errors.append(f"model artifact missing: {model['task']} -> {artifact}")
        else:
            bundle = joblib.load(artifact)
            check["input_column"] = bundle.get("input_column", "reaction_smiles")
            if model["task"] in CONDITION_TASKS and check["input_column"] != "canonical_reaction_smiles":
                errors.append(f"condition model leakage guard failed: {model['task']}")
        if not check["portable_path"]:
            errors.append(f"non-portable model path: {model['artifact_path']}")
        model_checks.append(check)

    index_path = settings.index_dir / "index_manifest.json"
    index_manifest = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    if not index_path.exists():
        errors.append("retrieval index manifest is missing")
    elif int(index_manifest.get("rows", 0)) <= 0:
        errors.append("retrieval index contains no rows")

    report = {
        "valid": not errors,
        "errors": errors,
        "manifest_dataset_version": manifest.get("dataset_version"),
        "storage_format": manifest.get("storage_format"),
        "canonical_rows": counts,
        "registered_models": len(models),
        "production_models": len(production_models),
        "model_checks": model_checks,
        "index_ready": index_path.exists(),
        "index_rows": index_manifest.get("rows"),
        "index_shards": len(index_manifest.get("shards", [])),
    }
    output = ROOT / "reports" / "release_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
