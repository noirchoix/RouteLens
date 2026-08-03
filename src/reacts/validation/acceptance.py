from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib

from reacts.ml.environment import runtime_environment, validate_runtime_environment
from reacts.ml.registry import Registry
from reacts.retrieval.contextual_index import ContextualFingerprintIndex
from reacts.science.hashing import hash_dataset_columns, hash_paths, sha256_file
from reacts.settings import Settings
from reacts.storage.tabular import iter_dataset
from reacts.validation.leakage import LeakageAuditor

PRODUCT_TWO_DATASET = "uspto_multistep_contextual_v2"
REQUIRED_PRODUCT_TWO_TASKS = {
    "parse_failure_class",
    "reaction_family",
    "solvent_multilabel",
    "solvent_family_multilabel",
    "time_regression",
    "temperature_regression",
    "agent_family_multilabel",
    "catalyst_family_multilabel",
}


class ScientificAcceptanceValidator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry = Registry(settings.registry_db, read_only=True)

    @staticmethod
    def _load(path: Path) -> dict[str, Any] | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _current_split_sha(self) -> str | None:
        try:
            return hash_dataset_columns(
                self.settings.canonical_v2_dir,
                "steps",
                [
                    "step_id",
                    "patent_document_id",
                    "reaction_signature",
                    "split_component_id",
                    "split",
                ],
            )
        except (FileNotFoundError, KeyError, ValueError):
            return None

    def _pipeline_state(self) -> dict[str, Any]:
        context_path = self.settings.canonical_v2_context_dir / "dataset_manifest.json"
        mapping_path = self.settings.mapping_v2_dir / "mapping_manifest.json"
        derivation_path = self.settings.derivation_v2_dir / "derivation_manifest.json"
        final_path = self.settings.canonical_v2_dir / "dataset_manifest.json"
        split_path = self.settings.canonical_v2_dir / "split_manifest.json"

        context = self._load(context_path)
        mapping = self._load(mapping_path)
        derivation = self._load(derivation_path)
        final = self._load(final_path)
        split = self._load(split_path)
        mapping_summary = (mapping or {}).get("queue_summary", {})
        derivation_summary = (derivation or {}).get("queue_summary", {})
        preflight = (mapping or {}).get("preflight", {}) or {}
        segregated = all(
            (self.settings.mapping_v2_dir / name).exists()
            for name in ["reaction_mappings_rxnmapper", "reaction_mappings_rejected"]
        )
        mapping_complete = bool(mapping) and all(
            mapping_summary.get(status, 0) == 0 for status in ("pending", "running")
        )
        derivation_complete = bool(derivation) and all(
            derivation_summary.get(status, 0) == 0 for status in ("pending", "running")
        )
        primary_is_rxnmapper = preflight.get("primary_backend") == "rxnmapper"
        split_pass = bool((split or {}).get("invariants", {}).get("strict_pass"))
        passed = all(
            [
                context,
                mapping,
                derivation,
                final,
                split,
                mapping_complete,
                derivation_complete,
                segregated,
                primary_is_rxnmapper,
                split_pass,
            ]
        )
        return {
            "context_manifest": context_path.exists(),
            "mapping_manifest": mapping_path.exists(),
            "derivation_manifest": derivation_path.exists(),
            "final_manifest": final_path.exists(),
            "split_manifest": split_path.exists(),
            "mapping_queue_summary": mapping_summary,
            "derivation_queue_summary": derivation_summary,
            "mapping_complete": mapping_complete,
            "derivation_complete": derivation_complete,
            "primary_backend": preflight.get("primary_backend"),
            "rxnmapper_required_for_strict_release": primary_is_rxnmapper,
            "mapping_outputs_segregated": segregated,
            "split_algorithm": (split or {}).get("algorithm"),
            "split_seed": (split or {}).get("seed"),
            "split_invariants": (split or {}).get("invariants", {}),
            "mapping_rebuilt_by_v209": False,
            "derivation_rebuilt_by_v209": False,
            "pass": passed,
        }

    def _dataset_integrity(self) -> dict[str, Any]:
        final_root = self.settings.canonical_v2_dir
        if not (final_root / "steps").exists() or not (final_root / "routes").exists():
            return {"rows": 0, "pass": False, "reason": "final canonical v2 is not materialized"}

        step_instances: set[str] = set()
        step_to_route: dict[str, str] = {}
        duplicate_steps = 0
        rows = 0
        route_splits: dict[str, str] = {}
        route_components: dict[str, str] = {}
        split_conflicts = 0
        component_conflicts = 0
        resolution_counts: Counter[str] = Counter()
        parse_valid = 0
        columns = [
            "step_id",
            "step_instance_id",
            "route_id",
            "route_instance_id",
            "split",
            "split_component_id",
            "contextual_parse_ok",
            "resolution_status",
        ]
        for chunk in iter_dataset(final_root, "steps", columns=columns):
            for row in chunk.to_dict(orient="records"):
                step_id = str(row.get("step_instance_id") or row["step_id"])
                route_id = str(row.get("route_instance_id") or row["route_id"])
                duplicate_steps += int(step_id in step_instances)
                step_instances.add(step_id)
                step_to_route[step_id] = route_id
                rows += 1
                parse_valid += int(bool(row.get("contextual_parse_ok")))
                resolution_counts[str(row.get("resolution_status"))] += 1
                split = str(row.get("split"))
                component = str(row.get("split_component_id"))
                if route_id in route_splits and route_splits[route_id] != split:
                    split_conflicts += 1
                if route_id in route_components and route_components[route_id] != component:
                    component_conflicts += 1
                route_splits[route_id] = split
                route_components[route_id] = component

        route_instances: set[str] = set()
        source_route_counts: Counter[str] = Counter()
        route_rows = 0
        for chunk in iter_dataset(
            final_root,
            "routes",
            columns=["route_id", "route_instance_id", "source_route_id", "split", "split_component_id"],
        ):
            for row in chunk.to_dict(orient="records"):
                route_rows += 1
                instance = str(row.get("route_instance_id") or row["route_id"])
                source = str(row.get("source_route_id") or row["route_id"])
                route_instances.add(instance)
                source_route_counts[source] += 1
                if route_splits.get(instance) != str(row.get("split")):
                    split_conflicts += 1
                if route_components.get(instance) != str(row.get("split_component_id")):
                    component_conflicts += 1

        cross_variant_edges = 0
        try:
            for chunk in iter_dataset(
                final_root,
                "route_edges",
                columns=["route_instance_id", "source_step_instance_id", "target_step_instance_id"],
            ):
                for row in chunk.to_dict(orient="records"):
                    route = str(row["route_instance_id"])
                    if step_to_route.get(str(row["source_step_instance_id"])) != route:
                        cross_variant_edges += 1
                    if step_to_route.get(str(row["target_step_instance_id"])) != route:
                        cross_variant_edges += 1
        except FileNotFoundError:
            pass

        expected_route_rows = sum(
            len(chunk) for chunk in iter_dataset(self.settings.canonical_dir, "routes")
        )
        expected_source_routes: set[str] = set()
        for chunk in iter_dataset(self.settings.canonical_dir, "routes", columns=["route_id"]):
            expected_source_routes.update(chunk["route_id"].astype(str))

        duplicate_source_groups = sum(count > 1 for count in source_route_counts.values())
        preserved_variant_rows = route_rows - len(source_route_counts)
        passed = all(
            [
                duplicate_steps == 0,
                split_conflicts == 0,
                component_conflicts == 0,
                cross_variant_edges == 0,
                rows > 0,
                route_rows == len(route_instances) == expected_route_rows,
                len(source_route_counts) == len(expected_source_routes),
            ]
        )
        return {
            "rows": rows,
            "unique_step_instance_ids": len(step_instances),
            "duplicate_step_instance_ids": duplicate_steps,
            "routes_total": route_rows,
            "unique_route_instance_ids": len(route_instances),
            "unique_source_route_ids": len(source_route_counts),
            "duplicate_source_route_groups": duplicate_source_groups,
            "preserved_conflicting_variant_rows": preserved_variant_rows,
            "cross_variant_edges": cross_variant_edges,
            "route_split_conflicts": split_conflicts,
            "route_component_conflicts": component_conflicts,
            "contextual_parse_valid": parse_valid,
            "resolution_counts": dict(resolution_counts),
            "expected_product_one_route_rows": expected_route_rows,
            "pass": passed,
        }

    def _index_state(self, current_split_sha: str | None) -> dict[str, Any]:
        reaction_path = self.settings.index_v2_dir / "index_manifest.json"
        route_path = self.settings.index_v2_dir / "routes" / "route_index_manifest.json"
        reaction = self._load(reaction_path)
        route = self._load(route_path)
        reaction_match = bool(reaction and current_split_sha and reaction.get("training_split_sha256") == current_split_sha)
        route_match = bool(route and current_split_sha and route.get("training_split_sha256") == current_split_sha)
        return {
            "reaction_manifest": reaction_path.exists(),
            "route_manifest": route_path.exists(),
            "reaction_rows": (reaction or {}).get("rows"),
            "route_rows": (route or {}).get("rows"),
            "current_training_split_sha256": current_split_sha,
            "reaction_training_split_sha256": (reaction or {}).get("training_split_sha256"),
            "route_training_split_sha256": (route or {}).get("training_split_sha256"),
            "reaction_split_matches": reaction_match,
            "route_split_matches": route_match,
            "pass": bool(reaction and route and reaction_match and route_match),
        }

    def _model_smoke(self, current_split_sha: str | None) -> dict[str, Any]:
        all_records = self.registry.list_models(dataset_version=PRODUCT_TWO_DATASET)
        runtime_records = self.registry.list_models(
            runtime_only=True, dataset_version=PRODUCT_TWO_DATASET
        )
        audits = self.registry.list_task_audits(dataset_version=PRODUCT_TWO_DATASET)
        latest_audit_by_task: dict[str, dict[str, Any]] = {}
        for audit in audits:
            latest_audit_by_task.setdefault(str(audit["task"]), audit)

        loaded: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        seen_tasks: Counter[str] = Counter()
        satisfied_tasks: set[str] = set()
        for record in runtime_records:
            task = str(record["task"])
            seen_tasks[task] += 1
            path = self.registry.resolve_artifact_path(record["artifact_path"])
            checks: dict[str, Any] = {
                "artifact_exists": path.exists(),
                "artifact_sha256_matches": False,
                "split_sha256_matches": bool(
                    current_split_sha and record.get("split_sha256") == current_split_sha
                ),
                "environment": validate_runtime_environment(record.get("training_environment")),
                "model_card_exists": self.registry.resolve_artifact_path(
                    record.get("model_card_path") or path.with_suffix(".model_card.json")
                ).exists(),
            }
            if path.exists():
                checks["artifact_sha256_matches"] = bool(
                    record.get("artifact_sha256")
                    and sha256_file(path) == record.get("artifact_sha256")
                )
            prefit = (record.get("metrics") or {}).get("prefit_support")
            if isinstance(prefit, dict):
                checks["prefit_trainable"] = bool(prefit.get("trainable"))
            else:
                checks["prefit_trainable"] = True
            checks["pass_before_load"] = all(
                [
                    checks["artifact_exists"],
                    checks["artifact_sha256_matches"],
                    checks["split_sha256_matches"],
                    checks["environment"]["pass"],
                    checks["model_card_exists"],
                    checks["prefit_trainable"],
                ]
            )
            if not checks["pass_before_load"]:
                failures.append(
                    {"model_id": record["model_id"], "task": task, "checks": checks}
                )
                continue
            try:
                bundle = joblib.load(path)
                bundle_environment = validate_runtime_environment(
                    bundle.get("training_environment")
                )
                if not bundle_environment["pass"]:
                    raise RuntimeError(
                        f"Serialized training environment mismatch: {bundle_environment}"
                    )
                loaded.append(
                    {
                        "model_id": record["model_id"],
                        "stage": record.get("effective_release_stage") or record["stage"],
                        "lifecycle_state": record.get("lifecycle_state"),
                        "task": bundle.get("task", task),
                        "checks": checks,
                    }
                )
                satisfied_tasks.add(task)
            except Exception as exc:
                failures.append(
                    {"model_id": record["model_id"], "task": task, "error": str(exc)}
                )

        current_audits: list[dict[str, Any]] = []
        for task, audit in latest_audit_by_task.items():
            if current_split_sha and audit.get("split_sha256") == current_split_sha:
                current_audits.append(
                    {
                        "task": task,
                        "audit_id": audit["audit_id"],
                        "reason_code": audit["reason_code"],
                        "lifecycle_state": audit["lifecycle_state"],
                    }
                )
                satisfied_tasks.add(task)

        duplicate_runtime_tasks = {
            task: count for task, count in seen_tasks.items() if count > 1
        }
        missing_tasks = sorted(REQUIRED_PRODUCT_TWO_TASKS - satisfied_tasks)
        archived = [
            {
                "model_id": record["model_id"],
                "task": record["task"],
                "lifecycle_state": record.get("lifecycle_state"),
                "runtime_load_required": record.get("runtime_load_required"),
            }
            for record in all_records
            if not record.get("runtime_load_required")
        ]
        runtime = runtime_environment()
        registry_json = self.settings.model_dir / "model_registry.json"
        passed = all(
            [
                registry_json.exists(),
                runtime.get("scikit_learn") == "1.9.0",
                not failures,
                not duplicate_runtime_tasks,
                not missing_tasks,
            ]
        )
        return {
            "registry_json": registry_json.exists(),
            "registry_json_sha256": sha256_file(registry_json) if registry_json.exists() else None,
            "runtime_environment": runtime,
            "loaded": loaded,
            "current_task_audits": current_audits,
            "archived_or_superseded": archived,
            "archived_artifacts_deserialized": 0,
            "duplicate_runtime_tasks": duplicate_runtime_tasks,
            "required_tasks": sorted(REQUIRED_PRODUCT_TWO_TASKS),
            "missing_tasks": missing_tasks,
            "failures": failures,
            "pass": passed,
        }

    def _api_smoke(self) -> dict[str, Any]:
        environment_key = "REACTS_API_READ_ONLY_REGISTRY"
        previous = os.environ.get(environment_key)
        os.environ[environment_key] = "1"
        try:
            # Importing reacts.api.main constructs its module-level ASGI app. During
            # acceptance validation that cold-import app must also use a read-only
            # registry, not only the explicit test app created below.
            from reacts.api.main import create_app
        finally:
            if previous is None:
                os.environ.pop(environment_key, None)
            else:
                os.environ[environment_key] = previous

        from fastapi.testclient import TestClient

        client = TestClient(create_app(self.settings, read_only_registry=True))
        health = client.get("/health")
        v2_health = client.get("/api/v2/health")
        reaction = "CCO>>CC=O"
        payload = {
            "reaction_smiles": reaction,
            "include_evidence": False,
            "evidence_k": 0,
            "tasks": [],
            "allow_experimental": True,
        }
        headers = {"X-REACTS-Allow-Experimental": "true"}
        single = client.post("/api/v2/inference/contextual", json=payload, headers=headers)
        batch = client.post(
            "/api/v2/inference/batch",
            json={
                "reactions": [reaction],
                "include_evidence": False,
                "evidence_k": 0,
                "tasks": [],
                "allow_experimental": True,
            },
            headers=headers,
        )
        equivalent = False
        if single.status_code == 200 and batch.status_code == 200:
            batch_rows = batch.json().get("results", [])
            equivalent = bool(
                batch_rows and batch_rows[0]["parse_ok"] == single.json()["parse_ok"]
            )
        statuses = [health.status_code, v2_health.status_code, single.status_code, batch.status_code]
        return {
            "health_status": health.status_code,
            "v2_health_status": v2_health.status_code,
            "single_status": single.status_code,
            "batch_status": batch.status_code,
            "online_batch_equivalent": equivalent,
            "pass": all(code == 200 for code in statuses) and equivalent,
        }

    def _retrieval_benchmark(self, sample_size: int = 50) -> dict[str, Any]:
        manifest = self.settings.index_v2_dir / "index_manifest.json"
        if not manifest.exists():
            return {"pass": False, "reason": "contextual index missing"}
        queries: list[dict[str, Any]] = []
        columns = [
            "step_id",
            "canonical_resolved_reaction_smiles",
            "reaction_family",
            "eligible_retrieval_v2",
        ]
        for chunk in iter_dataset(self.settings.canonical_v2_dir, "steps", columns=columns):
            subset = chunk.loc[chunk["eligible_retrieval_v2"].fillna(False).astype(bool)]
            for row in subset.to_dict(orient="records"):
                queries.append(
                    {
                        "step_id": row["step_id"],
                        "reaction_smiles": row["canonical_resolved_reaction_smiles"],
                        "reaction_family": row.get("reaction_family"),
                    }
                )
                if len(queries) >= sample_size:
                    break
            if len(queries) >= sample_size:
                break
        index = ContextualFingerprintIndex(self.settings.index_v2_dir)
        report = index.benchmark(queries, k=10)
        report["pass"] = (
            report["self_recall_at_k"] >= 0.95
            and report["latency_ms"]["p95"] < 10_000
        )
        return report

    def validate(self) -> dict[str, Any]:
        started = time.time()
        registry_json = self.settings.model_dir / "model_registry.json"
        registry_before = {
            "database_sha256": (
                sha256_file(self.settings.registry_db)
                if self.settings.registry_db.exists()
                else None
            ),
            "json_sha256": sha256_file(registry_json) if registry_json.exists() else None,
        }
        pipeline = self._pipeline_state()
        integrity = self._dataset_integrity()
        current_split_sha = self._current_split_sha()
        leakage = (
            LeakageAuditor(self.settings.canonical_v2_dir).audit()
            if pipeline["final_manifest"]
            else {"strict_pass": False, "reason": "final canonical v2 is not materialized"}
        )
        indexes = self._index_state(current_split_sha)
        models = self._model_smoke(current_split_sha)
        api = self._api_smoke()
        retrieval = self._retrieval_benchmark()
        files = [
            path for path in self.settings.canonical_v2_dir.rglob("*") if path.is_file()
        ]
        registry_after = {
            "database_sha256": (
                sha256_file(self.settings.registry_db)
                if self.settings.registry_db.exists()
                else None
            ),
            "json_sha256": sha256_file(registry_json) if registry_json.exists() else None,
        }
        registry_stable = registry_before == registry_after
        report = {
            "pipeline_state": pipeline,
            "dataset_integrity": integrity,
            "leakage": leakage,
            "index_state": indexes,
            "model_loading": models,
            "api_smoke": api,
            "retrieval": retrieval,
            "reproducibility": {
                "canonical_v2_tree_sha256": (
                    hash_paths(files, root=self.settings.project_root) if files else None
                ),
                "training_split_sha256": current_split_sha,
                "registry_database_sha256": registry_after["database_sha256"],
                "registry_json_sha256": registry_after["json_sha256"],
                "registry_read_only_validation": {
                    "before": registry_before,
                    "after": registry_after,
                    "pass": registry_stable,
                },
            },
            "elapsed_seconds": time.time() - started,
        }
        report["strict_pass"] = all(
            [
                pipeline["pass"],
                integrity["pass"],
                leakage["strict_pass"],
                indexes["pass"],
                models["pass"],
                api["pass"],
                retrieval.get("pass", False),
                registry_stable,
            ]
        )
        return report

    def write(self, path: Path) -> dict[str, Any]:
        report = self.validate()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return report
