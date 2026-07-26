from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from reacts.contracts import ModelStage
from reacts.ml.environment import runtime_environment
from reacts.science.hashing import portable_path, sha256_file

ACTIVE_LIFECYCLES = ("active", "candidate")
TERMINAL_LIFECYCLES = ("superseded", "archived_incompatible", "audit_non_trainable")

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS dataset_versions (
  dataset_version TEXT PRIMARY KEY,
  manifest_path TEXT NOT NULL,
  registered_at_utc TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  manifest_sha256 TEXT
);
CREATE TABLE IF NOT EXISTS training_runs (
  run_id TEXT PRIMARY KEY,
  task TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at_utc TEXT NOT NULL,
  completed_at_utc TEXT,
  config_json TEXT NOT NULL,
  metrics_json TEXT,
  error TEXT,
  reproducibility_json TEXT
);
CREATE TABLE IF NOT EXISTS model_versions (
  model_id TEXT PRIMARY KEY,
  task TEXT NOT NULL,
  version TEXT NOT NULL,
  stage TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  config_json TEXT NOT NULL,
  artifact_sha256 TEXT,
  dataset_sha256 TEXT,
  feature_sha256 TEXT,
  split_sha256 TEXT,
  release_decision_json TEXT,
  lifecycle_state TEXT,
  runtime_load_required INTEGER,
  effective_release_stage TEXT,
  model_card_path TEXT,
  training_environment_json TEXT,
  superseded_by TEXT,
  UNIQUE(task, version)
);
CREATE INDEX IF NOT EXISTS idx_model_task_stage ON model_versions(task, stage);
CREATE TABLE IF NOT EXISTS task_audits (
  audit_id TEXT PRIMARY KEY,
  task TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  lifecycle_state TEXT NOT NULL,
  audit_path TEXT NOT NULL,
  reason_code TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  split_sha256 TEXT,
  created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  detail_json TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS release_snapshots (
  release_id TEXT PRIMARY KEY,
  release_type TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  manifest_path TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  locked INTEGER NOT NULL DEFAULT 1
);
"""


class Registry:
    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = Path(path).resolve()
        self.read_only = bool(read_only)
        self.project_root = self.path.parents[2] if len(self.path.parents) >= 3 else self.path.parent
        self.model_dir = self.project_root / "data" / "models"
        self.json_path = self.model_dir / "model_registry.json"
        if self.read_only:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_runtime "
                "ON model_versions(dataset_version, runtime_load_required, lifecycle_state)"
            )
            self._normalize_existing_lifecycle(conn)
        if self.model_dir.exists():
            self.sync_json()

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    def _migrate(self, conn: sqlite3.Connection) -> None:
        additions = {
            "dataset_versions": {"manifest_sha256": "TEXT"},
            "training_runs": {"reproducibility_json": "TEXT"},
            "model_versions": {
                "artifact_sha256": "TEXT",
                "dataset_sha256": "TEXT",
                "feature_sha256": "TEXT",
                "split_sha256": "TEXT",
                "release_decision_json": "TEXT",
                "lifecycle_state": "TEXT",
                "runtime_load_required": "INTEGER",
                "effective_release_stage": "TEXT",
                "model_card_path": "TEXT",
                "training_environment_json": "TEXT",
                "superseded_by": "TEXT",
            },
        }
        for table, columns in additions.items():
            existing = self._columns(conn, table)
            for name, sql_type in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

    def _normalize_existing_lifecycle(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT model_id, task, dataset_version, stage, artifact_path, created_at_utc, "
            "lifecycle_state, runtime_load_required, effective_release_stage, model_card_path, "
            "training_environment_json "
            "FROM model_versions ORDER BY task, dataset_version, created_at_utc DESC"
        ).fetchall()
        seen: set[tuple[str, str]] = set()
        for row in rows:
            group = (str(row["task"]), str(row["dataset_version"]))
            is_latest = group not in seen
            seen.add(group)
            if row["lifecycle_state"]:
                lifecycle = str(row["lifecycle_state"])
            elif is_latest:
                lifecycle = "active"
            elif not row["training_environment_json"]:
                lifecycle = "archived_incompatible"
            else:
                lifecycle = "superseded"
            runtime_required = row["runtime_load_required"]
            if runtime_required is None:
                runtime_required = int(is_latest and lifecycle in ACTIVE_LIFECYCLES)
            effective_stage = str(row["effective_release_stage"] or row["stage"])
            artifact = self.resolve_artifact_path(str(row["artifact_path"]))
            card = artifact.with_suffix(".model_card.json")
            card_path = str(row["model_card_path"] or portable_path(card, self.project_root))
            environment_json = row["training_environment_json"] or json.dumps(
                {
                    "status": "unknown_legacy",
                    "scikit_learn": None,
                    "note": "Training environment was not persisted by the pre-v2.0.9 registry.",
                }
            )
            conn.execute(
                "UPDATE model_versions SET lifecycle_state=?, runtime_load_required=?, "
                "effective_release_stage=?, model_card_path=?, training_environment_json=? "
                "WHERE model_id=?",
                (
                    lifecycle,
                    int(runtime_required),
                    effective_stage,
                    card_path,
                    environment_json,
                    row["model_id"],
                ),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            if not self.path.exists():
                raise FileNotFoundError(f"Registry database does not exist: {self.path}")
            conn = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only=ON")
        else:
            conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            if not self.read_only:
                conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _decode_record(record: dict[str, Any]) -> dict[str, Any]:
        decoded = dict(record)
        for source, target in [
            ("metrics_json", "metrics"),
            ("config_json", "config"),
            ("release_decision_json", "release_decision"),
            ("training_environment_json", "training_environment"),
        ]:
            value = decoded.pop(source, None)
            try:
                decoded[target] = json.loads(value) if value else {}
            except (TypeError, json.JSONDecodeError):
                decoded[target] = {}
        decoded["runtime_load_required"] = bool(decoded.get("runtime_load_required"))
        return decoded

    def sync_json(self) -> Path:
        if self.read_only:
            raise RuntimeError("Cannot synchronize the model registry in read-only mode.")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            model_rows = [self._decode_record(dict(row)) for row in conn.execute(
                "SELECT * FROM model_versions ORDER BY task, dataset_version, created_at_utc DESC"
            )]
            audit_rows = []
            for row in conn.execute("SELECT * FROM task_audits ORDER BY created_at_utc DESC"):
                item = dict(row)
                try:
                    item["metrics"] = json.loads(item.pop("metrics_json"))
                except (TypeError, json.JSONDecodeError):
                    item["metrics"] = {}
                audit_rows.append(item)
        payload = {
            "schema_version": "2.0.9-model-registry-v1",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "registry_database": portable_path(self.path, self.project_root),
            "runtime_environment": runtime_environment(),
            "models": model_rows,
            "task_audits": audit_rows,
            "contract": {
                "runtime_loads_only_active_records": True,
                "superseded_artifacts_are_preserved_without_deserialization": True,
                "training_environment_is_persisted_per_model": True,
                "one_runtime_required_model_per_task_and_dataset": True,
            },
        }
        temporary = self.json_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temporary, self.json_path)
        return self.json_path

    def register_dataset(self, dataset_version: str, manifest_path: Path, metadata: dict[str, Any]) -> None:
        resolved = Path(manifest_path).resolve()
        stored_manifest = portable_path(resolved, self.project_root)
        manifest_hash = sha256_file(resolved) if resolved.exists() else None
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO dataset_versions(dataset_version, manifest_path, registered_at_utc, metadata_json, manifest_sha256)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dataset_version) DO UPDATE SET
                  manifest_path=excluded.manifest_path,
                  registered_at_utc=excluded.registered_at_utc,
                  metadata_json=excluded.metadata_json,
                  manifest_sha256=excluded.manifest_sha256
                """,
                (
                    dataset_version,
                    stored_manifest,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(metadata, default=str),
                    manifest_hash,
                ),
            )

    def dataset(self, dataset_version: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM dataset_versions WHERE dataset_version=?", (dataset_version,)).fetchone()
        return dict(row) if row else None

    def start_run(self, task: str, dataset_version: str, config: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO training_runs(run_id, task, dataset_version, status, started_at_utc, config_json) VALUES (?, ?, ?, 'running', ?, ?)",
                (run_id, task, dataset_version, datetime.now(timezone.utc).isoformat(), json.dumps(config, default=str)),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        metrics: dict[str, Any],
        error: str | None = None,
        reproducibility: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET status=?, completed_at_utc=?, metrics_json=?, error=?, reproducibility_json=?
                WHERE run_id=?
                """,
                (
                    "failed" if error else "completed",
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(metrics, default=str),
                    error,
                    json.dumps(reproducibility or {}, default=str),
                    run_id,
                ),
            )

    def register_task_audit(
        self,
        *,
        task: str,
        dataset_version: str,
        audit_path: Path,
        reason_code: str,
        metrics: dict[str, Any],
        split_sha256: str | None,
    ) -> dict[str, Any]:
        audit_id = uuid.uuid4().hex
        stored_path = portable_path(Path(audit_path).resolve(), self.project_root)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO task_audits VALUES (?, ?, ?, 'audit_non_trainable', ?, ?, ?, ?, ?)",
                (
                    audit_id,
                    task,
                    dataset_version,
                    stored_path,
                    reason_code,
                    json.dumps(metrics, default=str),
                    split_sha256,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        self.sync_json()
        return {
            "audit_id": audit_id,
            "task": task,
            "dataset_version": dataset_version,
            "lifecycle_state": "audit_non_trainable",
            "audit_path": stored_path,
            "reason_code": reason_code,
            "split_sha256": split_sha256,
        }

    def register_model(
        self,
        *,
        task: str,
        artifact_path: Path,
        dataset_version: str,
        metrics: dict[str, Any],
        config: dict[str, Any],
        stage: ModelStage = ModelStage.CANDIDATE,
        dataset_sha256: str | None = None,
        feature_sha256: str | None = None,
        split_sha256: str | None = None,
        release_decision: dict[str, Any] | None = None,
        training_environment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created = datetime.now(timezone.utc)
        version = created.strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        model_id = f"{task}:{version}"
        resolved_artifact = Path(artifact_path).resolve()
        stored_artifact = portable_path(resolved_artifact, self.project_root)
        card_path = portable_path(resolved_artifact.with_suffix(".model_card.json"), self.project_root)
        artifact_hash = sha256_file(resolved_artifact)
        environment = training_environment or runtime_environment()
        with self.connect() as conn:
            conn.execute(
                "UPDATE model_versions SET lifecycle_state='superseded', runtime_load_required=0, "
                "superseded_by=? WHERE task=? AND dataset_version=? AND runtime_load_required=1",
                (model_id, task, dataset_version),
            )
            conn.execute(
                """
                INSERT INTO model_versions(
                    model_id, task, version, stage, artifact_path, dataset_version, created_at_utc,
                    metrics_json, config_json, artifact_sha256, dataset_sha256, feature_sha256,
                    split_sha256, release_decision_json, lifecycle_state, runtime_load_required,
                    effective_release_stage, model_card_path, training_environment_json, superseded_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, ?, ?, NULL)
                """,
                (
                    model_id,
                    task,
                    version,
                    stage.value,
                    stored_artifact,
                    dataset_version,
                    created.isoformat(),
                    json.dumps(metrics, default=str),
                    json.dumps(config, default=str),
                    artifact_hash,
                    dataset_sha256,
                    feature_sha256,
                    split_sha256,
                    json.dumps(release_decision or {}, default=str),
                    stage.value,
                    card_path,
                    json.dumps(environment, default=str),
                ),
            )
        self.sync_json()
        return {
            "model_id": model_id,
            "task": task,
            "version": version,
            "stage": stage.value,
            "effective_release_stage": stage.value,
            "lifecycle_state": "active",
            "runtime_load_required": True,
            "artifact_path": stored_artifact,
            "model_card_path": card_path,
            "artifact_sha256": artifact_hash,
            "dataset_sha256": dataset_sha256,
            "feature_sha256": feature_sha256,
            "split_sha256": split_sha256,
            "training_environment": environment,
            "release_decision": release_decision or {},
        }

    def invalidate_models_for_split(self, dataset_version: str, split_sha256: str) -> int:
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE model_versions SET lifecycle_state='superseded', runtime_load_required=0, "
                "superseded_by='split-governance-v2.0.9' WHERE dataset_version=? "
                "AND COALESCE(split_sha256, '') <> ? AND runtime_load_required=1",
                (dataset_version, split_sha256),
            )
            count = int(result.rowcount)
        self.sync_json()
        return count

    def promote(
        self,
        model_id: str,
        stage: ModelStage,
        *,
        release_decision: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        if stage in {ModelStage.PRODUCTION, ModelStage.STAGING, ModelStage.SCREENING} and not force:
            if not release_decision or not release_decision.get("approved"):
                raise PermissionError("Promotion requires an approved task-specific release decision.")
        with self.connect() as conn:
            row = conn.execute("SELECT task, dataset_version FROM model_versions WHERE model_id=?", (model_id,)).fetchone()
            if row is None:
                raise KeyError(model_id)
            if stage == ModelStage.PRODUCTION:
                conn.execute(
                    "UPDATE model_versions SET stage=?, effective_release_stage=?, lifecycle_state='superseded', "
                    "runtime_load_required=0, superseded_by=? WHERE task=? AND dataset_version=? AND stage=?",
                    (
                        ModelStage.DEPRECATED.value,
                        ModelStage.DEPRECATED.value,
                        model_id,
                        row["task"],
                        row["dataset_version"],
                        ModelStage.PRODUCTION.value,
                    ),
                )
            conn.execute(
                "UPDATE model_versions SET stage=?, effective_release_stage=?, lifecycle_state='active', "
                "runtime_load_required=1, release_decision_json=COALESCE(?, release_decision_json) WHERE model_id=?",
                (
                    stage.value,
                    stage.value,
                    json.dumps(release_decision, default=str) if release_decision else None,
                    model_id,
                ),
            )
        self.sync_json()

    def set_stage(self, model_id: str, stage: ModelStage) -> None:
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE model_versions SET stage=?, effective_release_stage=? WHERE model_id=?",
                (stage.value, stage.value, model_id),
            )
            if result.rowcount == 0:
                raise KeyError(model_id)
        self.sync_json()

    def resolve_artifact_path(self, artifact_path: str | Path) -> Path:
        text = str(artifact_path)
        if text.startswith("external://"):
            raise FileNotFoundError(f"External artifact must be mounted explicitly: {text}")
        path = Path(text)
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def model_for_task(
        self,
        task: str,
        stage: ModelStage = ModelStage.PRODUCTION,
        *,
        allow_fallback: bool = False,
        allowed_stages: tuple[ModelStage, ...] | None = None,
    ) -> dict[str, Any] | None:
        stages = allowed_stages or (stage,)
        with self.connect() as conn:
            placeholders = ",".join("?" for _ in stages)
            row = conn.execute(
                f"SELECT * FROM model_versions WHERE task=? AND stage IN ({placeholders}) "
                "AND runtime_load_required=1 AND lifecycle_state IN ('active','candidate') "
                "ORDER BY created_at_utc DESC LIMIT 1",
                (task, *(item.value for item in stages)),
            ).fetchone()
            if row is None and allow_fallback:
                row = conn.execute(
                    "SELECT * FROM model_versions WHERE task=? AND runtime_load_required=1 "
                    "AND lifecycle_state IN ('active','candidate') ORDER BY created_at_utc DESC LIMIT 1",
                    (task,),
                ).fetchone()
        return self._decode_record(dict(row)) if row else None

    def list_models(
        self,
        *,
        runtime_only: bool = False,
        dataset_version: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if runtime_only:
            clauses.append("runtime_load_required=1")
            clauses.append("lifecycle_state IN ('active','candidate')")
        if dataset_version:
            clauses.append("dataset_version=?")
            params.append(dataset_version)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            return [
                self._decode_record(dict(row))
                for row in conn.execute(
                    f"SELECT * FROM model_versions{where} ORDER BY created_at_utc DESC",
                    params,
                )
            ]

    def list_task_audits(
        self,
        *,
        dataset_version: str | None = None,
    ) -> list[dict[str, Any]]:
        where = " WHERE dataset_version=?" if dataset_version else ""
        params: tuple[Any, ...] = (dataset_version,) if dataset_version else ()
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM task_audits{where} ORDER BY created_at_utc DESC",
                params,
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["metrics"] = json.loads(item.pop("metrics_json"))
            except (TypeError, json.JSONDecodeError):
                item["metrics"] = {}
            output.append(item)
        return output

    def create_release_snapshot(
        self,
        release_id: str,
        release_type: str,
        manifest_path: Path,
        metadata: dict[str, Any],
        *,
        locked: bool = True,
    ) -> dict[str, Any]:
        resolved = Path(manifest_path).resolve()
        stored = portable_path(resolved, self.project_root)
        digest = sha256_file(resolved)
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO release_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    release_id,
                    release_type,
                    datetime.now(timezone.utc).isoformat(),
                    stored,
                    digest,
                    json.dumps(metadata, default=str),
                    int(locked),
                ),
            )
        return {
            "release_id": release_id,
            "release_type": release_type,
            "manifest_path": stored,
            "manifest_sha256": digest,
            "locked": locked,
        }

    def release_snapshot(self, release_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM release_snapshots WHERE release_id=?", (release_id,)).fetchone()
        return dict(row) if row else None

    def create_job(self, job_type: str, detail: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, 'queued', ?, ?, ?)",
                (job_id, job_type, json.dumps(detail, default=str), now, now),
            )
        return job_id

    def update_job(self, job_id: str, status: str, detail: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, detail_json=?, updated_at_utc=? WHERE job_id=?",
                (status, json.dumps(detail, default=str), datetime.now(timezone.utc).isoformat(), job_id),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["detail"] = json.loads(result.pop("detail_json"))
        return result
