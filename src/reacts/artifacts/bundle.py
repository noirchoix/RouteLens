from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from reacts import __version__
from reacts.artifacts.errors import (
    ArtifactContractError,
    ArtifactIntegrityError,
)
from reacts.ml.environment import SCIKIT_LEARN_PIN, runtime_environment
from reacts.science.hashing import sha256_file
from reacts.settings import Settings

ARTIFACT_SCHEMA_VERSION = "2.1.0-artifact-bundle-v1"
RELEASE_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
FORBIDDEN_NAMES = {
    ".venv",
    "venv",
    "training_cache",
    "mapping_queue.sqlite3",
    "derivation_queue.sqlite3",
}
FORBIDDEN_PARTS = {"mapping_v2", "derivation_v2", "state", "__pycache__"}
RUNTIME_PACKAGE_KEYS = ("scikit_learn", "numpy", "scipy", "joblib", "rdkit")


def _python_major_minor(value: Any) -> str:
    parts = str(value or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(value or "")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactContractError(f"Expected a JSON object: {path}")
    return value


def _safe_relative(path: str | Path) -> Path:
    relative = Path(str(path).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactContractError(f"Unsafe artifact path: {path}")
    return relative


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ArtifactContractError(f"Required runtime artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _runtime_models(registry: dict[str, Any]) -> list[dict[str, Any]]:
    models = registry.get("models") or []
    output = [
        dict(model)
        for model in models
        if bool(model.get("runtime_load_required"))
        and model.get("lifecycle_state") in {"active", "candidate"}
    ]
    if not output:
        raise ArtifactContractError("The model registry contains no runtime-loadable models.")
    tasks: dict[str, str] = {}
    for model in output:
        task = str(model.get("task") or "")
        if not task:
            raise ArtifactContractError("A runtime model is missing its task identifier.")
        if task in tasks:
            raise ArtifactContractError(
                f"Duplicate runtime model for task {task}: {tasks[task]} and {model.get('model_id')}"
            )
        tasks[task] = str(model.get("model_id"))
    return output


def _index_files(index_dir: Path) -> list[Path]:
    reaction_manifest = _read_json(index_dir / "index_manifest.json")
    files = [index_dir / "index_manifest.json"]
    for shard in reaction_manifest.get("shards") or []:
        files.extend([index_dir / str(shard["vectors"]), index_dir / str(shard["metadata"])])
    route_root = index_dir / "routes"
    route_manifest = _read_json(route_root / "route_index_manifest.json")
    files.extend(
        [
            route_root / "route_index_manifest.json",
            route_root / "route_embeddings.npz",
            route_root / "route_metadata.jsonl",
        ]
    )
    # Make sure manifest references remain meaningful even if names evolve.
    for field in ("vectors", "metadata"):
        name = route_manifest.get(field)
        if name:
            files.append(route_root / str(name))
    unique: dict[str, Path] = {}
    for path in files:
        unique[path.resolve().as_posix()] = path
    return list(unique.values())




def _copy_sqlite_snapshot(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ArtifactContractError(f"Required registry database is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    # sqlite3.Connection.__exit__ commits or rolls back but does not close the
    # connection. Explicit closing is required so Windows releases the source
    # and destination handles before the staged bundle is moved or removed.
    with closing(sqlite3.connect(source_uri, uri=True)) as input_db:
        with closing(sqlite3.connect(destination)) as output_db:
            input_db.backup(output_db)
            output_db.commit()


def _rebase_registry_database(path: Path, model_paths: dict[str, tuple[str, str | None]]) -> None:
    # Keep the packaged registry as one self-contained SQLite file. A copied
    # WAL-mode database can create -wal/-shm files when reopened; checkpoint it
    # and return to DELETE mode before closing the final writable connection.
    with closing(sqlite3.connect(path)) as conn:
        for model_id, (artifact_path, card_path) in model_paths.items():
            conn.execute(
                "UPDATE model_versions SET artifact_path=?, model_card_path=? WHERE model_id=?",
                (artifact_path, card_path, model_id),
            )
        conn.commit()
        journal_mode_row = conn.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(journal_mode_row[0]).lower() if journal_mode_row else ""
        if journal_mode == "wal":
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            conn.execute("PRAGMA journal_mode=DELETE").fetchone()
        conn.commit()


def _read_runtime_registry_database(path: Path) -> list[dict[str, Any]]:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT model_id, task, artifact_path, model_card_path, split_sha256 "
            "FROM model_versions WHERE runtime_load_required=1 "
            "AND lifecycle_state IN ('active','candidate') ORDER BY task"
        ).fetchall()
        return [dict(row) for row in rows]


def _write_checksums(root: Path) -> Path:
    checksum_path = root / "SHA256SUMS"
    lines: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if not path.is_file() or path == checksum_path:
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


class ArtifactBundlePublisher:
    """Package only the immutable files needed by Product Two inference."""

    def __init__(self, settings: Settings):
        self.settings = settings.resolve()

    def package(
        self,
        *,
        release: str,
        destination: Path,
        compatible_service_version: str = ">=2.1.0,<2.2.0",
        archive: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if not RELEASE_PATTERN.fullmatch(release):
            raise ArtifactContractError(f"Invalid artifact release identifier: {release}")
        destination = Path(destination).resolve()
        final_root = destination / release
        if final_root.exists() and not overwrite:
            raise FileExistsError(final_root)
        destination.mkdir(parents=True, exist_ok=True)

        registry_json_source = self.settings.model_dir / "model_registry.json"
        registry_db_source = self.settings.registry_db
        registry = _read_json(registry_json_source)
        runtime_models = _runtime_models(registry)
        split_manifest_source = self.settings.canonical_v2_dir / "split_manifest.json"
        final_manifest_source = self.settings.canonical_v2_dir / "dataset_manifest.json"
        split_manifest = _read_json(split_manifest_source)
        final_manifest = _read_json(final_manifest_source)
        reaction_manifest_source = self.settings.index_v2_dir / "index_manifest.json"
        route_manifest_source = self.settings.index_v2_dir / "routes" / "route_index_manifest.json"
        reaction_manifest = _read_json(reaction_manifest_source)
        route_manifest = _read_json(route_manifest_source)

        with tempfile.TemporaryDirectory(prefix=f".{release}-", dir=destination) as temporary:
            stage = Path(temporary) / release
            (stage / "models").mkdir(parents=True)
            (stage / "indexes").mkdir(parents=True)
            (stage / "registry").mkdir(parents=True)
            (stage / "contracts").mkdir(parents=True)
            (stage / "environment").mkdir(parents=True)

            rebased_models: list[dict[str, Any]] = []
            database_paths: dict[str, tuple[str, str | None]] = {}
            for model in runtime_models:
                source_artifact = Path(str(model["artifact_path"]))
                if not source_artifact.is_absolute():
                    source_artifact = self.settings.project_root / source_artifact
                task = str(model["task"])
                target_artifact = Path("models") / task / source_artifact.name
                _copy_file(source_artifact, stage / target_artifact)
                if model.get("artifact_sha256") and sha256_file(stage / target_artifact) != model["artifact_sha256"]:
                    raise ArtifactIntegrityError(f"Model checksum mismatch before packaging: {model['model_id']}")

                target_card: Path | None = None
                if model.get("model_card_path"):
                    source_card = Path(str(model["model_card_path"]))
                    if not source_card.is_absolute():
                        source_card = self.settings.project_root / source_card
                    target_card = Path("models") / task / source_card.name
                    _copy_file(source_card, stage / target_card)

                rebased = dict(model)
                rebased["artifact_path"] = target_artifact.as_posix()
                rebased["model_card_path"] = target_card.as_posix() if target_card else None
                rebased_models.append(rebased)
                database_paths[str(model["model_id"])] = (
                    target_artifact.as_posix(),
                    target_card.as_posix() if target_card else None,
                )

            packaged_registry = dict(registry)
            packaged_registry["models"] = rebased_models
            packaged_registry["registry_database"] = "registry/reacts.sqlite3"
            packaged_registry["artifact_release"] = release
            packaged_registry_path = stage / "models" / "model_registry.json"
            packaged_registry_path.write_text(
                json.dumps(packaged_registry, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )

            packaged_db = stage / "registry" / "reacts.sqlite3"
            _copy_sqlite_snapshot(registry_db_source, packaged_db)
            _rebase_registry_database(packaged_db, database_paths)
            for suffix in ("-wal", "-shm"):
                sidecar = packaged_db.with_name(packaged_db.name + suffix)
                if sidecar.exists():
                    sidecar.unlink()

            for source in _index_files(self.settings.index_v2_dir):
                relative = source.relative_to(self.settings.index_v2_dir)
                _copy_file(source, stage / "indexes" / relative)

            contract_targets = {
                "dataset_manifest.json": final_manifest_source,
                "split_manifest.json": split_manifest_source,
                "reaction_index_manifest.json": reaction_manifest_source,
                "route_index_manifest.json": route_manifest_source,
            }
            for name, source in contract_targets.items():
                _copy_file(source, stage / "contracts" / name)

            environment = registry.get("runtime_environment") or runtime_environment()
            (stage / "environment" / "runtime_versions.json").write_text(
                json.dumps(environment, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )

            manifest = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_release": release,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "compatible_service_version": compatible_service_version,
                "source_service_version": __version__,
                "dataset_version": final_manifest.get("dataset_version", "uspto_multistep_contextual_v2"),
                "training_split_sha256": split_manifest.get("training_split_sha256"),
                "canonical_manifest_sha256": sha256_file(stage / "contracts" / "dataset_manifest.json"),
                "split_manifest_sha256": sha256_file(stage / "contracts" / "split_manifest.json"),
                "model_registry_sha256": sha256_file(packaged_registry_path),
                "registry_database_sha256": sha256_file(packaged_db),
                "reaction_index_sha256": sha256_file(stage / "contracts" / "reaction_index_manifest.json"),
                "route_index_sha256": sha256_file(stage / "contracts" / "route_index_manifest.json"),
                "required_scikit_learn": SCIKIT_LEARN_PIN,
                "required_runtime": {
                    "python_major_minor": _python_major_minor(environment.get("python")),
                    "python_implementation": environment.get("python_implementation"),
                    **{key: environment.get(key) for key in RUNTIME_PACKAGE_KEYS},
                },
                "required_tasks": sorted(str(model["task"]) for model in rebased_models),
                "runtime_model_count": len(rebased_models),
                "paths": {
                    "model_registry": "models/model_registry.json",
                    "registry_database": "registry/reacts.sqlite3",
                    "reaction_index": "indexes/index_manifest.json",
                    "route_index": "indexes/routes/route_index_manifest.json",
                    "dataset_manifest": "contracts/dataset_manifest.json",
                    "split_manifest": "contracts/split_manifest.json",
                    "runtime_environment": "environment/runtime_versions.json",
                },
                "contract": {
                    "immutable": True,
                    "read_only_runtime": True,
                    "mapping_queues_included": False,
                    "derivation_queues_included": False,
                    "training_caches_included": False,
                    "superseded_models_included": False,
                    "canonical_source_rows_included": False,
                },
                "source_hashes": {
                    "source_registry_json_sha256": sha256_file(registry_json_source),
                    "source_registry_database_sha256": sha256_file(registry_db_source),
                    "source_final_manifest_sha256": sha256_file(final_manifest_source),
                    "source_split_manifest_sha256": sha256_file(split_manifest_source),
                    "source_reaction_index_manifest_sha256": sha256_file(reaction_manifest_source),
                    "source_route_index_manifest_sha256": sha256_file(route_manifest_source),
                },
                "index_contract": {
                    "reaction_training_split_sha256": reaction_manifest.get("training_split_sha256"),
                    "route_training_split_sha256": route_manifest.get("training_split_sha256"),
                },
            }
            (stage / "artifact_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            _write_checksums(stage)
            validation = ArtifactBundleValidator(stage).validate(service_version="2.1.0")
            if not validation["pass"]:
                raise ArtifactContractError(f"Packaged artifact bundle failed validation: {validation['failures']}")

            if final_root.exists():
                shutil.rmtree(final_root)
            os.replace(stage, final_root)

        archive_path: Path | None = None
        archive_checksum_path: Path | None = None
        archive_sha256: str | None = None
        if archive:
            archive_path = Path(
                shutil.make_archive(str(destination / release), "zip", root_dir=destination, base_dir=release)
            )
            archive_sha256 = sha256_file(archive_path)
            archive_checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
            archive_checksum_path.write_text(
                f"{archive_sha256}  {archive_path.name}\n",
                encoding="utf-8",
            )
        return {
            "artifact_release": release,
            "bundle_path": final_root.as_posix(),
            "archive_path": archive_path.as_posix() if archive_path else None,
            "archive_sha256": archive_sha256,
            "archive_checksum_path": archive_checksum_path.as_posix() if archive_checksum_path else None,
            "artifact_manifest_sha256": sha256_file(final_root / "artifact_manifest.json"),
            "sha256sums_sha256": sha256_file(final_root / "SHA256SUMS"),
            "runtime_models": len(runtime_models),
            "validation": ArtifactBundleValidator(final_root).validate(service_version="2.1.0"),
        }


class ArtifactBundleValidator:
    """Fail-closed validator for an installed or publishable artifact bundle."""

    def __init__(self, bundle_root: Path):
        self.root = Path(bundle_root).resolve()

    def _checksums(self) -> tuple[dict[str, str], list[str]]:
        checksum_path = self.root / "SHA256SUMS"
        if not checksum_path.is_file():
            return {}, ["SHA256SUMS is missing."]
        expected: dict[str, str] = {}
        failures: list[str] = []
        for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                failures.append(f"Malformed SHA256SUMS line {line_number}.")
                continue
            digest, raw_path = parts
            relative = raw_path.lstrip("* ")
            try:
                safe = _safe_relative(relative)
            except ArtifactContractError as exc:
                failures.append(str(exc))
                continue
            expected[safe.as_posix()] = digest
            target = (self.root / safe).resolve()
            if not target.is_relative_to(self.root):
                failures.append(f"Checksum path escapes bundle root: {relative}")
            elif not target.is_file():
                failures.append(f"Checksummed file is missing: {relative}")
            elif sha256_file(target) != digest:
                failures.append(f"Checksum mismatch: {relative}")
        actual = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        missing_from_sums = sorted(actual - set(expected))
        if missing_from_sums:
            failures.append(f"Files missing from SHA256SUMS: {missing_from_sums}")
        return expected, failures

    def validate(self, *, service_version: str | None = None) -> dict[str, Any]:
        failures: list[str] = []
        manifest_path = self.root / "artifact_manifest.json"
        if not manifest_path.is_file():
            return {"pass": False, "failures": ["artifact_manifest.json is missing."], "bundle_root": str(self.root)}
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            failures.append(f"Unsupported artifact schema: {manifest.get('schema_version')}")
        if not RELEASE_PATTERN.fullmatch(str(manifest.get("artifact_release") or "")):
            failures.append("artifact_release is missing or invalid.")

        _, checksum_failures = self._checksums()
        failures.extend(checksum_failures)

        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if any(part in FORBIDDEN_PARTS for part in relative.parts) or path.name in FORBIDDEN_NAMES:
                failures.append(f"Forbidden runtime content is present: {relative.as_posix()}")

        paths = manifest.get("paths") or {}
        resolved: dict[str, Path] = {}
        for key in (
            "model_registry",
            "registry_database",
            "reaction_index",
            "route_index",
            "dataset_manifest",
            "split_manifest",
            "runtime_environment",
        ):
            try:
                relative = _safe_relative(paths[key])
            except (KeyError, ArtifactContractError) as exc:
                failures.append(f"Required manifest path is invalid: {key}: {exc}")
                continue
            resolved[key] = self.root / relative
            if not resolved[key].is_file():
                failures.append(f"Required artifact is missing: {relative.as_posix()}")

        if service_version and manifest.get("compatible_service_version"):
            try:
                specifier = SpecifierSet(str(manifest["compatible_service_version"]))
                if Version(service_version) not in specifier:
                    failures.append(
                        f"Service {service_version} is outside {manifest['compatible_service_version']}."
                    )
            except (InvalidSpecifier, InvalidVersion) as exc:
                failures.append(f"Invalid service compatibility contract: {exc}")

        runtime = _read_json(resolved["runtime_environment"]) if resolved.get("runtime_environment", Path()).is_file() else {}
        expected_sklearn = str(manifest.get("required_scikit_learn") or "")
        required_runtime = dict(manifest.get("required_runtime") or {})
        if expected_sklearn != SCIKIT_LEARN_PIN:
            failures.append(f"Bundle requires unsupported scikit-learn {expected_sklearn}.")
        if required_runtime.get("scikit_learn") != expected_sklearn:
            failures.append("Artifact manifest runtime and scikit-learn contracts disagree.")
        if _python_major_minor(runtime.get("python")) != str(required_runtime.get("python_major_minor") or ""):
            failures.append("Runtime environment Python major/minor does not match the artifact contract.")
        if runtime.get("python_implementation") != required_runtime.get("python_implementation"):
            failures.append("Runtime Python implementation does not match the artifact contract.")
        for key in RUNTIME_PACKAGE_KEYS:
            if runtime.get(key) != required_runtime.get(key):
                failures.append(f"Runtime environment and artifact {'scikit-learn' if key == 'scikit_learn' else key} contracts disagree.")

        host_runtime = runtime_environment()
        if service_version:
            if _python_major_minor(host_runtime.get("python")) != str(required_runtime.get("python_major_minor") or ""):
                failures.append(
                    f"Host Python {host_runtime.get('python')} does not match required major/minor "
                    f"{required_runtime.get('python_major_minor')}."
                )
            if host_runtime.get("python_implementation") != required_runtime.get("python_implementation"):
                failures.append("Host Python implementation does not match the artifact contract.")
            for key in RUNTIME_PACKAGE_KEYS:
                if host_runtime.get(key) != required_runtime.get(key):
                    failures.append(
                        f"Host runtime {'scikit-learn' if key == 'scikit_learn' else key} {host_runtime.get(key)} does not match required {required_runtime.get(key)}."
                    )

        registry = _read_json(resolved["model_registry"]) if resolved.get("model_registry", Path()).is_file() else {}
        models: list[dict[str, Any]] = []
        try:
            models = _runtime_models(registry)
        except ArtifactContractError as exc:
            failures.append(str(exc))
        required_tasks = sorted(str(value) for value in manifest.get("required_tasks") or [])
        actual_tasks = sorted(str(model.get("task")) for model in models)
        if required_tasks != actual_tasks:
            failures.append(f"Required task set mismatch: manifest={required_tasks}, registry={actual_tasks}")

        database_models: list[dict[str, Any]] = []
        database_path = resolved.get("registry_database")
        if database_path and database_path.is_file():
            try:
                database_models = _read_runtime_registry_database(database_path)
            except (sqlite3.Error, OSError) as exc:
                failures.append(f"Registry database cannot be read in query-only mode: {exc}")
        json_by_id = {str(model.get("model_id")): model for model in models}
        database_by_id = {str(model.get("model_id")): model for model in database_models}
        if set(json_by_id) != set(database_by_id):
            failures.append("Registry database and JSON runtime model IDs disagree.")
        for model_id in sorted(set(json_by_id) & set(database_by_id)):
            json_model = json_by_id[model_id]
            database_model = database_by_id[model_id]
            for field in ("task", "artifact_path", "model_card_path", "split_sha256"):
                if (json_model.get(field) or None) != (database_model.get(field) or None):
                    failures.append(f"Registry database/JSON mismatch for {model_id}: {field}")

        split = _read_json(resolved["split_manifest"]) if resolved.get("split_manifest", Path()).is_file() else {}
        reaction = _read_json(resolved["reaction_index"]) if resolved.get("reaction_index", Path()).is_file() else {}
        route = _read_json(resolved["route_index"]) if resolved.get("route_index", Path()).is_file() else {}
        split_hash = str(manifest.get("training_split_sha256") or "")
        if not split_hash or split.get("training_split_sha256") != split_hash:
            failures.append("Artifact and split-manifest training split hashes disagree.")
        if reaction.get("training_split_sha256") != split_hash:
            failures.append("Reaction index is not bound to the artifact training split.")
        if route.get("training_split_sha256") != split_hash:
            failures.append("Route index is not bound to the artifact training split.")

        for model in models:
            if model.get("split_sha256") != split_hash:
                failures.append(f"Model split mismatch: {model.get('model_id')}")
            try:
                artifact_path = self.root / _safe_relative(model["artifact_path"])
            except (KeyError, ArtifactContractError) as exc:
                failures.append(f"Invalid model artifact path for {model.get('model_id')}: {exc}")
                continue
            if not artifact_path.is_file():
                failures.append(f"Model artifact is missing: {model.get('model_id')}")
            elif model.get("artifact_sha256") and sha256_file(artifact_path) != model["artifact_sha256"]:
                failures.append(f"Model artifact hash mismatch: {model.get('model_id')}")
            card = model.get("model_card_path")
            if card and not (self.root / _safe_relative(card)).is_file():
                failures.append(f"Model card is missing: {model.get('model_id')}")
            training_environment = model.get("training_environment") or {}
            if _python_major_minor(training_environment.get("python")) != str(required_runtime.get("python_major_minor") or ""):
                failures.append(f"Model Python environment mismatch: {model.get('model_id')}")
            for key in RUNTIME_PACKAGE_KEYS:
                if training_environment.get(key) != required_runtime.get(key):
                    failures.append(f"Model {'scikit-learn' if key == 'scikit_learn' else key} environment mismatch: {model.get('model_id')}")

        high_level_hashes = {
            "model_registry_sha256": resolved.get("model_registry"),
            "registry_database_sha256": resolved.get("registry_database"),
            "canonical_manifest_sha256": resolved.get("dataset_manifest"),
            "split_manifest_sha256": resolved.get("split_manifest"),
            "reaction_index_sha256": self.root / "contracts" / "reaction_index_manifest.json",
            "route_index_sha256": self.root / "contracts" / "route_index_manifest.json",
        }
        for field, path in high_level_hashes.items():
            if path and path.is_file() and manifest.get(field) != sha256_file(path):
                failures.append(f"High-level manifest hash mismatch: {field}")

        return {
            "pass": not failures,
            "failures": failures,
            "bundle_root": self.root.as_posix(),
            "artifact_release": manifest.get("artifact_release"),
            "schema_version": manifest.get("schema_version"),
            "service_version": service_version,
            "runtime_models": len(models),
            "required_tasks": required_tasks,
            "training_split_sha256": split_hash or None,
            "manifest": manifest,
            "host_runtime": runtime_environment(),
        }


def extract_zip_safely(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ArtifactContractError(f"Archive member escapes destination: {member.filename}")
        handle.extractall(destination)
