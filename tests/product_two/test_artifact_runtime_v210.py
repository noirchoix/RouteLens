from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from reacts.api.main import create_app
from reacts.artifacts.bundle import ArtifactBundlePublisher, ArtifactBundleValidator
from reacts.artifacts.resolver import ArtifactResolver
from reacts.chemistry.reactions import reaction_fingerprint
from reacts.contracts import ModelStage
from reacts.ml.environment import SCIKIT_LEARN_PIN, runtime_environment
from reacts.ml.registry import Registry
from reacts.science.hashing import sha256_file
from reacts.settings import Settings

REACTION = "CCO>>CC=O"
GOLDEN = json.loads((Path(__file__).parents[1] / "fixtures" / "product_two_v210_golden.json").read_text(encoding="utf-8"))
SPLIT_SHA = GOLDEN["training_split_sha256"]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _source_runtime(root: Path) -> Settings:
    settings = Settings(project_root=root).resolve()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.index_v2_dir.mkdir(parents=True, exist_ok=True)
    settings.canonical_v2_dir.mkdir(parents=True, exist_ok=True)

    artifact = settings.model_dir / "reaction_family" / "model.joblib"
    artifact.parent.mkdir(parents=True)
    vectorizer = HashingVectorizer(n_features=64, alternate_sign=False)
    training = [REACTION, "CCBr>>CCO", "CC(=O)O>>CC(=O)Cl", "CCN>>CC=N"]
    labels = ["oxidation", "substitution", "activation", "oxidation"]
    model = LogisticRegression(max_iter=200, random_state=42).fit(vectorizer.transform(training), labels)
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "input_column": "canonical_reaction_smiles",
            "task_type": "classification",
            "calibrator": None,
            "metrics": {"validation": {}},
        },
        artifact,
    )
    _write_json(artifact.with_suffix(".model_card.json"), {"task": "reaction_family"})

    registry = Registry(settings.registry_db)
    registry.register_model(
        task="reaction_family",
        artifact_path=artifact,
        dataset_version="uspto_multistep_contextual_v2",
        metrics={"validation": {"rows": 2}},
        config={"test_fixture": True},
        stage=ModelStage.CANDIDATE,
        dataset_sha256="b" * 64,
        feature_sha256="c" * 64,
        split_sha256=SPLIT_SHA,
        release_decision={
            "approved": False,
            "permitted_use": "Top-k retrieval-backed suggestions with explicit provenance.",
        },
        training_environment={**runtime_environment(), "scikit_learn": SCIKIT_LEARN_PIN},
    )
    registry_json_path = settings.model_dir / "model_registry.json"
    registry_json = json.loads(registry_json_path.read_text(encoding="utf-8"))
    registry_json["runtime_environment"] = {**runtime_environment(), "scikit_learn": SCIKIT_LEARN_PIN}
    registry_json_path.write_text(json.dumps(registry_json, indent=2) + "\n", encoding="utf-8")

    rfp, pfp = reaction_fingerprint(REACTION, 2048)
    vectors = settings.index_v2_dir / "shard-00000.npz"
    np.savez_compressed(
        vectors,
        reactants=np.packbits(np.vstack([rfp]), axis=1),
        products=np.packbits(np.vstack([pfp]), axis=1),
        centres=np.zeros((1, 256), dtype=np.uint8),
    )
    metadata = settings.index_v2_dir / "shard-00000.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "step_id": "step-1",
                "route_id": "route-1",
                "route_instance_id": "route-1",
                "patent_document_id": "patent-1",
                "reaction_smiles": REACTION,
                "quality_score": 1.0,
                "solvent_primary": "water",
                "solvents": ["water"],
                "agents": [],
                "time_bucket": "short",
                "temperature_bucket": "ambient",
                "reaction_family": "oxidation",
                "reaction_centre_fingerprint": None,
                "resolution_status": "not_required",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reaction_manifest = {
        "index_version": "reaction_contextual_morgan_v2-test",
        "dataset_version": "uspto_multistep_contextual_v2",
        "n_bits": 2048,
        "rows": 1,
        "weights": {"reactant": 0.4, "product": 0.45, "centre": 0.15},
        "training_split_sha256": SPLIT_SHA,
        "shards": [
            {
                "shard": 0,
                "rows": 1,
                "vectors": vectors.name,
                "metadata": metadata.name,
                "vectors_sha256": sha256_file(vectors),
                "metadata_sha256": sha256_file(metadata),
            }
        ],
    }
    _write_json(settings.index_v2_dir / "index_manifest.json", reaction_manifest)

    route_root = settings.index_v2_dir / "routes"
    route_root.mkdir()
    route_vector = np.concatenate([rfp, pfp]).astype(np.float32)
    route_vector /= max(float(np.linalg.norm(route_vector)), 1.0)
    np.savez_compressed(route_root / "route_embeddings.npz", vectors=np.vstack([route_vector]))
    (route_root / "route_metadata.jsonl").write_text(
        json.dumps(
            {
                "route_id": "route-1",
                "route_instance_id": "route-1",
                "source_route_id": "route-1",
                "patent_document_id": "patent-1",
                "split": "test",
                "split_component_id": "component-1",
                "step_count": 1,
                "reaction_families": ["oxidation"],
                "quality_score": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        route_root / "route_index_manifest.json",
        {
            "index_version": "route_aggregate_fingerprint_v2-test",
            "rows": 1,
            "dimensions": 4096,
            "training_split_sha256": SPLIT_SHA,
            "vectors_sha256": sha256_file(route_root / "route_embeddings.npz"),
            "metadata_sha256": sha256_file(route_root / "route_metadata.jsonl"),
        },
    )

    _write_json(
        settings.canonical_v2_dir / "dataset_manifest.json",
        {
            "dataset_version": "uspto_multistep_contextual_v2",
            "manifest_sha256": "d" * 64,
        },
    )
    _write_json(
        settings.canonical_v2_dir / "split_manifest.json",
        {
            "schema_version": "2.0.9-split-governance-v1",
            "training_split_sha256": SPLIT_SHA,
            "invariants": {
                "patent_document_id_overlapping_keys": 0,
                "reaction_signature_overlapping_keys": 0,
                "route_split_conflicts": 0,
                "strict_pass": True,
            },
        },
    )
    return settings


def _package(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    declared = {**runtime_environment(), "scikit_learn": SCIKIT_LEARN_PIN}
    monkeypatch.setattr("reacts.artifacts.bundle.runtime_environment", lambda: declared)
    source = _source_runtime(tmp_path / "source")
    release = GOLDEN["artifact_release"]
    result = ArtifactBundlePublisher(source).package(
        release=release,
        destination=tmp_path / "dist",
        archive=True,
    )
    return Path(result["bundle_path"]), release


def test_publisher_scopes_runtime_registry_to_canonical_dataset(tmp_path: Path, monkeypatch) -> None:
    declared = {**runtime_environment(), "scikit_learn": SCIKIT_LEARN_PIN}
    monkeypatch.setattr("reacts.artifacts.bundle.runtime_environment", lambda: declared)
    source = _source_runtime(tmp_path / "source-mixed-registry")

    current_artifact = source.model_dir / "reaction_family" / "model.joblib"
    legacy_artifact = source.model_dir / "agent_presence" / "legacy.joblib"
    legacy_artifact.parent.mkdir(parents=True)
    shutil.copy2(current_artifact, legacy_artifact)
    _write_json(legacy_artifact.with_suffix(".model_card.json"), {"task": "agent_presence"})

    registry = Registry(source.registry_db)
    legacy = registry.register_model(
        task="agent_presence",
        artifact_path=legacy_artifact,
        dataset_version="uspto_multistep_canonical_v1",
        metrics={"validation": {"rows": 2}},
        config={"legacy_fixture": True},
        stage=ModelStage.SCREENING,
        split_sha256=None,
        training_environment={
            "python": "3.11.0",
            "python_implementation": "CPython",
            "platform": "legacy",
            "scikit_learn": "1.4.0",
            "numpy": "1.26.0",
            "scipy": "1.11.0",
            "joblib": "1.3.0",
            "rdkit": "2023.9.1",
        },
    )
    registry_json = json.loads((source.model_dir / "model_registry.json").read_text(encoding="utf-8"))
    registry_json["runtime_environment"] = declared
    (source.model_dir / "model_registry.json").write_text(
        json.dumps(registry_json, indent=2) + "\n",
        encoding="utf-8",
    )

    source_registry_hash = sha256_file(source.registry_db)
    source_json_hash = sha256_file(source.model_dir / "model_registry.json")
    result = ArtifactBundlePublisher(source).package(
        release="product-two-artifacts-mixed-registry-test",
        destination=tmp_path / "dist",
        archive=False,
    )

    bundle = Path(result["bundle_path"])
    validation = ArtifactBundleValidator(bundle).validate(service_version="2.1.0")
    assert validation["pass"] is True
    assert validation["required_tasks"] == ["reaction_family"]
    selection = validation["manifest"]["model_selection"]
    assert selection == {
        "dataset_version": "uspto_multistep_contextual_v2",
        "runtime_load_required": True,
        "lifecycle_states": ["active", "candidate"],
        "source_runtime_model_count": 2,
        "selected_runtime_model_count": 1,
    }

    packaged_registry = json.loads((bundle / "models" / "model_registry.json").read_text(encoding="utf-8"))
    assert [model["task"] for model in packaged_registry["models"]] == ["reaction_family"]
    assert legacy["model_id"] not in {model["model_id"] for model in packaged_registry["models"]}
    assert not (bundle / "models" / "agent_presence").exists()

    with sqlite3.connect(bundle / "registry" / "reacts.sqlite3") as connection:
        runtime_rows = connection.execute(
            "SELECT model_id, task, dataset_version FROM model_versions "
            "WHERE runtime_load_required=1 ORDER BY task"
        ).fetchall()
        all_rows = connection.execute(
            "SELECT model_id, task, dataset_version FROM model_versions ORDER BY task"
        ).fetchall()
    assert runtime_rows == all_rows
    assert len(all_rows) == 1
    assert all_rows[0][1:] == ("reaction_family", "uspto_multistep_contextual_v2")

    assert sha256_file(source.registry_db) == source_registry_hash
    assert sha256_file(source.model_dir / "model_registry.json") == source_json_hash


def test_clean_room_bundle_resolution_readiness_and_golden_contract(tmp_path: Path, monkeypatch) -> None:
    bundle, release = _package(tmp_path, monkeypatch)
    monkeypatch.setattr("reacts.ml.inference.validate_runtime_environment", lambda _: {"pass": True})
    validation = ArtifactBundleValidator(bundle).validate(service_version="2.1.0")
    assert validation["pass"] is True

    clean = tmp_path / "clean-service"
    settings = Settings(
        project_root=clean,
        artifact_uri=str(bundle.parent),
        artifact_release=release,
        artifact_cache_dir=clean / "cache",
        artifact_required=True,
        artifact_verify_sha256=True,
        artifact_warmup=True,
        rate_limit_requests_per_minute=0,
    ).resolve()
    client = TestClient(create_app(settings))
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True

    installed = settings.artifact_cache_dir / release
    before_db = sha256_file(installed / "registry" / "reacts.sqlite3")
    before_json = sha256_file(installed / "models" / "model_registry.json")
    response = client.post(
        "/api/v2/inference/contextual",
        json={
            "reaction_smiles": REACTION,
            "tasks": ["reaction_family"],
            "include_evidence": True,
            "evidence_k": 1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    task = payload["tasks"][0]
    assert task["model_id"].startswith("reaction_family:")
    assert task["model_stage"] == GOLDEN["model_stage"]
    assert task["lifecycle_state"] == GOLDEN["lifecycle_state"]
    assert task["artifact_release"] == release
    assert task["training_split_sha256"] == SPLIT_SHA
    assert GOLDEN["permitted_use_contains"] in task["permitted_use"]
    assert payload["provenance"]["artifact_release"] == release
    assert sha256_file(installed / "registry" / "reacts.sqlite3") == before_db
    assert sha256_file(installed / "models" / "model_registry.json") == before_json

    reaction_results = client.post(
        "/api/v2/retrieval/reactions", json={"reaction_smiles": REACTION, "k": 1}
    )
    route_results = client.post(
        "/api/v2/retrieval/routes", json={"reaction_smiles": REACTION, "k": 1}
    )
    assert reaction_results.status_code == 200
    assert route_results.status_code == 200
    assert reaction_results.json()["results"][0]["step_id"] == "step-1"
    assert route_results.json()["results"][0]["route_id"] == "route-1"
    route_summary = client.get("/api/v2/routes/route-1")
    assert route_summary.status_code == 200
    assert route_summary.json()["artifact_backed_summary"] is True
    assert client.post("/api/v2/jobs/freeze-baseline").status_code == 409


def test_corrupted_bundle_is_rejected(tmp_path: Path, monkeypatch) -> None:
    bundle, _ = _package(tmp_path, monkeypatch)
    artifact = next((bundle / "models").rglob("*.joblib"))
    artifact.write_bytes(artifact.read_bytes() + b"corruption")
    result = ArtifactBundleValidator(bundle).validate(service_version="2.1.0")
    assert result["pass"] is False
    assert any("Checksum mismatch" in failure or "hash mismatch" in failure for failure in result["failures"])


def test_offline_mode_requires_exact_cached_release(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "older-release").mkdir(parents=True)
    resolver = ArtifactResolver(
        uri=None,
        release="required-release",
        cache_dir=cache,
        offline_mode=True,
        service_version="2.1.0",
    )
    try:
        resolver.resolve()
    except Exception as exc:
        assert getattr(exc, "reason_code", None) == "artifact_not_cached_offline"
    else:  # pragma: no cover
        raise AssertionError("The resolver silently used a stale release.")


def test_failed_artifact_startup_keeps_health_but_blocks_inference(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        artifact_release="missing-release",
        artifact_cache_dir=tmp_path / "cache",
        artifact_required=True,
        offline_mode=True,
        rate_limit_requests_per_minute=0,
    ).resolve()
    client = TestClient(create_app(settings))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    response = client.post(
        "/api/v2/inference/contextual",
        json={"reaction_smiles": REACTION, "tasks": [], "include_evidence": False},
    )
    assert response.status_code == 503


def test_restart_uses_verified_exact_cache(tmp_path: Path, monkeypatch) -> None:
    bundle, release = _package(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    first = ArtifactResolver(
        uri=str(bundle.parent), release=release, cache_dir=cache, service_version="2.1.0"
    ).resolve()
    second = ArtifactResolver(
        uri="/unavailable/source", release=release, cache_dir=cache, offline_mode=True, service_version="2.1.0"
    ).resolve()
    assert first[0] == second[0]
    assert first[2] is False
    assert second[2] is True


def _refresh_checksums(bundle: Path) -> None:
    from reacts.artifacts.bundle import _write_checksums

    _write_checksums(bundle)


def test_missing_required_file_is_rejected(tmp_path: Path, monkeypatch) -> None:
    bundle, _ = _package(tmp_path, monkeypatch)
    route_root = bundle / "indexes" / "routes"
    route_manifest = json.loads((route_root / "route_index_manifest.json").read_text(encoding="utf-8"))
    (route_root / route_manifest["vectors"]).unlink()
    result = ArtifactBundleValidator(bundle).validate(service_version="2.1.0")
    assert result["pass"] is False
    assert any("missing" in failure.lower() for failure in result["failures"])


def test_wrong_index_split_is_rejected(tmp_path: Path, monkeypatch) -> None:
    bundle, _ = _package(tmp_path, monkeypatch)
    manifest_path = bundle / "indexes" / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_split_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _refresh_checksums(bundle)
    result = ArtifactBundleValidator(bundle).validate(service_version="2.1.0")
    assert result["pass"] is False
    assert "Reaction index is not bound" in " ".join(result["failures"])


def test_wrong_scikit_learn_contract_is_rejected(tmp_path: Path, monkeypatch) -> None:
    bundle, _ = _package(tmp_path, monkeypatch)
    environment_path = bundle / "environment" / "runtime_versions.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["scikit_learn"] = "0.0.0"
    environment_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    _refresh_checksums(bundle)
    result = ArtifactBundleValidator(bundle).validate(service_version="2.1.0")
    assert result["pass"] is False
    assert "scikit-learn contracts disagree" in " ".join(result["failures"])


def test_duplicate_active_task_is_rejected(tmp_path: Path, monkeypatch) -> None:
    bundle, _ = _package(tmp_path, monkeypatch)
    registry_path = bundle / "models" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    duplicate = dict(registry["models"][0])
    duplicate["model_id"] = duplicate["model_id"] + "-duplicate"
    registry["models"].append(duplicate)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    artifact_manifest_path = bundle / "artifact_manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    artifact_manifest["model_registry_sha256"] = sha256_file(registry_path)
    artifact_manifest_path.write_text(json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8")
    _refresh_checksums(bundle)
    result = ArtifactBundleValidator(bundle).validate(service_version="2.1.0")
    assert result["pass"] is False
    assert "Duplicate runtime model" in " ".join(result["failures"])


def test_required_endpoint_surface_matches_golden(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, rate_limit_requests_per_minute=0).resolve()
    client = TestClient(create_app(settings))
    paths = set(client.get('/openapi.json').json()['paths'])
    assert set(GOLDEN['required_endpoints']).issubset(paths)


def test_batch_and_single_structural_equivalence(tmp_path: Path, monkeypatch) -> None:
    bundle, release = _package(tmp_path, monkeypatch)
    monkeypatch.setattr('reacts.ml.inference.validate_runtime_environment', lambda _: {'pass': True})
    settings = Settings(
        project_root=tmp_path / 'service',
        artifact_uri=str(bundle.parent),
        artifact_release=release,
        artifact_cache_dir=tmp_path / 'cache',
        artifact_required=True,
        artifact_warmup=True,
        rate_limit_requests_per_minute=0,
    ).resolve()
    client = TestClient(create_app(settings))
    request = {
        'reaction_smiles': REACTION,
        'tasks': ['reaction_family'],
        'include_evidence': True,
        'evidence_k': 1,
    }
    single = client.post('/api/v2/inference/contextual', json=request)
    batch = client.post(
        '/api/v2/inference/batch',
        json={
            'reactions': [REACTION],
            'tasks': ['reaction_family'],
            'include_evidence': True,
            'evidence_k': 1,
        },
    )
    assert single.status_code == 200
    assert batch.status_code == 200
    single_task = single.json()['tasks'][0]
    batch_task = batch.json()['results'][0]['tasks'][0]
    for field in (
        'task',
        'model_id',
        'model_stage',
        'lifecycle_state',
        'permitted_use',
        'artifact_release',
        'training_split_sha256',
    ):
        assert batch_task[field] == single_task[field]


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    import zipfile

    from reacts.artifacts.bundle import extract_zip_safely
    from reacts.artifacts.errors import ArtifactContractError

    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('../escape.txt', 'not allowed')
    try:
        extract_zip_safely(archive, tmp_path / 'extract')
    except ArtifactContractError as exc:
        assert 'escapes destination' in str(exc)
    else:  # pragma: no cover
        raise AssertionError('Unsafe ZIP member was extracted.')


def test_checksum_verification_cannot_be_disabled_for_artifact_startup(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        artifact_release='release',
        artifact_required=True,
        artifact_verify_sha256=False,
        rate_limit_requests_per_minute=0,
    ).resolve()
    client = TestClient(create_app(settings))
    ready = client.get('/ready')
    assert ready.status_code == 503
    assert ready.json()['reason_code'] == 'artifact_verification_disabled'


def test_unavailable_remote_does_not_fall_back_to_other_release(tmp_path: Path, monkeypatch) -> None:
    from reacts.artifacts.errors import ArtifactUnavailableError

    cache = tmp_path / 'cache'
    (cache / 'older-release').mkdir(parents=True)

    def fail(*args, **kwargs):
        raise OSError('network unavailable')

    monkeypatch.setattr('urllib.request.urlopen', fail)
    resolver = ArtifactResolver(
        uri='https://example.invalid/{release}.zip',
        release='required-release',
        cache_dir=cache,
        service_version='2.1.0',
    )
    try:
        resolver.resolve()
    except ArtifactUnavailableError:
        pass
    else:  # pragma: no cover
        raise AssertionError('Resolver silently used a different cached release.')
    assert not (cache / 'required-release').exists()


def test_artifact_publisher_and_validator_cli_commands(tmp_path: Path, monkeypatch, capsys) -> None:
    from reacts.cli import main

    declared = {**runtime_environment(), 'scikit_learn': SCIKIT_LEARN_PIN}
    monkeypatch.setattr('reacts.artifacts.bundle.runtime_environment', lambda: declared)
    source = _source_runtime(tmp_path / 'source-cli')
    destination = tmp_path / 'cli-dist'
    release = 'product-two-artifacts-cli-test'
    assert main([
        '--project-root', str(source.project_root),
        'package-product-two-artifacts',
        '--release', release,
        '--destination', str(destination),
    ]) == 0
    capsys.readouterr()
    assert (destination / release / 'artifact_manifest.json').is_file()
    assert (destination / f'{release}.zip').is_file()
    assert (destination / f'{release}.zip.sha256').is_file()
    assert main([
        '--project-root', str(source.project_root),
        'validate-artifact-bundle',
        '--bundle', str(destination / release),
        '--service-version', '2.1.0',
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['pass'] is True


def test_interrupted_download_leaves_no_installed_release(tmp_path: Path, monkeypatch) -> None:
    import io

    cache = tmp_path / 'cache'

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: Response(b'partial'))

    def interrupted(*args, **kwargs):
        raise OSError('download interrupted')

    monkeypatch.setattr('shutil.copyfileobj', interrupted)
    resolver = ArtifactResolver(
        uri='https://example.invalid/{release}.zip',
        release='interrupted-release',
        cache_dir=cache,
        service_version='2.1.0',
    )
    try:
        resolver.resolve()
    except Exception as exc:
        assert 'download' in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError('Interrupted download unexpectedly installed a release.')
    assert not (cache / 'interrupted-release').exists()
    assert not list(cache.glob('.interrupted-release.installing-*'))


def test_incompatible_service_version_is_rejected(tmp_path: Path, monkeypatch) -> None:
    bundle, _ = _package(tmp_path, monkeypatch)
    result = ArtifactBundleValidator(bundle).validate(service_version='2.2.0')
    assert result['pass'] is False
    assert 'outside' in ' '.join(result['failures'])


def test_request_body_limit_returns_structured_413(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        max_request_bytes=64,
        rate_limit_requests_per_minute=0,
    ).resolve()
    client = TestClient(create_app(settings))
    response = client.post(
        '/api/v2/inference/contextual',
        content=b'{' + b'"padding":"' + (b'x' * 256) + b'"}',
        headers={'Content-Type': 'application/json', 'X-Request-ID': 'body-limit-test'},
    )
    assert response.status_code == 413
    assert response.json()['error']['code'] == 'request_too_large'
    assert response.headers['X-Request-ID'] == 'body-limit-test'


def test_serve_cli_exports_exact_artifact_contract_before_asgi_import(tmp_path: Path, monkeypatch) -> None:
    import os

    from reacts import cli

    invoked = {}
    keys = [
        'REACTS_PROJECT_ROOT',
        'REACTS_ARTIFACT_URI',
        'REACTS_ARTIFACT_RELEASE',
        'REACTS_ARTIFACT_CACHE_DIR',
        'REACTS_ARTIFACT_REQUIRED',
        'REACTS_ARTIFACT_VERIFY_SHA256',
        'REACTS_ARTIFACT_WARMUP',
        'REACTS_OFFLINE_MODE',
        'REACTS_API_READ_ONLY_REGISTRY',
    ]
    before = {key: os.environ.get(key) for key in keys}

    def fake_run(app, **kwargs):
        invoked['app'] = app
        invoked.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, 'run', fake_run)
    try:
        assert cli.main([
            '--project-root', str(tmp_path),
            'serve',
            '--artifact-uri', str(tmp_path / 'dist'),
            '--artifact-release', 'product-two-artifacts-v2.0.12',
            '--artifact-cache-dir', str(tmp_path / 'cache'),
            '--require-artifacts',
            '--offline',
            '--port', '8765',
        ]) == 0
        assert invoked['app'] == 'reacts.api.main:app'
        assert invoked['port'] == 8765
        assert invoked['reload'] is False
        assert os.environ['REACTS_ARTIFACT_RELEASE'] == 'product-two-artifacts-v2.0.12'
        assert os.environ['REACTS_ARTIFACT_REQUIRED'] == 'true'
        assert os.environ['REACTS_API_READ_ONLY_REGISTRY'] == 'true'
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
