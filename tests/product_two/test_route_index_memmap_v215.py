from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from reacts.api.main import create_app
from reacts.retrieval.route_index import RouteEmbeddingIndex
from reacts.settings import Settings
from test_artifact_runtime_v210 import REACTION, _package


def test_published_route_index_is_memory_mapped_and_searchable(tmp_path: Path, monkeypatch) -> None:
    bundle, _ = _package(tmp_path, monkeypatch)
    route_root = bundle / "indexes" / "routes"
    manifest = json.loads((route_root / "route_index_manifest.json").read_text(encoding="utf-8"))

    assert manifest["vectors_format"] == "npy_memmap_v1"
    assert manifest["vectors"] == "route_embeddings.npy"
    assert (route_root / manifest["vectors"]).is_file()
    assert not (route_root / "route_embeddings.npz").exists()

    index = RouteEmbeddingIndex(bundle / "indexes")
    try:
        info = index.storage_info(sample=True)
        assert info["memory_mapped"] is True
        assert info["rows"] == 1
        assert info["dimensions"] == 4096
        assert isinstance(index._ensure_vectors(), np.memmap)
        results = index.search_reaction(REACTION, k=1)
        assert results[0]["route_id"] == "route-1"
    finally:
        index.close()


def test_artifact_warmup_reports_memory_mapped_route_storage(tmp_path: Path, monkeypatch) -> None:
    bundle, release = _package(tmp_path, monkeypatch)
    monkeypatch.setattr("reacts.ml.inference.validate_runtime_environment", lambda _: {"pass": True})
    settings = Settings(
        project_root=tmp_path / "service",
        artifact_uri=str(bundle.parent),
        artifact_release=release,
        artifact_cache_dir=tmp_path / "cache",
        artifact_required=True,
        artifact_warmup=True,
        rate_limit_requests_per_minute=0,
    ).resolve()

    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        route_storage = ready.json()["warmup"]["route_index_storage"]
        assert route_storage["memory_mapped"] is True
        assert route_storage["vectors_format"] == "npy_memmap_v1"


def test_route_index_warmup_failure_is_not_misattributed_to_last_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, release = _package(tmp_path, monkeypatch)
    monkeypatch.setattr("reacts.ml.inference.validate_runtime_environment", lambda _: {"pass": True})

    def fail_storage(self, *, sample: bool = True):
        raise MemoryError("route storage unavailable")

    monkeypatch.setattr(RouteEmbeddingIndex, "storage_info", fail_storage)
    settings = Settings(
        project_root=tmp_path / "service-failure",
        artifact_uri=str(bundle.parent),
        artifact_release=release,
        artifact_cache_dir=tmp_path / "cache-failure",
        artifact_required=True,
        artifact_warmup=True,
        rate_limit_requests_per_minute=0,
    ).resolve()

    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        warmup = ready.json()["warmup"]
        assert warmup["failed_component"] == "route_index"
        assert "failed_task" not in warmup
