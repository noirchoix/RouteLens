from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path

from reacts.ml.registry import Registry
from reacts.settings import Settings
from reacts.validation.acceptance import ScientificAcceptanceValidator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(root: Path) -> Settings:
    return Settings(project_root=root).resolve()


def test_api_smoke_cold_import_does_not_rewrite_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    writable = Registry(settings.registry_db)
    writable.sync_json()

    database_before = _sha256(settings.registry_db)
    json_before = _sha256(settings.model_dir / "model_registry.json")

    # Force the same cold import path used by the CLI validator. Without the
    # v2.0.11 guard, importing reacts.api.main constructs a writable global app
    # and rewrites model_registry.json before the explicit read-only app exists.
    sys.modules.pop("reacts.api.main", None)
    monkeypatch.delenv("REACTS_API_READ_ONLY_REGISTRY", raising=False)

    validator = ScientificAcceptanceValidator(settings)
    result = validator._api_smoke()

    assert result["pass"] is True
    imported = importlib.import_module("reacts.api.main")
    assert imported.app.state.application.registry.read_only is True
    assert "REACTS_API_READ_ONLY_REGISTRY" not in os.environ
    assert _sha256(settings.registry_db) == database_before
    assert _sha256(settings.model_dir / "model_registry.json") == json_before
