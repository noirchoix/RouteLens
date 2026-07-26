from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reacts.api.main import create_app
from reacts.ml.registry import Registry
from reacts.settings import Settings
from reacts.validation.acceptance import ScientificAcceptanceValidator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(root: Path) -> Settings:
    return Settings(project_root=root).resolve()


def test_read_only_registry_queries_do_not_mutate_database_or_json(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    writable = Registry(settings.registry_db)
    writable.sync_json()

    database_before = _sha256(settings.registry_db)
    json_before = _sha256(settings.model_dir / "model_registry.json")

    readonly = Registry(settings.registry_db, read_only=True)
    assert readonly.list_models() == []
    assert readonly.list_task_audits() == []
    with pytest.raises(RuntimeError, match="read-only"):
        readonly.sync_json()

    assert _sha256(settings.registry_db) == database_before
    assert _sha256(settings.model_dir / "model_registry.json") == json_before


def test_validator_and_api_smoke_use_read_only_registry(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    writable = Registry(settings.registry_db)
    writable.sync_json()

    database_before = _sha256(settings.registry_db)
    json_before = _sha256(settings.model_dir / "model_registry.json")

    validator = ScientificAcceptanceValidator(settings)
    assert validator.registry.read_only is True

    app = create_app(settings, read_only_registry=True)
    assert app.state.application.registry.read_only is True
    assert len(app.state.application.registry.list_models()) == 0

    assert _sha256(settings.registry_db) == database_before
    assert _sha256(settings.model_dir / "model_registry.json") == json_before


def test_validation_report_makes_registry_mutation_a_strict_failure(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    registry = Registry(settings.registry_db)
    registry.sync_json()
    validator = ScientificAcceptanceValidator(settings)

    monkeypatch.setattr(validator, "_pipeline_state", lambda: {"pass": True, "final_manifest": True})
    monkeypatch.setattr(validator, "_dataset_integrity", lambda: {"pass": True})
    monkeypatch.setattr(validator, "_current_split_sha", lambda: "split")
    monkeypatch.setattr(validator, "_index_state", lambda split: {"pass": True})
    monkeypatch.setattr(validator, "_model_smoke", lambda split: {"pass": True})
    monkeypatch.setattr(validator, "_api_smoke", lambda: {"pass": True})
    monkeypatch.setattr(validator, "_retrieval_benchmark", lambda: {"pass": True})

    class _Leakage:
        def audit(self):
            return {"strict_pass": True}

    monkeypatch.setattr("reacts.validation.acceptance.LeakageAuditor", lambda root: _Leakage())

    original_index_state = validator._index_state
    def mutate_then_pass(split):
        payload = json.loads((settings.model_dir / "model_registry.json").read_text(encoding="utf-8"))
        payload["unexpected_validation_write"] = True
        (settings.model_dir / "model_registry.json").write_text(json.dumps(payload), encoding="utf-8")
        return original_index_state(split)

    monkeypatch.setattr(validator, "_index_state", mutate_then_pass)
    report = validator.validate()
    assert report["reproducibility"]["registry_read_only_validation"]["pass"] is False
    assert report["strict_pass"] is False
