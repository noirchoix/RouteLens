from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reacts.chemistry.repair import deterministic_repair_candidates
from reacts.ml.registry import Registry
from reacts.ml.training import Trainer, TrainingConfig
from reacts.storage.tabular import DatasetWriter


def _repairability_fixture(root: Path) -> None:
    rows = []
    for index in range(12):
        split = "train" if index < 8 else ("val" if index < 10 else "test")
        rows.append(
            {
                "dataset_version": "fixture_v2",
                "step_id": f"s{index}",
                "route_id": f"r{index}",
                "patent_document_id": f"p{index}",
                "split": split,
                "original_reaction_smiles": "M1>>CCO",
                "contextual_parse_ok": False,
                "repairable": False,
            }
        )
    DatasetWriter(root, "steps", prefer_parquet=False).write(pd.DataFrame(rows))


def test_parse_valid_reaction_never_generates_repair_candidate():
    candidates = deterministic_repair_candidates(
        "CCO>>CC=O",
        contextual_candidate="[CH3][CH2]O>>[CH3]C=O",
        route_continuity_score=1.0,
    )
    assert candidates == []


def test_repairability_is_audited_without_model_artifact(tmp_path):
    canonical = tmp_path / "canonical"
    _repairability_fixture(canonical)
    registry = Registry(tmp_path / "data" / "registry" / "registry.sqlite3")
    trainer = Trainer(
        TrainingConfig(
            canonical_dir=canonical,
            model_dir=tmp_path / "data" / "models",
            dataset_version="fixture_v2",
            epochs=1,
            n_features=2**10,
        ),
        registry,
    )

    result = trainer.train_task("repairability", promote_validated=True)

    assert result["status"] == "skipped"
    assert result["model"] is None
    assert result["metrics"]["reason_code"] == "deterministic_audit_not_trainable"
    assert result["metrics"]["population_audit"]["class_counts"]["train"] == {"False": 8}
    assert not registry.list_models()
    audit = registry.resolve_artifact_path(result["task_audit_path"])
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["metrics"]["model_created"] is False


def test_train_many_continues_after_repairability_audit(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    _repairability_fixture(canonical)
    registry = Registry(tmp_path / "data" / "registry" / "registry.sqlite3")
    trainer = Trainer(
        TrainingConfig(
            canonical_dir=canonical,
            model_dir=tmp_path / "data" / "models",
            dataset_version="fixture_v2",
            epochs=1,
            n_features=2**10,
        ),
        registry,
    )

    called = []
    original = trainer.train_task

    def wrapped(task: str, promote_validated: bool = False):
        called.append(task)
        if task == "synthetic_followup":
            return {"status": "completed", "task": task}
        return original(task, promote_validated=promote_validated)

    monkeypatch.setattr(trainer, "train_task", wrapped)
    result = trainer.train_many(["repairability", "synthetic_followup"], promote_validated=True)
    assert called == ["repairability", "synthetic_followup"]
    assert result["repairability"]["status"] == "skipped"
    assert result["synthetic_followup"]["status"] == "completed"
