from pathlib import Path

import joblib
import pytest

from reacts.contracts import ModelStage
from reacts.ml.registry import Registry
from reacts.science.promotion import decide_promotion


def test_task_specific_promotion_rejects_weak_time_model():
    metrics = {
        "test": {
            "rows": 1000,
            "macro_f1": 0.25,
            "balanced_accuracy": 0.28,
            "accuracy": 0.29,
            "majority_accuracy": 0.39,
        }
    }
    decision = decide_promotion("time_bucket", metrics, calibration_error=0.08)
    assert not decision.approved
    assert decision.stage == ModelStage.EXPERIMENTAL
    assert decision.reasons


def test_registry_requires_release_decision_for_production(tmp_path):
    registry = Registry(tmp_path / "data" / "registry" / "reacts.sqlite3")
    artifact = tmp_path / "data" / "models" / "model.joblib"
    artifact.parent.mkdir(parents=True)
    joblib.dump({"model": "fixture"}, artifact)
    model = registry.register_model(
        task="fixture",
        artifact_path=artifact,
        dataset_version="fixture_v2",
        metrics={},
        config={},
    )
    with pytest.raises(PermissionError):
        registry.promote(model["model_id"], ModelStage.PRODUCTION)
    registry.promote(
        model["model_id"],
        ModelStage.PRODUCTION,
        release_decision={"approved": True, "stage": "production"},
    )
    assert registry.model_for_task("fixture", ModelStage.PRODUCTION)
