from __future__ import annotations

from pathlib import Path

from reacts.ml.anomaly import ConditionAnomalyModel
from reacts.services.application import Application
from reacts.settings import Settings


def test_settings_resolve_normalizes_relative_project_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings(project_root=Path(".")).resolve()

    assert settings.project_root == tmp_path.resolve()
    assert settings.model_dir == tmp_path.resolve() / "data" / "models"
    assert settings.reports_dir == tmp_path.resolve() / "reports"


def test_build_anomaly_model_reports_portable_path_with_relative_project_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    class DummyConditionAnomalyModel:
        statistics = {"__global__": {}}

        def save(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        ConditionAnomalyModel,
        "fit",
        classmethod(lambda cls, canonical_dir: DummyConditionAnomalyModel()),
    )

    app = Application(Settings(project_root=Path(".")))
    try:
        result = app.build_anomaly_model()
    finally:
        app.close()

    assert result == {
        "artifact_path": "data/models/condition_anomaly/robust_family_stats.json",
        "families": 1,
    }
