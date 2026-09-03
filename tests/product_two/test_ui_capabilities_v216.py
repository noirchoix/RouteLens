from __future__ import annotations

from fastapi.testclient import TestClient

from reacts.api.main import create_app
from reacts.settings import Settings


def _workflow(payload: dict, identifier: str) -> dict:
    return next(item for item in payload["workflows"] if item["id"] == identifier)


def test_capabilities_report_setup_required_before_condition_stats_exist(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, rate_limit_requests_per_minute=0).resolve()
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v2/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert _workflow(payload, "repair")["available"] is True
    assert _workflow(payload, "quality")["available"] is True
    anomaly = _workflow(payload, "anomaly")
    assert anomaly["available"] is False
    assert anomaly["state"] == "setup_required"
    assert anomaly["setup_command"] == "reacts --project-root . build-anomaly-model"


def test_capabilities_report_condition_stats_when_present(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, rate_limit_requests_per_minute=0).resolve()
    artifact = settings.model_dir / "condition_anomaly" / "robust_family_stats.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}\n", encoding="utf-8")
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v2/capabilities")
    assert response.status_code == 200
    anomaly = _workflow(response.json(), "anomaly")
    assert anomaly["available"] is True
    assert anomaly["state"] == "available"
