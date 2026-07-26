from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reacts.api.main import create_app  # noqa: E402
from reacts.settings import Settings  # noqa: E402


def main() -> int:
    client = TestClient(create_app(Settings(project_root=ROOT).resolve()))
    checks = {
        "health": client.get("/api/v2/health"),
        "repair": client.post(
            "/api/v2/inference/repair",
            json={"reaction_smiles": "CCO>O>CC=O", "route_continuity_score": 0.0},
        ),
        "quality": client.post(
            "/api/v2/inference/route-quality",
            json={
                "parse": 1.0,
                "resolution": 1.0,
                "route_continuity": 1.0,
                "condition_completeness": 0.5,
                "condition_plausibility": 1.0,
                "mapping": 0.5,
            },
        ),
    }
    report = {
        name: {"status_code": response.status_code, "body": response.json()}
        for name, response in checks.items()
    }
    report["pass"] = all(item["status_code"] == 200 for item in report.values() if isinstance(item, dict) and "status_code" in item)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
