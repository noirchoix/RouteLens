from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from reacts.api.main import create_app
from reacts.ml.specialists import SpecialistTrainer
from reacts.settings import Settings


def test_multilabel_calibration_is_fitted_per_label():
    probabilities = np.asarray(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.25, 0.75],
            [0.10, 0.90],
        ]
    )
    labels = [{"water"}, {"water"}, {"alcohol"}, {"alcohol"}]
    calibrators, report = SpecialistTrainer._fit_multilabel_calibrators(
        probabilities, labels, ["water", "alcohol"]
    )
    calibrated = SpecialistTrainer._apply_multilabel_calibrators(
        probabilities, ["water", "alcohol"], calibrators
    )
    assert set(calibrators) == {"water", "alcohol"}
    assert calibrated.shape == probabilities.shape
    assert 0.0 <= report["mean_calibrated_ece"] <= 1.0


def test_product_two_repair_and_route_quality_endpoints(tmp_path: Path):
    settings = Settings(project_root=tmp_path).resolve()
    client = TestClient(create_app(settings))
    repair = client.post(
        "/api/v2/inference/repair",
        json={
            "reaction_smiles": " CCO >> CC=O ",
            "route_continuity_score": 0.0,
        },
    )
    quality = client.post(
        "/api/v2/inference/route-quality",
        json={
            "parse": 1.0,
            "resolution": 1.0,
            "route_continuity": 1.0,
            "condition_completeness": 0.5,
            "condition_plausibility": 1.0,
            "mapping": 0.5,
        },
    )
    assert repair.status_code == 200
    assert repair.json()["contract"]["strict_post_repair_validation"]
    assert quality.status_code == 200
    assert quality.json()["score"] == 0.875
