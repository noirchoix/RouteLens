from pathlib import Path

import numpy as np
import pandas as pd

from reacts.science.calibration import calibration_report, fit_temperature_scaler
from reacts.storage.tabular import DatasetWriter
from reacts.validation.leakage import LeakageAuditor


def test_temperature_calibration_produces_reliability_curves():
    classes = np.array(["a", "b"], dtype=object)
    y = np.array(["a", "a", "b", "b"], dtype=object)
    probabilities = np.array([[0.99, 0.01], [0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    calibrator = fit_temperature_scaler(y, probabilities, classes)
    report = calibration_report(y, probabilities, classes, calibrator)
    assert report["calibrated_reliability_curve"]
    assert report["calibrated_ece"] >= 0


def test_patent_leakage_report_detects_cross_split_group(tmp_path):
    rows = [
        {"step_id": "s1", "split": "train", "patent_document_id": "p1", "canonical_resolved_reaction_smiles": "CCO>>CC=O", "canonical_reaction_smiles": "CCO>>CC=O"},
        {"step_id": "s2", "split": "test", "patent_document_id": "p1", "canonical_resolved_reaction_smiles": "CCN>>CC=N", "canonical_reaction_smiles": "CCN>>CC=N"},
    ]
    DatasetWriter(tmp_path, "steps", prefer_parquet=False).write(pd.DataFrame(rows))
    report = LeakageAuditor(tmp_path).audit()
    assert not report["strict_pass"]
    assert report["summaries"]["patent_document_id"]["overlapping_keys"] == 1
