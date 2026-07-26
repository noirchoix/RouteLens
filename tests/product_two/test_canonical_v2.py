import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reacts.data.canonical_v2 import ContextualBuildConfig, ContextualCanonicalBuilder
from reacts.storage.tabular import DatasetWriter, iter_dataset


def _v1_fixture(root: Path) -> None:
    rows = [
        {
            "dataset_version": "fixture_v1",
            "step_id": "r1::000",
            "route_id": "r1",
            "patent_document_id": "p1",
            "split": "train",
            "step_index": 0,
            "raw_reaction_text": "CCO>>CC=O",
            "reaction_smiles": "CCO>>CC=O",
            "canonical_reaction_smiles": "CCO>>CC=O",
            "reactants": ["CCO"],
            "products": ["CC=O"],
            "parse_ok": True,
            "parse_failure_class": "valid",
            "input_intermediate": None,
            "output_intermediate": "M1",
            "solvents": ["O"],
            "agents": [],
            "solvent_primary": "O",
            "agent_primary": None,
            "agent_present": False,
            "condition_extraction_method": "source",
            "condition_extraction_confidence": "high",
            "condition_numeric_tokens": [],
            "temperature_observed_c": 25.0,
            "temperature_c": 25.0,
            "temperature_valid": True,
            "temperature_bucket": "0-25",
            "time_observed_h": 1.0,
            "time_h": 1.0,
            "time_valid": True,
            "time_bucket": "1-4h",
            "condition_status": "valid",
            "quality_issues": [],
            "quality_score": 1.0,
        },
        {
            "dataset_version": "fixture_v1",
            "step_id": "r1::001",
            "route_id": "r1",
            "patent_document_id": "p1",
            "split": "train",
            "step_index": 1,
            "raw_reaction_text": "M1.N>>CCN",
            "reaction_smiles": "M1.N>>CCN",
            "canonical_reaction_smiles": "M1.N>>CCN",
            "reactants": ["M1", "N"],
            "products": ["CCN"],
            "parse_ok": False,
            "parse_failure_class": "symbolic_intermediate",
            "input_intermediate": "M1",
            "output_intermediate": None,
            "solvents": ["CO"],
            "agents": ["[Pd]"],
            "solvent_primary": "CO",
            "agent_primary": "[Pd]",
            "agent_present": True,
            "condition_extraction_method": "source",
            "condition_extraction_confidence": "high",
            "condition_numeric_tokens": [],
            "temperature_observed_c": 50.0,
            "temperature_c": 50.0,
            "temperature_valid": True,
            "temperature_bucket": "25-60",
            "time_observed_h": 2.0,
            "time_h": 2.0,
            "time_valid": True,
            "time_bucket": "1-4h",
            "condition_status": "valid",
            "quality_issues": [],
            "quality_score": 0.5,
        },
    ]
    DatasetWriter(root, "steps", prefer_parquet=False).write(pd.DataFrame(rows))
    (root / "dataset_manifest.json").write_text(json.dumps({"dataset_version": "fixture_v1"}), encoding="utf-8")


def test_contextual_builder_emits_normalized_v2_tables(tmp_path):
    v1 = tmp_path / "canonical"
    v2 = tmp_path / "canonical_v2"
    _v1_fixture(v1)
    result = ContextualCanonicalBuilder(
        v1,
        v2,
        ContextualBuildConfig(prefer_parquet=False, map_reactions=False),
    ).build()
    steps = pd.concat(list(iter_dataset(v2, "steps")), ignore_index=True)
    edges = pd.concat(list(iter_dataset(v2, "route_edges")), ignore_index=True)
    resolutions = pd.concat(list(iter_dataset(v2, "intermediate_resolution")), ignore_index=True)
    conditions = pd.concat(list(iter_dataset(v2, "condition_evidence")), ignore_index=True)
    molecules = pd.concat(list(iter_dataset(v2, "molecules")), ignore_index=True)
    repairs = pd.concat(list(iter_dataset(v2, "repair_candidates")), ignore_index=True)

    assert result["dataset_version"] == "uspto_multistep_contextual_v2"
    assert steps.loc[steps["step_id"] == "r1::001", "contextual_parse_ok"].iloc[0]
    assert steps.loc[steps["step_id"] == "r1::001", "resolution_status"].iloc[0] == "resolved"
    assert edges.iloc[0]["continuity_status"] == "resolved"
    assert resolutions.iloc[0]["evidence_step_id"] == "r1::000"
    assert {"temperature", "time", "solvent", "agent"}.issubset(set(conditions["condition_type"]))
    assert molecules["molecule_id"].is_unique
    assert repairs["accepted"].any()
    assert not Path(result["manifest_path"]).is_absolute()


def test_condition_evidence_has_type_stable_arrow_columns(tmp_path):
    builder = ContextualCanonicalBuilder(
        tmp_path / "canonical",
        tmp_path / "canonical_v2",
        ContextualBuildConfig(prefer_parquet=True, map_reactions=False),
    )
    step = {
        "step_id": "r1::000",
        "route_id": "r1",
        "condition_extraction_confidence": "high",
        "condition_extraction_method": "source",
        "temperature_observed_c": 25.0,
        "temperature_c": 25.0,
        "temperature_valid": True,
        "time_observed_h": 1.5,
        "time_h": 1.5,
        "time_valid": True,
        "solvents": np.array(["O", "CO"], dtype=object),
        "agents": np.array(["[Pd]"], dtype=object),
    }
    frame = pd.DataFrame.from_records(builder._condition_rows(step))

    assert set(frame["normalized_value_type"]) == {"numeric", "categorical"}
    assert all(isinstance(value, str) for value in frame["normalized_value"].dropna())
    assert frame.loc[frame["normalized_value_type"] == "numeric", "normalized_numeric_value"].notna().all()
    assert frame.loc[frame["normalized_value_type"] == "categorical", "normalized_text_value"].notna().all()
    assert set(frame.loc[frame["condition_type"] == "solvent", "normalized_text_value"]) == {"O", "CO"}

    if pytest.importorskip("pyarrow"):
        target = tmp_path / "condition_evidence.parquet"
        frame.to_parquet(target, index=False, compression="zstd")
        assert target.exists()


def test_staged_route_generator_is_closed_when_processing_fails(tmp_path, monkeypatch):
    builder = ContextualCanonicalBuilder(
        tmp_path / "canonical",
        tmp_path / "canonical_v2",
        ContextualBuildConfig(prefer_parquet=False, map_reactions=False),
    )

    class Routes:
        def __init__(self):
            self.closed = False
            self._yielded = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._yielded:
                raise StopIteration
            self._yielded = True
            return [{"route_id": "r1"}]

        def close(self):
            self.closed = True

    routes = Routes()
    monkeypatch.setattr(builder, "_iter_routes", lambda _database: routes)
    monkeypatch.setattr(builder, "_process_route", lambda _rows, _writers: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        builder._process_staged_routes(tmp_path / "route_context.sqlite3", {})

    assert routes.closed
