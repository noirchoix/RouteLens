from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reacts.storage import tabular
from reacts.storage.tabular import DatasetWriter, iter_dataset


def test_parquet_json_contract_columns_are_serialized_before_arrow(monkeypatch, tmp_path: Path):
    captured: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(tabular, "parquet_available", lambda: True)

    def fake_to_parquet(self, path, *, index, compression):
        captured["frame"] = self.copy()
        Path(path).touch()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    DatasetWriter(tmp_path, "routes", prefer_parquet=True).write(
        pd.DataFrame.from_records(
            [
                {
                    "route_id": "r-empty",
                    "family_distribution": {},
                    "quality_components": {"parse": 1.0, "mapping": None},
                },
                {
                    "route_id": "r-populated",
                    "family_distribution": {"oxidation": 2, "reduction": 1},
                    "quality_components": {"parse": 0.5, "mapping": 1.0},
                },
            ]
        )
    )

    frame = captured["frame"]
    assert frame["family_distribution"].tolist() == [
        "{}",
        '{"oxidation": 2, "reduction": 1}',
    ]
    assert json.loads(frame.loc[0, "quality_components"]) == {
        "parse": 1.0,
        "mapping": None,
    }


def test_parquet_reader_restores_json_contract_columns(monkeypatch, tmp_path: Path):
    dataset_dir = tmp_path / "routes"
    dataset_dir.mkdir(parents=True)
    part = dataset_dir / "part-00000.parquet"
    part.touch()

    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: pd.DataFrame.from_records(
            [
                {
                    "route_id": "r1",
                    "family_distribution": '{"oxidation": 2}',
                    "quality_components": '{"parse": 1.0}',
                },
                {
                    "route_id": "r2",
                    "family_distribution": "{}",
                    "quality_components": None,
                },
            ]
        ),
    )

    frame = next(iter_dataset(tmp_path, "routes"))
    assert frame.loc[0, "family_distribution"] == {"oxidation": 2}
    assert frame.loc[1, "family_distribution"] == {}
    assert frame.loc[0, "quality_components"] == {"parse": 1.0}
    assert frame.loc[1, "quality_components"] == {}


def test_csv_json_contract_behavior_remains_round_trip_safe(tmp_path: Path):
    DatasetWriter(tmp_path, "routes", prefer_parquet=False).write(
        pd.DataFrame.from_records(
            [
                {
                    "route_id": "r1",
                    "reaction_families": ["oxidation"],
                    "family_distribution": {"oxidation": 1},
                }
            ]
        )
    )
    frame = next(iter_dataset(tmp_path, "routes"))
    assert frame.loc[0, "reaction_families"] == ["oxidation"]
    assert frame.loc[0, "family_distribution"] == {"oxidation": 1}


def test_parquet_reader_normalizes_arrow_numpy_json_containers(monkeypatch, tmp_path: Path):
    dataset_dir = tmp_path / "reaction_mappings_rxnmapper"
    dataset_dir.mkdir(parents=True)
    part = dataset_dir / "part-00000.parquet"
    part.touch()

    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: pd.DataFrame.from_records(
            [
                {
                    "step_id": "s-empty",
                    "diagnostics": np.array([], dtype=object),
                    "provenance": np.array([], dtype=object),
                },
                {
                    "step_id": "s-values",
                    "diagnostics": np.array(
                        ["low_confidence", "coverage_below_threshold"],
                        dtype=object,
                    ),
                    "provenance": '{"backend":"rxnmapper"}',
                },
                {
                    "step_id": "s-null",
                    "diagnostics": None,
                    "provenance": None,
                },
            ]
        ),
    )

    frame = next(iter_dataset(tmp_path, "reaction_mappings_rxnmapper"))
    assert frame.loc[0, "diagnostics"] == []
    assert frame.loc[1, "diagnostics"] == [
        "low_confidence",
        "coverage_below_threshold",
    ]
    assert frame.loc[2, "diagnostics"] == []
    assert frame.loc[0, "provenance"] == {}
    assert frame.loc[1, "provenance"] == {"backend": "rxnmapper"}
    assert frame.loc[2, "provenance"] == {}


def test_json_contract_serializer_accepts_numpy_arrays():
    frame = pd.DataFrame.from_records(
        [
            {
                "step_id": "s1",
                "diagnostics": np.array(["warning"], dtype=object),
            }
        ]
    )

    serial = tabular.serialize_json_contract_columns(frame)
    assert serial.loc[0, "diagnostics"] == '["warning"]'
