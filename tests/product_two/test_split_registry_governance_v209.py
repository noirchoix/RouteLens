from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reacts.contracts import ModelStage
from reacts.data.split_governance import ProductTwoSplitRebuilder
from reacts.ml.prefit import validate_classification_support, validate_multilabel_support
from reacts.ml.registry import Registry
from reacts.storage.tabular import DatasetWriter, iter_dataset
from reacts.validation.leakage import LeakageAuditor


def _canonical_fixture(root: Path) -> None:
    rows = [
        {
            "step_id": "s1", "step_instance_id": "s1", "route_id": "r1", "route_instance_id": "r1",
            "source_route_id": "r1", "patent_document_id": "p-a", "reaction_signature": "sig-a",
            "canonical_resolved_reaction_smiles": "CCO>>CC=O", "canonical_reaction_smiles": "CCO>>CC=O", "split": "train",
        },
        {
            "step_id": "s2", "step_instance_id": "s2", "route_id": "r2", "route_instance_id": "r2",
            "source_route_id": "r2", "patent_document_id": "p-a", "reaction_signature": "sig-b",
            "canonical_resolved_reaction_smiles": "CCN>>CC=N", "canonical_reaction_smiles": "CCN>>CC=N", "split": "val",
        },
        {
            "step_id": "s3", "step_instance_id": "s3", "route_id": "r3", "route_instance_id": "r3",
            "source_route_id": "r3", "patent_document_id": "p-b", "reaction_signature": "sig-b",
            "canonical_resolved_reaction_smiles": "CCC>>CC=C", "canonical_reaction_smiles": "CCC>>CC=C", "split": "test",
        },
        {
            "step_id": "s4", "step_instance_id": "s4", "route_id": "r4", "route_instance_id": "r4",
            "source_route_id": "r4", "patent_document_id": "p-c", "reaction_signature": "sig-c",
            "canonical_resolved_reaction_smiles": "CO>>C=O", "canonical_reaction_smiles": "CO>>C=O", "split": "train",
        },
        {
            "step_id": "s5", "step_instance_id": "s5", "route_id": "r5", "route_instance_id": "r5",
            "source_route_id": "r5", "patent_document_id": "p-d", "reaction_signature": "sig-d",
            "canonical_resolved_reaction_smiles": "CN>>C=N", "canonical_reaction_smiles": "CN>>C=N", "split": "val",
        },
    ]
    DatasetWriter(root, "steps", prefer_parquet=False).write(pd.DataFrame(rows))
    routes = [
        {
            "route_id": row["route_id"],
            "route_instance_id": row["route_instance_id"],
            "source_route_id": row["source_route_id"],
            "patent_document_id": row["patent_document_id"],
            "split": row["split"],
        }
        for row in rows
    ]
    DatasetWriter(root, "routes", prefer_parquet=False).write(pd.DataFrame(routes))
    (root / "dataset_manifest.json").write_text(
        json.dumps({"dataset_version": "uspto_multistep_contextual_v2", "contract": {}, "reproducibility": {}}),
        encoding="utf-8",
    )


def test_connected_component_split_is_deterministic_and_strict(tmp_path: Path) -> None:
    canonical = tmp_path / "data" / "canonical_v2"
    _canonical_fixture(canonical)
    marker = tmp_path / "data" / "mapping_v2" / "mapping_manifest.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("unchanged", encoding="utf-8")

    first = ProductTwoSplitRebuilder(canonical, random_seed=42, prefer_parquet=False).run()
    first_hash = first["training_split_sha256"]
    second = ProductTwoSplitRebuilder(canonical, random_seed=42, prefer_parquet=False).run()

    rows = pd.concat(list(iter_dataset(canonical, "steps")), ignore_index=True)
    connected = rows.loc[rows["route_instance_id"].isin(["r1", "r2", "r3"])]
    assert connected["split"].nunique() == 1
    assert connected["split_component_id"].nunique() == 1
    assert second["training_split_sha256"] == first_hash
    assert second["split_governance"]["invariants"] == {
        "patent_document_id_overlapping_keys": 0,
        "reaction_signature_overlapping_keys": 0,
        "route_split_conflicts": 0,
        "strict_pass": True,
    }
    assert second["mapping_rebuilt"] is False
    assert second["derivation_rebuilt"] is False
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_exact_reaction_signature_overlap_is_strict_failure(tmp_path: Path) -> None:
    rows = [
        {
            "step_id": "s1", "split": "train", "patent_document_id": "p1", "reaction_signature": "same",
            "canonical_resolved_reaction_smiles": "CCO>>CC=O", "canonical_reaction_smiles": "CCO>>CC=O",
        },
        {
            "step_id": "s2", "split": "test", "patent_document_id": "p2", "reaction_signature": "same",
            "canonical_resolved_reaction_smiles": "CCO>>CC=O", "canonical_reaction_smiles": "CCO>>CC=O",
        },
    ]
    DatasetWriter(tmp_path, "steps", prefer_parquet=False).write(pd.DataFrame(rows))
    report = LeakageAuditor(tmp_path).audit()
    assert report["summaries"]["patent_document_id"]["overlapping_keys"] == 0
    assert report["summaries"]["reaction_signature"]["overlapping_keys"] == 1
    assert report["strict_pass"] is False
    assert report["strict_invariants"]["product_scaffold_is_diagnostic_only"] is True


def test_prefit_support_rejects_one_class_and_retains_valid_multilabel_targets() -> None:
    binary = validate_classification_support(
        {"train": {"False": 100}, "val": {"False": 20}, "test": {"False": 20}},
        retained_classes=["False"],
        task_kind="binary",
    )
    assert binary.trainable is False
    assert binary.reason_code == "insufficient_class_support"

    multilabel = validate_multilabel_support(
        {"train": 100, "val": 20, "test": 20},
        {
            "train": {"valid": 30, "positive_only": 100},
            "val": {"valid": 5, "positive_only": 20},
            "test": {"valid": 5, "positive_only": 20},
        },
        candidate_labels=["valid", "positive_only"],
        minimum_positive_train=10,
        minimum_negative_train=10,
    )
    assert multilabel.trainable is True
    assert multilabel.retained_labels == ("valid",)
    assert multilabel.dropped_labels[0]["label"] == "positive_only"
    assert "insufficient_train_negative_support" in multilabel.dropped_labels[0]["reasons"]


def test_registry_supersedes_old_runtime_model_and_writes_json(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "data" / "registry" / "reacts.sqlite3")
    model_dir = tmp_path / "data" / "models" / "task"
    model_dir.mkdir(parents=True)
    first = model_dir / "first.joblib"
    second = model_dir / "second.joblib"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    environment = {
        "python": "3.13.0", "scikit_learn": "1.9.0", "numpy": "x", "scipy": "x", "joblib": "x", "rdkit": "x"
    }
    old = registry.register_model(
        task="task", artifact_path=first, dataset_version="v2", metrics={}, config={},
        stage=ModelStage.CANDIDATE, split_sha256="split", training_environment=environment,
    )
    new = registry.register_model(
        task="task", artifact_path=second, dataset_version="v2", metrics={}, config={},
        stage=ModelStage.CANDIDATE, split_sha256="split", training_environment=environment,
    )
    runtime = registry.list_models(runtime_only=True, dataset_version="v2")
    all_records = registry.list_models(dataset_version="v2")
    assert [item["model_id"] for item in runtime] == [new["model_id"]]
    older = next(item for item in all_records if item["model_id"] == old["model_id"])
    assert older["lifecycle_state"] == "superseded"
    assert older["runtime_load_required"] is False
    payload = json.loads((tmp_path / "data" / "models" / "model_registry.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0.9-model-registry-v1"
    assert payload["contract"]["superseded_artifacts_are_preserved_without_deserialization"] is True
