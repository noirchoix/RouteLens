from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from reacts.api.main import create_app
from reacts.ml.registry import Registry
from reacts.ml.training import Trainer, TrainingConfig
from reacts.settings import Settings
from reacts.storage.tabular import DatasetWriter


def _canonical_fixture(root: Path) -> None:
    rows = []
    reactions = [
        ("CCO.CC(=O)O>>CCOC(C)=O", "O", "1-4h", False),
        ("CCBr.N>>CCN", "CN(C)C=O", "4-16h", True),
        ("CC=O.[H-]>>CCO", "CO", "<1h", True),
        ("CCO>>CC=O", "ClCCl", "16-24h", False),
        ("CCOC(C)=O.O>>CC(=O)O.CCO", "O", "24h+", False),
    ]
    for i in range(175):
        reaction, solvent, bucket, agent = reactions[i % len(reactions)]
        split = "train" if i < 125 else ("val" if i < 150 else "test")
        rows.append(
            {
                "dataset_version": "fixture_v1",
                "step_id": f"s{i}",
                "route_id": f"r{i}",
                "patent_document_id": f"p{i}",
                "split": split,
                "step_index": 0,
                "reaction_smiles": reaction,
                "canonical_reaction_smiles": reaction,
                "parse_ok": True,
                "solvent_primary": solvent,
                "time_bucket": bucket,
                "temperature_bucket": "0-25",
                "agent_present": agent,
                "condition_extraction_confidence": "high",
                "eligible_condition_models": True,
                "eligible_retrieval": True,
            }
        )
    writer = DatasetWriter(root, "steps", prefer_parquet=False)
    writer.write(pd.DataFrame(rows))


def test_training_registry_and_health(tmp_path):
    canonical = tmp_path / "canonical"
    _canonical_fixture(canonical)
    registry = Registry(tmp_path / "registry.sqlite3")
    trainer = Trainer(
        TrainingConfig(canonical_dir=canonical, model_dir=tmp_path / "models", dataset_version="fixture_v1", epochs=1, n_features=2**12),
        registry,
    )
    result = trainer.train_task("time_bucket", promote_validated=True)
    assert result["metrics"]["test"]["rows"] == 25
    assert registry.list_models()
    assert not Path(registry.list_models()[0]["artifact_path"]).is_absolute()

    settings = Settings(
        project_root=tmp_path,
        canonical_dir=canonical,
        model_dir=tmp_path / "models",
        index_dir=tmp_path / "indexes",
        registry_db=tmp_path / "registry.sqlite3",
        source_artifact=tmp_path / "missing.zip",
    ).resolve()
    client = TestClient(create_app(settings))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["models"] >= 1


def test_condition_models_use_structure_only_input(tmp_path):
    canonical = tmp_path / "canonical"
    _canonical_fixture(canonical)
    registry = Registry(tmp_path / "data" / "registry" / "registry.sqlite3")
    trainer = Trainer(
        TrainingConfig(canonical_dir=canonical, model_dir=tmp_path / "data" / "models", dataset_version="fixture_v1", epochs=1, n_features=2**12),
        registry,
    )
    result = trainer.train_task("time_bucket", promote_validated=True)
    import joblib

    artifact = registry.resolve_artifact_path(result["model"]["artifact_path"])
    bundle = joblib.load(artifact)
    assert bundle["input_column"] == "canonical_reaction_smiles"
