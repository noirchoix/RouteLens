import json
from pathlib import Path

import pandas as pd

from reacts.data.canonical import CanonicalBuildConfig, CanonicalBuilder
from reacts.data.source import ArtifactSource
from reacts.storage.tabular import iter_dataset


def _source_tree(root: Path) -> Path:
    base = root / "uspto_llm_multistep_only"
    data = base / "multistep_csv"
    qc = base / "qc"
    data.mkdir(parents=True)
    qc.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "route_id": "20200101-US123A1-0001",
                "step_index": 0,
                "step_reaction_smiles": "CCO>O.25.3600>CC=O",
                "step_working_reaction_smiles": "CCO>O.25.3600>CC=O",
                "step_parse_ok": True,
                "step_solvent_list_norm": "['O']",
                "step_agent_or_spectator_list_norm": "[]",
                "temperature_c_primary": 25,
                "time_h_primary": 1,
            },
            {
                "route_id": "20200101-US123A1-0001",
                "step_index": 1,
                "step_reaction_smiles": "CC=O>>M1",
                "step_working_reaction_smiles": "CC=O>>M1",
                "step_parse_ok": False,
                "step_solvent_list_norm": "[]",
                "step_agent_or_spectator_list_norm": "['[Pd]']",
                "temperature_c_primary": -2100,
                "time_h_primary": 2,
            },
        ]
    ).to_csv(data / "step_table.csv", index=False)
    # Duplicate source headers reproduce the legacy route-summary defect.
    (data / "route_summary.csv").write_text(
        "route_id,route_id,route_source_id,multistep_reaction_text,route_step_count,route_parse_ok,route_parsed_step_count,route_is_atom_mapped,route_reactants_demapped,route_products_demapped,final_products_demapped\n"
        "20200101-US123A1-0001,20200101-US123A1-0001,20200101-US123A1-0001,route,2,False,1,False,['CCO'],['CC=O'],['M1']\n",
        encoding="utf-8",
    )
    for name in ["run_summary.json", "artifact_manifest.json"]:
        (base / name).write_text("{}", encoding="utf-8")
    (qc / "multistep_csv_qc.json").write_text("{}", encoding="utf-8")
    (qc / "multistep_csv_condition_qc.json").write_text("{}", encoding="utf-8")
    return root


def test_builder_repairs_ids_preserves_symbolic_records_and_groups_patents(tmp_path):
    source = ArtifactSource(_source_tree(tmp_path / "source"))
    out = tmp_path / "canonical"
    result = CanonicalBuilder(source, out, CanonicalBuildConfig(prefer_parquet=False, chunksize=1)).build()
    steps = pd.concat(list(iter_dataset(out, "steps")), ignore_index=True)
    assert steps["step_id"].tolist() == ["20200101-US123A1-0001::000", "20200101-US123A1-0001::001"]
    assert steps.loc[1, "parse_failure_class"] == "symbolic_intermediate"
    assert pd.isna(steps.loc[1, "temperature_c"])
    assert steps["patent_document_id"].nunique() == 1
    assert steps["split"].nunique() == 1
    assert result["metrics"]["steps_total"] == 2
