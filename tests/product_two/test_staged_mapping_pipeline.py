from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from reacts.data.canonical_v2 import ContextualBuildConfig, ContextualCanonicalBuilder
from reacts.mapping.derivation import DerivationConfig, ReactionCentreDeriver
from reacts.mapping.preflight import MappingPreflight, resolve_mapping_backend
from reacts.mapping.queue import MappingQueue
from reacts.mapping.runner import MappingRunConfig, ResumableMappingRunner
from reacts.storage.tabular import DatasetWriter, iter_dataset


def _context_fixture(root: Path, reactions: list[tuple[str, str, str]]) -> None:
    steps = []
    routes = []
    candidates = []
    for step_id, route_id, reaction in reactions:
        steps.append(
            {
                "dataset_version": "uspto_multistep_contextual_v2",
                "step_id": step_id,
                "route_id": route_id,
                "patent_document_id": f"pat-{route_id}",
                "split": "train",
                "step_index": 0,
                "canonical_resolved_reaction_smiles": reaction,
                "contextual_parse_ok": True,
                "contextual_parse_failure_class": "valid",
                "eligible_contextual_models": True,
                "eligible_mapping_models": False,
                "eligible_retrieval_v2": True,
                "mapping_status": "pending",
                "mapping_confidence": None,
                "reaction_family": None,
                "reaction_centre_fingerprint": None,
                "quality_components": {
                    "parse": 1.0,
                    "resolution": 1.0,
                    "route_continuity": 1.0,
                    "condition_completeness": 0.0,
                    "condition_plausibility": 1.0,
                    "mapping": None,
                },
                "contextual_quality_score": 0.83,
                "resolution_status": "not_required",
                "solvents": [],
                "agents": [],
                "solvent_primary": None,
                "time_bucket": None,
                "temperature_bucket": None,
            }
        )
        routes.append(
            {
                "dataset_version": "uspto_multistep_contextual_v2",
                "route_id": route_id,
                "patent_document_id": f"pat-{route_id}",
                "split": "train",
                "step_count": 1,
                "mapped_steps": 0,
                "reaction_families": [],
                "family_distribution": {},
            }
        )
        candidates.append(
            {
                "dataset_version": "uspto_multistep_contextual_v2",
                "step_id": step_id,
                "route_id": route_id,
                "patent_document_id": f"pat-{route_id}",
                "split": "train",
                "reaction_smiles": reaction,
                "reaction_signature": f"sig-{step_id}",
                "eligibility_status": "eligible",
                "eligibility_reason": "contextual_parse_valid",
                "mapping_status": "pending",
            }
        )
    DatasetWriter(root, "steps", prefer_parquet=False).write(pd.DataFrame(steps))
    DatasetWriter(root, "routes", prefer_parquet=False).write(pd.DataFrame(routes))
    DatasetWriter(root, "mapping_candidates", prefer_parquet=False).write(pd.DataFrame(candidates))
    (root / "dataset_manifest.json").write_text(json.dumps({"dataset_version": "uspto_multistep_contextual_v2"}))



def _v1_source_fixture(root: Path) -> None:
    rows = [
        {
            "dataset_version": "fixture_v1", "step_id": "r1::000", "route_id": "r1",
            "patent_document_id": "p1", "split": "train", "step_index": 0,
            "raw_reaction_text": "CCO>>CC=O", "reaction_smiles": "CCO>>CC=O",
            "canonical_reaction_smiles": "CCO>>CC=O", "reactants": ["CCO"], "products": ["CC=O"],
            "parse_ok": True, "parse_failure_class": "valid", "input_intermediate": None,
            "output_intermediate": "M1", "solvents": ["O"], "agents": [],
            "solvent_primary": "O", "agent_primary": None, "agent_present": False,
            "condition_extraction_method": "source", "condition_extraction_confidence": "high",
            "condition_numeric_tokens": [], "temperature_observed_c": 25.0, "temperature_c": 25.0,
            "temperature_valid": True, "temperature_bucket": "0-25", "time_observed_h": 1.0,
            "time_h": 1.0, "time_valid": True, "time_bucket": "1-4h", "condition_status": "valid",
            "quality_issues": [], "quality_score": 1.0,
        },
        {
            "dataset_version": "fixture_v1", "step_id": "r1::001", "route_id": "r1",
            "patent_document_id": "p1", "split": "train", "step_index": 1,
            "raw_reaction_text": "M1.N>>CCN", "reaction_smiles": "M1.N>>CCN",
            "canonical_reaction_smiles": "M1.N>>CCN", "reactants": ["M1", "N"], "products": ["CCN"],
            "parse_ok": False, "parse_failure_class": "symbolic_intermediate", "input_intermediate": "M1",
            "output_intermediate": None, "solvents": ["CO"], "agents": ["[Pd]"],
            "solvent_primary": "CO", "agent_primary": "[Pd]", "agent_present": True,
            "condition_extraction_method": "source", "condition_extraction_confidence": "high",
            "condition_numeric_tokens": [], "temperature_observed_c": 50.0, "temperature_c": 50.0,
            "temperature_valid": True, "temperature_bucket": "25-60", "time_observed_h": 2.0,
            "time_h": 2.0, "time_valid": True, "time_bucket": "1-4h", "condition_status": "valid",
            "quality_issues": [], "quality_score": 0.5,
        },
    ]
    DatasetWriter(root, "steps", prefer_parquet=False).write(pd.DataFrame(rows))
    (root / "dataset_manifest.json").write_text(json.dumps({"dataset_version": "fixture_v1"}))

def test_context_stage_emits_mapping_candidates_without_inline_mapping(tmp_path):
    source = tmp_path / "canonical"
    context = tmp_path / "context"
    _v1_source_fixture(source)
    result = ContextualCanonicalBuilder(
        source,
        context,
        ContextualBuildConfig(prefer_parquet=False, checkpoint_routes=1),
    ).build()
    candidates = pd.concat(list(iter_dataset(context, "mapping_candidates")), ignore_index=True)
    steps = pd.concat(list(iter_dataset(context, "steps")), ignore_index=True)
    assert set(candidates["mapping_status"]) == {"pending"}
    assert set(steps["mapping_status"]) == {"pending"}
    assert not (context / "reaction_mappings").exists()
    assert result["contract"]["atom_mapping_is_not_executed_in_this_stage"] is True
    assert (context / ".work" / "context_progress.json").exists()


def test_explicit_rxnmapper_preflight_fails_before_data_work(monkeypatch):
    module = types.ModuleType("rxnmapper")

    class Broken:
        def __init__(self):
            raise RuntimeError("missing runtime dependency")

    module.RXNMapper = Broken
    monkeypatch.setitem(sys.modules, "rxnmapper", module)
    with pytest.raises(RuntimeError, match="RXNMapper initialization failed"):
        resolve_mapping_backend("rxnmapper")


def test_mapping_queue_closes_database_and_recovers_stale(tmp_path):
    context = tmp_path / "context"
    _context_fixture(context, [("s1", "r1", "CCO>>CC=O")])
    database = tmp_path / "state" / "queue.sqlite3"
    queue = MappingQueue(database)
    assert queue.populate_from_context(context)["inserted"] == 1
    claimed = queue.claim_batch(1, backend_requested="rxnmapper", max_attempts=2)
    assert claimed[0].step_id == "s1"
    # A fresh queue instance can open and close the same database, proving no
    # process-global connection remains locked (the Windows regression case).
    assert MappingQueue(database).summary()["running"] == 1
    database.rename(database.with_name("queue-renamed.sqlite3"))


def test_batched_mapping_is_resumable_and_segregates_fallback(tmp_path, monkeypatch):
    context = tmp_path / "context"
    _context_fixture(
        context,
        [("s1", "r1", "CCO>>CC=O"), ("s2", "r2", "CCN>>CC=N")],
    )

    calls: list[int] = []

    class FakeMapper:
        def get_attention_guided_atom_maps(self, reactions):
            calls.append(len(reactions))
            output = []
            for reaction in reactions:
                if reaction.startswith("CCN"):
                    raise RuntimeError("synthetic mapper failure")
                output.append(
                    {
                        "confidence": 0.95,
                        "mapped_rxn": "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
                    }
                )
            return output

    preflight = MappingPreflight(
        backend_requested="rxnmapper",
        primary_backend="rxnmapper",
        fallback_backend="mcs_fallback",
        rxnmapper_available=True,
        rxnmapper_version="fixture",
        setuptools_version="fixture",
        device="cpu",
        initialization_status="successful",
    )
    monkeypatch.setattr(
        "reacts.mapping.runner.resolve_mapping_backend",
        lambda *args, **kwargs: (preflight, FakeMapper()),
    )

    config = MappingRunConfig(
        context_dir=context,
        output_dir=tmp_path / "mapping",
        queue_db=tmp_path / "state" / "mapping.sqlite3",
        reports_dir=tmp_path / "reports",
        backend="rxnmapper",
        batch_size=2,
        shard_size=1,
        resume=False,
        prefer_parquet=False,
    )
    first = ResumableMappingRunner(config).run()
    assert first["queue_summary"]["total"] == 2
    assert calls[0] == 2
    assert (config.output_dir / "reaction_mappings_rxnmapper").exists()
    assert (config.output_dir / "reaction_mappings_mcs_fallback").exists()
    assert (config.output_dir / "reaction_mappings_rejected").exists()

    # A resume has no work to repeat.
    config.resume = True
    second = ResumableMappingRunner(config).run()
    assert second["queue_summary"] == first["queue_summary"]


def test_independent_derivation_materializes_final_scientific_view(tmp_path):
    context = tmp_path / "context"
    _context_fixture(context, [("s1", "r1", "CCO>>CC=O")])
    mapping = tmp_path / "mapping"
    DatasetWriter(mapping, "reaction_mappings_rxnmapper", prefer_parquet=False).write(
        pd.DataFrame(
            [
                {
                    "dataset_version": "uspto_multistep_contextual_v2",
                    "step_id": "s1",
                    "route_id": "r1",
                    "mapping_status": "mapped",
                    "mapped_reaction_smiles": "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
                    "backend": "rxnmapper",
                    "confidence": 0.95,
                    "validation_status": "passed",
                    "scientific_eligibility": True,
                }
            ]
        )
    )
    (mapping / "mapping_manifest.json").write_text(json.dumps({"backend": "rxnmapper"}))
    result = ReactionCentreDeriver(
        DerivationConfig(
            context_dir=context,
            mapping_dir=mapping,
            derivation_dir=tmp_path / "derivation",
            final_canonical_dir=tmp_path / "canonical_v2",
            queue_db=tmp_path / "state" / "derivation.sqlite3",
            prefer_parquet=False,
        )
    ).run()
    final_steps = pd.concat(list(iter_dataset(tmp_path / "canonical_v2", "steps")), ignore_index=True)
    assert result["queue_summary"]["derived"] == 1
    assert final_steps.loc[0, "eligible_mapping_models"]
    assert final_steps.loc[0, "reaction_family"]
    assert (tmp_path / "canonical_v2" / "reaction_centres").exists()
    assert (tmp_path / "canonical_v2" / "reaction_families").exists()


def test_contextualization_resume_does_not_duplicate_committed_routes(tmp_path, monkeypatch):
    source = tmp_path / "canonical"
    context = tmp_path / "context"
    _v1_source_fixture(source)
    # Add a second independent route.
    first = pd.concat(list(iter_dataset(source, "steps")), ignore_index=True).iloc[0].to_dict()
    first.update(
        {
            "step_id": "r2::000",
            "route_id": "r2",
            "patent_document_id": "p2",
            "raw_reaction_text": "CCN>>CC=N",
            "reaction_smiles": "CCN>>CC=N",
            "canonical_reaction_smiles": "CCN>>CC=N",
            "reactants": ["CCN"],
            "products": ["CC=N"],
            "output_intermediate": None,
        }
    )
    DatasetWriter(source, "steps", prefer_parquet=False, part_index=1).write(pd.DataFrame([first]))

    builder = ContextualCanonicalBuilder(
        source,
        context,
        ContextualBuildConfig(prefer_parquet=False, checkpoint_routes=1),
    )
    original = builder._process_route
    calls = {"count": 0}

    def interrupt_second(rows, writers):
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt()
        return original(rows, writers)

    monkeypatch.setattr(builder, "_process_route", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        builder.build()

    resumed = ContextualCanonicalBuilder(
        source,
        context,
        ContextualBuildConfig(prefer_parquet=False, checkpoint_routes=1, resume=True),
    ).build()
    steps = pd.concat(list(iter_dataset(context, "steps")), ignore_index=True)
    assert resumed["metrics"]["routes_total"] == 2
    assert steps["step_id"].is_unique
    assert set(steps["route_id"]) == {"r1", "r2"}


def test_route_instance_identity_and_symbolic_evidence_are_preserved(tmp_path):
    source = tmp_path / "canonical"
    context = tmp_path / "context"

    def step(step_id, index, raw, *, parse_ok, failure, input_label=None, output_label=None):
        return {
            "dataset_version": "fixture_v1",
            "step_id": step_id,
            "route_id": "rdup",
            "patent_document_id": "pdup",
            "split": "train",
            "step_index": index,
            "raw_reaction_text": raw,
            "reaction_smiles": raw,
            "canonical_reaction_smiles": raw,
            "reactants": raw.split(">>")[0].split("."),
            "products": raw.split(">>")[1].split("."),
            "parse_ok": parse_ok,
            "parse_failure_class": failure,
            "input_intermediate": input_label,
            "output_intermediate": output_label,
            "solvents": [],
            "agents": [],
            "solvent_primary": None,
            "agent_primary": None,
            "agent_present": False,
            "condition_extraction_method": "source",
            "condition_extraction_confidence": "high",
            "condition_numeric_tokens": [],
            "temperature_observed_c": None,
            "temperature_c": None,
            "temperature_valid": False,
            "temperature_bucket": None,
            "time_observed_h": None,
            "time_h": None,
            "time_valid": False,
            "time_bucket": None,
            "condition_status": "missing",
            "quality_issues": [],
            "quality_score": 0.5,
        }

    steps = [
        step("rdup::000", 0, "CCO>>CC=O", parse_ok=True, failure="valid"),
        step("rdup::000", 0, "CCO>>CC=O", parse_ok=True, failure="valid"),
        step("rdup::000::variant", 0, "CC>>M1", parse_ok=False, failure="symbolic_intermediate", output_label="M1"),
        step("rdup::001", 1, "M1>>CCC", parse_ok=False, failure="symbolic_intermediate", input_label="M1"),
    ]
    routes = [
        {
            "dataset_version": "fixture_v1", "route_uid": "rdup", "route_id": "rdup",
            "patent_document_id": "pdup", "split": "train", "source_content_hash": "h1",
            "multistep_reaction_text": "CCO>>CC=O", "step_count": 1,
        },
        {
            "dataset_version": "fixture_v1", "route_uid": "rdup", "route_id": "rdup",
            "patent_document_id": "pdup", "split": "train", "source_content_hash": "h1",
            "multistep_reaction_text": "CCO>>CC=O", "step_count": 1,
        },
        {
            "dataset_version": "fixture_v1", "route_uid": "rdup::v2", "route_id": "rdup",
            "patent_document_id": "pdup", "split": "train", "source_content_hash": "h2",
            "multistep_reaction_text": "CC>>M1\nM1>>CCC", "step_count": 2,
        },
    ]
    DatasetWriter(source, "steps", prefer_parquet=False).write(pd.DataFrame(steps))
    DatasetWriter(source, "routes", prefer_parquet=False).write(pd.DataFrame(routes))
    (source / "dataset_manifest.json").write_text(json.dumps({"dataset_version": "fixture_v1"}))

    manifest = ContextualCanonicalBuilder(
        source,
        context,
        ContextualBuildConfig(prefer_parquet=False, checkpoint_routes=1),
    ).build()

    route_rows = pd.concat(list(iter_dataset(context, "routes")), ignore_index=True)
    step_rows = pd.concat(list(iter_dataset(context, "steps")), ignore_index=True)
    edges = pd.concat(list(iter_dataset(context, "route_edges")), ignore_index=True)
    resolutions = pd.concat(list(iter_dataset(context, "intermediate_resolution")), ignore_index=True)
    candidates = pd.concat(list(iter_dataset(context, "mapping_candidates")), ignore_index=True)
    quarantine = pd.concat(list(iter_dataset(context, "quarantine")), ignore_index=True)

    assert len(route_rows) == route_rows["route_instance_id"].nunique() == 3
    assert route_rows["source_route_id"].nunique() == 1
    assert len(step_rows) == step_rows["step_instance_id"].nunique() == 4
    assert set(step_rows["route_instance_id"]) == set(route_rows["route_instance_id"])
    assert len(candidates) == 4
    assert int((candidates["eligibility_status"] == "eligible").sum()) == 2
    assert set(resolutions["evidence_status"]) == {
        "unsupported_product_placeholder",
        "unresolved_label_only",
    }
    assert not resolutions["resolved_molecules"].map(bool).any()
    assert set(quarantine["reason"]) == {"unresolved_symbolic_intermediate"}
    assert all(
        row.source_step_instance_id in set(step_rows.loc[step_rows.route_instance_id == row.route_instance_id, "step_instance_id"])
        and row.target_step_instance_id in set(step_rows.loc[step_rows.route_instance_id == row.route_instance_id, "step_instance_id"])
        for row in edges.itertuples()
    )

    report = manifest["scientific_identity_report"]
    assert report["routes_total"] == 3
    assert report["unique_route_instance_id"] == 3
    assert report["unique_source_route_id"] == 1
    assert report["duplicate_source_route_groups"] == 1
    assert report["preserved_conflicting_variant_rows"] == 2
    assert report["steps_total"] == 4
    assert report["duplicate_step_instance_ids"] == 0
    assert report["cross_variant_edges"] == 0
    assert report["mapping_candidates_total"] == 4
    assert report["strict_mapping_eligible"] == 2
    assert report["unsupported_symbolic_auto_resolved"] == 0
    assert report["symbolic_placeholder_occurrences"] == 2
    assert report["symbolic_steps"] == 2
    assert report["label_connected_edges"] == 1
    assert report["observed_structure_resolutions"] == 0
    assert report["inferred_structure_hypotheses"] == 0
    assert report["unresolved_symbolic_steps"] == 2
    assert report["invalid_non_symbolic_steps"] == 0
    assert manifest["scientific_identity_pass"] is True


def _write_identity_assignment_fixture(root: Path, *, unique_routes: int = 25) -> int:
    steps: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []

    def add_step(route_id: str, step_id: str, step_index: int, reaction: str) -> None:
        steps.append(
            {
                "dataset_version": "fixture_v1",
                "step_id": step_id,
                "route_id": route_id,
                "patent_document_id": f"pat-{route_id}",
                "split": "train",
                "step_index": step_index,
                "raw_reaction_text": reaction,
                "reaction_smiles": reaction,
                "canonical_reaction_smiles": reaction,
                "reactants": reaction.split(">>")[0].split("."),
                "products": reaction.split(">>")[1].split("."),
                "parse_ok": True,
                "parse_failure_class": "valid",
                "input_intermediate": None,
                "output_intermediate": None,
                "solvents": [],
                "agents": [],
                "solvent_primary": None,
                "agent_primary": None,
                "agent_present": False,
                "condition_extraction_method": "source",
                "condition_extraction_confidence": "high",
                "condition_numeric_tokens": [],
                "temperature_observed_c": None,
                "temperature_c": None,
                "temperature_valid": False,
                "temperature_bucket": None,
                "time_observed_h": None,
                "time_h": None,
                "time_valid": False,
                "time_bucket": None,
                "condition_status": "missing",
                "quality_issues": [],
                "quality_score": 1.0,
            }
        )

    for index in range(unique_routes):
        route_id = f"unique-{index:04d}"
        reaction = f"C{'C' * (index % 4)}O>>C{'C' * (index % 4)}=O"
        add_step(route_id, f"{route_id}::000", 0, reaction)
        routes.append(
            {
                "dataset_version": "fixture_v1",
                "route_uid": route_id,
                "route_id": route_id,
                "patent_document_id": f"pat-{route_id}",
                "split": "train",
                "source_content_hash": f"hash-{route_id}",
                "multistep_reaction_text": reaction,
                "step_count": 1,
            }
        )

    duplicate_id = "duplicate-source"
    duplicate_reactions = ["CCO>>CC=O", "CC>>CCC", "CCC>>CCCC"]
    add_step(duplicate_id, f"{duplicate_id}::000::a", 0, duplicate_reactions[0])
    add_step(duplicate_id, f"{duplicate_id}::000::b", 0, duplicate_reactions[1])
    add_step(duplicate_id, f"{duplicate_id}::001::b", 1, duplicate_reactions[2])
    routes.extend(
        [
            {
                "dataset_version": "fixture_v1",
                "route_uid": f"{duplicate_id}::a",
                "route_id": duplicate_id,
                "patent_document_id": "pat-duplicate",
                "split": "train",
                "source_content_hash": "hash-a",
                "multistep_reaction_text": duplicate_reactions[0],
                "step_count": 1,
            },
            {
                "dataset_version": "fixture_v1",
                "route_uid": f"{duplicate_id}::b",
                "route_id": duplicate_id,
                "patent_document_id": "pat-duplicate",
                "split": "train",
                "source_content_hash": "hash-b",
                "multistep_reaction_text": "\n".join(duplicate_reactions[1:]),
                "step_count": 2,
            },
        ]
    )

    DatasetWriter(root, "steps", prefer_parquet=False).write(pd.DataFrame(steps))
    DatasetWriter(root, "routes", prefer_parquet=False).write(pd.DataFrame(routes))
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "fixture_v1",
                "metrics": {
                    "steps_total": len(steps),
                    "routes_total": len(routes),
                },
            }
        ),
        encoding="utf-8",
    )
    return len(steps)


def test_indexed_two_path_route_assignment_is_bounded_to_duplicate_groups(tmp_path):
    import sqlite3

    source = tmp_path / "canonical"
    context = tmp_path / "context"
    total_steps = _write_identity_assignment_fixture(source, unique_routes=100)
    builder = ContextualCanonicalBuilder(
        source,
        context,
        ContextualBuildConfig(prefer_parquet=False),
    )
    builder._prepare_output()
    builder._stage_steps(builder.route_database)

    connection = sqlite3.connect(builder.route_database)
    try:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert {
            "idx_route_instance_id",
            "idx_route_source_id",
            "idx_steps_source_route",
            "idx_steps_source_route_order",
            "idx_unique_route_assignment_source",
            "idx_route_assignment_instance_step",
            "idx_route_assignment_step_instance",
            "idx_route_assignment_route_instance",
        }.issubset(indexes)
        assert connection.execute(
            "SELECT COUNT(*) FROM route_instance_assignment"
        ).fetchone()[0] == total_steps
        method_counts = dict(
            connection.execute(
                "SELECT assignment_method, COUNT(*) FROM route_instance_assignment GROUP BY assignment_method"
            ).fetchall()
        )
    finally:
        connection.close()

    manifest = json.loads(builder.identity_assignment_manifest.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["unique_source_routes"] == 100
    assert manifest["duplicate_source_route_groups"] == 1
    assert manifest["duplicate_route_instances"] == 2
    assert manifest["duplicate_group_steps"] == 3
    assert manifest["duplicate_group_steps_assigned"] == 3
    assert manifest["expensive_match_calls"] == 3
    assert manifest["expensive_match_calls"] <= manifest["duplicate_group_steps"]
    assert manifest["unassigned_steps"] == 0
    assert manifest["duplicate_step_instance_ids"] == 0
    assert method_counts["unique_source_route"] == 100
    assert method_counts["exact_route_text_match"] == 3


def test_route_assignment_resume_reuses_staging_database(tmp_path, monkeypatch):
    import reacts.data.canonical_v2 as canonical_v2_module

    source = tmp_path / "canonical"
    context = tmp_path / "context"
    _write_identity_assignment_fixture(source, unique_routes=10)
    first = ContextualCanonicalBuilder(
        source,
        context,
        ContextualBuildConfig(prefer_parquet=False),
    )
    first._prepare_output()

    def interrupt_assignment(*args, **kwargs):
        raise RuntimeError("simulated interruption after source staging")

    monkeypatch.setattr(first, "_assign_route_instances", interrupt_assignment)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        first._stage_steps(first.route_database)

    def forbid_source_restage(*args, **kwargs):
        raise AssertionError("Product One source data was restaged during resume")

    monkeypatch.setattr(canonical_v2_module, "iter_dataset", forbid_source_restage)
    resumed = ContextualCanonicalBuilder(
        source,
        context,
        ContextualBuildConfig(prefer_parquet=False, resume=True),
    )
    resumed._stage_steps(resumed.route_database)
    manifest = json.loads(resumed.identity_assignment_manifest.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["assignment_failures"] == 0
    assert manifest["unassigned_steps"] == 0


def test_pre_v205_staged_database_is_adopted_without_restage(tmp_path, monkeypatch):
    import sqlite3
    import reacts.data.canonical_v2 as canonical_v2_module

    source = tmp_path / "canonical"
    context = tmp_path / "context"
    _write_identity_assignment_fixture(source, unique_routes=12)
    first = ContextualCanonicalBuilder(
        source,
        context,
        ContextualBuildConfig(prefer_parquet=False),
    )
    first._prepare_output()

    def interrupt_assignment(*args, **kwargs):
        raise RuntimeError("simulated legacy pre-assignment stop")

    monkeypatch.setattr(first, "_assign_route_instances", interrupt_assignment)
    with pytest.raises(RuntimeError, match="simulated legacy"):
        first._stage_steps(first.route_database)

    connection = sqlite3.connect(first.route_database)
    try:
        connection.execute("DROP TABLE build_metadata")
        connection.commit()
    finally:
        connection.close()

    def forbid_source_restage(*args, **kwargs):
        raise AssertionError("legacy staging database was not adopted")

    monkeypatch.setattr(canonical_v2_module, "iter_dataset", forbid_source_restage)
    resumed = ContextualCanonicalBuilder(
        source,
        context,
        ContextualBuildConfig(prefer_parquet=False, resume=True),
    )
    resumed._stage_steps(resumed.route_database)
    manifest = json.loads(resumed.identity_assignment_manifest.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["unassigned_steps"] == 0
    assert manifest["assignment_failures"] == 0


def _sleeping_mcs_worker(connection, reaction, min_coverage, rdkit_timeout_seconds):
    import time

    time.sleep(5)
    connection.close()


class _LengthControlledTokenizer:
    model_max_length = 512

    def tokenize(self, reaction):
        # Canonicalized CCN reaction is the exceptional fixture.
        size = 600 if "CCN" in reaction else 510
        return ["tok"] * size

    def num_special_tokens_to_add(self, pair=False):
        return 2


class _GuardedFakeMapper:
    def __init__(self):
        self.tokenizer = _LengthControlledTokenizer()
        self.calls: list[list[str]] = []

    def get_attention_guided_atom_maps(self, reactions):
        self.calls.append(list(reactions))
        assert all("CCN" not in reaction for reaction in reactions), "oversized record reached RXNMapper"
        return [
            {
                "confidence": 0.95,
                "mapped_rxn": "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
            }
            for _ in reactions
        ]


def test_rxnmapper_token_guard_accepts_512_and_rejects_513_without_inference():
    from reacts.mapping.backends import RXNMapperBackend

    mapper = _GuardedFakeMapper()
    backend = RXNMapperBackend(mapper=mapper, max_token_length=512)
    results = backend.map_batch(["CCO>>CC=O", "CCN>>CC=N"])

    assert results[0].status.value == "mapped"
    assert results[0].rxnmapper_token_count == 512
    assert results[0].rxnmapper_eligible is True
    assert results[1].status.value == "failed"
    assert results[1].error_code == "rxnmapper_sequence_too_long"
    assert results[1].rxnmapper_token_count == 602
    assert results[1].rxnmapper_token_limit == 512
    assert results[1].rxnmapper_eligible is False
    assert mapper.calls == [["CCO>>CC=O"]]


def test_mcs_process_timeout_is_hard_and_deterministic():
    import time

    from reacts.mapping.backends import MCSBackend

    backend = MCSBackend(
        process_timeout_seconds=1,
        worker_target=_sleeping_mcs_worker,
    )
    started = time.perf_counter()
    result = backend.map_batch(["CCO>>CC=O"])[0]
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert result.status.value == "failed"
    assert result.error_code == "mcs_timeout"
    assert result.fallback_status == "timeout"


def test_exceptional_record_is_quarantined_without_blocking_normal_commit(tmp_path, monkeypatch):
    from reacts.mapping.backends import MCSBackend as RealMCSBackend

    context = tmp_path / "context"
    _context_fixture(
        context,
        [("s1", "r1", "CCO>>CC=O"), ("s2", "r2", "CCN>>CC=N")],
    )
    mapper = _GuardedFakeMapper()
    preflight = MappingPreflight(
        backend_requested="rxnmapper",
        primary_backend="rxnmapper",
        fallback_backend="mcs_fallback",
        rxnmapper_available=True,
        rxnmapper_version="fixture",
        setuptools_version="fixture",
        device="cpu",
        initialization_status="successful",
    )
    monkeypatch.setattr(
        "reacts.mapping.runner.resolve_mapping_backend",
        lambda *args, **kwargs: (preflight, mapper),
    )
    monkeypatch.setattr(
        "reacts.mapping.runner.MCSBackend",
        lambda **kwargs: RealMCSBackend(
            min_coverage=kwargs.get("min_coverage", 0.6),
            timeout_seconds=kwargs.get("timeout_seconds", 3),
            process_timeout_seconds=1,
            worker_target=_sleeping_mcs_worker,
        ),
    )

    config = MappingRunConfig(
        context_dir=context,
        output_dir=tmp_path / "mapping",
        queue_db=tmp_path / "state" / "mapping.sqlite3",
        reports_dir=tmp_path / "reports",
        backend="rxnmapper",
        fallback_backend="mcs",
        batch_size=2,
        shard_size=100,
        fallback_process_timeout_seconds=1,
        resume=False,
        prefer_parquet=False,
    )
    manifest = ResumableMappingRunner(config).run()

    assert manifest["queue_summary"]["mapped"] == 1
    assert manifest["queue_summary"]["failed"] == 1
    metrics = manifest["exceptional_mapping_metrics"]
    assert metrics["rxnmapper_sequence_too_long"] == 1
    assert metrics["mcs_fallback_timeout"] == 1
    assert metrics["mapping_exception_quarantined"] == 1

    queue = MappingQueue(config.queue_db)
    with queue.connection() as connection:
        normal = connection.execute(
            "SELECT output_shard, mapping_status FROM reaction_mapping_queue WHERE step_id='s1'"
        ).fetchone()
        exceptional = connection.execute(
            """
            SELECT output_shard, mapping_status, validation_status, error_code,
                   rxnmapper_token_count, rxnmapper_eligible, fallback_status
            FROM reaction_mapping_queue WHERE step_id='s2'
            """
        ).fetchone()
    assert normal["mapping_status"] == "mapped"
    assert exceptional["mapping_status"] == "failed"
    assert exceptional["validation_status"] == "quarantined"
    assert exceptional["error_code"] == "mcs_timeout"
    assert exceptional["rxnmapper_token_count"] == 602
    assert exceptional["rxnmapper_eligible"] == 0
    assert exceptional["fallback_status"] == "timeout"
    assert normal["output_shard"] != exceptional["output_shard"]
    assert (config.output_dir / "reaction_mapping_exceptions").exists()

    # Quarantined deterministic failures are terminal and are not reclaimed.
    config.resume = True
    second = ResumableMappingRunner(config).run()
    assert second["queue_summary"] == manifest["queue_summary"]
    assert len(mapper.calls) == 1


def test_v205_queue_is_migrated_in_place_and_running_rows_are_recoverable(tmp_path):
    import sqlite3

    database = tmp_path / "product_two_mapping.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE reaction_mapping_queue (
                step_id TEXT PRIMARY KEY,
                route_id TEXT NOT NULL,
                source_step_id TEXT,
                source_route_id TEXT,
                reaction_smiles TEXT NOT NULL,
                reaction_signature TEXT NOT NULL,
                eligibility_status TEXT NOT NULL,
                mapping_status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                backend_requested TEXT,
                backend_used TEXT,
                confidence REAL,
                validation_status TEXT,
                scientific_eligibility INTEGER NOT NULL DEFAULT 0,
                error_code TEXT,
                error_message TEXT,
                started_at TEXT,
                completed_at TEXT,
                output_shard TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO reaction_mapping_queue (
                step_id, route_id, reaction_smiles, reaction_signature,
                eligibility_status, mapping_status, attempt_count,
                started_at, created_at, updated_at
            ) VALUES (
                's1', 'r1', 'CCN>>CC=N', 'sig1',
                'eligible', 'running', 1,
                '2026-07-24T00:00:00+00:00',
                '2026-07-24T00:00:00+00:00',
                '2026-07-24T00:00:00+00:00'
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    queue = MappingQueue(database)
    with queue.connection() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(reaction_mapping_queue)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(reaction_mapping_queue)")}
    assert {
        "rxnmapper_token_count",
        "rxnmapper_token_limit",
        "rxnmapper_eligible",
        "fallback_status",
        "fallback_attempt_count",
        "exceptional_reason",
    }.issubset(columns)
    assert "idx_mapping_queue_rxn_eligible" in indexes
    assert "idx_mapping_queue_fallback_status" in indexes
    assert queue.recover_stale(0) == 1
    claimed = queue.claim_batch(1, backend_requested="rxnmapper", max_attempts=2)
    assert [item.step_id for item in claimed] == ["s1"]


def test_resume_preserves_committed_v205_shards_and_recovers_active_rows(tmp_path, monkeypatch):
    from reacts.mapping.backends import MCSBackend as RealMCSBackend

    context = tmp_path / "context"
    _context_fixture(
        context,
        [
            ("s0", "r0", "CCC>>CC=C"),
            ("s1", "r1", "CCN>>CC=N"),
            ("s2", "r2", "CCO>>CC=O"),
        ],
    )
    queue_db = tmp_path / "state" / "mapping.sqlite3"
    queue = MappingQueue(queue_db)
    assert queue.populate_from_context(context)["inserted"] == 3
    with queue.connection() as connection:
        connection.execute(
            """
            UPDATE reaction_mapping_queue
            SET mapping_status='mapped', backend_used='rxnmapper', confidence=0.9,
                validation_status='passed', scientific_eligibility=1,
                output_shard='part-00006', completed_at='2026-07-24T06:00:00+00:00'
            WHERE step_id='s0'
            """
        )
        connection.execute(
            """
            UPDATE reaction_mapping_queue
            SET mapping_status='running', attempt_count=1,
                started_at='2026-07-24T06:30:00+00:00'
            WHERE step_id='s1'
            """
        )

    output_dir = tmp_path / "mapping"
    output_dir.mkdir(parents=True)
    (output_dir / "mapping_manifest.json").write_text(
        json.dumps({"last_completed_shard": 6}),
        encoding="utf-8",
    )

    mapper = _GuardedFakeMapper()
    preflight = MappingPreflight(
        backend_requested="rxnmapper",
        primary_backend="rxnmapper",
        fallback_backend="mcs_fallback",
        rxnmapper_available=True,
        rxnmapper_version="fixture",
        setuptools_version="fixture",
        device="cpu",
        initialization_status="successful",
    )
    monkeypatch.setattr(
        "reacts.mapping.runner.resolve_mapping_backend",
        lambda *args, **kwargs: (preflight, mapper),
    )
    monkeypatch.setattr(
        "reacts.mapping.runner.MCSBackend",
        lambda **kwargs: RealMCSBackend(
            min_coverage=kwargs.get("min_coverage", 0.6),
            timeout_seconds=kwargs.get("timeout_seconds", 3),
            process_timeout_seconds=1,
            worker_target=_sleeping_mcs_worker,
        ),
    )

    manifest = ResumableMappingRunner(
        MappingRunConfig(
            context_dir=context,
            output_dir=output_dir,
            queue_db=queue_db,
            reports_dir=tmp_path / "reports",
            backend="rxnmapper",
            fallback_backend="mcs",
            batch_size=8,
            shard_size=5000,
            fallback_process_timeout_seconds=1,
            resume=True,
            prefer_parquet=False,
        )
    ).run()

    assert manifest["queue_summary"]["mapped"] == 2
    assert manifest["queue_summary"]["failed"] == 1
    assert manifest["last_completed_shard"] == 8
    with MappingQueue(queue_db).connection() as connection:
        rows = {
            row["step_id"]: row
            for row in connection.execute(
                """
                SELECT step_id, mapping_status, output_shard, validation_status,
                       rxnmapper_eligible, fallback_status
                FROM reaction_mapping_queue
                ORDER BY step_id
                """
            )
        }
    assert rows["s0"]["mapping_status"] == "mapped"
    assert rows["s0"]["output_shard"] == "part-00006"
    assert rows["s1"]["mapping_status"] == "failed"
    assert rows["s1"]["output_shard"] == "part-00007"
    assert rows["s1"]["validation_status"] == "quarantined"
    assert rows["s1"]["rxnmapper_eligible"] == 0
    assert rows["s1"]["fallback_status"] == "timeout"
    assert rows["s2"]["mapping_status"] == "mapped"
    assert rows["s2"]["output_shard"] == "part-00008"
    assert mapper.calls == [["CCO>>CC=O"]]
