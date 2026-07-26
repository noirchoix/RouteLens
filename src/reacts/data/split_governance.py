from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reacts.data.parsing import stable_hash
from reacts.science.hashing import canonical_json_hash, hash_dataset_columns, hash_paths, sha256_file
from reacts.storage.tabular import DatasetWriter, iter_dataset, parquet_available

SPLIT_ALGORITHM = "patent_reaction_connected_components_v1"
SPLIT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}
SPLIT_NAMES = ("train", "val", "test")


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none", "null"} else text




def _dataset_columns(root: Path, dataset_name: str) -> set[str]:
    directory = Path(root) / dataset_name
    parquet_parts = sorted(directory.glob("*.parquet"))
    if parquet_parts:
        try:
            import pyarrow.parquet as pq

            return set(pq.read_schema(parquet_parts[0]).names)
        except (ImportError, OSError):
            return set(pd.read_parquet(parquet_parts[0]).columns)
    csv_parts = sorted(directory.glob("*.csv.gz"))
    if csv_parts:
        return set(pd.read_csv(csv_parts[0], nrows=0).columns)
    raise FileNotFoundError(f"No dataset parts found in {directory}")

def _signature(row: dict[str, Any]) -> str | None:
    existing = _clean(row.get("reaction_signature"))
    if existing:
        return existing
    reaction = _clean(row.get("canonical_resolved_reaction_smiles")) or _clean(
        row.get("canonical_reaction_smiles")
    )
    return stable_hash(reaction, 40) if reaction else None


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


@dataclass(frozen=True)
class SplitAssignment:
    split: str
    component_id: str
    component_rows: int
    component_routes: int


class ProductTwoSplitRebuilder:
    def __init__(
        self,
        canonical_dir: Path,
        *,
        random_seed: int = 42,
        prefer_parquet: bool = True,
    ) -> None:
        self.canonical_dir = Path(canonical_dir)
        self.random_seed = int(random_seed)
        self.prefer_parquet = bool(prefer_parquet and parquet_available())

    def _build_components(self) -> tuple[dict[str, SplitAssignment], dict[str, Any]]:
        union = _UnionFind()
        patent_owner: dict[str, str] = {}
        signature_owner: dict[str, str] = {}
        route_rows: Counter[str] = Counter()
        row_count = 0
        columns = [
            "route_id",
            "route_instance_id",
            "patent_document_id",
            "canonical_resolved_reaction_smiles",
            "canonical_reaction_smiles",
        ]
        if "reaction_signature" in _dataset_columns(self.canonical_dir, "steps"):
            columns.append("reaction_signature")
        for chunk in iter_dataset(self.canonical_dir, "steps", columns=columns):
            for row in chunk.to_dict(orient="records"):
                route_id = _clean(row.get("route_instance_id")) or _clean(row.get("route_id"))
                if not route_id:
                    raise ValueError("Every Product Two step must have a route instance identifier.")
                union.add(route_id)
                route_rows[route_id] += 1
                row_count += 1
                patent = _clean(row.get("patent_document_id"))
                signature = _signature(row)
                if patent:
                    owner = patent_owner.setdefault(patent, route_id)
                    union.union(route_id, owner)
                if signature:
                    owner = signature_owner.setdefault(signature, route_id)
                    union.union(route_id, owner)

        component_routes: dict[str, list[str]] = defaultdict(list)
        component_rows: Counter[str] = Counter()
        for route_id, rows in route_rows.items():
            root = union.find(route_id)
            component_routes[root].append(route_id)
            component_rows[root] += rows

        components: list[dict[str, Any]] = []
        for root, routes in component_routes.items():
            ordered_routes = sorted(routes)
            digest = hashlib.sha256()
            for route_id in ordered_routes:
                digest.update(route_id.encode("utf-8"))
                digest.update(b"\n")
            component_id = f"splitcc_{digest.hexdigest()[:24]}"
            components.append(
                {
                    "root": root,
                    "component_id": component_id,
                    "rows": int(component_rows[root]),
                    "routes": ordered_routes,
                    "route_count": len(ordered_routes),
                    "tie": hashlib.sha256(
                        f"{self.random_seed}|{component_id}".encode("utf-8")
                    ).hexdigest(),
                }
            )

        components.sort(key=lambda item: (-item["rows"], item["tie"], item["component_id"]))
        targets = {name: row_count * fraction for name, fraction in SPLIT_FRACTIONS.items()}
        assigned_rows: Counter[str] = Counter()
        component_counts: Counter[str] = Counter()
        assignments: dict[str, SplitAssignment] = {}

        for component in components:
            size = component["rows"]

            def score(split: str) -> tuple[float, float, str]:
                target = max(targets[split], 1.0)
                projected_ratio = (assigned_rows[split] + size) / target
                current_ratio = assigned_rows[split] / target
                tie = hashlib.sha256(
                    f"{self.random_seed}|{component['component_id']}|{split}".encode("utf-8")
                ).hexdigest()
                return projected_ratio, current_ratio, tie

            selected = min(SPLIT_NAMES, key=score)
            assigned_rows[selected] += size
            component_counts[selected] += 1
            assignment = SplitAssignment(
                split=selected,
                component_id=component["component_id"],
                component_rows=size,
                component_routes=component["route_count"],
            )
            for route_id in component["routes"]:
                assignments[route_id] = assignment

        # Release owner maps before the verification pass; the full corpus contains
        # hundreds of thousands of exact-reaction keys.
        del patent_owner, signature_owner, union
        patent_splits: dict[str, str] = {}
        signature_splits: dict[str, str] = {}
        route_splits: dict[str, str] = {}
        patent_conflicts: set[str] = set()
        signature_conflicts: set[str] = set()
        route_conflicts: set[str] = set()
        for chunk in iter_dataset(self.canonical_dir, "steps", columns=columns):
            for row in chunk.to_dict(orient="records"):
                route_id = _clean(row.get("route_instance_id")) or _clean(row.get("route_id"))
                assignment = assignments[str(route_id)]
                route_key = str(route_id)
                prior_route_split = route_splits.setdefault(route_key, assignment.split)
                if prior_route_split != assignment.split:
                    route_conflicts.add(route_key)
                patent = _clean(row.get("patent_document_id"))
                signature = _signature(row)
                if patent:
                    prior = patent_splits.setdefault(patent, assignment.split)
                    if prior != assignment.split:
                        patent_conflicts.add(patent)
                if signature:
                    prior = signature_splits.setdefault(signature, assignment.split)
                    if prior != assignment.split:
                        signature_conflicts.add(signature)

        patent_overlap = len(patent_conflicts)
        signature_overlap = len(signature_conflicts)
        route_overlap = len(route_conflicts)
        if patent_overlap or signature_overlap or route_overlap:
            raise RuntimeError(
                "Connected-component split construction failed invariants: "
                f"patent={patent_overlap}, reaction_signature={signature_overlap}, route={route_overlap}"
            )

        assignment_digest = hashlib.sha256()
        for route_id, assignment in sorted(assignments.items()):
            assignment_digest.update(
                canonical_json_hash(
                    {
                        "route_instance_id": route_id,
                        "split": assignment.split,
                        "split_component_id": assignment.component_id,
                        "split_component_rows": assignment.component_rows,
                        "split_component_routes": assignment.component_routes,
                    }
                ).encode("ascii")
            )
        assignment_sha = assignment_digest.hexdigest()
        report = {
            "algorithm": SPLIT_ALGORITHM,
            "seed": self.random_seed,
            "fractions": SPLIT_FRACTIONS,
            "rows": row_count,
            "routes": len(route_rows),
            "components": len(components),
            "component_counts": dict(component_counts),
            "target_rows": {name: int(round(value)) for name, value in targets.items()},
            "actual_rows": {name: int(assigned_rows[name]) for name in SPLIT_NAMES},
            "assignment_sha256": assignment_sha,
            "invariants": {
                "patent_document_id_overlapping_keys": int(patent_overlap),
                "reaction_signature_overlapping_keys": int(signature_overlap),
                "route_split_conflicts": int(route_overlap),
                "strict_pass": patent_overlap == signature_overlap == route_overlap == 0,
            },
            "largest_component_rows": max((item["rows"] for item in components), default=0),
            "largest_component_routes": max((item["route_count"] for item in components), default=0),
        }
        return assignments, report

    def _stage_dataset(
        self,
        staging_root: Path,
        dataset_name: str,
        assignments: dict[str, SplitAssignment],
    ) -> list[Path]:
        writer = DatasetWriter(
            staging_root,
            dataset_name,
            prefer_parquet=self.prefer_parquet,
        )
        outputs: list[Path] = []
        for chunk in iter_dataset(self.canonical_dir, dataset_name):
            route_column = "route_instance_id" if "route_instance_id" in chunk.columns else "route_id"
            route_ids = chunk[route_column].fillna(chunk.get("route_id", "")).astype(str)
            missing = sorted({route_id for route_id in route_ids if route_id not in assignments})
            if missing:
                raise KeyError(f"Split assignment missing {len(missing)} routes; examples={missing[:5]}")
            chunk = chunk.copy()
            if dataset_name in {"steps", "mapping_candidates"}:
                existing = (
                    chunk["reaction_signature"]
                    if "reaction_signature" in chunk.columns
                    else pd.Series([None] * len(chunk), index=chunk.index)
                )
                reactions = []
                for row in chunk.to_dict(orient="records"):
                    reaction = (
                        _clean(row.get("canonical_resolved_reaction_smiles"))
                        or _clean(row.get("canonical_reaction_smiles"))
                        or _clean(row.get("reaction_smiles"))
                    )
                    reactions.append(stable_hash(reaction, 40) if reaction else None)
                derived = pd.Series(reactions, index=chunk.index)
                chunk["reaction_signature"] = existing.where(
                    existing.map(_clean).notna(), derived
                )
            chunk["split"] = route_ids.map(lambda route_id: assignments[route_id].split)
            chunk["split_component_id"] = route_ids.map(
                lambda route_id: assignments[route_id].component_id
            )
            chunk["split_component_rows"] = route_ids.map(
                lambda route_id: assignments[route_id].component_rows
            )
            chunk["split_component_routes"] = route_ids.map(
                lambda route_id: assignments[route_id].component_routes
            )
            chunk["split_algorithm"] = SPLIT_ALGORITHM
            chunk["split_seed"] = self.random_seed
            outputs.append(writer.write(chunk))
        return outputs

    def _swap_staged(self, staging_root: Path, dataset_names: tuple[str, ...]) -> None:
        swaps: list[tuple[Path, Path]] = []
        try:
            for name in dataset_names:
                target = self.canonical_dir / name
                staged = staging_root / name
                backup = self.canonical_dir / f".{name}.pre_v209"
                if not target.exists():
                    raise FileNotFoundError(f"Canonical dataset directory is missing: {target}")
                if not staged.exists():
                    raise FileNotFoundError(f"Staged dataset directory is missing: {staged}")
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(target, backup)
                swaps.append((target, backup))
                os.replace(staged, target)
        except Exception:
            for target, backup in reversed(swaps):
                if target.exists():
                    shutil.rmtree(target)
                if backup.exists():
                    os.replace(backup, target)
            raise
        else:
            for _, backup in swaps:
                shutil.rmtree(backup, ignore_errors=True)

    def _update_manifest(self, split_report: dict[str, Any]) -> dict[str, Any]:
        manifest_path = self.canonical_dir / "dataset_manifest.json"
        parent_final_manifest_sha256 = (
            sha256_file(manifest_path) if manifest_path.exists() else None
        )
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"dataset_version": "uspto_multistep_contextual_v2"}
        )
        data_root = self.canonical_dir.parent
        mapping_manifest = data_root / "mapping_v2" / "mapping_manifest.json"
        derivation_manifest = data_root / "derivation_v2" / "derivation_manifest.json"
        split_manifest = {
            "schema_version": "2.0.9-split-governance-v1",
            "dataset_version": manifest.get("dataset_version", "uspto_multistep_contextual_v2"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_final_manifest_sha256": parent_final_manifest_sha256,
            "upstream_manifests": {
                "mapping_manifest_sha256": (
                    sha256_file(mapping_manifest) if mapping_manifest.exists() else None
                ),
                "derivation_manifest_sha256": (
                    sha256_file(derivation_manifest) if derivation_manifest.exists() else None
                ),
            },
            **split_report,
            "contract": {
                "patent_document_id_is_grouped": True,
                "reaction_signature_is_grouped": True,
                "route_instance_is_atomic": True,
                "product_scaffold_is_diagnostic_only": True,
                "mapping_outputs_were_not_rebuilt": True,
                "derivation_outputs_were_not_rebuilt": True,
            },
        }
        split_path = self.canonical_dir / "split_manifest.json"
        temporary_split = split_path.with_suffix(".json.tmp")
        temporary_split.write_text(
            json.dumps(split_manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(temporary_split, split_path)
        split_sha = sha256_file(split_path)
        manifest["split_governance"] = {
            "algorithm": SPLIT_ALGORITHM,
            "seed": self.random_seed,
            "manifest_path": "split_manifest.json",
            "manifest_sha256": split_sha,
            "assignment_sha256": split_report["assignment_sha256"],
            "actual_rows": split_report["actual_rows"],
            "components": split_report["components"],
            "invariants": split_report["invariants"],
        }
        manifest.setdefault("contract", {})["exact_reaction_signature_leakage_is_forbidden"] = True
        manifest["contract"]["product_scaffold_overlap_is_diagnostic_only"] = True
        manifest.setdefault("reproducibility", {})["split_manifest_sha256"] = split_sha
        files = [
            path
            for path in self.canonical_dir.rglob("*")
            if path.is_file() and path != manifest_path
        ]
        manifest["reproducibility"]["output_tree_sha256"] = (
            hash_paths(files, root=self.canonical_dir) if files else None
        )
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        return {
            "split_manifest": split_manifest,
            "split_manifest_sha256": split_sha,
            "dataset_manifest_sha256": sha256_file(manifest_path),
        }

    def run(self) -> dict[str, Any]:
        if not (self.canonical_dir / "steps").exists():
            raise FileNotFoundError(
                "Product Two final canonical dataset is not materialized. "
                "Reaction-centre derivation must complete before split rebuilding."
            )
        assignments, report = self._build_components()
        staging_root = self.canonical_dir / ".split_v209_staging"
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        dataset_names = ["steps", "routes"]
        if (self.canonical_dir / "mapping_candidates").exists():
            dataset_names.append("mapping_candidates")
        selected_datasets = tuple(dataset_names)
        try:
            staged_outputs = {
                name: self._stage_dataset(staging_root, name, assignments)
                for name in selected_datasets
            }
            self._swap_staged(staging_root, selected_datasets)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
        training_split_sha = hash_dataset_columns(
            self.canonical_dir,
            "steps",
            [
                "step_id",
                "patent_document_id",
                "reaction_signature",
                "split_component_id",
                "split",
            ],
        )
        report["training_split_sha256"] = training_split_sha
        manifests = self._update_manifest(report)
        return {
            "stage": "product_two_connected_component_resplit",
            "dataset_version": "uspto_multistep_contextual_v2",
            "canonical_dir": str(self.canonical_dir),
            "mapping_rebuilt": False,
            "derivation_rebuilt": False,
            "training_split_sha256": training_split_sha,
            "split_governance": report,
            "outputs": {
                name: [path.name for path in paths]
                for name, paths in staged_outputs.items()
            },
            **manifests,
        }
