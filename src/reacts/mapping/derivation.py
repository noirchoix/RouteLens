from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from reacts.chemistry.mapping import derive_reaction_centre
from reacts.science.hashing import canonical_json_hash, hash_paths, sha256_file
from reacts.storage.tabular import DatasetWriter, iter_dataset, parquet_available

LOGGER = logging.getLogger(__name__)


@dataclass
class DerivationConfig:
    context_dir: Path
    mapping_dir: Path
    derivation_dir: Path
    final_canonical_dir: Path
    queue_db: Path
    dataset_version: str = "uspto_multistep_contextual_v2"
    min_confidence: float = 0.50
    include_mcs: bool = False
    batch_size: int = 256
    shard_size: int = 5_000
    resume: bool = False
    prefer_parquet: bool = True
    max_rows: int | None = None
    max_attempts: int = 2


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(directory: Path, shard_id: int, records: list[dict[str, Any]], prefer_parquet: bool) -> Path | None:
    if not records:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    use_parquet = prefer_parquet and parquet_available()
    suffix = ".parquet" if use_parquet else ".csv.gz"
    target = directory / f"part-{shard_id:05d}{suffix}"
    temporary = directory / f".{target.name}.tmp"
    frame = pd.DataFrame.from_records(records)
    if use_parquet:
        frame.to_parquet(temporary, index=False, compression="zstd")
        rows = len(pd.read_parquet(temporary, columns=["step_id"]))
    else:
        serial = frame.copy()
        for column in [
            "formed_bonds",
            "broken_bonds",
            "changed_bonds",
            "changed_atom_maps",
            "atom_environment_changes",
        ]:
            if column in serial:
                serial[column] = serial[column].map(lambda value: json.dumps(value, ensure_ascii=False, default=str))
        serial.to_csv(temporary, index=False, compression="gzip")
        rows = len(pd.read_csv(temporary, usecols=["step_id"], compression="gzip"))
    if rows != len(frame):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Derivation shard validation failed for {target}")
    os.replace(temporary, target)
    return target


class DerivationQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=60)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS derivation_queue (
                    step_id TEXT PRIMARY KEY,
                    route_id TEXT NOT NULL,
                    source_step_id TEXT,
                    source_route_id TEXT,
                    mapped_reaction_smiles TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    reaction_family TEXT,
                    reaction_centre_fingerprint TEXT,
                    reaction_template TEXT,
                    error_message TEXT,
                    output_shard TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_derivation_status ON derivation_queue(status);
                CREATE INDEX IF NOT EXISTS idx_derivation_route ON derivation_queue(route_id);
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(derivation_queue)").fetchall()}
            if "source_step_id" not in columns:
                conn.execute("ALTER TABLE derivation_queue ADD COLUMN source_step_id TEXT")
            if "source_route_id" not in columns:
                conn.execute("ALTER TABLE derivation_queue ADD COLUMN source_route_id TEXT")

    def reset(self) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM derivation_queue")

    def populate(self, mapping_dir: Path, *, min_confidence: float, include_mcs: bool) -> dict[str, int]:
        sources = [("reaction_mappings_rxnmapper", True)]
        if include_mcs:
            sources.append(("reaction_mappings_mcs_fallback", False))
        inserted = 0
        seen = 0
        with self.connection() as conn:
            for dataset_name, strict_source in sources:
                try:
                    chunks = iter_dataset(mapping_dir, dataset_name)
                    for chunk in chunks:
                        for row in chunk.to_dict(orient="records"):
                            seen += 1
                            status = str(row.get("mapping_status"))
                            confidence = float(row.get("confidence") or 0.0)
                            validation = str(row.get("validation_status"))
                            mapped = row.get("mapped_reaction_smiles")
                            eligible = (
                                status in {"mapped", "existing"}
                                and validation == "passed"
                                and confidence >= min_confidence
                                and bool(mapped)
                            )
                            if not eligible:
                                continue
                            before = conn.total_changes
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO derivation_queue(
                                    step_id, route_id, source_step_id, source_route_id,
                                    mapped_reaction_smiles, backend, confidence, status, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                                """,
                                (
                                    str(row.get("step_instance_id") or row["step_id"]),
                                    str(row.get("route_instance_id") or row["route_id"]),
                                    str(row.get("source_step_id") or row["step_id"]),
                                    str(row.get("source_route_id") or row["route_id"]),
                                    str(mapped),
                                    str(row.get("backend") or dataset_name),
                                    confidence,
                                    _utcnow(),
                                ),
                            )
                            inserted += conn.total_changes - before
                except FileNotFoundError:
                    continue
        return {"seen": seen, "inserted": inserted}

    def claim(self, limit: int, *, max_attempts: int) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM derivation_queue
                WHERE status='pending' OR (status='failed' AND attempt_count < ?)
                ORDER BY step_id LIMIT ?
                """,
                (max_attempts, limit),
            ).fetchall()
            if not rows:
                conn.execute("COMMIT")
                return []
            ids = [str(row["step_id"]) for row in rows]
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE derivation_queue SET status='running', attempt_count=attempt_count+1, updated_at=? WHERE step_id IN ({placeholders})",
                (_utcnow(), *ids),
            )
            conn.execute("COMMIT")
            return [dict(row) for row in rows]
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def complete(self, rows: list[dict[str, Any]], shard_name: str) -> None:
        with self.connection() as conn:
            conn.executemany(
                """
                UPDATE derivation_queue
                SET status=?, reaction_family=?, reaction_centre_fingerprint=?,
                    reaction_template=?, error_message=?, output_shard=?, updated_at=?
                WHERE step_id=?
                """,
                [
                    (
                        row["derivation_status"],
                        row.get("reaction_family"),
                        row.get("reaction_centre_fingerprint"),
                        row.get("reaction_template"),
                        row.get("error_message"),
                        shard_name,
                        _utcnow(),
                        row["step_id"],
                    )
                    for row in rows
                ],
            )

    def summary(self) -> dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute("SELECT status, COUNT(*) n FROM derivation_queue GROUP BY status").fetchall()
            output = {str(row["status"]): int(row["n"]) for row in rows}
            output["total"] = sum(output.values())
            return output

    def lookup(self, step_ids: list[str]) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        with self.connection() as conn:
            for offset in range(0, len(step_ids), 900):
                batch = step_ids[offset : offset + 900]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT * FROM derivation_queue WHERE step_id IN ({placeholders})",
                    batch,
                ).fetchall()
                output.update({str(row["step_id"]): dict(row) for row in rows})
        return output


class ReactionCentreDeriver:
    CONTEXT_TABLES = [
        "molecules",
        "step_molecules",
        "route_edges",
        "intermediate_resolution",
        "condition_evidence",
        "repair_candidates",
        "mapping_candidates",
        "quarantine",
    ]

    def __init__(self, config: DerivationConfig):
        self.config = config
        self.queue = DerivationQueue(config.queue_db)
        self.buffer: list[dict[str, Any]] = []
        self.shard_id = 0
        self.manifest_path = self.config.derivation_dir / "derivation_manifest.json"

    def _prepare(self) -> None:
        self.config.derivation_dir.mkdir(parents=True, exist_ok=True)
        if not self.config.resume:
            for name in ["reaction_mappings", "reaction_centres", "reaction_families", "reaction_templates", "rejected_derivations"]:
                target = self.config.derivation_dir / name
                if target.exists():
                    shutil.rmtree(target)
            self.queue.reset()
            self.manifest_path.unlink(missing_ok=True)
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.shard_id = int(manifest.get("last_completed_shard", -1)) + 1
        self.queue.populate(
            self.config.mapping_dir,
            min_confidence=self.config.min_confidence,
            include_mcs=self.config.include_mcs,
        )

    def _derive(self, row: dict[str, Any]) -> dict[str, Any]:
        base = {
            "dataset_version": self.config.dataset_version,
            "step_id": str(row["step_id"]),
            "step_instance_id": str(row["step_id"]),
            "source_step_id": str(row.get("source_step_id") or row["step_id"]),
            "route_id": str(row["route_id"]),
            "route_instance_id": str(row["route_id"]),
            "source_route_id": str(row.get("source_route_id") or row["route_id"]),
            "mapped_reaction_smiles": str(row["mapped_reaction_smiles"]),
            "backend": str(row["backend"]),
            "mapping_confidence": float(row["confidence"]),
        }
        try:
            centre = derive_reaction_centre(str(row["mapped_reaction_smiles"]))
            return {
                **base,
                "derivation_status": "derived",
                "formed_bonds": list(centre.formed_bonds),
                "broken_bonds": list(centre.broken_bonds),
                "changed_bonds": list(centre.changed_bonds),
                "changed_atom_maps": list(centre.changed_atom_maps),
                "atom_environment_changes": list(centre.atom_environment_changes),
                "reaction_centre_fingerprint": centre.fingerprint,
                "reaction_template": centre.reaction_template,
                "reaction_family": centre.structural_family,
                "family_method": "mapped_bond_change_rules_v2",
                "family_confidence": float(row["confidence"]),
                "family_cluster_id": f"{centre.structural_family}:{centre.fingerprint[:12]}",
                "error_message": None,
            }
        except Exception as exc:
            return {
                **base,
                "derivation_status": "failed",
                "reaction_centre_fingerprint": None,
                "reaction_template": None,
                "reaction_family": None,
                "error_message": f"{type(exc).__name__}: {exc}",
            }

    def _flush(self) -> None:
        if not self.buffer:
            return
        derived = [row for row in self.buffer if row["derivation_status"] == "derived"]
        rejected = [row for row in self.buffer if row["derivation_status"] != "derived"]
        mappings = [
            {
                "dataset_version": row["dataset_version"],
                "step_id": row["step_id"],
                "step_instance_id": row["step_instance_id"],
                "source_step_id": row["source_step_id"],
                "route_id": row["route_id"],
                "route_instance_id": row["route_instance_id"],
                "source_route_id": row["source_route_id"],
                "mapped_reaction_smiles": row["mapped_reaction_smiles"],
                "backend": row["backend"],
                "confidence": row["mapping_confidence"],
                "mapping_status": "mapped",
                "validation_status": "passed",
                "scientific_eligibility": row["backend"] in {"rxnmapper", "existing"},
            }
            for row in derived
        ]
        centres = [
            {
                key: row[key]
                for key in [
                    "dataset_version", "step_id", "step_instance_id", "source_step_id",
                    "route_id", "route_instance_id", "source_route_id", "formed_bonds", "broken_bonds",
                    "changed_bonds", "changed_atom_maps", "atom_environment_changes",
                    "reaction_centre_fingerprint", "reaction_template",
                ]
            }
            for row in derived
        ]
        families = [
            {
                key: row[key]
                for key in [
                    "dataset_version", "step_id", "step_instance_id", "source_step_id",
                    "route_id", "route_instance_id", "source_route_id", "reaction_family", "family_method",
                    "family_confidence", "reaction_centre_fingerprint", "family_cluster_id",
                ]
            }
            for row in derived
        ]
        templates = [
            {
                "dataset_version": row["dataset_version"],
                "step_id": row["step_id"],
                "step_instance_id": row["step_instance_id"],
                "source_step_id": row["source_step_id"],
                "route_id": row["route_id"],
                "route_instance_id": row["route_instance_id"],
                "source_route_id": row["source_route_id"],
                "reaction_template": row["reaction_template"],
                "reaction_family": row["reaction_family"],
                "reaction_centre_fingerprint": row["reaction_centre_fingerprint"],
            }
            for row in derived
        ]
        _atomic_write(self.config.derivation_dir / "reaction_mappings", self.shard_id, mappings, self.config.prefer_parquet)
        _atomic_write(self.config.derivation_dir / "reaction_centres", self.shard_id, centres, self.config.prefer_parquet)
        _atomic_write(self.config.derivation_dir / "reaction_families", self.shard_id, families, self.config.prefer_parquet)
        _atomic_write(self.config.derivation_dir / "reaction_templates", self.shard_id, templates, self.config.prefer_parquet)
        _atomic_write(self.config.derivation_dir / "rejected_derivations", self.shard_id, rejected, self.config.prefer_parquet)
        self.queue.complete(self.buffer, f"part-{self.shard_id:05d}")
        LOGGER.info("Committed reaction-centre shard %s (%s rows)", self.shard_id, len(self.buffer))
        self.buffer.clear()
        self.shard_id += 1
        self._write_manifest()

    def _write_manifest(self) -> dict[str, Any]:
        outputs = [path for path in self.config.derivation_dir.rglob("part-*") if path.is_file()]
        mapping_manifest = self.config.mapping_dir / "mapping_manifest.json"
        manifest = {
            "stage": "independent_reaction_centre_derivation",
            "dataset_version": self.config.dataset_version,
            "updated_at_utc": _utcnow(),
            "configuration": {
                key: (value.as_posix() if isinstance(value, Path) else value)
                for key, value in asdict(self.config).items()
            },
            "queue_summary": self.queue.summary(),
            "last_completed_shard": self.shard_id - 1,
            "reproducibility": {
                "mapping_manifest_sha256": sha256_file(mapping_manifest) if mapping_manifest.exists() else None,
                "outputs_sha256": hash_paths(outputs, root=self.config.derivation_dir) if outputs else None,
                "config_sha256": canonical_json_hash(
                    {
                        key: (value.as_posix() if isinstance(value, Path) else value)
                        for key, value in asdict(self.config).items()
                    }
                ),
            },
            "contract": {
                "mapping_and_derivation_are_independent": True,
                "default_training_population_is_rxnmapper_only": not self.config.include_mcs,
                "derivation_is_resumable": True,
            },
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temporary, self.manifest_path)
        return manifest

    @staticmethod
    def _link_or_copy_tree(source: Path, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if not path.is_file():
                continue
            destination = target / path.name
            try:
                os.link(path, destination)
            except OSError:
                shutil.copy2(path, destination)

    def _materialize_final(self) -> dict[str, Any]:
        final = self.config.final_canonical_dir
        if final.exists():
            shutil.rmtree(final)
        final.mkdir(parents=True, exist_ok=True)
        for name in self.CONTEXT_TABLES:
            source = self.config.context_dir / name
            if source.exists():
                self._link_or_copy_tree(source, final / name)
        for name in ["reaction_mappings", "reaction_centres", "reaction_families", "reaction_templates"]:
            source = self.config.derivation_dir / name
            if source.exists():
                self._link_or_copy_tree(source, final / name)

        step_writer = DatasetWriter(final, "steps", prefer_parquet=self.config.prefer_parquet)
        route_aggregates: dict[str, dict[str, Any]] = {}
        for chunk in iter_dataset(self.config.context_dir, "steps"):
            step_ids = chunk["step_id"].astype(str).tolist()
            lookup = self.queue.lookup(step_ids)
            records: list[dict[str, Any]] = []
            for row in chunk.to_dict(orient="records"):
                step_id = str(row["step_id"])
                mapped = lookup.get(step_id)
                components = row.get("quality_components")
                if not isinstance(components, dict):
                    components = {}
                mapping_component = 0.0
                if mapped and mapped.get("status") == "derived":
                    mapping_component = 1.0 if mapped.get("backend") in {"rxnmapper", "existing"} else 0.4
                components = {**components, "mapping": mapping_component}
                contextual_score = float(row.get("contextual_quality_score") or 0.0)
                final_score = 0.90 * contextual_score + 0.10 * mapping_component
                record = {
                    **row,
                    "dataset_version": self.config.dataset_version,
                    "mapping_status": "mapped" if mapped and mapped.get("status") == "derived" else row.get("mapping_status"),
                    "mapping_confidence": mapped.get("confidence") if mapped else None,
                    "mapping_backend": mapped.get("backend") if mapped else None,
                    "reaction_family": mapped.get("reaction_family") if mapped else None,
                    "reaction_centre_fingerprint": mapped.get("reaction_centre_fingerprint") if mapped else None,
                    "quality_components": components,
                    "contextual_quality_score": round(final_score, 6),
                    "eligible_mapping_models": bool(
                        mapped
                        and mapped.get("status") == "derived"
                        and mapped.get("backend") in {"rxnmapper", "existing"}
                    ),
                }
                records.append(record)
                route_id = str(row["route_id"])
                aggregate = route_aggregates.setdefault(route_id, {"mapped": 0, "families": {}})
                if record["eligible_mapping_models"]:
                    aggregate["mapped"] += 1
                    family = record.get("reaction_family")
                    if family:
                        aggregate["families"][family] = aggregate["families"].get(family, 0) + 1
            step_writer.write(pd.DataFrame.from_records(records))

        route_writer = DatasetWriter(final, "routes", prefer_parquet=self.config.prefer_parquet)
        for chunk in iter_dataset(self.config.context_dir, "routes"):
            records = []
            for row in chunk.to_dict(orient="records"):
                aggregate = route_aggregates.get(str(row["route_id"]), {"mapped": 0, "families": {}})
                records.append(
                    {
                        **row,
                        "dataset_version": self.config.dataset_version,
                        "mapped_steps": aggregate["mapped"],
                        "reaction_families": sorted(aggregate["families"]),
                        "family_distribution": aggregate["families"],
                    }
                )
            route_writer.write(pd.DataFrame.from_records(records))

        context_manifest = self.config.context_dir / "dataset_manifest.json"
        mapping_manifest = self.config.mapping_dir / "mapping_manifest.json"
        derivation_manifest = self.config.derivation_dir / "derivation_manifest.json"
        files = [path for path in final.rglob("*") if path.is_file()]
        manifest = {
            "dataset_version": self.config.dataset_version,
            "stage": "qualified_contextual_structural_dataset",
            "built_at_utc": _utcnow(),
            "metrics": self.queue.summary(),
            "reproducibility": {
                "context_manifest_sha256": sha256_file(context_manifest),
                "mapping_manifest_sha256": sha256_file(mapping_manifest),
                "derivation_manifest_sha256": sha256_file(derivation_manifest),
                "output_tree_sha256": hash_paths(files, root=final) if files else None,
            },
            "contract": {
                "contextualization_mapping_and_derivation_are_separate": True,
                "product_one_inputs_are_read_only": True,
                "mcs_fallback_is_excluded_from_strict_structural_training": not self.config.include_mcs,
                "final_steps_are_enriched_only_after_completed_derivation": True,
            },
        }
        manifest_path = final / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        return manifest

    def run(self) -> dict[str, Any]:
        self._prepare()
        processed = 0
        try:
            while True:
                if self.config.max_rows is not None and processed >= self.config.max_rows:
                    break
                limit = self.config.batch_size
                if self.config.max_rows is not None:
                    limit = min(limit, self.config.max_rows - processed)
                rows = self.queue.claim(limit, max_attempts=self.config.max_attempts)
                if not rows:
                    break
                derived = [self._derive(row) for row in rows]
                self.buffer.extend(derived)
                processed += len(derived)
                if len(self.buffer) >= self.config.shard_size:
                    self._flush()
        except KeyboardInterrupt:
            LOGGER.warning("Derivation interrupted; committing completed rows.")
            self._flush()
            raise
        self._flush()
        manifest = self._write_manifest()
        summary = self.queue.summary()
        if summary.get("pending", 0) == 0 and summary.get("running", 0) == 0:
            manifest["final_canonical"] = self._materialize_final()
        else:
            manifest["final_canonical"] = {
                "materialized": False,
                "reason": "derivation queue is incomplete",
            }
        return manifest
