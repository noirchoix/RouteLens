from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from reacts.chemistry.reactions import canonicalize_molecule, parse_reaction
from reacts.chemistry.repair import deterministic_repair_candidates
from reacts.chemistry.taxonomy import agent_family, catalyst_family, solvent_family
from reacts.context.route_resolution import ResolvedStep, SymbolicIntermediateResolver
from reacts.contracts import ResolutionStatus
from reacts.data.parsing import INTERMEDIATE_RE, parse_list, stable_hash
from reacts.science.hashing import canonical_json_hash, hash_paths, sha256_file
from reacts.storage.tabular import DatasetWriter, iter_dataset

LOGGER = logging.getLogger(__name__)


@dataclass
class ContextualBuildConfig:
    dataset_version: str = "uspto_multistep_contextual_v2"
    prefer_parquet: bool = True
    batch_rows: int = 10_000
    checkpoint_routes: int = 5_000
    resume: bool = False
    clean: bool = False
    preserve_work_database: bool = True
    # Compatibility fields retained for callers of v2.0.0/v2.0.1. Mapping is
    # deliberately disabled in the contextual stage as of v2.0.2.
    map_reactions: bool = False
    mapping_backend: str = "disabled_context_only"
    mapping_min_coverage: float = 0.60
    mapping_timeout_seconds: int = 3
    mapping_max_rows: int | None = None


class _BufferedWriter:
    def __init__(self, root: Path, name: str, prefer_parquet: bool, batch_rows: int, *, resume: bool = False):
        self.writer = DatasetWriter(root, name, prefer_parquet=prefer_parquet)
        self.batch_rows = batch_rows
        self.rows: list[dict[str, Any]] = []
        existing = sorted(self.writer.output_dir.glob("part-*")) if resume else []
        self.outputs: list[Path] = existing
        self.writer.part_index = len(existing)

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def extend(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        self.outputs.append(self.writer.write(pd.DataFrame.from_records(self.rows)))
        self.rows.clear()


class MoleculeCatalog:
    """Disk-backed, idempotent molecule catalogue suitable for resume runs."""

    def __init__(self, database: Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.database)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS molecules (
                molecule_id TEXT PRIMARY KEY,
                canonical_smiles TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def register(self, smiles: str) -> str | None:
        canonical = canonicalize_molecule(str(smiles))
        if not canonical or INTERMEDIATE_RE.fullmatch(canonical):
            return None
        existing = self.conn.execute(
            "SELECT molecule_id FROM molecules WHERE canonical_smiles=?", (canonical,)
        ).fetchone()
        if existing:
            return str(existing[0])
        molecule_id = f"mol_{stable_hash(canonical, 24)}"
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            return None
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = ""
        record = {
            "molecule_id": molecule_id,
            "canonical_smiles": canonical,
            "isomeric_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
            "inchikey": Chem.MolToInchiKey(mol),
            "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "ring_count": int(Lipinski.RingCount(mol)),
            "formal_charge": int(Chem.GetFormalCharge(mol)),
            "murcko_scaffold": scaffold,
        }
        self.conn.execute(
            "INSERT OR IGNORE INTO molecules(molecule_id, canonical_smiles, payload_json) VALUES (?, ?, ?)",
            (molecule_id, canonical, json.dumps(record, ensure_ascii=False)),
        )
        return molecule_id

    def commit(self) -> None:
        self.conn.commit()

    def iter_records(self, batch_size: int = 10_000) -> Iterator[list[dict[str, Any]]]:
        cursor = self.conn.execute("SELECT payload_json FROM molecules ORDER BY molecule_id")
        try:
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield [json.loads(row[0]) for row in rows]
        finally:
            cursor.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


class ContextualCanonicalBuilder:
    CONTEXT_SCHEMA_VERSION = "2.0.5-indexed-route-assignment-v1"
    REQUIRED_STEP_COLUMNS = [
        "dataset_version",
        "step_id",
        "route_id",
        "patent_document_id",
        "split",
        "step_index",
        "raw_reaction_text",
        "reaction_smiles",
        "canonical_reaction_smiles",
        "reactants",
        "products",
        "parse_ok",
        "parse_failure_class",
        "input_intermediate",
        "output_intermediate",
        "solvents",
        "agents",
        "solvent_primary",
        "agent_primary",
        "agent_present",
        "condition_extraction_method",
        "condition_extraction_confidence",
        "condition_numeric_tokens",
        "temperature_observed_c",
        "temperature_c",
        "temperature_valid",
        "temperature_bucket",
        "time_observed_h",
        "time_h",
        "time_valid",
        "time_bucket",
        "condition_status",
        "quality_issues",
        "quality_score",
    ]


    REQUIRED_ROUTE_COLUMNS = [
        "dataset_version",
        "route_uid",
        "route_id",
        "patent_document_id",
        "split",
        "source_content_hash",
        "multistep_reaction_text",
        "step_count",
    ]

    def __init__(self, canonical_v1_dir: Path, output_root: Path, config: ContextualBuildConfig | None = None):
        self.canonical_v1_dir = Path(canonical_v1_dir)
        self.output_root = Path(output_root)
        self.config = config or ContextualBuildConfig()
        self.metrics: Counter[str] = Counter()
        self.resolver = SymbolicIntermediateResolver()
        self.molecules: MoleculeCatalog | None = None
        self.progress_path = self.output_root / ".work" / "context_progress.json"
        self.route_database = self.output_root / ".work" / "route_context.sqlite3"
        self.molecule_database = self.output_root / ".work" / "molecule_catalog.sqlite3"
        self.identity_assignment_manifest = self.output_root / ".work" / "identity_assignment_manifest.json"

    CONTEXT_TABLES = [
        "steps",
        "routes",
        "molecules",
        "step_molecules",
        "route_edges",
        "intermediate_resolution",
        "condition_evidence",
        "repair_candidates",
        "mapping_candidates",
        "quarantine",
    ]

    def _prepare_output(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        work_root = self.output_root / ".work"
        work_root.mkdir(parents=True, exist_ok=True)
        if self.config.clean or not self.config.resume:
            for name in self.CONTEXT_TABLES:
                target = self.output_root / name
                if target.exists():
                    shutil.rmtree(target)
            for path in [
                self.progress_path,
                self.route_database,
                self.molecule_database,
                self.identity_assignment_manifest,
            ]:
                path.unlink(missing_ok=True)

    def _progress(self) -> dict[str, Any]:
        if not self.config.resume or not self.progress_path.exists():
            return {
                "context_schema_version": self.CONTEXT_SCHEMA_VERSION,
                "routes_completed": 0,
                "last_route_instance_id": None,
                "writer_part_counts": {},
            }
        payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        if payload.get("context_schema_version") != self.CONTEXT_SCHEMA_VERSION:
            return {
                "context_schema_version": self.CONTEXT_SCHEMA_VERSION,
                "routes_completed": 0,
                "last_route_instance_id": None,
                "writer_part_counts": {},
                "incompatible_previous_progress": True,
            }
        return payload

    def _reconcile_resume_outputs(self, progress: dict[str, Any]) -> None:
        counts = progress.get("writer_part_counts", {})
        for name in self.CONTEXT_TABLES:
            keep = int(counts.get(name, 0))
            directory = self.output_root / name
            if not directory.exists():
                continue
            parts = sorted(directory.glob("part-*"))
            for part in parts[keep:]:
                part.unlink(missing_ok=True)

    def _checkpoint(
        self,
        writers: dict[str, _BufferedWriter],
        *,
        last_route_instance_id: str,
        routes_completed: int,
    ) -> None:
        for name, writer in writers.items():
            if name != "molecules":
                writer.flush()
        if self.molecules is not None:
            self.molecules.commit()
        payload = {
            "context_schema_version": self.CONTEXT_SCHEMA_VERSION,
            "dataset_version": self.config.dataset_version,
            "routes_completed": routes_completed,
            "last_route_instance_id": last_route_instance_id,
            "writer_part_counts": {name: len(writer.outputs) for name, writer in writers.items()},
            "metrics": dict(self.metrics),
            "complete": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        temporary = self.progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.progress_path)
        LOGGER.info("Committed contextual checkpoint at %s routes", routes_completed)

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone() is not None

    @staticmethod
    def _read_build_metadata(conn: sqlite3.Connection) -> dict[str, str]:
        if not ContextualCanonicalBuilder._table_exists(conn, "build_metadata"):
            return {}
        return {
            str(key): str(value)
            for key, value in conn.execute("SELECT key, value FROM build_metadata").fetchall()
        }

    def _expected_source_counts(self) -> tuple[int | None, int | None]:
        manifest_path = self.canonical_v1_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            return None, None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            steps = metrics.get("steps_total")
            routes = metrics.get("routes_total")
            return (
                int(steps) if steps is not None else None,
                int(routes) if routes is not None else None,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None, None

    def _can_adopt_staged_database(self, conn: sqlite3.Connection) -> bool:
        if not all(self._table_exists(conn, name) for name in {"steps", "route_instances"}):
            return False
        expected_steps, expected_routes = self._expected_source_counts()
        if expected_steps is None or expected_routes is None:
            return False
        try:
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                return False
            steps_total = int(conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0])
            routes_total = int(conn.execute("SELECT COUNT(*) FROM route_instances").fetchone()[0])
            step_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(steps)").fetchall()
            }
            route_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(route_instances)").fetchall()
            }
        except sqlite3.Error:
            return False
        required_step_columns = {
            "source_step_id", "source_route_id", "source_row_order",
            "step_index", "raw_reaction_text",
        }
        required_route_columns = {
            "route_instance_id", "source_route_id", "source_route_uid",
            "route_variant_index", "multistep_reaction_text", "step_count", "route_order",
        }
        return (
            steps_total == expected_steps
            and routes_total == expected_routes
            and required_step_columns.issubset(step_columns)
            and required_route_columns.issubset(route_columns)
        )

    def _identity_manifest_payload(self) -> dict[str, Any] | None:
        if not self.identity_assignment_manifest.exists():
            return None
        try:
            payload = json.loads(self.identity_assignment_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _identity_assignment_complete(
        self,
        conn: sqlite3.Connection,
        *,
        source_hash: str,
    ) -> bool:
        payload = self._identity_manifest_payload()
        if not payload or not payload.get("complete"):
            return False
        if payload.get("context_schema_version") != self.CONTEXT_SCHEMA_VERSION:
            return False
        if payload.get("source_manifest_sha256") != source_hash:
            return False
        required_tables = {"steps", "route_instances", "route_instance_assignment"}
        if not all(self._table_exists(conn, name) for name in required_tables):
            return False
        expected_steps = int(payload.get("steps_total", -1))
        expected_routes = int(payload.get("routes_total", -1))
        steps_total = int(conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0])
        routes_total = int(conn.execute("SELECT COUNT(*) FROM route_instances").fetchone()[0])
        assignments_total = int(
            conn.execute("SELECT COUNT(*) FROM route_instance_assignment").fetchone()[0]
        )
        return (
            expected_steps == steps_total == assignments_total
            and expected_routes == routes_total
            and int(payload.get("unassigned_steps", -1)) == 0
            and int(payload.get("duplicate_step_instance_ids", -1)) == 0
            and int(payload.get("assignment_failures", -1)) == 0
        )

    def _write_identity_assignment_manifest(self, payload: dict[str, Any]) -> None:
        self.identity_assignment_manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.identity_assignment_manifest.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, self.identity_assignment_manifest)

    @staticmethod
    def _ensure_assignment_indexes(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_route_instance_id
                ON route_instances(route_instance_id);
            CREATE INDEX IF NOT EXISTS idx_route_source_id
                ON route_instances(source_route_id);
            CREATE INDEX IF NOT EXISTS idx_route_source_order
                ON route_instances(source_route_id, route_order);
            CREATE INDEX IF NOT EXISTS idx_steps_source_route
                ON steps(source_route_id);
            CREATE INDEX IF NOT EXISTS idx_steps_source_route_order
                ON steps(source_route_id, step_index, source_row_order);
            """
        )

    @staticmethod
    def _create_assignment_table(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TABLE IF EXISTS route_instance_assignment;
            CREATE TABLE route_instance_assignment (
                source_rowid INTEGER PRIMARY KEY,
                source_route_id TEXT NOT NULL,
                source_step_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                route_instance_id TEXT NOT NULL,
                step_instance_id TEXT NOT NULL,
                assignment_method TEXT NOT NULL,
                assignment_confidence REAL NOT NULL
            );
            """
        )

    @staticmethod
    def _bulk_assign_unique_routes(conn: sqlite3.Connection) -> int:
        conn.executescript(
            """
            DROP TABLE IF EXISTS unique_route_assignment;
            CREATE TABLE unique_route_assignment AS
            SELECT source_route_id, MIN(route_instance_id) AS route_instance_id
            FROM route_instances
            GROUP BY source_route_id
            HAVING COUNT(*) = 1;
            CREATE UNIQUE INDEX idx_unique_route_assignment_source
                ON unique_route_assignment(source_route_id);
            """
        )
        conn.execute(
            """
            INSERT INTO route_instance_assignment (
                source_rowid,
                source_route_id,
                source_step_id,
                step_index,
                route_instance_id,
                step_instance_id,
                assignment_method,
                assignment_confidence
            )
            SELECT
                s.rowid,
                s.source_route_id,
                s.source_step_id,
                CAST(s.step_index AS INTEGER),
                u.route_instance_id,
                CASE
                    WHEN s.source_step_id LIKE s.source_route_id || '::%'
                    THEN u.route_instance_id || substr(s.source_step_id, length(s.source_route_id) + 1)
                    ELSE u.route_instance_id || '::' || printf('%03d', CAST(s.step_index AS INTEGER))
                         || '::row::' || printf('%09d', CAST(s.source_row_order AS INTEGER))
                END,
                'unique_source_route',
                1.0
            FROM steps s
            JOIN unique_route_assignment u
              ON u.source_route_id = s.source_route_id
            """
        )
        return int(conn.execute("SELECT changes()").fetchone()[0])

    def _assign_route_instances(
        self,
        conn: sqlite3.Connection,
        *,
        source_hash: str,
    ) -> dict[str, Any]:
        started = perf_counter()
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA temp_store=FILE")
        self._ensure_assignment_indexes(conn)
        conn.execute(
            "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('stage_state', 'assignment_started')"
        )
        conn.commit()

        self._create_assignment_table(conn)
        direct_steps = self._bulk_assign_unique_routes(conn)
        unique_source_routes = int(
            conn.execute("SELECT COUNT(*) FROM unique_route_assignment").fetchone()[0]
        )
        duplicate_source_ids = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT source_route_id
                FROM route_instances
                GROUP BY source_route_id
                HAVING COUNT(*) > 1
                ORDER BY source_route_id
                """
            ).fetchall()
        ]
        duplicate_route_instances = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM route_instances
                WHERE source_route_id IN (
                    SELECT source_route_id
                    FROM route_instances
                    GROUP BY source_route_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        duplicate_group_steps = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM steps
                WHERE source_route_id IN (
                    SELECT source_route_id
                    FROM route_instances
                    GROUP BY source_route_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        LOGGER.info("Loaded %s route instances", conn.execute("SELECT COUNT(*) FROM route_instances").fetchone()[0])
        LOGGER.info("Unique source routes assigned by indexed join: %s", unique_source_routes)
        LOGGER.info("Unique-route steps assigned by indexed join: %s", direct_steps)
        LOGGER.info("Duplicate source-route groups requiring reconstruction: %s", len(duplicate_source_ids))
        LOGGER.info("Duplicate-group route instances: %s", duplicate_route_instances)
        LOGGER.info("Duplicate-group steps requiring reconstruction: %s", duplicate_group_steps)

        def normalize_text(value: object) -> str:
            return str(value or "").strip().replace("\r\n", "\n").replace("\r", "\n")

        route_columns = [
            str(item[1])
            for item in conn.execute("PRAGMA table_info(route_instances)").fetchall()
        ]
        expensive_match_calls = 0
        duplicate_group_steps_assigned = 0
        assignment_failures: list[dict[str, Any]] = []
        duplicate_assignments: list[tuple[Any, ...]] = []
        for source_route_id in duplicate_source_ids:
            instances = conn.execute(
                "SELECT * FROM route_instances WHERE source_route_id=? ORDER BY route_order",
                (source_route_id,),
            ).fetchall()
            instance_rows = [dict(zip(route_columns, row)) for row in instances]
            source_steps = [
                {
                    "rowid": int(row[0]),
                    "source_step_id": str(row[1]),
                    "step_index": int(row[2]),
                    "text": normalize_text(row[3]),
                    "source_row_order": int(row[4]),
                }
                for row in conn.execute(
                    """
                    SELECT rowid, source_step_id, step_index, raw_reaction_text, source_row_order
                    FROM steps
                    WHERE source_route_id=?
                    ORDER BY source_row_order
                    """,
                    (source_route_id,),
                ).fetchall()
            ]
            by_exact: dict[tuple[int, str], deque[dict[str, Any]]] = defaultdict(deque)
            by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
            by_text: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
            for item in source_steps:
                by_exact[(item["step_index"], item["text"])].append(item)
                by_index[item["step_index"]].append(item)
                by_text[item["text"]].append(item)
            used: set[int] = set()
            for instance in instance_rows:
                lines = [
                    line.strip()
                    for line in normalize_text(instance["multistep_reaction_text"]).split("\n")
                    if line.strip()
                ]
                expected = int(instance["step_count"])
                if len(lines) != expected:
                    assignment_failures.append(
                        {
                            "source_route_id": source_route_id,
                            "route_instance_id": instance["route_instance_id"],
                            "reason": "line_count_mismatch",
                            "observed": len(lines),
                            "expected": expected,
                        }
                    )
                    continue
                for step_index, line in enumerate(lines):
                    expensive_match_calls += 1
                    candidate: dict[str, Any] | None = None
                    method = ""
                    confidence = 0.0
                    queue = by_exact[(step_index, line)]
                    while queue and queue[0]["rowid"] in used:
                        queue.popleft()
                    if queue:
                        candidate = queue[0]
                        method = "exact_route_text_match"
                        confidence = 1.0
                    if candidate is None:
                        choices = [
                            item
                            for item in by_index[step_index]
                            if item["rowid"] not in used
                        ]
                        if len(choices) == 1:
                            candidate = choices[0]
                            method = "unique_step_index_match"
                            confidence = 0.90
                    if candidate is None:
                        queue = by_text[line]
                        while queue and queue[0]["rowid"] in used:
                            queue.popleft()
                        if queue:
                            candidate = queue[0]
                            method = "exact_step_text_match"
                            confidence = 0.85
                    if candidate is None:
                        assignment_failures.append(
                            {
                                "source_route_id": source_route_id,
                                "route_instance_id": instance["route_instance_id"],
                                "step_index": step_index,
                                "reason": "no_candidate_step",
                            }
                        )
                        continue
                    used.add(candidate["rowid"])
                    duplicate_group_steps_assigned += 1
                    source_step_id = str(candidate["source_step_id"])
                    source_prefix = f"{source_route_id}::"
                    if source_step_id.startswith(source_prefix):
                        step_suffix = source_step_id[len(source_route_id):]
                        step_instance_id = f"{instance['route_instance_id']}{step_suffix}"
                    else:
                        step_instance_id = (
                            f"{instance['route_instance_id']}::{step_index:03d}"
                            f"::row::{candidate['source_row_order']:09d}"
                        )
                    duplicate_assignments.append(
                        (
                            candidate["rowid"],
                            source_route_id,
                            candidate["source_step_id"],
                            step_index,
                            instance["route_instance_id"],
                            step_instance_id,
                            method,
                            confidence,
                        )
                    )
            if len(used) != len(source_steps):
                assignment_failures.append(
                    {
                        "source_route_id": source_route_id,
                        "reason": "unassigned_source_steps",
                        "count": len(source_steps) - len(used),
                    }
                )

        if assignment_failures:
            raise RuntimeError(
                f"Route-instance assignment failed for {len(assignment_failures)} records: "
                f"{assignment_failures[:5]}"
            )
        if duplicate_assignments:
            conn.executemany(
                """
                INSERT INTO route_instance_assignment (
                    source_rowid,
                    source_route_id,
                    source_step_id,
                    step_index,
                    route_instance_id,
                    step_instance_id,
                    assignment_method,
                    assignment_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                duplicate_assignments,
            )

        conn.executescript(
            """
            CREATE INDEX idx_route_assignment_instance_step
                ON route_instance_assignment(route_instance_id, step_index, step_instance_id);
            CREATE UNIQUE INDEX idx_route_assignment_step_instance
                ON route_instance_assignment(step_instance_id);
            CREATE INDEX idx_route_assignment_source_route
                ON route_instance_assignment(source_route_id);
            CREATE INDEX idx_route_assignment_route_instance
                ON route_instance_assignment(route_instance_id);
            """
        )

        routes_total = int(conn.execute("SELECT COUNT(*) FROM route_instances").fetchone()[0])
        unique_instances = int(
            conn.execute("SELECT COUNT(DISTINCT route_instance_id) FROM route_instances").fetchone()[0]
        )
        unique_source_route_id = int(
            conn.execute("SELECT COUNT(DISTINCT source_route_id) FROM route_instances").fetchone()[0]
        )
        duplicate_groups = len(duplicate_source_ids)
        steps_total = int(conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0])
        assignment_rows = int(
            conn.execute("SELECT COUNT(*) FROM route_instance_assignment").fetchone()[0]
        )
        unassigned = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM steps s
                LEFT JOIN route_instance_assignment a ON a.source_rowid=s.rowid
                WHERE a.source_rowid IS NULL
                """
            ).fetchone()[0]
        )
        duplicate_steps = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT step_instance_id
                    FROM route_instance_assignment
                    GROUP BY step_instance_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        method_counts = {
            str(method): int(count)
            for method, count in conn.execute(
                """
                SELECT assignment_method, COUNT(*)
                FROM route_instance_assignment
                GROUP BY assignment_method
                """
            ).fetchall()
        }
        elapsed = perf_counter() - started
        stage_metrics = {
            "routes_total": routes_total,
            "unique_route_instance_id": unique_instances,
            "unique_source_route_id": unique_source_route_id,
            "duplicate_source_route_groups": duplicate_groups,
            "preserved_conflicting_variant_rows": routes_total - unique_source_route_id,
            "steps_source_total": steps_total,
            "duplicate_step_instance_ids": duplicate_steps,
            "cross_variant_edges": 0,
            "unsupported_symbolic_auto_resolved": 0,
            "inferred_structure_hypotheses": 0,
            "unique_route_assignments": unique_source_routes,
            "unique_route_steps_assigned": direct_steps,
            "duplicate_route_instances": duplicate_route_instances,
            "duplicate_group_steps": duplicate_group_steps,
            "duplicate_group_steps_assigned": duplicate_group_steps_assigned,
            "expensive_match_calls": expensive_match_calls,
            "unassigned_steps": unassigned,
            "route_instance_assignment_rows": assignment_rows,
        }
        if routes_total != unique_instances:
            raise RuntimeError("route_instance_id is not unique")
        if assignment_rows != steps_total:
            raise RuntimeError(
                f"route assignment row count mismatch: assignments={assignment_rows}, steps={steps_total}"
            )
        if unassigned or duplicate_steps:
            raise RuntimeError(
                f"Route-instance assignment invalid: unassigned={unassigned}, "
                f"duplicate_step_instance_ids={duplicate_steps}"
            )
        if duplicate_group_steps_assigned != duplicate_group_steps:
            raise RuntimeError(
                "Duplicate-group assignment incomplete: "
                f"assigned={duplicate_group_steps_assigned}, expected={duplicate_group_steps}"
            )
        if expensive_match_calls > duplicate_group_steps:
            raise RuntimeError(
                f"Expensive matching escaped duplicate groups: calls={expensive_match_calls}, "
                f"duplicate_group_steps={duplicate_group_steps}"
            )

        conn.commit()
        manifest = {
            "context_schema_version": self.CONTEXT_SCHEMA_VERSION,
            "source_manifest_sha256": source_hash,
            "routes_total": routes_total,
            "steps_total": steps_total,
            "unique_source_routes": unique_source_routes,
            "unique_route_steps_assigned": direct_steps,
            "duplicate_source_route_groups": duplicate_groups,
            "duplicate_route_instances": duplicate_route_instances,
            "duplicate_group_steps": duplicate_group_steps,
            "duplicate_group_steps_assigned": duplicate_group_steps_assigned,
            "expensive_match_calls": expensive_match_calls,
            "assignment_method_counts": method_counts,
            "unassigned_steps": unassigned,
            "duplicate_step_instance_ids": duplicate_steps,
            "assignment_failures": 0,
            "route_instance_assignment_rows": assignment_rows,
            "elapsed_seconds": round(elapsed, 6),
            "complete": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write_identity_assignment_manifest(manifest)
        identity_hash = sha256_file(self.identity_assignment_manifest)
        for key, value in stage_metrics.items():
            self.metrics[key] = value
        conn.execute(
            "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('stage_metrics_json', ?)",
            (json.dumps(stage_metrics, sort_keys=True),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('identity_manifest_sha256', ?)",
            (identity_hash,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('stage_state', 'assignment_complete')"
        )
        conn.commit()
        LOGGER.info("Duplicate-group steps assigned: %s", duplicate_group_steps_assigned)
        LOGGER.info("Unassigned steps: %s", unassigned)
        LOGGER.info("Duplicate step-instance IDs: %s", duplicate_steps)
        LOGGER.info("Identity assignment completed in %.2f seconds", elapsed)
        LOGGER.info("Starting contextual route processing")
        return stage_metrics

    def _stage_steps(self, database: Path) -> None:
        source_manifest = self.canonical_v1_dir / "dataset_manifest.json"
        source_hash = sha256_file(source_manifest) if source_manifest.exists() else "missing"
        if self.config.resume and database.exists():
            conn = sqlite3.connect(database, timeout=60)
            try:
                metadata = self._read_build_metadata(conn)
                matching_source = (
                    metadata.get("source_manifest_sha256") == source_hash
                    and metadata.get("context_schema_version") == self.CONTEXT_SCHEMA_VERSION
                )
                if matching_source and self._identity_assignment_complete(
                    conn,
                    source_hash=source_hash,
                ):
                    LOGGER.info("Reusing completed indexed route assignment from %s", database)
                    for key, value in json.loads(metadata.get("stage_metrics_json", "{}")).items():
                        self.metrics[key] = int(value)
                    return
                if matching_source and metadata.get("stage_state") in {
                    "source_staged",
                    "assignment_started",
                }:
                    LOGGER.info("Reusing staged Product One steps from %s", database)
                    LOGGER.info("Resuming indexed route-instance assignment")
                    self._assign_route_instances(conn, source_hash=source_hash)
                    return
                if self._can_adopt_staged_database(conn):
                    LOGGER.info(
                        "Adopting complete pre-v2.0.5 staged Product One database from %s",
                        database,
                    )
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS build_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('source_manifest_sha256', ?)",
                        (source_hash,),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('context_schema_version', ?)",
                        (self.CONTEXT_SCHEMA_VERSION,),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('stage_state', 'source_staged')"
                    )
                    conn.commit()
                    self.identity_assignment_manifest.unlink(missing_ok=True)
                    self._assign_route_instances(conn, source_hash=source_hash)
                    return
            except (sqlite3.Error, json.JSONDecodeError, OSError):
                LOGGER.exception("Could not resume staged route assignment; rebuilding staging database")
            finally:
                conn.close()
            database.unlink(missing_ok=True)
            self.identity_assignment_manifest.unlink(missing_ok=True)

        conn = sqlite3.connect(database, timeout=60)
        try:
            conn.execute("PRAGMA busy_timeout=60000")
            first = True
            source_order = 0
            for chunk_number, chunk in enumerate(
                iter_dataset(self.canonical_v1_dir, "steps", columns=self.REQUIRED_STEP_COLUMNS), start=1
            ):
                serial = chunk.copy()
                for column in [
                    "reactants", "products", "solvents", "agents",
                    "condition_numeric_tokens", "quality_issues",
                ]:
                    if column in serial:
                        serial[column] = serial[column].map(
                            lambda value: json.dumps(parse_list(value), ensure_ascii=False, default=str)
                        )
                serial["source_step_id"] = serial["step_id"].astype(str)
                serial["source_route_id"] = serial["route_id"].astype(str)
                serial["source_row_order"] = range(source_order, source_order + len(serial))
                source_order += len(serial)
                serial.to_sql("steps", conn, if_exists="replace" if first else "append", index=False, chunksize=5000)
                first = False
                LOGGER.info("Staged Product One step chunk %s (%s rows)", chunk_number, len(serial))

            route_records: list[dict[str, Any]] = []
            source_route_counts: Counter[str] = Counter()
            route_uid_counts: Counter[str] = Counter()
            route_order = 0
            try:
                route_chunks = iter_dataset(
                    self.canonical_v1_dir,
                    "routes",
                    columns=self.REQUIRED_ROUTE_COLUMNS,
                )
                for chunk in route_chunks:
                    for row in chunk.to_dict(orient="records"):
                        source_route_id = str(row.get("route_id") or "")
                        source_route_counts[source_route_id] += 1
                        variant_index = source_route_counts[source_route_id] - 1
                        base_uid = str(row.get("route_uid") or source_route_id)
                        route_uid_counts[base_uid] += 1
                        occurrence = route_uid_counts[base_uid]
                        route_instance_id = (
                            base_uid if occurrence == 1 else f"{base_uid}::instance::{occurrence:03d}"
                        )
                        route_records.append(
                            {
                                "route_instance_id": route_instance_id,
                                "source_route_id": source_route_id,
                                "source_route_uid": base_uid,
                                "route_variant_index": variant_index,
                                "route_content_hash": str(row.get("source_content_hash") or ""),
                                "multistep_reaction_text": str(row.get("multistep_reaction_text") or ""),
                                "step_count": int(row.get("step_count") or 0),
                                "patent_document_id": row.get("patent_document_id"),
                                "split": row.get("split"),
                                "route_order": route_order,
                            }
                        )
                        route_order += 1
            except FileNotFoundError:
                rows = conn.execute(
                    "SELECT DISTINCT source_route_id, patent_document_id, split FROM steps ORDER BY source_route_id"
                ).fetchall()
                for source_route_id, patent_document_id, split in rows:
                    step_rows = conn.execute(
                        "SELECT raw_reaction_text FROM steps WHERE source_route_id=? ORDER BY step_index, source_row_order",
                        (source_route_id,),
                    ).fetchall()
                    route_records.append(
                        {
                            "route_instance_id": str(source_route_id),
                            "source_route_id": str(source_route_id),
                            "source_route_uid": str(source_route_id),
                            "route_variant_index": 0,
                            "route_content_hash": stable_hash("\n".join(str(item[0]) for item in step_rows), 16),
                            "multistep_reaction_text": "\n".join(str(item[0]) for item in step_rows),
                            "step_count": len(step_rows),
                            "patent_document_id": patent_document_id,
                            "split": split,
                            "route_order": route_order,
                        }
                    )
                    source_route_counts[str(source_route_id)] = 1
                    route_order += 1

            pd.DataFrame.from_records(route_records).to_sql(
                "route_instances", conn, if_exists="replace", index=False, chunksize=5000
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(steps)").fetchall()
            }
            for definition in [
                "route_instance_id TEXT",
                "step_instance_id TEXT",
                "source_route_uid TEXT",
                "route_variant_index INTEGER",
                "route_content_hash TEXT",
                "assignment_method TEXT",
                "assignment_confidence REAL",
            ]:
                name = definition.split()[0]
                if name not in columns:
                    conn.execute(f"ALTER TABLE steps ADD COLUMN {definition}")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS build_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('source_manifest_sha256', ?)",
                (source_hash,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('context_schema_version', ?)",
                (self.CONTEXT_SCHEMA_VERSION,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('source_rows', ?)",
                (str(source_order),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO build_metadata(key, value) VALUES ('stage_state', 'source_staged')"
            )
            self._ensure_assignment_indexes(conn)
            conn.commit()
            LOGGER.info("Staged Product One steps: %s", source_order)
            self._assign_route_instances(conn, source_hash=source_hash)
        finally:
            conn.close()

    @staticmethod
    def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
        for column in [
            "reactants",
            "products",
            "solvents",
            "agents",
            "condition_numeric_tokens",
            "quality_issues",
        ]:
            row[column] = parse_list(row.get(column))
        return row

    def _iter_routes(
        self,
        database: Path,
        after_route_instance_id: str | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        conn = sqlite3.connect(database)
        cursor: sqlite3.Cursor | None = None
        try:
            conn.row_factory = sqlite3.Row
            base_query = """
                SELECT
                    s.*,
                    a.route_instance_id AS assigned_route_instance_id,
                    a.step_instance_id AS assigned_step_instance_id,
                    a.assignment_method AS route_assignment_method,
                    a.assignment_confidence AS route_assignment_confidence,
                    r.source_route_uid AS assigned_source_route_uid,
                    r.route_variant_index AS assigned_route_variant_index,
                    r.route_content_hash AS assigned_route_content_hash
                FROM route_instance_assignment a
                JOIN steps s ON s.rowid=a.source_rowid
                JOIN route_instances r ON r.route_instance_id=a.route_instance_id
            """
            if after_route_instance_id:
                cursor = conn.execute(
                    base_query
                    + """
                    WHERE a.route_instance_id > ?
                    ORDER BY a.route_instance_id, a.step_index, a.step_instance_id
                    """,
                    (after_route_instance_id,),
                )
            else:
                cursor = conn.execute(
                    base_query
                    + """
                    ORDER BY a.route_instance_id, a.step_index, a.step_instance_id
                    """
                )
            current_route: str | None = None
            group: list[dict[str, Any]] = []
            for sqlite_row in cursor:
                row = dict(sqlite_row)
                route_instance_id = str(row.pop("assigned_route_instance_id"))
                step_instance_id = str(row.pop("assigned_step_instance_id"))
                row["route_instance_id"] = route_instance_id
                row["step_instance_id"] = step_instance_id
                row["source_route_uid"] = row.pop("assigned_source_route_uid")
                row["route_variant_index"] = row.pop("assigned_route_variant_index")
                row["route_content_hash"] = row.pop("assigned_route_content_hash")
                row["assignment_method"] = row.pop("route_assignment_method")
                row["assignment_confidence"] = row.pop("route_assignment_confidence")
                row["route_id"] = route_instance_id
                row["step_id"] = step_instance_id
                row = self._decode_row(row)
                if current_route is not None and route_instance_id != current_route:
                    yield group
                    group = []
                group.append(row)
                current_route = route_instance_id
            if group:
                yield group
        finally:
            if cursor is not None:
                cursor.close()
            conn.close()

    @staticmethod
    def _condition_confidence_numeric(value: object) -> float:
        mapping = {"high": 1.0, "medium": 0.75, "low": 0.4, "none": 0.0}
        return mapping.get(str(value).lower(), 0.5)

    @staticmethod
    def _continuity_score(edges: list[Any]) -> float:
        if not edges:
            return 1.0
        return float(sum(edge.continuity_confidence for edge in edges) / len(edges))

    def _condition_rows(self, step: dict[str, Any]) -> list[dict[str, Any]]:
        step_instance_id = str(step.get("step_instance_id") or step["step_id"])
        route_instance_id = str(step.get("route_instance_id") or step["route_id"])
        base = {
            "dataset_version": self.config.dataset_version,
            "step_id": step_instance_id,
            "step_instance_id": step_instance_id,
            "source_step_id": str(step.get("source_step_id") or step["step_id"]),
            "route_id": route_instance_id,
            "route_instance_id": route_instance_id,
            "source_route_id": str(step.get("source_route_id") or step["route_id"]),
        }
        confidence = self._condition_confidence_numeric(step.get("condition_extraction_confidence"))
        inferred = str(step.get("condition_extraction_method")) not in {"legacy", "source", "not_three_field"}
        rows: list[dict[str, Any]] = []
        for kind, observed, normalized, valid, unit in [
            ("temperature", step.get("temperature_observed_c"), step.get("temperature_c"), step.get("temperature_valid"), "degC"),
            ("time", step.get("time_observed_h"), step.get("time_h"), step.get("time_valid"), "h"),
        ]:
            if pd.notna(observed) or pd.notna(normalized):
                rows.append(
                    {
                        **base,
                        "condition_type": kind,
                        "original_token": None if pd.isna(observed) else str(observed),
                        # Keep the generic value column homogeneously textual
                        # across numeric and categorical conditions. Typed
                        # consumers should use normalized_numeric_value or
                        # normalized_text_value. This prevents Arrow from trying
                        # to coerce solvent/agent strings to double.
                        "normalized_value": None if pd.isna(normalized) else str(float(normalized)),
                        "normalized_value_type": "numeric",
                        "normalized_numeric_value": None if pd.isna(normalized) else float(normalized),
                        "normalized_text_value": None,
                        "normalized_unit": unit,
                        "extraction_rule": step.get("condition_extraction_method"),
                        "confidence": confidence,
                        "plausibility_status": "valid" if bool(valid) else "suspicious",
                        "evidence_origin": "inferred" if inferred else "directly_observed",
                    }
                )
        for kind, values in [("solvent", step.get("solvents", [])), ("agent", step.get("agents", []))]:
            for value in parse_list(values):
                rows.append(
                    {
                        **base,
                        "condition_type": kind,
                        "original_token": value,
                        "normalized_value": str(value),
                        "normalized_value_type": "categorical",
                        "normalized_numeric_value": None,
                        "normalized_text_value": str(value),
                        "normalized_unit": None,
                        "extraction_rule": "normalized_source_list",
                        "confidence": 0.9,
                        "plausibility_status": "not_numerically_evaluated",
                        "evidence_origin": "directly_observed",
                    }
                )
        return rows

    def _molecule_rows(self, step: dict[str, Any], reaction: str | None) -> list[dict[str, Any]]:
        if not reaction:
            return []
        parsed = parse_reaction(reaction)
        rows: list[dict[str, Any]] = []
        positions: dict[str, int] = {"reactant": 0, "product": 0, "solvent": 0, "agent": 0}
        role_values = [
            ("reactant", parsed.reactants),
            ("product", parsed.products),
            ("solvent", tuple(parse_list(step.get("solvents")))),
            ("agent", tuple(parse_list(step.get("agents")))),
        ]
        for role, values in role_values:
            for value in values:
                molecule_id = self.molecules.register(value)
                if molecule_id is None:
                    continue
                rows.append(
                    {
                        "dataset_version": self.config.dataset_version,
                        "step_id": str(step.get("step_instance_id") or step["step_id"]),
                        "step_instance_id": str(step.get("step_instance_id") or step["step_id"]),
                        "source_step_id": str(step.get("source_step_id") or step["step_id"]),
                        "route_instance_id": str(step.get("route_instance_id") or step["route_id"]),
                        "source_route_id": str(step.get("source_route_id") or step["route_id"]),
                        "molecule_id": molecule_id,
                        "role": role,
                        "position": positions[role],
                        "source_role": role,
                        "role_confidence": 1.0 if role in {"reactant", "product"} else 0.9,
                    }
                )
                positions[role] += 1
        return rows

    def _mapping_candidate(
        self,
        step: dict[str, Any],
        reaction: str | None,
        *,
        resolution_status: ResolutionStatus,
    ) -> dict[str, Any]:
        eligible = bool(reaction and parse_reaction(reaction).parse_ok)
        if eligible and bool(step.get("parse_ok")):
            eligibility_source = "original_parse_valid"
            structure_evidence_type = "observed_step_structure"
        elif eligible and resolution_status in {ResolutionStatus.RESOLVED, ResolutionStatus.PARTIALLY_RESOLVED}:
            eligibility_source = "resolved_observed_intermediate"
            structure_evidence_type = "observed_route_metadata"
        else:
            eligibility_source = "not_eligible"
            structure_evidence_type = "symbolic_only" if step.get("input_intermediate") or step.get("output_intermediate") else "unavailable"
        return {
            "dataset_version": self.config.dataset_version,
            "step_id": step["step_id"],
            "step_instance_id": step["step_instance_id"],
            "source_step_id": step["source_step_id"],
            "route_id": step["route_id"],
            "route_instance_id": step["route_instance_id"],
            "source_route_id": step["source_route_id"],
            "patent_document_id": step.get("patent_document_id"),
            "split": step.get("split"),
            "reaction_smiles": reaction or "",
            "reaction_signature": stable_hash(reaction or str(step.get("reaction_smiles") or ""), 40),
            "eligibility_status": "eligible" if eligible else "not_eligible",
            "eligibility_reason": eligibility_source,
            "eligibility_source": eligibility_source,
            "structure_evidence_type": structure_evidence_type,
            "mapping_status": "pending" if eligible else "not_eligible",
        }

    def _process_route(self, rows: list[dict[str, Any]], writers: dict[str, _BufferedWriter]) -> None:
        route_instance_id = str(rows[0]["route_instance_id"])
        source_route_id = str(rows[0]["source_route_id"])
        if any(str(row["route_instance_id"]) != route_instance_id for row in rows):
            self.metrics["cross_variant_edges"] += 1
            raise RuntimeError("Cross-variant route group detected")
        resolved_steps, edges = self.resolver.resolve_route(rows)
        continuity = self._continuity_score(edges)
        for edge in edges:
            if edge.route_id != route_instance_id:
                self.metrics["cross_variant_edges"] += 1
                raise RuntimeError("Cross-variant edge detected")
            connecting_ids = [
                molecule_id for value in edge.connecting_molecules
                if (molecule_id := self.molecules.register(value)) is not None
            ]
            writers["route_edges"].append(
                {
                    "dataset_version": self.config.dataset_version,
                    **asdict(edge),
                    "route_instance_id": route_instance_id,
                    "source_route_id": source_route_id,
                    "source_step_instance_id": edge.source_step_id,
                    "target_step_instance_id": edge.target_step_id,
                    "connecting_molecules": list(edge.connecting_molecules),
                    "connecting_molecule_ids": connecting_ids,
                }
            )
            self.metrics[f"route_edge::{edge.continuity_status}"] += 1
            if edge.edge_evidence_type in {"exact_symbolic_label", "explicit_structural_identity"}:
                self.metrics["label_connected_edges"] += 1

        route_quality_values: list[float] = []
        resolved_count = 0
        mapping_eligible_count = 0
        for resolved_step in resolved_steps:
            source = resolved_step.source
            step_id = source["step_id"]
            symbolic_step = bool(resolved_step.resolutions)
            if symbolic_step:
                self.metrics["symbolic_steps"] += 1
            for resolution in resolved_step.resolutions:
                record = {
                    "dataset_version": self.config.dataset_version,
                    **asdict(resolution),
                    "route_instance_id": route_instance_id,
                    "source_route_id": source_route_id,
                    "step_instance_id": source["step_instance_id"],
                    "source_step_id": source["source_step_id"],
                    "status": resolution.status.value,
                    "resolved_molecules": list(resolution.resolved_molecules),
                }
                writers["intermediate_resolution"].append(record)
                self.metrics["symbolic_placeholder_occurrences"] += 1
                self.metrics[f"intermediate::{resolution.evidence_status}"] += 1
                if resolution.evidence_status == "resolved_observed_structure":
                    self.metrics["observed_structure_resolutions"] += 1

            effective_reaction = resolved_step.canonical_resolved_reaction_smiles
            if not effective_reaction and bool(source.get("parse_ok")):
                effective_reaction = source.get("canonical_reaction_smiles")
            effective_parse = parse_reaction(effective_reaction or "")
            if resolved_step.resolution_status == ResolutionStatus.RESOLVED:
                resolved_count += 1

            repair_candidates = deterministic_repair_candidates(
                str(source.get("reaction_smiles") or ""),
                contextual_candidate=resolved_step.resolved_reaction_smiles,
                route_continuity_score=continuity,
            )
            for rank, candidate in enumerate(repair_candidates, start=1):
                writers["repair_candidates"].append(
                    {
                        "dataset_version": self.config.dataset_version,
                        "step_id": step_id,
                        "step_instance_id": source["step_instance_id"],
                        "source_step_id": source["source_step_id"],
                        "route_id": route_instance_id,
                        "route_instance_id": route_instance_id,
                        "source_route_id": source_route_id,
                        "rank": rank,
                        **asdict(candidate),
                    }
                )

            mapping_candidate = self._mapping_candidate(
                source,
                effective_reaction if effective_parse.parse_ok else None,
                resolution_status=resolved_step.resolution_status,
            )
            writers["mapping_candidates"].append(mapping_candidate)
            self.metrics["mapping_candidates_total"] += 1
            is_eligible = mapping_candidate["eligibility_status"] == "eligible"
            mapping_eligible_count += int(is_eligible)
            self.metrics["strict_mapping_eligible"] += int(is_eligible)
            self.metrics[f"mapping_eligibility_source::{mapping_candidate['eligibility_source']}"] += 1

            condition_completeness = sum([
                bool(parse_list(source.get("solvents"))),
                pd.notna(source.get("temperature_c")),
                pd.notna(source.get("time_h")),
            ]) / 3.0
            parse_component = 1.0 if effective_parse.parse_ok else 0.0
            resolution_component = {
                ResolutionStatus.NOT_REQUIRED: 1.0,
                ResolutionStatus.RESOLVED: 1.0,
                ResolutionStatus.PARTIALLY_RESOLVED: 0.6,
                ResolutionStatus.AMBIGUOUS: 0.2,
                ResolutionStatus.UNRESOLVED: 0.0,
                ResolutionStatus.INVALID_AFTER_RESOLUTION: 0.0,
            }[resolved_step.resolution_status]
            anomaly_component = 1.0 if str(source.get("condition_status")) != "suspicious" else 0.0
            quality_components = {
                "parse": parse_component,
                "resolution": resolution_component,
                "route_continuity": continuity,
                "condition_completeness": condition_completeness,
                "condition_plausibility": anomaly_component,
                "mapping": None,
            }
            route_quality = (
                0.33 * parse_component + 0.22 * resolution_component + 0.17 * continuity
                + 0.17 * condition_completeness + 0.11 * anomaly_component
            )
            route_quality_values.append(route_quality)

            repairable = any(candidate.accepted for candidate in repair_candidates)
            solvents = parse_list(source.get("solvents"))
            agents = parse_list(source.get("agents"))
            solvent_families = sorted({solvent_family(item) for item in solvents})
            agent_families = sorted({agent_family(item) for item in agents})
            catalyst_families = sorted({catalyst for item in agents if (catalyst := catalyst_family(item))})
            if bool(source.get("parse_ok")):
                structure_evidence_type = "observed_step_structure"
            elif resolved_step.resolution_status in {ResolutionStatus.RESOLVED, ResolutionStatus.PARTIALLY_RESOLVED}:
                structure_evidence_type = "observed_route_metadata"
            elif symbolic_step:
                structure_evidence_type = "symbolic_only"
            else:
                structure_evidence_type = "unavailable"
            step_record = {
                **source,
                "source_dataset_version": source.get("dataset_version"),
                "dataset_version": self.config.dataset_version,
                "route_id": route_instance_id,
                "route_instance_id": route_instance_id,
                "source_route_id": source_route_id,
                "step_id": source["step_instance_id"],
                "step_instance_id": source["step_instance_id"],
                "source_step_id": source["source_step_id"],
                "structure_evidence_type": structure_evidence_type,
                "original_reaction_smiles": source.get("reaction_smiles"),
                "resolved_reaction_smiles": resolved_step.resolved_reaction_smiles,
                "canonical_resolved_reaction_smiles": effective_reaction,
                "resolution_status": resolved_step.resolution_status.value,
                "resolution_confidence": resolved_step.resolution_confidence,
                "contextual_parse_ok": effective_parse.parse_ok,
                "contextual_parse_failure_class": effective_parse.failure_class.value,
                "repairable": repairable,
                "mapping_status": mapping_candidate["mapping_status"],
                "mapping_confidence": None,
                "mapping_backend": None,
                "reaction_family": None,
                "solvent_families": solvent_families,
                "agent_families": agent_families,
                "catalyst_families": catalyst_families,
                "reaction_centre_fingerprint": None,
                "route_continuity_score": continuity,
                "quality_components": quality_components,
                "contextual_quality_score": round(route_quality, 6),
                "eligible_contextual_models": effective_parse.parse_ok,
                "eligible_mapping_models": False,
                "eligible_retrieval_v2": effective_parse.parse_ok,
            }
            writers["steps"].append(step_record)
            writers["condition_evidence"].extend(self._condition_rows(source))
            writers["step_molecules"].extend(self._molecule_rows(source, effective_reaction))

            if symbolic_step and resolved_step.resolution_status in {ResolutionStatus.UNRESOLVED, ResolutionStatus.AMBIGUOUS}:
                quarantine_reason = "unresolved_symbolic_intermediate"
                self.metrics["unresolved_symbolic_steps"] += 1
            elif not effective_parse.parse_ok:
                quarantine_reason = str(source.get("parse_failure_class") or effective_parse.failure_class.value)
                self.metrics["invalid_non_symbolic_steps"] += 1
            else:
                quarantine_reason = None
            if quarantine_reason:
                writers["quarantine"].append(
                    {
                        "dataset_version": self.config.dataset_version,
                        "step_id": source["step_instance_id"],
                        "step_instance_id": source["step_instance_id"],
                        "source_step_id": source["source_step_id"],
                        "route_id": route_instance_id,
                        "route_instance_id": route_instance_id,
                        "source_route_id": source_route_id,
                        "reason": quarantine_reason,
                        "reason_code": quarantine_reason,
                        "structure_evidence_type": structure_evidence_type,
                        "original_reaction_smiles": source.get("reaction_smiles"),
                        "candidate_reaction_smiles": resolved_step.resolved_reaction_smiles,
                    }
                )

            self.metrics["steps_total"] += 1
            self.metrics["steps_contextual_parse_valid"] += int(effective_parse.parse_ok)
            self.metrics["steps_repairable"] += int(repairable)
            self.metrics[f"resolution_status::{resolved_step.resolution_status.value}"] += 1
            self.metrics[f"mapping_queue::{mapping_candidate['mapping_status']}"] += 1

        writers["routes"].append(
            {
                "dataset_version": self.config.dataset_version,
                "route_id": route_instance_id,
                "route_instance_id": route_instance_id,
                "source_route_id": source_route_id,
                "source_route_uid": rows[0].get("source_route_uid"),
                "route_variant_index": int(rows[0].get("route_variant_index") or 0),
                "route_content_hash": rows[0].get("route_content_hash"),
                "patent_document_id": rows[0].get("patent_document_id"),
                "split": rows[0].get("split"),
                "step_count": len(rows),
                "contextual_parse_valid_steps": sum(
                    1 for item in resolved_steps
                    if parse_reaction(item.canonical_resolved_reaction_smiles or "").parse_ok
                ),
                "resolved_intermediate_steps": resolved_count,
                "mapping_eligible_steps": mapping_eligible_count,
                "mapped_steps": 0,
                "route_continuity_score": continuity,
                "route_quality_score": round(sum(route_quality_values) / max(len(route_quality_values), 1), 6),
                "reaction_families": [],
                "family_distribution": {},
            }
        )
        self.metrics["routes_contextualized"] += 1

    def _write_molecules(self, writer: _BufferedWriter) -> None:
        if self.molecules is None:
            return
        for records in self.molecules.iter_records(self.config.batch_rows):
            writer.extend(records)
            writer.flush()

    def _manifest(self, writers: dict[str, _BufferedWriter]) -> dict[str, Any]:
        source_manifest = self.canonical_v1_dir / "dataset_manifest.json"
        outputs = {
            name: [path.relative_to(self.output_root).as_posix() for path in writer.outputs]
            for name, writer in writers.items()
        }
        all_output_paths = [path for writer in writers.values() for path in writer.outputs]
        split_files = writers["steps"].outputs
        required = {
            "routes_total": int(self.metrics.get("routes_total", 0)),
            "unique_route_instance_id": int(self.metrics.get("unique_route_instance_id", 0)),
            "unique_source_route_id": int(self.metrics.get("unique_source_route_id", 0)),
            "duplicate_source_route_groups": int(self.metrics.get("duplicate_source_route_groups", 0)),
            "preserved_conflicting_variant_rows": int(self.metrics.get("preserved_conflicting_variant_rows", 0)),
            "steps_total": int(self.metrics.get("steps_total", 0)),
            "cross_variant_edges": int(self.metrics.get("cross_variant_edges", 0)),
            "duplicate_step_instance_ids": int(self.metrics.get("duplicate_step_instance_ids", 0)),
            "mapping_candidates_total": int(self.metrics.get("mapping_candidates_total", 0)),
            "strict_mapping_eligible": int(self.metrics.get("strict_mapping_eligible", 0)),
            "unsupported_symbolic_auto_resolved": int(self.metrics.get("unsupported_symbolic_auto_resolved", 0)),
            "symbolic_placeholder_occurrences": int(self.metrics.get("symbolic_placeholder_occurrences", 0)),
            "symbolic_steps": int(self.metrics.get("symbolic_steps", 0)),
            "label_connected_edges": int(self.metrics.get("label_connected_edges", 0)),
            "observed_structure_resolutions": int(self.metrics.get("observed_structure_resolutions", 0)),
            "inferred_structure_hypotheses": int(self.metrics.get("inferred_structure_hypotheses", 0)),
            "unresolved_symbolic_steps": int(self.metrics.get("unresolved_symbolic_steps", 0)),
            "invalid_non_symbolic_steps": int(self.metrics.get("invalid_non_symbolic_steps", 0)),
        }
        identity_pass = (
            required["routes_total"] == required["unique_route_instance_id"]
            and required["steps_total"] == required["mapping_candidates_total"]
            and required["cross_variant_edges"] == 0
            and required["duplicate_step_instance_ids"] == 0
            and required["unsupported_symbolic_auto_resolved"] == 0
        )
        manifest = {
            "context_schema_version": self.CONTEXT_SCHEMA_VERSION,
            "dataset_version": self.config.dataset_version,
            "stage": "context_only_canonicalization",
            "source_dataset_version": "uspto_multistep_canonical_v1",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "metrics": dict(sorted(self.metrics.items())),
            "scientific_identity_report": required,
            "scientific_identity_pass": identity_pass,
            "identity_assignment": {
                "path": self.identity_assignment_manifest.relative_to(self.output_root).as_posix(),
                "sha256": sha256_file(self.identity_assignment_manifest)
                if self.identity_assignment_manifest.exists() else None,
                "complete": bool((self._identity_manifest_payload() or {}).get("complete")),
            },
            "outputs": outputs,
            "reproducibility": {
                "source_manifest_sha256": sha256_file(source_manifest) if source_manifest.exists() else None,
                "output_tree_sha256": hash_paths(all_output_paths, root=self.output_root) if all_output_paths else None,
                "steps_parts_sha256": hash_paths(split_files, root=self.output_root) if split_files else None,
                "config_sha256": canonical_json_hash(asdict(self.config)),
            },
            "contract": {
                "product_one_inputs_are_read_only": True,
                "route_instance_identity_is_preserved": True,
                "identity_assignment_is_indexed_and_checkpointed": True,
                "expensive_matching_is_limited_to_duplicate_source_routes": True,
                "source_step_rows_remain_immutable_during_assignment": True,
                "compatibility_route_id_is_route_instance_id": True,
                "compatibility_step_id_is_step_instance_id": True,
                "source_identifiers_are_preserved_explicitly": True,
                "symbolic_labels_establish_logical_not_structural_continuity": True,
                "symbolic_resolution_requires_observed_unique_explicit_anchor": True,
                "unsupported_symbolic_auto_resolution_is_forbidden": True,
                "model_inferred_intermediates_are_not_canonical_observations": True,
                "unresolved_and_ambiguous_records_are_quarantined": True,
                "original_reactions_are_preserved": True,
                "atom_mapping_is_not_executed_in_this_stage": True,
                "mapping_candidates_are_persisted_for_a_resumable_queue": True,
                "condition_values_preserve_observed_and_normalized_evidence": True,
                "molecule_catalog_is_disk_backed": True,
                "all_output_paths_are_relative": True,
            },
        }
        if not identity_pass:
            raise RuntimeError(f"Contextual identity invariants failed: {required}")
        path = self.output_root / "dataset_manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temporary, path)
        manifest["manifest_path"] = "dataset_manifest.json"
        manifest["manifest_sha256"] = sha256_file(path)
        return manifest

    def _process_staged_routes(
        self,
        database: Path,
        writers: dict[str, _BufferedWriter],
        progress: dict[str, Any] | None = None,
    ) -> tuple[int, str | None]:
        progress = progress or {}
        routes_completed = int(progress.get("routes_completed", 0))
        last_route_instance_id = progress.get("last_route_instance_id")
        routes = (
            self._iter_routes(database, after_route_instance_id=last_route_instance_id)
            if last_route_instance_id else self._iter_routes(database)
        )
        try:
            for rows in routes:
                self._process_route(rows, writers)
                routes_completed += 1
                last_route_instance_id = str(rows[0]["route_instance_id"])
                if routes_completed % self.config.checkpoint_routes == 0:
                    self._checkpoint(
                        writers,
                        last_route_instance_id=last_route_instance_id,
                        routes_completed=routes_completed,
                    )
                    LOGGER.info("Contextualized %s routes", routes_completed)
        finally:
            routes.close()
        return routes_completed, last_route_instance_id

    def build(self) -> dict[str, Any]:
        manifest_path = self.output_root / "dataset_manifest.json"
        if self.config.resume and manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_progress = self._progress()
            if (
                existing.get("context_schema_version") == self.CONTEXT_SCHEMA_VERSION
                and existing_progress.get("complete")
            ):
                return existing
            if existing.get("context_schema_version") != self.CONTEXT_SCHEMA_VERSION:
                LOGGER.warning("Existing contextual output uses an incompatible identity contract; rebuilding active context directory.")
                self.config.clean = True
                self.config.resume = False
        self._prepare_output()
        progress = self._progress()
        self.metrics.update(progress.get("metrics", {}))
        self._reconcile_resume_outputs(progress)
        writers = {
            name: _BufferedWriter(
                self.output_root, name, self.config.prefer_parquet,
                self.config.batch_rows, resume=self.config.resume,
            )
            for name in self.CONTEXT_TABLES
        }
        self._stage_steps(self.route_database)
        self.molecules = MoleculeCatalog(self.molecule_database)
        try:
            routes_completed, last_route_instance_id = self._process_staged_routes(
                self.route_database, writers, progress,
            )
            if last_route_instance_id is not None:
                self._checkpoint(
                    writers,
                    last_route_instance_id=last_route_instance_id,
                    routes_completed=routes_completed,
                )
            self._write_molecules(writers["molecules"])
            for writer in writers.values():
                writer.flush()
            manifest = self._manifest(writers)
            complete_progress = {
                "context_schema_version": self.CONTEXT_SCHEMA_VERSION,
                "dataset_version": self.config.dataset_version,
                "routes_completed": routes_completed,
                "last_route_instance_id": last_route_instance_id,
                "writer_part_counts": {name: len(writer.outputs) for name, writer in writers.items()},
                "metrics": dict(self.metrics),
                "complete": True,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            temporary = self.progress_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(complete_progress, indent=2), encoding="utf-8")
            os.replace(temporary, self.progress_path)
            return manifest
        finally:
            if self.molecules is not None:
                self.molecules.close()
