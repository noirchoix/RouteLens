from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

from reacts.chemistry.mapping import MappingResult
from reacts.mapping.contracts import MappingBatchItem, MappingQueueItem
from reacts.storage.tabular import iter_dataset


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class MappingQueue:
    def __init__(self, database: Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database, timeout=60)
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
                CREATE TABLE IF NOT EXISTS reaction_mapping_queue (
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
                    rxnmapper_token_count INTEGER,
                    rxnmapper_token_limit INTEGER,
                    rxnmapper_eligible INTEGER,
                    fallback_status TEXT,
                    fallback_attempt_count INTEGER NOT NULL DEFAULT 0,
                    exceptional_reason TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    output_shard TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mapping_queue_status
                    ON reaction_mapping_queue(mapping_status);
                CREATE INDEX IF NOT EXISTS idx_mapping_queue_eligibility
                    ON reaction_mapping_queue(eligibility_status);
                CREATE INDEX IF NOT EXISTS idx_mapping_queue_signature
                    ON reaction_mapping_queue(reaction_signature);
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(reaction_mapping_queue)").fetchall()}
            migrations = {
                "source_step_id": "TEXT",
                "source_route_id": "TEXT",
                "rxnmapper_token_count": "INTEGER",
                "rxnmapper_token_limit": "INTEGER",
                "rxnmapper_eligible": "INTEGER",
                "fallback_status": "TEXT",
                "fallback_attempt_count": "INTEGER NOT NULL DEFAULT 0",
                "exceptional_reason": "TEXT",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE reaction_mapping_queue ADD COLUMN {name} {declaration}")
            # Existing v2.0.5 queues are migrated in place. These indexes are
            # created after the columns so interrupted production queues remain
            # resumable without rebuilding committed shards.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mapping_queue_rxn_eligible "
                "ON reaction_mapping_queue(rxnmapper_eligible)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mapping_queue_fallback_status "
                "ON reaction_mapping_queue(fallback_status)"
            )

    def reset(self) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM reaction_mapping_queue")

    def populate_from_context(self, context_dir: Path) -> dict[str, int]:
        inserted = 0
        seen = 0
        now = _utcnow()
        with self.connection() as conn:
            for chunk in iter_dataset(context_dir, "mapping_candidates"):
                rows = []
                for row in chunk.to_dict(orient="records"):
                    seen += 1
                    eligible = str(row.get("eligibility_status") or "not_eligible")
                    initial = "pending" if eligible == "eligible" else "not_eligible"
                    rows.append(
                        (
                            str(row.get("step_instance_id") or row["step_id"]),
                            str(row.get("route_instance_id") or row["route_id"]),
                            str(row.get("source_step_id") or row["step_id"]),
                            str(row.get("source_route_id") or row["route_id"]),
                            str(row["reaction_smiles"]),
                            str(row["reaction_signature"]),
                            eligible,
                            initial,
                            now,
                            now,
                        )
                    )
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO reaction_mapping_queue (
                        step_id, route_id, source_step_id, source_route_id,
                        reaction_smiles, reaction_signature,
                        eligibility_status, mapping_status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                inserted += conn.total_changes - before
        return {"seen": seen, "inserted": inserted}

    def recover_stale(self, stale_after_minutes: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=stale_after_minutes)).isoformat()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE reaction_mapping_queue
                SET mapping_status='pending', error_code='stale_running_recovered',
                    error_message=NULL, started_at=NULL, updated_at=?
                WHERE mapping_status='running' AND started_at < ?
                """,
                (_utcnow(), cutoff),
            )
            return int(cursor.rowcount)

    def claim_batch(
        self,
        batch_size: int,
        *,
        backend_requested: str,
        max_attempts: int,
    ) -> list[MappingQueueItem]:
        conn = sqlite3.connect(self.database, timeout=60, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT step_id, route_id, source_step_id, source_route_id,
                       reaction_smiles, reaction_signature, attempt_count,
                       rxnmapper_token_count, rxnmapper_token_limit,
                       rxnmapper_eligible, fallback_status, fallback_attempt_count
                FROM reaction_mapping_queue
                WHERE eligibility_status='eligible'
                  AND (
                    mapping_status='pending'
                    OR (
                        mapping_status='failed'
                        AND attempt_count < ?
                        AND COALESCE(validation_status, '') != 'quarantined'
                    )
                  )
                ORDER BY step_id
                LIMIT ?
                """,
                (max_attempts, batch_size),
            ).fetchall()
            if not rows:
                conn.execute("COMMIT")
                return []
            ids = [str(row["step_id"]) for row in rows]
            placeholders = ",".join("?" for _ in ids)
            now = _utcnow()
            conn.execute(
                f"""
                UPDATE reaction_mapping_queue
                SET mapping_status='running', attempt_count=attempt_count+1,
                    backend_requested=?, started_at=?, updated_at=?
                WHERE step_id IN ({placeholders})
                """,
                (backend_requested, now, now, *ids),
            )
            conn.execute("COMMIT")
            return [
                MappingQueueItem(
                    step_id=str(row["step_id"]),
                    route_id=str(row["route_id"]),
                    reaction_smiles=str(row["reaction_smiles"]),
                    reaction_signature=str(row["reaction_signature"]),
                    attempt_count=int(row["attempt_count"]) + 1,
                    source_step_id=str(row["source_step_id"]) if row["source_step_id"] is not None else None,
                    source_route_id=str(row["source_route_id"]) if row["source_route_id"] is not None else None,
                    rxnmapper_token_count=(
                        int(row["rxnmapper_token_count"])
                        if row["rxnmapper_token_count"] is not None
                        else None
                    ),
                    rxnmapper_token_limit=(
                        int(row["rxnmapper_token_limit"])
                        if row["rxnmapper_token_limit"] is not None
                        else None
                    ),
                    rxnmapper_eligible=(
                        bool(row["rxnmapper_eligible"])
                        if row["rxnmapper_eligible"] is not None
                        else None
                    ),
                    fallback_status=(
                        str(row["fallback_status"])
                        if row["fallback_status"] is not None
                        else None
                    ),
                    fallback_attempt_count=int(row["fallback_attempt_count"] or 0),
                )
                for row in rows
            ]
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def record_primary_results(
        self,
        queue_items: Sequence[MappingQueueItem],
        primary_results: Sequence[MappingResult],
    ) -> None:
        if len(queue_items) != len(primary_results):
            raise ValueError("Primary metadata update requires one result per queue item")
        now = _utcnow()
        with self.connection() as conn:
            conn.executemany(
                """
                UPDATE reaction_mapping_queue
                SET rxnmapper_token_count=COALESCE(?, rxnmapper_token_count),
                    rxnmapper_token_limit=COALESCE(?, rxnmapper_token_limit),
                    rxnmapper_eligible=COALESCE(?, rxnmapper_eligible),
                    exceptional_reason=COALESCE(?, exceptional_reason),
                    error_code=COALESCE(?, error_code),
                    error_message=?, updated_at=?
                WHERE step_id=?
                """,
                [
                    (
                        result.rxnmapper_token_count,
                        result.rxnmapper_token_limit,
                        int(result.rxnmapper_eligible) if result.rxnmapper_eligible is not None else None,
                        result.error_code if result.error_code == "rxnmapper_sequence_too_long" else None,
                        result.error_code,
                        json.dumps(list(result.diagnostics), ensure_ascii=False),
                        now,
                        item.step_id,
                    )
                    for item, result in zip(queue_items, primary_results)
                ],
            )

    def mark_fallback_attempts(self, step_ids: Sequence[str]) -> None:
        if not step_ids:
            return
        now = _utcnow()
        with self.connection() as conn:
            conn.executemany(
                """
                UPDATE reaction_mapping_queue
                SET fallback_attempt_count=fallback_attempt_count+1,
                    fallback_status='running', updated_at=?
                WHERE step_id=?
                """,
                [(now, step_id) for step_id in step_ids],
            )

    def complete_shard(self, items: list[MappingBatchItem], shard_name: str) -> None:
        now = _utcnow()
        with self.connection() as conn:
            conn.executemany(
                """
                UPDATE reaction_mapping_queue
                SET mapping_status=?, backend_used=?, confidence=?, validation_status=?,
                    scientific_eligibility=?, error_code=?, error_message=?,
                    rxnmapper_token_count=COALESCE(?, rxnmapper_token_count),
                    rxnmapper_token_limit=COALESCE(?, rxnmapper_token_limit),
                    rxnmapper_eligible=COALESCE(?, rxnmapper_eligible),
                    fallback_status=COALESCE(?, fallback_status),
                    fallback_attempt_count=MAX(fallback_attempt_count, ?),
                    exceptional_reason=COALESCE(?, exceptional_reason),
                    completed_at=?, output_shard=?, updated_at=?
                WHERE step_id=?
                """,
                [
                    (
                        item.status.value,
                        item.backend,
                        float(item.confidence),
                        item.validation_status,
                        int(item.scientific_eligibility),
                        item.error_code,
                        json.dumps(list(item.diagnostics), ensure_ascii=False),
                        item.rxnmapper_token_count,
                        item.rxnmapper_token_limit,
                        int(item.rxnmapper_eligible) if item.rxnmapper_eligible is not None else None,
                        item.fallback_status,
                        int(item.fallback_attempt_count),
                        item.exceptional_reason,
                        now,
                        shard_name,
                        now,
                        item.step_id,
                    )
                    for item in items
                ],
            )

    def summary(self) -> dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT mapping_status, COUNT(*) AS n FROM reaction_mapping_queue GROUP BY mapping_status"
            ).fetchall()
            output = {str(row["mapping_status"]): int(row["n"]) for row in rows}
            output["total"] = sum(output.values())
            return output

    def exceptional_summary(self) -> dict[str, int]:
        with self.connection() as conn:
            scalar = lambda query: int(conn.execute(query).fetchone()[0])
            return {
                "rxnmapper_token_eligible": scalar(
                    """
                    SELECT COUNT(*) FROM reaction_mapping_queue
                    WHERE rxnmapper_eligible=1
                       OR (rxnmapper_eligible IS NULL AND backend_used='rxnmapper')
                    """
                ),
                "rxnmapper_sequence_too_long": scalar(
                    "SELECT COUNT(*) FROM reaction_mapping_queue WHERE exceptional_reason='rxnmapper_sequence_too_long'"
                ),
                "mcs_fallback_attempted": scalar(
                    "SELECT COUNT(*) FROM reaction_mapping_queue WHERE fallback_attempt_count > 0 OR backend_used='mcs_fallback'"
                ),
                "mcs_fallback_mapped": scalar(
                    "SELECT COUNT(*) FROM reaction_mapping_queue WHERE fallback_status='mapped'"
                ),
                "mcs_fallback_low_confidence": scalar(
                    "SELECT COUNT(*) FROM reaction_mapping_queue WHERE fallback_status='low_confidence'"
                ),
                "mcs_fallback_timeout": scalar(
                    "SELECT COUNT(*) FROM reaction_mapping_queue WHERE fallback_status='timeout'"
                ),
                "mcs_fallback_failed": scalar(
                    "SELECT COUNT(*) FROM reaction_mapping_queue WHERE fallback_status='failed'"
                ),
                "mapping_exception_quarantined": scalar(
                    "SELECT COUNT(*) FROM reaction_mapping_queue WHERE validation_status='quarantined'"
                ),
            }
