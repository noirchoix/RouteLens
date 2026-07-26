from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reacts.contracts import MappingStatus
from reacts.mapping.backends import MCSBackend, RXNMapperBackend
from reacts.mapping.contracts import MappingBatchItem, MappingQueueItem
from reacts.mapping.preflight import MappingPreflight, resolve_mapping_backend
from reacts.mapping.queue import MappingQueue
from reacts.science.hashing import canonical_json_hash, hash_paths, sha256_file
from reacts.storage.tabular import (
    JSON_COLUMNS,
    LIST_COLUMNS,
    parquet_available,
    serialize_json_contract_columns,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class MappingRunConfig:
    context_dir: Path
    output_dir: Path
    queue_db: Path
    reports_dir: Path
    dataset_version: str = "uspto_multistep_contextual_v2"
    backend: str = "rxnmapper"
    fallback_backend: str | None = "mcs"
    allow_auto_fallback: bool = False
    batch_size: int = 16
    workers: int = 1
    prefetch_batches: int = 2
    shard_size: int = 5_000
    min_confidence: float = 0.50
    min_coverage: float = 0.60
    mcs_timeout_seconds: int = 3
    fallback_process_timeout_seconds: int = 30
    rxnmapper_token_limit: int = 512
    max_attempts: int = 2
    stale_after_minutes: int = 60
    resume: bool = False
    max_rows: int | None = None
    prefer_parquet: bool = True


@dataclass
class _Processed:
    final: MappingBatchItem
    primary: MappingBatchItem
    fallback: MappingBatchItem | None = None
    exceptional: bool = False


def _serialize_csv(frame: pd.DataFrame) -> pd.DataFrame:
    serial = frame.copy()
    for column in (LIST_COLUMNS | JSON_COLUMNS).intersection(serial.columns):
        serial[column] = serial[column].map(
            lambda value: json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, (list, tuple, dict))
            else value
        )
    return serial


def _atomic_write_records(directory: Path, shard_id: int, records: list[dict[str, Any]], prefer_parquet: bool) -> Path | None:
    if not records:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    use_parquet = prefer_parquet and parquet_available()
    suffix = ".parquet" if use_parquet else ".csv.gz"
    target = directory / f"part-{shard_id:05d}{suffix}"
    temporary = directory / f".{target.name}.tmp"
    frame = pd.DataFrame.from_records(records)
    if use_parquet:
        serial = serialize_json_contract_columns(frame)
        serial.to_parquet(temporary, index=False, compression="zstd")
        check = pd.read_parquet(temporary)
    else:
        _serialize_csv(frame).to_csv(temporary, index=False, compression="gzip")
        check = pd.read_csv(temporary, low_memory=False, compression="gzip")
    if len(check) != len(frame):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Atomic shard validation failed for {target}: row count mismatch")
    os.replace(temporary, target)
    return target


class ResumableMappingRunner:
    def __init__(self, config: MappingRunConfig):
        if config.workers != 1:
            raise ValueError("Product Two v2.0.2 supports one mapper process to avoid duplicate transformer memory.")
        if config.batch_size < 1 or config.shard_size < 1:
            raise ValueError("batch_size and shard_size must be positive")
        self.config = config
        self.queue = MappingQueue(config.queue_db)
        self.preflight: MappingPreflight | None = None
        self.primary = None
        self.fallback = None
        self._buffer: list[_Processed] = []
        self._processed_this_run = 0
        self._manifest_path = self.config.output_dir / "mapping_manifest.json"
        self._shard_id = 0

    def _prepare(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        output_names = [
            "reaction_mappings_rxnmapper",
            "reaction_mappings_mcs_fallback",
            "reaction_mappings_rejected",
            "reaction_mapping_exceptions",
        ]
        if not self.config.resume:
            for name in output_names:
                target = self.config.output_dir / name
                if target.exists():
                    shutil.rmtree(target)
            self.queue.reset()
            self._manifest_path.unlink(missing_ok=True)
        for name in output_names:
            (self.config.output_dir / name).mkdir(parents=True, exist_ok=True)
        if self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            self._shard_id = int(manifest.get("last_completed_shard", -1)) + 1
        else:
            self._shard_id = 0

        population = self.queue.populate_from_context(self.config.context_dir)
        recovered = self.queue.recover_stale(0 if self.config.resume else self.config.stale_after_minutes)
        LOGGER.info("Mapping queue: seen=%s inserted=%s stale_recovered=%s", population["seen"], population["inserted"], recovered)

    def _initialize_backends(self) -> None:
        self.preflight, mapper = resolve_mapping_backend(
            self.config.backend,
            fallback_backend=self.config.fallback_backend,
            allow_auto_fallback=self.config.allow_auto_fallback,
        )
        if self.preflight.primary_backend == "rxnmapper":
            self.primary = RXNMapperBackend(
                mapper=mapper,
                min_confidence=self.config.min_confidence,
                min_coverage=self.config.min_coverage,
                max_token_length=self.config.rxnmapper_token_limit,
            )
            if self.preflight.fallback_backend == "mcs_fallback":
                self.fallback = MCSBackend(
                    min_coverage=self.config.min_coverage,
                    timeout_seconds=self.config.mcs_timeout_seconds,
                    process_timeout_seconds=self.config.fallback_process_timeout_seconds,
                )
        else:
            self.primary = MCSBackend(
                min_coverage=self.config.min_coverage,
                timeout_seconds=self.config.mcs_timeout_seconds,
                process_timeout_seconds=self.config.fallback_process_timeout_seconds,
            )
            self.fallback = None
        LOGGER.info("RXNMapper token-length guard: %s tokens", self.config.rxnmapper_token_limit)
        if self.fallback is not None:
            LOGGER.info(
                "MCS fallback hard process timeout: %s seconds per record",
                self.config.fallback_process_timeout_seconds,
            )
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        preflight_path = self.config.reports_dir / "mapping_preflight.json"
        preflight_path.write_text(json.dumps(self.preflight.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def _to_item(
        queue_item: MappingQueueItem,
        result,
        elapsed_ms: float,
        *,
        strict_scientific: bool,
        primary_backend_error: str | None = None,
        token_source=None,
        fallback_attempt_count: int | None = None,
    ) -> MappingBatchItem:
        passed = result.status in {MappingStatus.MAPPED, MappingStatus.EXISTING}
        quarantined = bool(
            result.error_code in {
                "mcs_timeout",
                "mcs_worker_error",
                "mcs_worker_no_result",
                "mcs_process_error",
                "mcs_failed",
                "mcs_not_eligible",
            }
            or (result.backend == "mcs_fallback" and result.status == MappingStatus.FAILED)
        )
        validation_status = "quarantined" if quarantined else ("passed" if passed else result.status.value)
        scientific = bool(strict_scientific and passed)
        metadata = token_source or result
        exceptional_reason = None
        if getattr(metadata, "error_code", None) == "rxnmapper_sequence_too_long":
            exceptional_reason = "rxnmapper_sequence_too_long"
        elif quarantined:
            exceptional_reason = result.error_code or "mapping_exception"
        error_code = result.error_code
        if error_code is None and result.status == MappingStatus.LOW_CONFIDENCE:
            error_code = f"{result.backend}_low_confidence"
        if error_code is None and result.status == MappingStatus.FAILED:
            error_code = f"{result.backend}_failed"
        return MappingBatchItem(
            step_id=queue_item.step_id,
            route_id=queue_item.route_id,
            reaction_smiles=queue_item.reaction_smiles,
            reaction_signature=queue_item.reaction_signature,
            status=result.status,
            mapped_reaction_smiles=result.mapped_reaction_smiles,
            backend=result.backend,
            confidence=float(result.confidence),
            atom_coverage=float(result.atom_coverage),
            validation_status=validation_status,
            diagnostics=tuple(result.diagnostics),
            runtime_ms=elapsed_ms,
            scientific_eligibility=scientific,
            primary_backend_error=primary_backend_error,
            source_step_id=queue_item.source_step_id,
            source_route_id=queue_item.source_route_id,
            error_code=error_code,
            rxnmapper_token_count=(
                getattr(metadata, "rxnmapper_token_count", None)
                if getattr(metadata, "rxnmapper_token_count", None) is not None
                else queue_item.rxnmapper_token_count
            ),
            rxnmapper_token_limit=(
                getattr(metadata, "rxnmapper_token_limit", None)
                if getattr(metadata, "rxnmapper_token_limit", None) is not None
                else queue_item.rxnmapper_token_limit
            ),
            rxnmapper_eligible=(
                getattr(metadata, "rxnmapper_eligible", None)
                if getattr(metadata, "rxnmapper_eligible", None) is not None
                else queue_item.rxnmapper_eligible
            ),
            fallback_status=getattr(result, "fallback_status", None),
            fallback_attempt_count=(
                int(fallback_attempt_count)
                if fallback_attempt_count is not None
                else int(queue_item.fallback_attempt_count)
            ),
            exceptional_reason=exceptional_reason,
        )

    def _cached_primary_result(self, item: MappingQueueItem):
        if (
            self.preflight
            and self.preflight.primary_backend == "rxnmapper"
            and item.rxnmapper_eligible is False
            and item.rxnmapper_token_count is not None
            and item.rxnmapper_token_limit is not None
            and item.rxnmapper_token_count > item.rxnmapper_token_limit
        ):
            from reacts.chemistry.mapping import MappingResult

            return MappingResult(
                MappingStatus.FAILED,
                None,
                "rxnmapper",
                0.0,
                0.0,
                (
                    "rxnmapper_sequence_too_long",
                    f"token_count={item.rxnmapper_token_count}",
                    f"token_limit={item.rxnmapper_token_limit}",
                    "persisted_token_guard=true",
                ),
                error_code="rxnmapper_sequence_too_long",
                rxnmapper_token_count=item.rxnmapper_token_count,
                rxnmapper_token_limit=item.rxnmapper_token_limit,
                rxnmapper_eligible=False,
            )
        return None

    def _process_batch(self, queue_items: list[MappingQueueItem]) -> list[_Processed]:
        primary_results: list[Any | None] = [None] * len(queue_items)
        live_positions: list[int] = []
        live_reactions: list[str] = []
        for index, item in enumerate(queue_items):
            cached = self._cached_primary_result(item)
            if cached is None:
                live_positions.append(index)
                live_reactions.append(item.reaction_smiles)
            else:
                primary_results[index] = cached

        elapsed_each = 0.0
        if live_reactions:
            started = time.perf_counter()
            live_results = self.primary.map_batch(live_reactions)
            elapsed_each = ((time.perf_counter() - started) * 1000.0) / max(len(live_reactions), 1)
            if len(live_results) != len(live_positions):
                raise RuntimeError("Mapping backend broke output-order contract")
            for position, result in zip(live_positions, live_results):
                primary_results[position] = result

        if any(result is None for result in primary_results):
            raise RuntimeError("Primary mapping produced an incomplete result set")
        concrete_primary = [result for result in primary_results if result is not None]
        self.queue.record_primary_results(queue_items, concrete_primary)

        failed_positions = [
            index
            for index, result in enumerate(concrete_primary)
            if result.status == MappingStatus.FAILED
        ]
        fallback_by_position: dict[int, Any] = {}
        if self.fallback is not None and failed_positions:
            self.queue.mark_fallback_attempts([queue_items[index].step_id for index in failed_positions])
            for position in failed_positions:
                fallback_started = time.perf_counter()
                result = self.fallback.map_batch([queue_items[position].reaction_smiles])[0]
                fallback_elapsed = (time.perf_counter() - fallback_started) * 1000.0
                fallback_by_position[position] = (result, fallback_elapsed)

        processed: list[_Processed] = []
        for index, (queue_item, primary_result) in enumerate(zip(queue_items, concrete_primary)):
            primary_item = self._to_item(
                queue_item,
                primary_result,
                elapsed_each,
                strict_scientific=primary_result.backend in {"rxnmapper", "existing"},
            )
            fallback_item: MappingBatchItem | None = None
            final = primary_item
            exceptional = primary_result.error_code == "rxnmapper_sequence_too_long"
            if index in fallback_by_position:
                fallback_result, fallback_elapsed = fallback_by_position[index]
                fallback_item = self._to_item(
                    queue_item,
                    fallback_result,
                    fallback_elapsed,
                    strict_scientific=False,
                    primary_backend_error=";".join(primary_result.diagnostics),
                    token_source=primary_result,
                    fallback_attempt_count=queue_item.fallback_attempt_count + 1,
                )
                final = fallback_item
                exceptional = True
            processed.append(
                _Processed(
                    final=final,
                    primary=primary_item,
                    fallback=fallback_item,
                    exceptional=exceptional,
                )
            )
        return processed

    def _flush(self) -> None:
        if not self._buffer:
            return
        shard_name = f"part-{self._shard_id:05d}"
        primary_records = [item.primary.to_record(dataset_version=self.config.dataset_version) for item in self._buffer]
        fallback_records = [
            item.fallback.to_record(dataset_version=self.config.dataset_version)
            for item in self._buffer
            if item.fallback is not None
        ]
        rejected_records = [
            item.final.to_record(dataset_version=self.config.dataset_version)
            for item in self._buffer
            if not item.final.scientific_eligibility
        ]
        exceptional_records = [
            item.final.to_record(dataset_version=self.config.dataset_version)
            for item in self._buffer
            if item.exceptional
        ]
        primary_dir = (
            "reaction_mappings_rxnmapper"
            if self.preflight and self.preflight.primary_backend == "rxnmapper"
            else "reaction_mappings_mcs_fallback"
        )
        outputs = [
            _atomic_write_records(self.config.output_dir / primary_dir, self._shard_id, primary_records, self.config.prefer_parquet),
            _atomic_write_records(
                self.config.output_dir / "reaction_mappings_mcs_fallback",
                self._shard_id,
                fallback_records,
                self.config.prefer_parquet,
            ) if primary_dir != "reaction_mappings_mcs_fallback" else None,
            _atomic_write_records(
                self.config.output_dir / "reaction_mappings_rejected",
                self._shard_id,
                rejected_records,
                self.config.prefer_parquet,
            ),
            _atomic_write_records(
                self.config.output_dir / "reaction_mapping_exceptions",
                self._shard_id,
                exceptional_records,
                self.config.prefer_parquet,
            ),
        ]
        self.queue.complete_shard([item.final for item in self._buffer], shard_name)
        LOGGER.info("Committed mapping shard %s (%s queue rows)", self._shard_id, len(self._buffer))
        self._buffer.clear()
        self._shard_id += 1
        self._write_manifest([path for path in outputs if path is not None])

    def _write_manifest(self, new_outputs: list[Path] | None = None) -> dict[str, Any]:
        all_outputs = [path for path in self.config.output_dir.rglob("part-*") if path.is_file()]
        summary = self.queue.summary()
        exceptional_metrics = self.queue.exceptional_summary()
        context_manifest = self.config.context_dir / "dataset_manifest.json"
        manifest = {
            "run_type": "resumable_batched_atom_mapping",
            "dataset_version": self.config.dataset_version,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "preflight": self.preflight.to_dict() if self.preflight else None,
            "configuration": {
                key: (value.as_posix() if isinstance(value, Path) else value)
                for key, value in asdict(self.config).items()
            },
            "queue_summary": summary,
            "exceptional_mapping_metrics": exceptional_metrics,
            "completed_rows": summary.get("mapped", 0) + summary.get("low_confidence", 0) + summary.get("failed", 0),
            "last_completed_shard": self._shard_id - 1,
            "reproducibility": {
                "context_manifest_sha256": sha256_file(context_manifest) if context_manifest.exists() else None,
                "mapping_outputs_sha256": hash_paths(all_outputs, root=self.config.output_dir) if all_outputs else None,
                "configuration_sha256": canonical_json_hash(
                    {
                        key: (value.as_posix() if isinstance(value, Path) else value)
                        for key, value in asdict(self.config).items()
                    }
                ),
            },
            "contract": {
                "rxnmapper_and_mcs_outputs_are_segregated": True,
                "mcs_results_are_not_scientifically_eligible_by_default": True,
                "queue_is_persistent_and_resumable": True,
                "shards_are_atomic": True,
                "one_transformer_instance_per_process": True,
                "rxnmapper_token_limit_is_enforced_before_inference": True,
                "oversized_reactions_are_never_sent_to_rxnmapper": True,
                "mcs_fallback_isolated_per_record": True,
                "mcs_fallback_has_a_hard_process_timeout": True,
                "exceptional_records_are_deterministically_quarantined": True,
                "persisted_token_metadata_prevents_repeat_rxnmapper_attempts": True,
            },
        }
        temporary = self._manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temporary, self._manifest_path)
        return manifest

    def run(self) -> dict[str, Any]:
        # Resolve and initialize the requested backend before touching the
        # persistent queue or mapping outputs. Explicit RXNMapper failures are
        # therefore fail-fast and cannot start a misleading fallback run.
        self._initialize_backends()
        try:
            self._prepare()
            try:
                while True:
                    if self.config.max_rows is not None and self._processed_this_run >= self.config.max_rows:
                        break
                    requested = self.config.batch_size
                    if self.config.max_rows is not None:
                        requested = min(requested, self.config.max_rows - self._processed_this_run)
                    queue_items = self.queue.claim_batch(
                        requested,
                        backend_requested=self.preflight.primary_backend if self.preflight else self.config.backend,
                        max_attempts=self.config.max_attempts,
                    )
                    if not queue_items:
                        break
                    processed = self._process_batch(queue_items)
                    self._processed_this_run += len(processed)
                    for item in processed:
                        if item.exceptional:
                            # Commit normal completed rows before entering the
                            # exceptional lane, then commit the bounded fallback
                            # result independently. A pathological reaction can
                            # no longer hold thousands of normal mappings in RAM.
                            self._flush()
                            self._buffer.append(item)
                            self._flush()
                        else:
                            self._buffer.append(item)
                            if len(self._buffer) >= self.config.shard_size:
                                self._flush()
            except KeyboardInterrupt:
                LOGGER.warning("Mapping interrupted; committing completed in-memory results before exit.")
                self._flush()
                raise
            self._flush()
            return self._write_manifest()
        finally:
            if self.primary is not None:
                self.primary.close()
            if self.fallback is not None:
                self.fallback.close()
