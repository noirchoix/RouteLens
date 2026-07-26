from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reacts.contracts import MappingStatus
from reacts.mapping.backends import MCSBackend, RXNMapperBackend
from reacts.mapping.preflight import resolve_mapping_backend
from reacts.storage.tabular import iter_dataset


@dataclass
class MappingBenchmarkConfig:
    context_dir: Path
    report_path: Path
    backend: str = "rxnmapper"
    batch_sizes: tuple[int, ...] = (8, 16, 32, 64)
    sample_size: int = 512
    min_confidence: float = 0.50
    min_coverage: float = 0.60
    mcs_timeout_seconds: int = 3


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def benchmark_mapper(config: MappingBenchmarkConfig) -> dict[str, Any]:
    preflight, mapper = resolve_mapping_backend(config.backend, fallback_backend=None, allow_auto_fallback=False)
    if preflight.primary_backend == "rxnmapper":
        backend = RXNMapperBackend(mapper, min_confidence=config.min_confidence, min_coverage=config.min_coverage)
    else:
        backend = MCSBackend(min_coverage=config.min_coverage, timeout_seconds=config.mcs_timeout_seconds)

    reactions: list[str] = []
    for chunk in iter_dataset(config.context_dir, "mapping_candidates", columns=["reaction_smiles", "eligibility_status"]):
        subset = chunk.loc[chunk["eligibility_status"].astype(str) == "eligible"]
        reactions.extend(subset["reaction_smiles"].astype(str).tolist())
        if len(reactions) >= config.sample_size:
            reactions = reactions[: config.sample_size]
            break
    if not reactions:
        raise ValueError("No eligible mapping candidates are available for benchmarking")

    results: list[dict[str, Any]] = []
    try:
        for batch_size in config.batch_sizes:
            latencies: list[float] = []
            status_counts: dict[str, int] = {}
            confidences: list[float] = []
            before_memory = _rss_mb()
            started = time.perf_counter()
            for offset in range(0, len(reactions), batch_size):
                batch = reactions[offset : offset + batch_size]
                batch_started = time.perf_counter()
                mapped = backend.map_batch(batch)
                latencies.append((time.perf_counter() - batch_started) * 1000.0)
                for item in mapped:
                    status_counts[item.status.value] = status_counts.get(item.status.value, 0) + 1
                    confidences.append(float(item.confidence))
            elapsed = time.perf_counter() - started
            after_memory = _rss_mb()
            sorted_latency = sorted(latencies)
            p95_index = max(0, min(len(sorted_latency) - 1, int(len(sorted_latency) * 0.95) - 1))
            results.append(
                {
                    "batch_size": batch_size,
                    "rows": len(reactions),
                    "elapsed_seconds": elapsed,
                    "reactions_per_second": len(reactions) / max(elapsed, 1e-9),
                    "median_batch_latency_ms": statistics.median(latencies),
                    "p95_batch_latency_ms": sorted_latency[p95_index],
                    "rss_before_mb": before_memory,
                    "rss_after_mb": after_memory,
                    "rss_delta_mb": None if before_memory is None or after_memory is None else after_memory - before_memory,
                    "status_counts": status_counts,
                    "mapping_success_rate": status_counts.get(MappingStatus.MAPPED.value, 0) / len(reactions),
                    "mean_confidence": statistics.mean(confidences) if confidences else 0.0,
                }
            )
    finally:
        backend.close()

    eligible = [row for row in results if row["status_counts"].get("failed", 0) < len(reactions)]
    recommended = max(eligible, key=lambda row: row["reactions_per_second"])["batch_size"] if eligible else None
    report = {
        "benchmark_type": "rxnmapper_batch_size",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight.to_dict(),
        "sample_size": len(reactions),
        "results": results,
        "recommended_batch_size": recommended,
        "selection_rule": "highest successful throughput within the observed process memory envelope",
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, config.report_path)
    return report
