from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reacts.storage.tabular import iter_dataset


@dataclass(frozen=True)
class RobustConditionStats:
    count: int
    median: float
    mad: float
    q01: float
    q99: float


class ConditionAnomalyModel:
    """Family-conditional robust condition anomaly detector.

    The detector is intentionally transparent: it combines plausibility bounds with
    median absolute deviation and family quantiles. It does not claim experimental
    infeasibility; it flags records that are unusual relative to the corpus.
    """

    def __init__(self, statistics: dict[str, dict[str, RobustConditionStats]]):
        self.statistics = statistics

    @classmethod
    def fit(cls, canonical_dir: Path) -> "ConditionAnomalyModel":
        values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        columns = ["reaction_family", "temperature_c", "time_h", "split", "eligible_contextual_models"]
        for chunk in iter_dataset(canonical_dir, "steps", columns=columns):
            subset = chunk.loc[
                (chunk["split"] == "train")
                & chunk["eligible_contextual_models"].fillna(False).astype(bool)
            ]
            for _, row in subset.iterrows():
                family = str(row.get("reaction_family") or "unknown")
                for field in ["temperature_c", "time_h"]:
                    try:
                        value = float(row[field])
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(value):
                        continue
                    values[family][field].append(value)
                    values["__global__"][field].append(value)
        stats: dict[str, dict[str, RobustConditionStats]] = {}
        for family, fields in values.items():
            stats[family] = {}
            for field, raw in fields.items():
                array = np.asarray(raw, dtype=float)
                median = float(np.median(array))
                mad = float(np.median(np.abs(array - median)))
                stats[family][field] = RobustConditionStats(
                    count=len(array),
                    median=median,
                    mad=max(mad, 1e-6),
                    q01=float(np.quantile(array, 0.01)),
                    q99=float(np.quantile(array, 0.99)),
                )
        return cls(stats)

    def score(self, *, reaction_family: str | None, temperature_c: float | None, time_h: float | None) -> dict[str, Any]:
        family = reaction_family if reaction_family in self.statistics else "__global__"
        family_stats = self.statistics.get(family, self.statistics.get("__global__", {}))
        reasons: list[str] = []
        component_scores: dict[str, float] = {}
        for field, value in [("temperature_c", temperature_c), ("time_h", time_h)]:
            if value is None or field not in family_stats:
                continue
            stats = family_stats[field]
            robust_z = abs(float(value) - stats.median) / (1.4826 * stats.mad)
            quantile_outlier = float(value) < stats.q01 or float(value) > stats.q99
            score = min(1.0, robust_z / 8.0 + (0.25 if quantile_outlier else 0.0))
            component_scores[field] = score
            if quantile_outlier:
                reasons.append(f"{field} is outside the family 1st-99th percentile range")
            if robust_z > 6:
                reasons.append(f"{field} has robust z-score {robust_z:.2f}")
        anomaly_score = max(component_scores.values(), default=0.0)
        return {
            "anomaly_score": anomaly_score,
            "component_scores": component_scores,
            "reasons": reasons,
            "reference_family": family,
        }

    def save(self, path: Path) -> None:
        payload = {
            family: {field: asdict(stats) for field, stats in fields.items()}
            for family, fields in self.statistics.items()
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ConditionAnomalyModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            {
                family: {field: RobustConditionStats(**stats) for field, stats in fields.items()}
                for family, fields in payload.items()
            }
        )


class RouteQualityScorer:
    WEIGHTS = {
        "parse": 0.30,
        "resolution": 0.20,
        "route_continuity": 0.15,
        "condition_completeness": 0.15,
        "condition_plausibility": 0.10,
        "mapping": 0.10,
    }

    @classmethod
    def score(cls, components: dict[str, float]) -> dict[str, Any]:
        normalized = {key: min(1.0, max(0.0, float(components.get(key, 0.0)))) for key in cls.WEIGHTS}
        total = sum(cls.WEIGHTS[key] * normalized[key] for key in cls.WEIGHTS)
        return {"score": total, "components": normalized, "weights": cls.WEIGHTS}
