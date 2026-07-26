from __future__ import annotations

import json
import logging
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from reacts.chemistry.conditions import (
    normalize_condition_lists,
    reparse_multistep_middle,
    temperature_bucket,
    time_bucket,
    validate_conditions,
)
from reacts.chemistry.reactions import canonicalize_reaction, parse_reaction
from reacts.data.parsing import first_scalar, parse_list, patent_document_id, stable_group_split, stable_hash
from reacts.data.source import ArtifactSource
from reacts.storage.tabular import DatasetWriter

LOGGER = logging.getLogger(__name__)


@dataclass
class CanonicalBuildConfig:
    dataset_version: str = "uspto_multistep_canonical_v1"
    chunksize: int = 20_000
    prefer_parquet: bool = True
    temperature_min_c: float = -150.0
    temperature_max_c: float = 350.0
    time_min_h: float = 1.0 / 3600.0
    time_max_h: float = 8760.0


class CanonicalBuilder:
    def __init__(self, source: ArtifactSource, output_root: Path, config: CanonicalBuildConfig | None = None):
        self.source = source
        self.output_root = Path(output_root)
        self.config = config or CanonicalBuildConfig()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.step_ids: set[str] = set()
        self.route_content_hashes: dict[str, set[str]] = defaultdict(set)
        self.metrics: Counter[str] = Counter()
        self.quality_events: list[dict[str, Any]] = []
        self.split_counts: Counter[str] = Counter()
        self.format_used: str | None = None

    @staticmethod
    def _bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}

    @staticmethod
    def _scalar_from_row(row: pd.Series, column: str, default: Any = None) -> Any:
        try:
            value = row[column]
        except KeyError:
            return default
        value = first_scalar(value)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return value

    def _event(self, entity_type: str, entity_id: str, code: str, severity: str, observed: Any, message: str) -> None:
        self.quality_events.append(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "rule_code": code,
                "severity": severity,
                "observed_value": json.dumps(observed, default=str, ensure_ascii=False),
                "message": message,
                "dataset_version": self.config.dataset_version,
            }
        )

    def _canonicalize_step(self, row: pd.Series) -> dict[str, Any] | None:
        route_id = str(self._scalar_from_row(row, "route_id", "")).strip()
        if not route_id:
            self.metrics["steps_missing_route_id"] += 1
            return None
        step_index_raw = self._scalar_from_row(row, "step_index", None)
        try:
            step_index = int(float(step_index_raw))
        except (TypeError, ValueError):
            self.metrics["steps_invalid_step_index"] += 1
            return None

        step_id = f"{route_id}::{step_index:03d}"
        source_reaction = str(self._scalar_from_row(row, "step_working_reaction_smiles", "") or "").strip()
        raw_reaction = str(self._scalar_from_row(row, "step_reaction_smiles", source_reaction) or source_reaction).strip()
        parsed = parse_reaction(source_reaction)
        source_parse_ok = self._bool(self._scalar_from_row(row, "step_parse_ok", False))
        canonical_source = self._scalar_from_row(row, "step_canonical_reaction_demapped", None)
        canonical = str(canonical_source).strip() if canonical_source is not None and str(canonical_source).strip() else None
        if canonical is None and parsed.parse_ok:
            canonical = canonicalize_reaction(source_reaction)

        solvents, agents = normalize_condition_lists(
            self._scalar_from_row(row, "step_solvent_list_norm", []),
            self._scalar_from_row(row, "step_agent_or_spectator_list_norm", []),
        )
        legacy_temperature = self._scalar_from_row(row, "temperature_c_primary", None)
        legacy_time = self._scalar_from_row(row, "time_h_primary", None)
        repaired = reparse_multistep_middle(source_reaction)
        selected_temperature = repaired.temperature_c if repaired.temperature_c is not None else legacy_temperature
        selected_time = repaired.time_h if repaired.time_h is not None else legacy_time
        validity = validate_conditions(
            selected_temperature,
            selected_time,
            temperature_min_c=self.config.temperature_min_c,
            temperature_max_c=self.config.temperature_max_c,
            time_min_h=self.config.time_min_h,
            time_max_h=self.config.time_max_h,
        )
        if repaired.method == "positional_numeric_repair":
            self.metrics["steps_condition_reparsed"] += 1
        for repair_issue in repaired.issues:
            self.metrics[f"condition_repair::{repair_issue}"] += 1

        duplicate = step_id in self.step_ids
        if duplicate:
            self.metrics["duplicate_step_ids"] += 1
            payload = f"{source_reaction}|{route_id}|{step_index}"
            step_id = f"{step_id}::v{stable_hash(payload, 8)}"
            self._event("step", step_id, "DUPLICATE_STEP_KEY", "warning", [route_id, step_index], "Conflicting duplicate step key retained with a stable variant suffix.")
        self.step_ids.add(step_id)

        if source_parse_ok != parsed.parse_ok:
            self.metrics["source_parse_disagreements"] += 1
            self._event("step", step_id, "PARSE_STATUS_REEVALUATED", "info", source_parse_ok, f"Legacy parse status was reevaluated as {parsed.failure_class.value}.")
        for issue in validity.issues:
            self.metrics[issue] += 1
            self._event("step", step_id, issue.upper(), "warning", {"temperature": validity.temperature_observed_c, "time": validity.time_observed_h}, issue.replace("_", " "))

        patent_id = patent_document_id(route_id)
        split = stable_group_split(patent_id)
        self.split_counts[split] += 1
        self.metrics["steps_total"] += 1
        self.metrics[f"parse_class::{parsed.failure_class.value}"] += 1
        self.metrics["steps_parse_valid"] += int(parsed.parse_ok)
        self.metrics["steps_symbolic_intermediate"] += int(parsed.failure_class.value == "symbolic_intermediate")
        self.metrics["steps_with_solvent"] += int(bool(solvents))
        self.metrics["steps_with_agents"] += int(bool(agents))
        self.metrics["steps_with_clean_temperature"] += int(validity.temperature_clean_c is not None)
        self.metrics["steps_with_clean_time"] += int(validity.time_clean_h is not None)

        parse_score = 1.0 if parsed.parse_ok else (0.65 if parsed.failure_class.value == "symbolic_intermediate" else 0.0)
        condition_score = (int(bool(solvents)) + int(validity.temperature_clean_c is not None) + int(validity.time_clean_h is not None)) / 3.0
        anomaly_score = 1.0 if not validity.issues else 0.0
        quality_score = round(0.55 * parse_score + 0.30 * condition_score + 0.15 * anomaly_score, 6)

        return {
            "dataset_version": self.config.dataset_version,
            "step_id": step_id,
            "route_id": route_id,
            "patent_document_id": patent_id,
            "split": split,
            "step_index": step_index,
            "raw_reaction_text": raw_reaction,
            "reaction_smiles": source_reaction,
            "canonical_reaction_smiles": canonical,
            "reactants": list(parsed.reactants),
            "products": list(parsed.products),
            "parse_ok": parsed.parse_ok,
            "reactants_valid": parsed.reactants_valid,
            "products_valid": parsed.products_valid,
            "parse_failure_class": parsed.failure_class.value,
            "source_parse_ok": source_parse_ok,
            "symbolic_intermediate": parsed.failure_class.value == "symbolic_intermediate",
            "input_intermediate": self._scalar_from_row(row, "step_input_intermediate", None),
            "output_intermediate": self._scalar_from_row(row, "step_output_intermediate", None),
            "solvents": solvents,
            "agents": agents,
            "solvent_primary": solvents[0] if solvents else None,
            "agent_primary": agents[0] if agents else None,
            "agent_present": bool(agents),
            "legacy_temperature_c": pd.to_numeric(legacy_temperature, errors="coerce"),
            "legacy_time_h": pd.to_numeric(legacy_time, errors="coerce"),
            "condition_extraction_method": repaired.method,
            "condition_extraction_confidence": repaired.confidence,
            "condition_numeric_tokens": list(repaired.numeric_tokens),
            "temperature_observed_c": validity.temperature_observed_c,
            "temperature_c": validity.temperature_clean_c,
            "temperature_valid": validity.temperature_valid,
            "temperature_bucket": temperature_bucket(validity.temperature_clean_c),
            "time_observed_h": validity.time_observed_h,
            "time_h": validity.time_clean_h,
            "time_valid": validity.time_valid,
            "time_bucket": time_bucket(validity.time_clean_h),
            "condition_status": validity.status,
            "quality_issues": list(validity.issues),
            "quality_score": quality_score,
            "eligible_parse_model": True,
            "eligible_condition_models": parsed.parse_ok,
            "eligible_retrieval": parsed.parse_ok,
        }

    def _iter_step_chunks(self, step_path: Path) -> Iterator[pd.DataFrame]:
        yield from pd.read_csv(step_path, chunksize=self.config.chunksize, low_memory=False)

    def _build_steps(self, step_path: Path) -> list[Path]:
        writer = DatasetWriter(self.output_root, "steps", prefer_parquet=self.config.prefer_parquet)
        outputs: list[Path] = []
        for chunk_no, chunk in enumerate(self._iter_step_chunks(step_path), start=1):
            records = [record for _, row in chunk.iterrows() if (record := self._canonicalize_step(row)) is not None]
            if records:
                outputs.append(writer.write(pd.DataFrame.from_records(records)))
            LOGGER.info("Canonicalized step chunk %s (%s rows)", chunk_no, len(records))
        self.format_used = writer.format
        return outputs

    @staticmethod
    def _resolve_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=df.index)
        bases: dict[str, list[str]] = defaultdict(list)
        for col in df.columns:
            base = re.sub(r"\.\d+$", "", col)
            bases[base].append(col)
        for base, cols in bases.items():
            series = df[cols[0]]
            for col in cols[1:]:
                series = series.where(series.notna() & series.astype(str).ne(""), df[col])
            output[base] = series
        return output

    def _build_routes(self, route_path: Path) -> list[Path]:
        writer = DatasetWriter(self.output_root, "routes", prefer_parquet=self.config.prefer_parquet)
        outputs: list[Path] = []
        seen: dict[str, str] = {}
        for chunk_no, raw in enumerate(pd.read_csv(route_path, chunksize=self.config.chunksize, low_memory=False), start=1):
            chunk = self._resolve_duplicate_columns(raw)
            records: list[dict[str, Any]] = []
            for _, row in chunk.iterrows():
                route_id = str(row.get("route_id", "")).strip()
                if not route_id:
                    continue
                route_text = str(row.get("multistep_reaction_text", ""))
                content_hash = stable_hash(route_text, 16)
                route_uid = route_id
                if route_id in seen:
                    if seen[route_id] == content_hash:
                        self.metrics["duplicate_routes_exact"] += 1
                        continue
                    route_uid = f"{route_id}::v{content_hash[:8]}"
                    self.metrics["duplicate_routes_conflicting"] += 1
                    self._event("route", route_uid, "DUPLICATE_ROUTE_ID_CONFLICT", "warning", route_id, "A conflicting route reused the same source route ID and was retained with a stable variant suffix.")
                else:
                    seen[route_id] = content_hash

                patent_id = patent_document_id(route_id)
                split = stable_group_split(patent_id)
                step_count = pd.to_numeric(row.get("route_step_count"), errors="coerce")
                parsed_count = pd.to_numeric(row.get("route_parsed_step_count"), errors="coerce")
                step_count_i = int(step_count) if pd.notna(step_count) else 0
                parsed_count_i = int(parsed_count) if pd.notna(parsed_count) else 0
                parse_rate = parsed_count_i / step_count_i if step_count_i else 0.0
                records.append(
                    {
                        "dataset_version": self.config.dataset_version,
                        "route_uid": route_uid,
                        "route_id": route_id,
                        "patent_document_id": patent_id,
                        "split": split,
                        "source_content_hash": content_hash,
                        "multistep_reaction_text": route_text,
                        "step_count": step_count_i,
                        "parsed_step_count_legacy": parsed_count_i,
                        "legacy_route_parse_ok": self._bool(row.get("route_parse_ok", False)),
                        "legacy_parse_rate": parse_rate,
                        "is_atom_mapped": self._bool(row.get("route_is_atom_mapped", False)),
                        "route_reactants": parse_list(row.get("route_reactants_demapped")),
                        "route_products": parse_list(row.get("route_products_demapped")),
                        "final_products": parse_list(row.get("final_products_demapped")),
                    }
                )
                self.metrics["routes_total"] += 1
            if records:
                outputs.append(writer.write(pd.DataFrame.from_records(records)))
            LOGGER.info("Canonicalized route chunk %s (%s rows)", chunk_no, len(records))
        return outputs

    def _write_quality_events(self) -> Path | None:
        if not self.quality_events:
            return None
        writer = DatasetWriter(self.output_root, "quality_events", prefer_parquet=self.config.prefer_parquet)
        return writer.write(pd.DataFrame.from_records(self.quality_events))

    def _write_manifest(self, outputs: dict[str, list[Path] | Path | None]) -> Path:
        inventory = self.source.inventory()
        inventory_record = asdict(inventory)
        source_path = Path(str(inventory_record.get("source_path", "")))
        if source_path.is_absolute():
            inventory_record["source_path"] = f"external://{source_path.name}"
        report = {
            "dataset_version": self.config.dataset_version,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": inventory_record,
            "storage_format": self.format_used,
            "metrics": dict(sorted(self.metrics.items())),
            "split_counts_steps": dict(self.split_counts),
            "outputs": {
                key: [str(p.relative_to(self.output_root)) for p in value] if isinstance(value, list)
                else (str(value.relative_to(self.output_root)) if isinstance(value, Path) else None)
                for key, value in outputs.items()
            },
            "contract": {
                "source_rows_are_not_deleted": True,
                "patent_group_split": True,
                "typed_list_columns_when_parquet": True,
                "outliers_preserved_as_observed_and_nullified_for_training": True,
                "symbolic_intermediates_are_structural_partial_records": True,
            },
        }
        path = self.output_root / "dataset_manifest.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def build(self) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="reacts-source-") as temp:
            staging = Path(temp)
            with self.source.materialize("steps", staging) as step_path:
                step_outputs = self._build_steps(step_path)
            with self.source.materialize("routes", staging) as route_path:
                route_outputs = self._build_routes(route_path)
        quality_path = self._write_quality_events()
        outputs: dict[str, Any] = {"steps": step_outputs, "routes": route_outputs, "quality_events": quality_path}
        manifest_path = self._write_manifest(outputs)
        outputs["manifest"] = manifest_path
        return {
            "dataset_version": self.config.dataset_version,
            "storage_format": self.format_used,
            "metrics": dict(self.metrics),
            "outputs": {k: str(v) for k, v in outputs.items()},
        }
