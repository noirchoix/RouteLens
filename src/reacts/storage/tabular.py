from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


LIST_COLUMNS = {
    "reactants",
    "products",
    "solvents",
    "agents",
    "quality_issues",
    "route_reactants",
    "route_products",
    "final_products",
    "condition_numeric_tokens",
    "connecting_molecules",
    "connecting_molecule_ids",
    "resolved_molecules",
    "formed_bonds",
    "broken_bonds",
    "changed_bonds",
    "changed_atom_maps",
    "atom_environment_changes",
    "candidate_labels",
    "reaction_families",
    "solvent_families",
    "agent_families",
    "catalyst_families",
}
JSON_COLUMNS = {
    "diagnostics",
    "condition_payload",
    "quality_components",
    "family_distribution",
    "provenance",
}


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


JSON_COLUMN_DEFAULTS: dict[str, Any] = {
    "diagnostics": [],
}


def _fresh_default(default: Any) -> Any:
    if isinstance(default, dict):
        return dict(default)
    if isinstance(default, list):
        return list(default)
    return default


def _native_container(value: Any) -> Any:
    """Normalize Arrow/numpy containers without treating them as scalar nulls."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)

    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        converted = as_py()
        if converted is not value:
            return _native_container(converted)

    to_pylist = getattr(value, "to_pylist", None)
    if callable(to_pylist):
        return to_pylist()

    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes, bytearray)):
        converted = tolist()
        if isinstance(converted, tuple):
            return list(converted)
        return converted

    return value


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    # Container-valued pd.isna results are deliberately not collapsed with
    # bool(...), which raises for numpy/Arrow arrays and is not a scalar-null
    # test in any case.
    ndim = getattr(missing, "ndim", None)
    if ndim == 0:
        try:
            return bool(missing)
        except (TypeError, ValueError):
            return False
    return False


def _serialize(value: Any):
    normalized = _native_container(value)
    if isinstance(normalized, (list, tuple, dict)):
        return json.dumps(normalized, ensure_ascii=False, default=str)
    return normalized


def _deserialize(value: Any, *, default: Any):
    normalized = _native_container(value)
    if isinstance(normalized, dict):
        return normalized
    if isinstance(normalized, list):
        if not normalized and isinstance(default, dict):
            return _fresh_default(default)
        return normalized
    if _is_missing_scalar(normalized):
        return _fresh_default(default)
    if isinstance(normalized, str):
        text = normalized.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
    return normalized


def _json_default(column: str) -> Any:
    return JSON_COLUMN_DEFAULTS.get(column, {})


def serialize_json_contract_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with JSON contract columns encoded as canonical text."""
    serial = frame.copy()
    for column in JSON_COLUMNS.intersection(serial.columns):
        serial[column] = serial[column].map(_serialize)
    return serial


@dataclass
class DatasetWriter:
    root: Path
    dataset_name: str
    prefer_parquet: bool = True
    part_index: int = 0

    @property
    def output_dir(self) -> Path:
        path = self.root / self.dataset_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def format(self) -> str:
        return "parquet" if self.prefer_parquet and parquet_available() else "csv.gz"

    def write(self, df: pd.DataFrame) -> Path:
        suffix = ".parquet" if self.format == "parquet" else ".csv.gz"
        target = self.output_dir / f"part-{self.part_index:05d}{suffix}"
        self.part_index += 1

        # JSON contract columns are stored as canonical JSON text in every
        # physical format. Letting PyArrow infer arbitrary Python dictionaries
        # creates unstable struct schemas: an empty dictionary becomes a
        # zero-field struct, which Parquet cannot write, while non-empty
        # dictionaries may produce different child fields between shards.
        serial = serialize_json_contract_columns(df)

        if self.format == "parquet":
            serial.to_parquet(target, index=False, compression="zstd")
        else:
            for col in LIST_COLUMNS.intersection(serial.columns):
                serial[col] = serial[col].map(_serialize)
            serial.to_csv(target, index=False, compression="gzip")
        return target


def iter_dataset(
    root: Path,
    dataset_name: str,
    chunksize: int = 50_000,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    directory = Path(root) / dataset_name
    parquet_parts = sorted(directory.glob("*.parquet"))
    csv_parts = sorted(directory.glob("*.csv.gz"))
    if parquet_parts:
        for part in parquet_parts:
            frame = pd.read_parquet(part, columns=columns)
            for col in JSON_COLUMNS.intersection(frame.columns):
                frame[col] = frame[col].map(
                    lambda value, column=col: _deserialize(
                        value,
                        default=_json_default(column),
                    )
                )
            yield frame
        return
    if csv_parts:
        for part in csv_parts:
            for chunk in pd.read_csv(part, chunksize=chunksize, usecols=columns, low_memory=False):
                for col in LIST_COLUMNS.intersection(chunk.columns):
                    chunk[col] = chunk[col].map(lambda value: _deserialize(value, default=[]))
                for col in JSON_COLUMNS.intersection(chunk.columns):
                    chunk[col] = chunk[col].map(
                        lambda value, column=col: _deserialize(
                            value,
                            default=_json_default(column),
                        )
                    )
                yield chunk
        return
    raise FileNotFoundError(f"No dataset parts found in {directory}")


def dataset_rows(root: Path, dataset_name: str) -> int:
    return sum(len(chunk) for chunk in iter_dataset(root, dataset_name))
