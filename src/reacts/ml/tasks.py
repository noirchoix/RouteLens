from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from reacts.contracts import ParseFailureClass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_column: str
    eligibility: Callable[[pd.DataFrame], pd.Series]
    input_column: str = "reaction_smiles"
    required_columns: tuple[str, ...] = ()
    fixed_classes: tuple[str, ...] | None = None
    min_class_count: int = 100
    training_mode: str = "classifier"
    governance_note: str | None = None


PARSE_CLASSES = tuple(item.value for item in ParseFailureClass)


TASKS: dict[str, TaskSpec] = {
    "parse_validity": TaskSpec(
        name="parse_validity",
        target_column="parse_ok",
        eligibility=lambda df: pd.Series(True, index=df.index),
        required_columns=(),
        fixed_classes=("False", "True"),
        min_class_count=1,
    ),
    "parse_failure_class": TaskSpec(
        name="parse_failure_class",
        target_column="contextual_parse_failure_class",
        eligibility=lambda df: df["contextual_parse_failure_class"].notna(),
        input_column="original_reaction_smiles",
        required_columns=("contextual_parse_failure_class", "original_reaction_smiles"),
        fixed_classes=PARSE_CLASSES,
        min_class_count=20,
    ),
    "repairability": TaskSpec(
        name="repairability",
        target_column="repairable",
        eligibility=lambda df: ~df["contextual_parse_ok"].fillna(False).astype(bool),
        input_column="original_reaction_smiles",
        required_columns=("repairable", "contextual_parse_ok", "original_reaction_smiles"),
        fixed_classes=("False", "True"),
        min_class_count=1,
        training_mode="deterministic_audit",
        governance_note=(
            "Repairability is a deterministic repair-candidate audit, not a validated "
            "supervised classifier. The current parse-invalid population contains no "
            "eligible positive class after strict repair acceptance."
        ),
    ),
    "reaction_family": TaskSpec(
        name="reaction_family",
        target_column="reaction_family",
        eligibility=lambda df: df["eligible_mapping_models"].fillna(False).astype(bool) & df["reaction_family"].notna(),
        input_column="canonical_resolved_reaction_smiles",
        required_columns=("eligible_mapping_models", "reaction_family", "canonical_resolved_reaction_smiles"),
        fixed_classes=None,
        min_class_count=100,
    ),
    "primary_solvent": TaskSpec(
        name="primary_solvent",
        target_column="solvent_primary",
        eligibility=lambda df: df["eligible_condition_models"].fillna(False).astype(bool) & df["solvent_primary"].notna(),
        input_column="canonical_reaction_smiles",
        required_columns=("eligible_condition_models", "canonical_reaction_smiles"),
        fixed_classes=None,
        min_class_count=100,
    ),
    "time_bucket": TaskSpec(
        name="time_bucket",
        target_column="time_bucket",
        eligibility=lambda df: df["eligible_condition_models"].fillna(False).astype(bool)
        & df["time_bucket"].notna()
        & df["condition_extraction_confidence"].isin(["high", "medium"]),
        input_column="canonical_reaction_smiles",
        required_columns=("eligible_condition_models", "condition_extraction_confidence", "canonical_reaction_smiles"),
        fixed_classes=("<1h", "1-4h", "4-16h", "16-24h", "24h+"),
        min_class_count=20,
    ),
    "temperature_bucket": TaskSpec(
        name="temperature_bucket",
        target_column="temperature_bucket",
        eligibility=lambda df: df["eligible_condition_models"].fillna(False).astype(bool)
        & df["temperature_bucket"].notna()
        & df["condition_extraction_confidence"].isin(["high", "medium"]),
        input_column="canonical_reaction_smiles",
        required_columns=("eligible_condition_models", "condition_extraction_confidence", "canonical_reaction_smiles"),
        fixed_classes=("<0", "0-25", "25-60", "60-100", "100+"),
        min_class_count=20,
    ),
    "agent_presence": TaskSpec(
        name="agent_presence",
        target_column="agent_present",
        eligibility=lambda df: df["eligible_condition_models"].fillna(False).astype(bool),
        input_column="canonical_reaction_smiles",
        required_columns=("eligible_condition_models", "canonical_reaction_smiles"),
        fixed_classes=("False", "True"),
        min_class_count=1,
    ),
}
