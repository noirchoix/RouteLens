from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from reacts.contracts import ModelStage


@dataclass(frozen=True)
class PromotionRule:
    minimum_macro_f1: float
    minimum_balanced_accuracy: float
    maximum_calibration_error: float
    minimum_accuracy_ratio_to_majority: float = 0.75
    minimum_rows: int = 500
    requested_stage: ModelStage = ModelStage.PRODUCTION
    permitted_use: str = "general"


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    stage: ModelStage
    reasons: tuple[str, ...]
    rule: PromotionRule

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "stage": self.stage.value,
            "reasons": list(self.reasons),
            "rule": {**asdict(self.rule), "requested_stage": self.rule.requested_stage.value},
        }


DEFAULT_PROMOTION_RULES: dict[str, PromotionRule] = {
    "parse_validity": PromotionRule(0.90, 0.90, 0.08, 0.95, permitted_use="deterministic-parser triage"),
    "parse_failure_class": PromotionRule(0.70, 0.70, 0.10, 0.90, permitted_use="failure routing"),
    "repairability": PromotionRule(0.72, 0.72, 0.10, 0.90, permitted_use="repair queue ranking"),
    "primary_solvent": PromotionRule(0.40, 0.40, 0.12, 0.90, requested_stage=ModelStage.STAGING, permitted_use="top-k evidence-backed suggestions"),
    "solvent_multilabel": PromotionRule(0.40, 0.40, 0.12, 0.90, requested_stage=ModelStage.STAGING, permitted_use="top-k evidence-backed suggestions"),
    "agent_presence": PromotionRule(0.60, 0.65, 0.12, 0.75, requested_stage=ModelStage.SCREENING, permitted_use="high-recall screening"),
    "agent_multilabel": PromotionRule(0.35, 0.35, 0.15, 0.75, requested_stage=ModelStage.SCREENING, permitted_use="agent-family retrieval"),
    "time_bucket": PromotionRule(0.35, 0.35, 0.12, 0.90, requested_stage=ModelStage.EXPERIMENTAL, permitted_use="research only"),
    "temperature_bucket": PromotionRule(0.45, 0.50, 0.12, 0.90, requested_stage=ModelStage.EXPERIMENTAL, permitted_use="research only"),
    "time_regression": PromotionRule(0.0, 0.0, 0.15, 0.0, requested_stage=ModelStage.STAGING, permitted_use="prediction intervals"),
    "temperature_regression": PromotionRule(0.0, 0.0, 0.15, 0.0, requested_stage=ModelStage.STAGING, permitted_use="prediction intervals"),
    "reaction_family": PromotionRule(0.70, 0.70, 0.10, 0.90, permitted_use="structural taxonomy"),
}


BASELINE_STAGE_OVERRIDES: dict[str, ModelStage] = {
    "parse_validity": ModelStage.BASELINE,
    "primary_solvent": ModelStage.STAGING,
    "agent_presence": ModelStage.SCREENING,
    "time_bucket": ModelStage.EXPERIMENTAL,
    "temperature_bucket": ModelStage.EXPERIMENTAL,
}


def decide_promotion(
    task: str,
    metrics: dict[str, Any],
    *,
    calibration_error: float | None = None,
    rules: dict[str, PromotionRule] | None = None,
) -> PromotionDecision:
    rule = (rules or DEFAULT_PROMOTION_RULES).get(
        task,
        PromotionRule(0.60, 0.60, 0.10, 0.90, requested_stage=ModelStage.VALIDATED, permitted_use="task-specific validation required"),
    )
    evaluation = metrics.get("test") or metrics.get("validation") or metrics
    reasons: list[str] = []
    rows = int(evaluation.get("rows", 0))
    macro_f1 = float(evaluation.get("macro_f1", 0.0))
    balanced = float(evaluation.get("balanced_accuracy", 0.0))
    accuracy = float(evaluation.get("accuracy", 0.0))
    majority = float(evaluation.get("majority_accuracy", 0.0))
    if rows < rule.minimum_rows:
        reasons.append(f"evaluation rows {rows} below minimum {rule.minimum_rows}")
    if macro_f1 < rule.minimum_macro_f1:
        reasons.append(f"macro_f1 {macro_f1:.4f} below {rule.minimum_macro_f1:.4f}")
    if balanced < rule.minimum_balanced_accuracy:
        reasons.append(f"balanced_accuracy {balanced:.4f} below {rule.minimum_balanced_accuracy:.4f}")
    if majority > 0 and accuracy / majority < rule.minimum_accuracy_ratio_to_majority:
        reasons.append(
            f"accuracy-to-majority ratio {accuracy / majority:.4f} below {rule.minimum_accuracy_ratio_to_majority:.4f}"
        )
    if calibration_error is not None and calibration_error > rule.maximum_calibration_error:
        reasons.append(f"calibration error {calibration_error:.4f} above {rule.maximum_calibration_error:.4f}")
    approved = not reasons
    if approved:
        stage = rule.requested_stage
    elif rule.requested_stage in {ModelStage.PRODUCTION, ModelStage.STAGING, ModelStage.SCREENING}:
        stage = ModelStage.CANDIDATE
    else:
        stage = ModelStage.EXPERIMENTAL
    return PromotionDecision(approved=approved, stage=stage, reasons=tuple(reasons), rule=rule)
