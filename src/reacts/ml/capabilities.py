from __future__ import annotations

from typing import Any

STAGE_POLICIES: dict[str, dict[str, Any]] = {
    "production": {
        "enabled_by_default": True,
        "behavior": "direct_prediction",
        "permitted_use": "Direct prediction within the documented applicability domain.",
        "warning": None,
    },
    "staging": {
        "enabled_by_default": True,
        "behavior": "internal_preview",
        "permitted_use": "Internal preview and controlled staging evaluation.",
        "warning": "This model is in staging and has not been approved for unrestricted production use.",
    },
    "screening": {
        "enabled_by_default": True,
        "behavior": "screening_only",
        "permitted_use": "Screening and prioritisation only; not a definitive chemistry claim.",
        "warning": "This model is approved only for screening use.",
    },
    "baseline": {
        "enabled_by_default": True,
        "behavior": "baseline_reference",
        "permitted_use": "Baseline comparison and compatibility behavior.",
        "warning": "This is a baseline model, not a current production claim.",
    },
    "candidate": {
        "enabled_by_default": True,
        "behavior": "retrieval_backed_suggestion",
        "permitted_use": "Top-k retrieval-backed suggestions with explicit uncertainty and provenance.",
        "warning": "This candidate model has not met production promotion thresholds.",
    },
    "validated": {
        "enabled_by_default": False,
        "behavior": "controlled_validation",
        "permitted_use": "Controlled validation only.",
        "warning": "This model requires explicit experimental access.",
    },
    "experimental": {
        "enabled_by_default": False,
        "behavior": "experimental",
        "permitted_use": "Experimental research use only.",
        "warning": "This experimental model is disabled unless explicitly enabled.",
    },
    "audit_non_trainable": {
        "enabled_by_default": False,
        "behavior": "audit_only",
        "permitted_use": "Audit result only; no prediction is permitted.",
        "warning": "No trainable model exists for this task under the current evidence contract.",
    },
    "superseded": {
        "enabled_by_default": False,
        "behavior": "never_load",
        "permitted_use": "Historical provenance only.",
        "warning": "Superseded artifacts are never loaded.",
    },
    "archived_incompatible": {
        "enabled_by_default": False,
        "behavior": "never_load",
        "permitted_use": "Historical provenance only.",
        "warning": "Incompatible archived artifacts are never loaded.",
    },
}


def model_capability(record: dict[str, Any]) -> dict[str, Any]:
    stage = str(record.get("effective_release_stage") or record.get("stage") or "candidate")
    lifecycle = str(record.get("lifecycle_state") or "active")
    policy = dict(STAGE_POLICIES.get(stage, STAGE_POLICIES["candidate"]))
    if lifecycle in {"superseded", "archived_incompatible", "audit_non_trainable"}:
        policy = dict(STAGE_POLICIES[lifecycle])
    decision = record.get("release_decision") or {}
    return {
        "task": record.get("task"),
        "model_id": record.get("model_id"),
        "version": record.get("version"),
        "stage": stage,
        "lifecycle_state": lifecycle,
        "runtime_load_required": bool(record.get("runtime_load_required")),
        "release_approved": bool(decision.get("approved")),
        "enabled_by_default": bool(policy["enabled_by_default"]),
        "behavior": policy["behavior"],
        "permitted_use": decision.get("permitted_use") or policy["permitted_use"],
        "warning": policy["warning"],
        "dataset_version": record.get("dataset_version"),
        "training_split_sha256": record.get("split_sha256"),
        "artifact_sha256": record.get("artifact_sha256"),
    }
