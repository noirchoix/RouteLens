from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParseFailureClass(StrEnum):
    VALID = "valid"
    SYMBOLIC_INTERMEDIATE = "symbolic_intermediate"
    MALFORMED_DELIMITER = "malformed_delimiter"
    EMPTY_REACTION_SIDE = "empty_reaction_side"
    INVALID_REACTANT = "invalid_reactant_smiles"
    INVALID_PRODUCT = "invalid_product_smiles"
    INVALID_BOTH = "invalid_both_sides"
    UNKNOWN = "unknown_parse_failure"


class Applicability(StrEnum):
    IN_DOMAIN = "in_domain"
    WEAKLY_SUPPORTED = "weakly_supported"
    OUT_OF_DOMAIN = "out_of_domain"
    INVALID = "invalid"


class ModelStage(StrEnum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"
    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    SCREENING = "screening"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class ResolutionStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    INVALID_AFTER_RESOLUTION = "invalid_after_resolution"


class MappingStatus(StrEnum):
    MAPPED = "mapped"
    EXISTING = "existing"
    LOW_CONFIDENCE = "low_confidence"
    FAILED = "failed"
    NOT_ELIGIBLE = "not_eligible"


class ReactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reaction_smiles: str = Field(min_length=3, max_length=100_000)
    include_evidence: bool = True
    evidence_k: int = Field(default=5, ge=0, le=50)


class ConditionPredictionRequest(ReactionInput):
    tasks: list[str] = Field(
        default_factory=lambda: [
            "solvent_multilabel",
            "time_regression",
            "temperature_regression",
            "agent_multilabel",
        ]
    )
    allow_experimental: bool = False


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reactions: list[str] = Field(min_length=1)
    tasks: list[str] = Field(default_factory=lambda: ["parse_failure_class", "reaction_family"])
    include_evidence: bool = True
    evidence_k: int = Field(default=5, ge=0, le=25)
    allow_experimental: bool = False


class PredictionItem(BaseModel):
    label: str
    probability: float
    calibrated_probability: float | None = None
    model_probability: float | None = None
    neighbour_probability: float | None = None
    family_prior: float | None = None
    combined_score: float | None = None


class TaskPrediction(BaseModel):
    task: str
    predictions: list[PredictionItem]
    abstained: bool = False
    reason: str | None = None
    model_version: str | None = None
    model_stage: str | None = None
    calibration_error: float | None = None
    applicability: Applicability | None = None
    neighbour_support: int = 0
    reaction_family_agreement: float | None = None
    interval: tuple[float, float] | None = None
    point_estimate: float | None = None
    units: str | None = None


class EvidenceItem(BaseModel):
    step_id: str
    route_id: str
    patent_document_id: str | None = None
    reaction_smiles: str
    score: float
    quality_score: float | None = None
    solvent_primary: str | None = None
    solvents: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    time_bucket: str | None = None
    temperature_bucket: str | None = None
    reaction_family: str | None = None
    reaction_centre_fingerprint: str | None = None
    resolution_status: str | None = None


class InferenceResponse(BaseModel):
    input_reaction: str
    canonical_reaction: str | None = None
    parse_ok: bool
    parse_failure_class: str | None = None
    applicability: Applicability
    tasks: list[TaskPrediction]
    evidence: list[EvidenceItem] = Field(default_factory=list)
    neighbour_label_distributions: dict[str, dict[str, float]] = Field(default_factory=dict)
    reaction_family: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[str] = Field(default_factory=lambda: ["parse_failure_class", "reaction_family"])
    dataset_version: str = "uspto_multistep_contextual_v2"
    request_promotion: bool = False
    max_rows: int | None = Field(default=None, ge=100)


class ProductTwoBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prefer_parquet: bool = True
    resume: bool = False
    clean: bool = False


class MappingRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = Field(default="rxnmapper", pattern=r"^(auto|rxnmapper|mcs|mcs_fallback)$")
    fallback_backend: str | None = Field(default="mcs", pattern=r"^(mcs|mcs_fallback)$")
    allow_auto_fallback: bool = False
    batch_size: int = Field(default=16, ge=1, le=512)
    workers: int = Field(default=1, ge=1, le=1)
    prefetch_batches: int = Field(default=2, ge=1, le=16)
    shard_size: int = Field(default=5000, ge=100, le=100000)
    rxnmapper_token_limit: int = Field(default=512, ge=32, le=8192)
    fallback_process_timeout_seconds: int = Field(default=30, ge=1, le=600)
    resume: bool = False
    max_rows: int | None = Field(default=None, ge=1)
    prefer_parquet: bool = True


class DerivationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume: bool = False
    include_mcs: bool = False
    min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    max_rows: int | None = Field(default=None, ge=1)
    prefer_parquet: bool = True


class RepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reaction_smiles: str = Field(min_length=1, max_length=100_000)
    contextual_candidate: str | None = Field(default=None, max_length=100_000)
    route_continuity_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ConditionAnomalyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reaction_smiles: str = Field(min_length=3, max_length=100_000)
    temperature_c: float | None = None
    time_h: float | None = Field(default=None, ge=0.0)


class RouteQualityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parse: float = Field(ge=0.0, le=1.0)
    resolution: float = Field(ge=0.0, le=1.0)
    route_continuity: float = Field(ge=0.0, le=1.0)
    condition_completeness: float = Field(ge=0.0, le=1.0)
    condition_plausibility: float = Field(ge=0.0, le=1.0)
    mapping: float = Field(ge=0.0, le=1.0)


class ReleaseLockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    release_id: str = Field(default="v2.0.0", pattern=r"^[A-Za-z0-9._-]+$")


class JobResponse(BaseModel):
    job_id: str
    status: str
    detail: dict[str, Any] = Field(default_factory=dict)
