from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reacts.contracts import MappingStatus


@dataclass(frozen=True)
class MappingQueueItem:
    # step_id/route_id are compatibility aliases for instance identifiers.
    step_id: str
    route_id: str
    reaction_smiles: str
    reaction_signature: str
    attempt_count: int
    source_step_id: str | None = None
    source_route_id: str | None = None
    rxnmapper_token_count: int | None = None
    rxnmapper_token_limit: int | None = None
    rxnmapper_eligible: bool | None = None
    fallback_status: str | None = None
    fallback_attempt_count: int = 0

    @property
    def step_instance_id(self) -> str:
        return self.step_id

    @property
    def route_instance_id(self) -> str:
        return self.route_id


@dataclass(frozen=True)
class MappingBatchItem:
    step_id: str
    route_id: str
    reaction_smiles: str
    reaction_signature: str
    status: MappingStatus
    mapped_reaction_smiles: str | None
    backend: str
    confidence: float
    atom_coverage: float
    validation_status: str
    diagnostics: tuple[str, ...]
    runtime_ms: float
    scientific_eligibility: bool
    primary_backend_error: str | None = None
    source_step_id: str | None = None
    source_route_id: str | None = None
    error_code: str | None = None
    rxnmapper_token_count: int | None = None
    rxnmapper_token_limit: int | None = None
    rxnmapper_eligible: bool | None = None
    fallback_status: str | None = None
    fallback_attempt_count: int = 0
    exceptional_reason: str | None = None

    @property
    def step_instance_id(self) -> str:
        return self.step_id

    @property
    def route_instance_id(self) -> str:
        return self.route_id

    def to_record(self, *, dataset_version: str) -> dict[str, Any]:
        return {
            "dataset_version": dataset_version,
            "step_id": self.step_id,
            "step_instance_id": self.step_instance_id,
            "source_step_id": self.source_step_id,
            "route_id": self.route_id,
            "route_instance_id": self.route_instance_id,
            "source_route_id": self.source_route_id,
            "reaction_smiles": self.reaction_smiles,
            "reaction_signature": self.reaction_signature,
            "mapping_status": self.status.value,
            "mapped_reaction_smiles": self.mapped_reaction_smiles,
            "backend": self.backend,
            "confidence": float(self.confidence),
            "atom_coverage": float(self.atom_coverage),
            "validation_status": self.validation_status,
            "scientific_eligibility": bool(self.scientific_eligibility),
            "diagnostics": list(self.diagnostics),
            "runtime_ms": float(self.runtime_ms),
            "primary_backend_error": self.primary_backend_error,
            "error_code": self.error_code,
            "rxnmapper_token_count": self.rxnmapper_token_count,
            "rxnmapper_token_limit": self.rxnmapper_token_limit,
            "rxnmapper_eligible": self.rxnmapper_eligible,
            "fallback_status": self.fallback_status,
            "fallback_attempt_count": int(self.fallback_attempt_count),
            "exceptional_reason": self.exceptional_reason,
        }
