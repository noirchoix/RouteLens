from __future__ import annotations


class ArtifactError(RuntimeError):
    """Base error for immutable runtime artifact failures."""

    reason_code = "artifact_error"

    def __init__(self, message: str, *, reason_code: str | None = None):
        super().__init__(message)
        self.reason_code = reason_code or self.reason_code


class ArtifactUnavailableError(ArtifactError):
    reason_code = "artifact_unavailable"


class ArtifactIntegrityError(ArtifactError):
    reason_code = "artifact_hash_mismatch"


class ArtifactCompatibilityError(ArtifactError):
    reason_code = "artifact_incompatible"


class ArtifactContractError(ArtifactError):
    reason_code = "artifact_contract_invalid"
