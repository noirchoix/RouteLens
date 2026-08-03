from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reacts.artifacts.errors import ArtifactError
from reacts.artifacts.resolver import ArtifactResolver
from reacts.settings import Settings


@dataclass
class ArtifactRuntimeState:
    configured: bool = False
    ready: bool = False
    warmed_up: bool = False
    mode: str = "local_unmanaged"
    reason_code: str | None = None
    detail: str | None = None
    artifact_release: str | None = None
    artifact_root: Path | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    warmup: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "ready": self.ready,
            "warmed_up": self.warmed_up,
            "mode": self.mode,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "artifact_release": self.artifact_release,
            "artifact_root": self.artifact_root.as_posix() if self.artifact_root else None,
            "cache_hit": self.cache_hit,
            "training_split_sha256": self.manifest.get("training_split_sha256"),
            "schema_version": self.manifest.get("schema_version"),
            "compatible_service_version": self.manifest.get("compatible_service_version"),
            "runtime_model_count": self.manifest.get("runtime_model_count"),
            "required_tasks": self.manifest.get("required_tasks", []),
            "validation_pass": self.validation.get("pass") if self.validation else None,
            "warmup": self.warmup,
        }


def bind_settings_to_artifacts(settings: Settings, root: Path) -> Settings:
    cfg = settings.model_copy(deep=True)
    cfg.model_dir = root / "models"
    cfg.index_v2_dir = root / "indexes"
    cfg.registry_db = root / "registry" / "reacts.sqlite3"
    cfg.canonical_v2_dir = root / "contracts"
    return cfg


def bootstrap_artifact_runtime(settings: Settings, *, service_version: str) -> tuple[Settings, ArtifactRuntimeState]:
    release = settings.artifact_release
    configured = bool(release or settings.artifact_required)
    if not configured:
        return settings, ArtifactRuntimeState(
            configured=False,
            ready=True,
            warmed_up=True,
            mode="local_unmanaged",
            detail="Artifact resolution is disabled; local development paths are in use.",
        )
    if not release:
        state = ArtifactRuntimeState(
            configured=True,
            ready=False,
            mode="artifact_backed",
            reason_code="artifact_release_missing",
            detail="REACTS_ARTIFACT_RELEASE is required for artifact-backed startup.",
        )
        return settings, state
    if not settings.artifact_verify_sha256:
        state = ArtifactRuntimeState(
            configured=True,
            ready=False,
            mode="artifact_backed",
            reason_code="artifact_verification_disabled",
            detail="Artifact-backed startup requires SHA-256 verification.",
            artifact_release=release,
        )
        return settings, state
    try:
        root, validation, cache_hit = ArtifactResolver(
            uri=settings.artifact_uri,
            release=release,
            cache_dir=settings.artifact_cache_dir,
            verify_sha256=settings.artifact_verify_sha256,
            offline_mode=settings.offline_mode,
            service_version=service_version,
            lock_timeout_seconds=settings.artifact_lock_timeout_seconds,
        ).resolve()
        manifest = validation.get("manifest") or {}
        state = ArtifactRuntimeState(
            configured=True,
            ready=False,
            warmed_up=False,
            mode="artifact_backed",
            artifact_release=release,
            artifact_root=root,
            manifest=manifest,
            validation=validation,
            cache_hit=cache_hit,
        )
        return bind_settings_to_artifacts(settings, root), state
    except ArtifactError as exc:
        return settings, ArtifactRuntimeState(
            configured=True,
            ready=False,
            mode="artifact_backed",
            reason_code=exc.reason_code,
            detail=str(exc),
            artifact_release=release,
        )
    except Exception as exc:
        return settings, ArtifactRuntimeState(
            configured=True,
            ready=False,
            mode="artifact_backed",
            reason_code="artifact_startup_failed",
            detail=str(exc),
            artifact_release=release,
        )
