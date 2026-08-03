from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REACTS_", env_file=".env", extra="ignore")

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    source_artifact: Path = Path("data/source_artifacts/uspto_llm_multistep_only.zip")
    canonical_dir: Path = Path("data/canonical")
    canonical_v2_context_dir: Path = Path("data/canonical_v2_context")
    mapping_v2_dir: Path = Path("data/mapping_v2")
    derivation_v2_dir: Path = Path("data/derivation_v2")
    canonical_v2_dir: Path = Path("data/canonical_v2")
    mapping_queue_db: Path = Path("data/state/product_two_mapping.sqlite3")
    derivation_queue_db: Path = Path("data/state/product_two_derivation.sqlite3")
    model_dir: Path = Path("data/models")
    index_dir: Path = Path("data/indexes")
    index_v2_dir: Path = Path("data/indexes_v2")
    baseline_dir: Path = Path("data/baselines/v1.0.0-baseline")
    registry_db: Path = Path("data/registry/reacts.sqlite3")
    reports_dir: Path = Path("reports")
    releases_dir: Path = Path("data/releases")
    artifact_uri: str | None = None
    artifact_release: str | None = None
    artifact_cache_dir: Path = Path("data/artifact_cache")
    artifact_verify_sha256: bool = True
    artifact_required: bool = False
    artifact_warmup: bool = True
    artifact_lock_timeout_seconds: int = 120
    offline_mode: bool = False
    api_key: str | None = None
    require_api_key: bool = False
    max_batch_rows: int = 100_000
    inference_max_batch_rows: int = 256
    max_request_bytes: int = 1_048_576
    request_timeout_seconds: float = 30.0
    max_concurrent_requests: int = 8
    rate_limit_requests_per_minute: int = 120
    cors_origins: str = ""
    trusted_hosts: str = "localhost,127.0.0.1,testserver"
    allow_experimental_models: bool = False
    temperature_min_c: float = -150.0
    temperature_max_c: float = 350.0
    time_min_h: float = 1.0 / 3600.0
    time_max_h: float = 8760.0
    random_seed: int = 42
    mapping_backend: str = "auto"
    mapping_min_coverage: float = 0.60
    mapping_timeout_seconds: int = 3
    mapping_batch_size: int = 16
    mapping_shard_size: int = 5_000
    mapping_prefetch_batches: int = 2
    mapping_stale_after_minutes: int = 60
    evidence_in_domain_threshold: float = 0.65
    evidence_weak_threshold: float = 0.35
    inference_abstention_threshold: float = 0.35

    def resolve(self) -> "Settings":
        root = self.project_root.resolve()
        for field in [
            "source_artifact",
            "canonical_dir",
            "canonical_v2_context_dir",
            "mapping_v2_dir",
            "derivation_v2_dir",
            "canonical_v2_dir",
            "mapping_queue_db",
            "derivation_queue_db",
            "model_dir",
            "index_dir",
            "index_v2_dir",
            "baseline_dir",
            "registry_db",
            "reports_dir",
            "releases_dir",
            "artifact_cache_dir",
        ]:
            value = getattr(self, field)
            if not value.is_absolute():
                setattr(self, field, root / value)
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def trusted_host_list(self) -> list[str]:
        values = [item.strip() for item in self.trusted_hosts.split(",") if item.strip()]
        return values or ["*"]


settings = Settings().resolve()
