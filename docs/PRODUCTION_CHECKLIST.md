# Production-readiness checklist

## Included in Product One

- Immutable source resolution and SHA-256 inventory.
- Chunked complete-corpus canonicalization.
- Parquet-first storage and explicit compatibility fallback.
- Deterministic IDs, conflict suffixes and patent-grouped splits.
- Quality-event audit trail and task eligibility flags.
- Incremental model training and held-out validation/test metrics.
- Immutable model artifacts, stages and explicit promotion.
- Sharded chemical retrieval and evidence-grounded inference.
- Applicability classification and abstention.
- Single and batch REST inference.
- Background jobs, API-key mode, operator UI, Docker and tests.

## Required before internet-facing deployment

- Replace SQLite with PostgreSQL for multi-instance coordination.
- Move background execution to Celery/RQ/Arq and Redis.
- Add organization/workspace tables and row-level tenancy.
- Store artifacts in S3-compatible object storage.
- Add OIDC, secret manager, TLS termination and rate-limit enforcement.
- Add OpenTelemetry traces and centralized logs.
- Add malware scanning for uploaded batch files.
- Run chemistry-domain review of repair confidence thresholds.
- Train and validate release models on the complete canonical Parquet build.
- Publish dataset/model cards and license review before Hugging Face release.
