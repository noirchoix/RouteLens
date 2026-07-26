# REACTS Product One architecture

## Deployment shape

Product One is a modular monolith with isolated data, ML, retrieval, registry, and API boundaries. It can run on one workstation and can later extract workers or inference runtimes into independent services without changing contracts.

```text
Browser / API client
        │
        ▼
FastAPI + embedded operator UI
        │
        ├── Dataset registry and audit
        ├── Route and step repository
        ├── Training job orchestration
        ├── Model registry and promotion
        ├── Applicability-gated inference
        └── Evidence retrieval
        │
        ├───────────────┬─────────────────┐
        ▼               ▼                 ▼
Canonical Parquet   SQLite registry   Sharded Morgan index
        │               │                 │
        └────────── ML training ──────────┘
```

## Runtime modules

- `reacts.data.source`: read-only ZIP or extracted-directory resolver.
- `reacts.data.canonical`: chunked canonical build, deduplication, quality events, patent-grouped splits.
- `reacts.chemistry`: reaction parsing, condition repair, plausibility and Morgan fingerprints.
- `reacts.ml.training`: incremental full-corpus baselines using stateless hashed reaction features.
- `reacts.ml.registry`: immutable model versions, stages, jobs, training runs, and dataset registration.
- `reacts.retrieval`: sharded reactant/product Morgan indexes with weighted Tanimoto ranking.
- `reacts.ml.inference`: model loading, abstention, evidence attachment and applicability classification.
- `reacts.api`: REST surface and embedded operator interface.

## Storage responsibilities

- **Parquet/Zstandard:** canonical routes, steps, and quality events. Native list columns are retained.
- **SQLite/WAL:** metadata, jobs, training runs, model versions, model stages and audit state.
- **Joblib:** immutable classical-model bundles.
- **NPZ + JSONL shards:** packed Morgan fingerprints and evidence metadata.
- **Hugging Face:** intended public distribution location for canonical data and model releases; publication is intentionally not performed by the local build.

## Scale policy

The canonical build always processes the complete source artifact. `max_rows` exists only for CI, smoke runs, and explicit experiment configuration. It is never silently applied.

## Reliability boundaries

- All source artifacts are read-only.
- Every model records dataset version, model version, task, metrics and artifact path.
- Condition and structural models abstain on invalid or symbolic reactions.
- Patent grouping is deterministic and independent of row order.
- Online and batch inference call the same service code.
