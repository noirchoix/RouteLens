# REACTS Product Two

**Contextual Reaction Intelligence**

REACTS Product Two upgrades the stored Product One run into a scientifically governed reaction-intelligence platform. It preserves the complete Product One corpus and model artifacts as an immutable `v1.0.0-baseline`, builds a separate contextual canonical dataset, resolves only evidence-supported symbolic intermediates, derives route and structural chemistry records, trains redesigned specialist models, and serves retrieval-augmented inference with applicability and abstention controls.

Product Two is a code upgrade. It does not require copying the large Product One data, model, or index artifacts into the patch archive. Apply the patch over the existing Product One project and retain its `data/` directory.

## Scientific contract

Product Two implements the following release sequence:

```text
freeze Product One baseline
→ enforce task-specific model gates
→ audit patent/signature/scaffold leakage
→ resolve route-context intermediates conservatively
→ build canonical dataset v2
→ atom-map confidence-qualified reactions
→ derive reaction centres and structural families
→ build reaction and route evidence indexes
→ train specialist models
→ add retrieval-augmented inference
→ score anomalies and route quality
→ run strict full-corpus acceptance
→ lock only qualified Product Two models
```

No source value is overwritten. Every contextual reconstruction, condition normalization, mapping result, repair candidate, and quarantine decision is written as a separate auditable record.

## Stored Product One prerequisites

The existing Product One project should retain:

```text
data/canonical/
data/models/
data/indexes/
data/registry/reacts.sqlite3
reports/product_one_run.json          # recommended
```

The original USPTO artifact is only required when rebuilding Product One from source. Product Two reads the stored Product One canonical data directly.

## Installation

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,mapping]"
```

The mapping extra is pinned to the verified compatibility pair:

```text
rxnmapper==0.4.3
setuptools==81.0.0
```

## Product Two v2.1.0 artifact-backed inference

The v2.1.0 service can start from a clean code checkout with no local model, index, or registry artifacts. It resolves one exact immutable artifact release, verifies the complete checksum and scientific contract, installs it atomically into a cache, opens the registry read-only, warms required models and indexes, and exposes readiness only after the runtime is usable.

Package the locked v2.0.12 runtime artifacts:

```bash
reacts --project-root . package-product-two-artifacts \
  --release product-two-artifacts-v2.0.12-r1 \
  --destination dist/artifacts

reacts --project-root . validate-artifact-bundle \
  --bundle dist/artifacts/product-two-artifacts-v2.0.12-r1
```

Start the v2.1.0 service against that exact release:

```bash
reacts --project-root . serve \
  --artifact-uri dist/artifacts \
  --artifact-release product-two-artifacts-v2.0.12-r1 \
  --require-artifacts \
  --port 8000
```

`GET /health` reports process health. `GET /ready` returns success only after artifact verification and model/index warm-up. Artifact-dependent inference and retrieval endpoints return HTTP 503 while readiness is false.

Product Two v2.1.5 converts the locked compressed route matrix into a standalone NPY payload while publishing the artifact bundle. The service memory-maps that matrix and scores it in bounded row chunks, so startup does not allocate the full 156,076 × 4,096 float32 route matrix. The `-r1` artifact suffix identifies this inference-storage correction without changing the underlying v2.0.12 models, routes, split, or scientific index values.

The service distinguishes runtime readiness from scientific model promotion. Candidate and staging models retain explicit permitted-use declarations, warnings, lifecycle state, artifact release, and training-split provenance in every response. Superseded and incompatible artifacts are never loaded.

See `docs/ARTIFACT_BACKED_INFERENCE_RUNTIME.md`, `docs/STAGING_AND_ROLLBACK.md`, and `docs/PERFORMANCE_ENVELOPE.md`.

## Product Two staged run

The locked Product Two v2.0.12 lineage adds deterministic patent/reaction connected-component splits, strict exact-chemistry leakage gates, universal pre-fit class-support validation, a lifecycle-aware model registry, an exact scikit-learn 1.9.0 model/runtime contract, and observational read-only acceptance validation. Mapping and derivation remain frozen when upgrading from v2.0.8.

```bash
reacts --project-root . build-contextual-v2 --resume

reacts --project-root . benchmark-mapper \
  --backend rxnmapper \
  --batch-sizes 8 16 32 64 \
  --sample-size 512

reacts --project-root . map-reactions \
  --backend rxnmapper \
  --fallback-backend mcs \
  --batch-size 8 \
  --workers 1 \
  --shard-size 5000 \
  --rxnmapper-token-limit 512 \
  --fallback-timeout-seconds 30 \
  --resume

reacts --project-root . derive-reaction-centres --resume
reacts --project-root . rebuild-product-two-splits --seed 42
reacts --project-root . build-product-two-indexes --resume
reacts --project-root . train-product-two --request-promotion
reacts --project-root . validate-product-two
```

Only after strict validation succeeds:

```bash
reacts --project-root . lock-product-two --release-id v2.0.12
```


For an existing completed v2.0.8 corpus, do not rerun contextualization, mapping, or derivation. Start at the split boundary:

```bash
reacts --project-root . rebuild-product-two-splits --seed 42
reacts --project-root . build-product-two-indexes --resume
reacts --project-root . train-product-two --request-promotion
reacts --project-root . validate-product-two
```

The wrapper is resumable and stage-addressable:

```bash
reacts --project-root . product-two \
  --from-stage mapping \
  --stop-after derivation \
  --mapping-backend rxnmapper \
  --batch-size 16 \
  --resume
```

`auto` mapping is strict. If RXNMapper cannot initialize, the run fails before queue mutation. `--allow-auto-fallback` is required for an explicitly labelled MCS baseline.

## Product Two data lineage

```text
data/canonical_v2_context/                 context-only canonical data
data/state/product_two_mapping.sqlite3     persistent mapping queue
data/mapping_v2/                           segregated mapping products
data/state/product_two_derivation.sqlite3  persistent derivation queue
data/derivation_v2/                        structural derivation products
data/canonical_v2/                         final qualified scientific view
```

The context stage writes:

```text
steps/
routes/
molecules/
step_molecules/
route_edges/
intermediate_resolution/
condition_evidence/
repair_candidates/
mapping_candidates/
quarantine/
dataset_manifest.json
```

The mapping stage writes separate RXNMapper, MCS fallback, and rejected datasets. The derivation stage writes reaction mappings, centres, templates, and families, then materializes the final scientific view only when its queue is complete.

The key rules are:

- Product One canonical records remain read-only.
- Route context and symbolic recovery are independent from atom mapping.
- RXNMapper is initialized once and receives bounded batches.
- MCS runs only after RXNMapper failure and is excluded from strict training by default.
- Mapping and derivation queues are persistent and immediately resumable after interruption.
- Output shards are written to temporary files, validated, atomically renamed, and then committed to queue state.
- Original reactions, reconstructed reactions, mapping diagnostics, and rejection reasons remain auditable.
- The archived partial MCS run is never merged automatically.

See `docs/PRODUCT_TWO_STAGED_PIPELINE.md` for the complete execution contract.

## Redesigned ML tasks

Classification tasks:

- `parse_failure_class`
- `repairability` — deterministic audit only; no classifier is trained without two eligible parse-invalid classes
- `reaction_family`

Specialist tasks:

- `solvent_multilabel`
- `solvent_family_multilabel`
- `agent_family_multilabel`
- `catalyst_family_multilabel`
- `time_regression`
- `temperature_regression`

Additional deterministic or statistical services:

- ranked deterministic parse repair;
- condition anomaly detection;
- decomposable route-quality scoring;
- route-continuity scoring in canonical v2;
- reaction and route similarity retrieval.

Every trained model records unique training rows, epochs, total examples seen, calibration or interval coverage, dataset/feature/split hashes, model hash, release decision, permitted use, and immutable artifact path.

## Release governance

`--request-promotion` requests evaluation; it does not override policy. Each task has an independent gate covering:

- minimum evaluation population;
- macro-F1 or interval metrics;
- balanced accuracy where applicable;
- accuracy relative to a majority baseline;
- calibration error;
- permitted stage and product use.

Product One models are frozen with the following baseline stages:

```text
parse_validity      → baseline
primary_solvent     → staging
agent_presence      → screening
time_bucket         → experimental
temperature_bucket  → experimental
```

A Product Two release lock requires strict acceptance and excludes experimental or candidate models.

## Retrieval-augmented inference

The contextual inference response combines:

```text
model probability
+ nearest-reaction evidence
+ reaction-family prior
+ applicability support
```

Responses include:

- calibrated/model probability;
- evidence and family contributions;
- ranked top-k results;
- analogous patent reaction records;
- neighbour label distributions;
- reaction-family agreement;
- applicability status;
- abstention reason;
- dataset, index, feature, split, and model provenance.

Malformed reactions can still be submitted to parse-intelligence tasks. Condition and structural tasks abstain until the reaction is chemically complete.

## Principal API endpoints

```text
GET  /health
GET  /ready
GET  /api/v2/health
GET  /api/v2/artifacts
GET  /api/v2/models
GET  /api/v2/datasets
GET  /api/v2/routes/{route_id}
POST /api/v2/inference/contextual
POST /api/v2/inference/batch
POST /api/v2/retrieval/reactions
POST /api/v2/retrieval/routes
POST /api/v2/inference/repair
POST /api/v2/inference/anomaly
POST /api/v2/inference/route-quality
POST /api/v2/jobs/freeze-baseline
POST /api/v2/jobs/build-contextual
POST /api/v2/jobs/map-reactions
POST /api/v2/jobs/derive-reaction-centres
POST /api/v2/jobs/build-index
POST /api/v2/jobs/train
POST /api/v2/jobs/validate
POST /api/v2/jobs/lock-release
GET  /api/v2/jobs/{job_id}
```

Start the service:

```bash
reacts --project-root . serve --port 8000
```

OpenAPI is available at `http://localhost:8000/docs`.

## Validation and release artifacts

Strict validation writes:

```text
reports/product_two_scientific_acceptance.json
```

The report contains:

- canonical-v2 key and route/split integrity;
- patent, reaction-signature, and scaffold leakage;
- model loading and checksum verification;
- API and batch/online equivalence smoke tests;
- retrieval latency, self-recall, and family relevance;
- canonical and registry reproducibility hashes.

A locked release writes:

```text
data/releases/v2.0.5/release_manifest.json
```

The release manifest links the Product One baseline, canonical v2, contextual retrieval index, acceptance report, and every qualified model by portable path and SHA-256 hash.

## Tests

```bash
pytest
```

The suite covers Product One compatibility plus Product Two governance, calibration, leakage, symbolic resolution, canonical-v2 tables, mapping boundaries, reaction-centre derivation, deterministic repair, API startup, and inference contracts.

## Scientific boundaries

- Retrieval evidence and predictions are patent-corpus intelligence, not experimental validation.
- MCS fallback mapping is low-confidence and must not be represented as equivalent to learned atom mapping.
- Condition anomalies indicate corpus-relative unusualness, not chemical impossibility.
- No yield or reaction-success model is exposed because the source artifacts do not contain reliable outcome labels.
- Automatic symbolic substitution is prohibited when the route anchor is ambiguous or absent.

See `docs/PRODUCT_TWO_IMPLEMENTATION.md`, `docs/SCIENTIFIC_GOVERNANCE.md`, `docs/CANONICAL_V2_CONTRACT.md`, and `docs/PATCH_MIGRATION.md`.
