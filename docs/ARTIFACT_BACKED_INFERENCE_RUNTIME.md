# Product Two v2.1.0 — Artifact-Backed Inference Runtime

## Purpose

Product Two v2.1.0 turns the validated v2.0.12 research runtime into a reproducible inference service. The service no longer assumes that `data/models`, `data/indexes`, or the registry database are present in a Git clone. It resolves one exact immutable artifact release, verifies the complete release contract, installs it atomically into a cache, binds all runtime paths to that bundle, and opens the registry read-only.

This feature does not retrain models and does not rebuild canonical data, mapping, reaction-centre derivation, splits, or indexes.

## Artifact bundle

The publisher creates:

```text
<release>/
├── artifact_manifest.json
├── SHA256SUMS
├── models/
│   ├── model_registry.json
│   └── <task>/<active model and model card>
├── indexes/
│   ├── index_manifest.json
│   ├── shard-*.npz
│   ├── shard-*.jsonl
│   └── routes/
├── registry/
│   └── reacts.sqlite3
├── contracts/
│   ├── dataset_manifest.json
│   ├── split_manifest.json
│   ├── reaction_index_manifest.json
│   └── route_index_manifest.json
└── environment/
    └── runtime_versions.json
```

The SQLite registry is included because inference needs the authoritative queryable lifecycle records. It is copied through SQLite's backup API, rebased to bundle-relative model paths, and opened with immutable query-only connections at runtime.

The formal manifest schema is `docs/schemas/product_two_artifact_manifest_v1.schema.json`.

The publisher excludes:

- mapping and derivation queues;
- training caches;
- superseded and incompatible model binaries;
- canonical source rows not needed for inference;
- mutable state databases;
- virtual environments and temporary reports.

## Publish and validate

Run these commands from the locked runtime checkout that contains the verified v2.0.12 models and indexes:

```bash
reacts --project-root . package-product-two-artifacts \
  --release product-two-artifacts-v2.0.12 \
  --destination dist/artifacts
```

The command creates a directory, a ZIP archive, and a companion ZIP SHA-256 file by default. Publish the ZIP together with its external digest, then validate the extracted bundle before deployment:

```bash
reacts --project-root . validate-artifact-bundle \
  --bundle dist/artifacts/product-two-artifacts-v2.0.12
```

The validator fails closed for:

- absent or altered checksummed files;
- files omitted from `SHA256SUMS`;
- unsafe or forbidden paths;
- incompatible service, Python major/minor, scikit-learn, NumPy, SciPy, joblib, or RDKit versions;
- duplicate runtime models for one task;
- missing model cards or model artifacts;
- model artifact hash mismatch;
- disagreement between the SQLite and JSON registries;
- registry, split, reaction-index, and route-index split mismatch;
- missing required runtime tasks.

## Resolve and serve

Configuration:

```text
REACTS_ARTIFACT_URI
REACTS_ARTIFACT_RELEASE
REACTS_ARTIFACT_CACHE_DIR
REACTS_ARTIFACT_VERIFY_SHA256=true
REACTS_ARTIFACT_REQUIRED=true
REACTS_ARTIFACT_WARMUP=true
REACTS_OFFLINE_MODE=false
```

`REACTS_ARTIFACT_URI` supports:

- a local bundle directory;
- a local parent directory containing `<release>/` or `<release>.zip`;
- a `file://` URI;
- an HTTP or HTTPS ZIP URL;
- a URL template containing `{release}`.

S3, Azure Blob, Google Cloud Storage, GitHub Releases, and Hugging Face assets can be consumed through an HTTPS or pre-signed URL. Native cloud SDKs are deliberately not required by the runtime.

Start the service:

```bash
reacts --project-root . serve \
  --artifact-uri dist/artifacts \
  --artifact-release product-two-artifacts-v2.0.12 \
  --require-artifacts \
  --port 8000
```

Offline startup requires the exact release to exist in the cache:

```bash
reacts --project-root . serve \
  --artifact-release product-two-artifacts-v2.0.12 \
  --artifact-cache-dir data/artifact_cache \
  --offline \
  --require-artifacts
```

The resolver never substitutes an older release. Installation uses a per-release lock, a temporary staging directory, full verification before and after copying, and an atomic directory rename.

The runtime contract compares Python major/minor and exact scikit-learn, NumPy, SciPy, joblib, and RDKit versions against the bundle. Platform is retained as provenance but is not required to match, allowing the locked Windows-trained environment to be served from a Linux container with the same software stack.

## Health and readiness

`GET /health` confirms that the process is alive. It remains available when artifact resolution fails.

`GET /ready` returns HTTP 200 only after:

- the exact bundle is installed and verified;
- the runtime registry is readable and immutable;
- required active models are present and deserializable;
- reaction and route indexes are loadable;
- registry and index split hashes match;
- warm-up has completed.

A failed artifact startup returns HTTP 503 from `/ready` and from artifact-dependent inference/retrieval endpoints. The process does not rebuild or regenerate missing runtime artifacts.

`GET /api/v2/artifacts` exposes the installed artifact release, contract, cache status, split hash, and warm-up result.

`GET /api/v2/models` exposes each model's stage, lifecycle state, permitted use, warning, artifact hash, and training split.

## Scientific capability policy

Service readiness is separate from model promotion. The response contract preserves model maturity:

| Stage or lifecycle | Runtime behavior |
| --- | --- |
| `production` | Direct prediction within documented scope |
| `staging` | Internal or controlled preview with warning |
| `candidate` | Retrieval-backed suggestion with warning |
| `experimental` | Disabled unless explicitly enabled |
| `audit_non_trainable` | Audit only; no prediction |
| `superseded` | Never loaded |
| `archived_incompatible` | Never loaded |

Every model-backed task response includes:

- `model_id`;
- model stage;
- lifecycle state;
- permitted use;
- artifact release;
- training split SHA-256;
- warnings;
- evidence and applicability status.

A candidate output without supporting retrieval evidence abstains rather than being represented as a direct validated prediction.

Experimental access requires both an explicit request and either the server setting `REACTS_ALLOW_EXPERIMENTAL_MODELS=true` or the request header:

```text
X-REACTS-Allow-Experimental: true
```

## Runtime hardening

The v2.1.0 API supports:

- request IDs returned through `X-Request-ID`;
- request body size limits;
- bounded inference batches;
- per-request timeout;
- concurrency semaphore;
- in-memory per-client rate limiting;
- configured CORS origins;
- configured trusted hosts;
- API-key protection where configured;
- read-only artifact-backed build/training job rejection;
- structured error codes;
- no raw reaction payloads in default telemetry.

Important settings:

```text
REACTS_MAX_REQUEST_BYTES
REACTS_INFERENCE_MAX_BATCH_ROWS
REACTS_REQUEST_TIMEOUT_SECONDS
REACTS_MAX_CONCURRENT_REQUESTS
REACTS_RATE_LIMIT_REQUESTS_PER_MINUTE
REACTS_CORS_ORIGINS
REACTS_TRUSTED_HOSTS
REACTS_API_KEY
```

The default trusted-host list is limited to `localhost`, `127.0.0.1`, and `testserver`. Production deployments must explicitly add their public or internal service hostname.

## Observability

Prometheus metrics are exposed at `/metrics`:

```text
inference_requests_total
inference_failures_total
inference_latency_seconds
retrieval_latency_seconds
artifact_verification_failures_total
model_load_failures_total
readiness_state
active_model_info
```

HTTP request totals and latency are also recorded. Logs should retain request IDs, service and artifact versions, task/model identifiers, stage, batch size, result count, latency, and structured failure code. Raw proprietary reaction strings should not be logged by default.

## Deployment forms

`Dockerfile` is the thin image. It downloads or resolves an exact release at startup and persists only the artifact cache volume.

`Dockerfile.hermetic` bakes a prevalidated artifact bundle into the image and starts in offline mode. This is the recommended first production checkpoint because it eliminates artifact-store availability from startup.

See `docs/STAGING_AND_ROLLBACK.md` for release verification and rollback.
