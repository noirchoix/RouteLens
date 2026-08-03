# Product Two v2.1.0 Staging and Rollback

## Release pairing

Deploy an explicit pair:

```text
service-v2.1.0 + product-two-artifacts-v2.0.12-r1
```

Never deploy an unversioned `latest` artifact pointer. Keep the previous known-good service image and exact artifact release available until the new release completes staging soak.

## Pre-deployment gates

1. Validate the artifact bundle with `validate-artifact-bundle`.
2. Verify the service image digest and artifact ZIP/directory SHA-256.
3. Confirm `required_scikit_learn` equals the service runtime pin.
4. Confirm the bundle service compatibility includes `2.1.0`.
5. Confirm the exact release name is configured in the deployment.
6. Confirm runtime credentials can only read the artifact object and cache path.

## Staging sequence

### Cold cache

Start with an empty artifact cache and verify:

- exact-release resolution;
- checksum verification;
- atomic cache installation;
- `/health` remains 200 during initialization;
- `/ready` remains 503 until warm-up succeeds;
- `/ready` becomes 200 after all required models and indexes load;
- `/api/v2/artifacts` reports the intended release and split hash;
- `/api/v2/models` reports the expected governed capability set;
- readiness reports `route_index_storage.memory_mapped=true`;
- resident memory does not increase by the full 2.38 GiB route-matrix size.

Ensure the staging host has enough free disk for the compressed source, temporary extraction, expanded route matrix, and installed cache. Use `product-two-artifacts-v2.0.12-r1`; do not reuse a cache directory containing the superseded non-mmap release under the old identifier.

### Warm cache restart

Restart without network access and verify:

- the same exact cached release is selected;
- no artifact file hash changes;
- registry SQLite and JSON files remain byte-identical;
- readiness completes using the cache;
- golden single and batch requests retain structural equivalence.

### Failure injections

Verify fail-closed behavior for:

- corrupted model binary;
- altered manifest;
- missing index shard;
- wrong split hash;
- wrong scikit-learn version;
- duplicate active task;
- unavailable artifact store with an empty cache;
- incomplete ZIP download;
- path-traversal ZIP member;
- stale cache containing only a different release.

Inference and retrieval must return 503 while `/health` remains available.

### Load and security checks

- Measure single, batch, and retrieval p50/p95 latency.
- Measure idle and warmed memory.
- Exercise the configured concurrency and batch caps.
- Verify request-size, timeout, API-key, rate-limit, CORS, and trusted-host controls.
- Confirm the container runs non-root and writes only to its cache/log/tmp paths.
- Confirm build, training, mapping, and derivation jobs are rejected in artifact mode.

## Hermetic image build

Prepare a validated directory bundle, then build:

```bash
docker build \
  -f Dockerfile.hermetic \
  --build-arg ARTIFACT_RELEASE=product-two-artifacts-v2.0.12-r1 \
  --build-arg ARTIFACT_BUNDLE=dist/artifacts/product-two-artifacts-v2.0.12-r1 \
  -t routelens-service:2.1.0-artifacts-v2.0.12 .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  routelens-service:2.1.0-artifacts-v2.0.12
```

## Rollback

Rollback is a release-pair change, not an in-place file replacement:

1. Stop routing new requests to the affected deployment.
2. Deploy the previous known-good image and its exact artifact release.
3. Wait for `/ready` to return 200.
4. Run the golden smoke suite.
5. Restore traffic gradually.
6. Preserve the failed image, artifact manifest, logs, request IDs, and metrics for analysis.

Do not edit a cached bundle or replace files inside an existing release directory. Publish a new artifact release identifier for every corrected bundle.

## Release evidence

Retain:

- service Git commit and tag;
- image digest;
- artifact manifest and `SHA256SUMS`;
- artifact ZIP/object digest;
- bundle validation output;
- runtime version report;
- readiness/warm-up report;
- golden inference output;
- load-test report;
- rollback test result.
