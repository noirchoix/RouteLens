# Product Two v2.1.5 — Memory-Safe Artifact-Backed Inference Runtime

Product Two v2.1.5 finalizes the immutable, artifact-backed inference runtime for RouteLens.

## Release validation

* 90 tests passed on Windows 11 with Python 3.13.13.
* Eight governed Product Two runtime models were loaded successfully.
* Artifact validation, readiness warm-up, retrieval, and contextual inference passed.
* Windows SQLite lifecycle and artifact-path handling were validated.
* Exact-release artifact caching was validated.
* Candidate and staging governance, warnings, permitted-use statements, provenance, and abstention behavior remain enforced.

## Artifact runtime

* Artifact release: `product-two-artifacts-v2.0.12-r1`
* Dataset: `uspto_multistep_contextual_v2`
* Runtime models: 8
* Route-index rows: 156,076
* Route-index dimensions: 4,096
* Vector format: `npy_memmap_v1`
* Retrieval chunk size: 2,048 rows
* Training split SHA-256: `078f1f283c215660fdc4d62cf40487b84903bb6031a108e7ce21c7941c4a561e`

## Artifact integrity

* Archive SHA-256: `d9c27f762a723f89ae68bec91409687b6100921b5c52b93867f06f03f6379fa6`
* Artifact manifest SHA-256: `56af9fd411319f1310cea543c1b1e238e1eef25cadb3485a43588c3245914e6f`
* SHA256SUMS SHA-256: `0b7b233f07c8c50a52a77fffa966297b567bb4468cecddeeab8d53a51c1dfb60`

## Major changes

* Dataset-scoped immutable model packaging.
* Preservation of legacy registry records without including them in the deployment bundle.
* Memory-mapped route-vector storage.
* Bounded route retrieval instead of eager 2.38 GiB dense allocation.
* Correct route-index warm-up diagnostics.
* Windows-safe SQLite artifact publication.
* Windows drive-letter and UNC artifact-path resolution.
* Read-only artifact-backed serving mode.
* Fail-closed checksum, split, environment, model, and release validation.

## Governance status

This release is suitable for internal preview, staging, research, and demonstration.

Several models remain in candidate or staging lifecycle states and have not met unrestricted-production promotion thresholds. Responses therefore retain explicit warnings, evidence provenance, calibrated applicability, and governed abstention behavior.
