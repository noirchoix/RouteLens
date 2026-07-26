# REACTS Product Two v2.0.12

- Makes the top-level `validate-product-two` CLI dispatch read-only before any application or registry object is constructed.
- Prevents pre-validation JSON registry normalization/export from occurring before the validator captures its internal baseline hashes.
- Aligns external before/after registry checks with `registry_read_only_validation.before` and `.after`.
- Updates default Product Two release identifiers to `v2.0.12`.
- No canonical, split, index, mapping, derivation, registry lifecycle, or model artifact rebuild is required.

# REACTS Product Two v2.0.11

- Fixed the remaining validation-time model-registry mutation caused by the cold import of `reacts.api.main`.
- The module-level ASGI application now honors `REACTS_API_READ_ONLY_REGISTRY=1`.
- Acceptance API smoke sets that mode only for the cold import, restores the prior environment, and still constructs its explicit application with `read_only_registry=True`.
- Registry database and JSON hashes must remain unchanged across complete validation.
- No canonical, split, index, mapping, derivation, or model artifact rebuild is required.

# REACTS Product Two v2.0.10

## Read-only acceptance validation hotfix

- Acceptance validation opens the SQLite model registry in query-only mode.
- API smoke tests use a read-only registry and cannot migrate, normalize, or export registry state.
- Validation no longer calls `sync_json()`.
- Registry JSON and database hashes are captured before and after validation; any mutation is a strict release failure.
- Mapping, derivation, canonical splits, indexes, and trained model artifacts are unchanged.

# REACTS Product Two v2.0.9 split and model-registry governance

## Fixed

- Rebuilds final Product Two train/validation/test assignments as deterministic connected components across patent document identity and exact reaction signature, with seed 42.
- Makes exact reaction-signature overlap a strict release failure while preserving product-scaffold overlap as a challenge diagnostic.
- Adds universal pre-fit class support validation for binary, multiclass, and multilabel tasks; non-trainable tasks create auditable non-model records.
- Preserves empty multilabel label sets as valid negative examples and prevents one-class catalyst metrics from being represented as discrimination evidence.
- Introduces a persistent JSON model registry mirrored from SQLite, explicit lifecycle states, one runtime-required model per task/dataset, and non-loading archived/superseded records.
- Pins scikit-learn 1.9.0 and persists the complete training environment in model artifacts, model cards, and the registry.
- Binds reaction and route indexes to the governed split hash and requires all active Product Two models to be retrained on that hash.
- Extends strict validation to split, index, environment, model-lifecycle, and pre-fit invariants.

## Rebuild boundary

- Does not rerun Product One ingestion, Product Two contextualization, atom mapping, or reaction-centre derivation.
- Rebuilds only final split metadata, split-aware indexes, active supervised models, the model registry, and scientific acceptance outputs.

# REACTS Product Two v2.0.8 repairability-task governance hotfix

- Prevents parse-valid reactions from being counted as repairable merely because an alternative canonical/contextual representation exists.
- Reclassifies `repairability` as a deterministic audited task rather than a supervised classifier for the current corpus.
- Removes `repairability` from default Product Two classification training.
- Explicit requests produce a reproducible task-audit JSON and no model artifact or promotion.
- Preserves existing canonical, mapping, derivation, and index artifacts; no rebuild is required for the current release workflow.

# REACTS Product Two v2.0.7 Arrow-container derivation hotfix

## Fixed

- Normalizes Arrow and NumPy list containers before scalar-null checks, preventing ambiguous truth-value failures while reading completed mapping Parquet shards.
- Reads both legacy native-list mapping shards and canonical JSON-text shards without rewriting or invalidating mapping outputs.
- Preserves the `diagnostics` field as a list contract, including empty and null values, while retaining dictionary defaults for metadata JSON columns.
- Serializes JSON contract columns before future mapping Parquet writes so the mixed physical representation does not recur.
- Adds exact regression coverage for empty and populated NumPy arrays returned by `pandas.read_parquet`.

# REACTS Product Two v2.0.6 bounded mapping exceptions

## Fixed

- Counts RXNMapper tokens before inference and never sends reactions exceeding the configured/model limit to the transformer.
- Persists token count, token limit and mapper eligibility so interrupted queues do not repeat deterministic oversized-input failures.
- Runs every MCS fallback record in an isolated spawned process with a hard parent-enforced timeout.
- Quarantines timeout and terminal fallback failures with explicit error taxonomy while preserving original reactions unchanged.
- Flushes normal mappings before committing exceptional records and writes a separate exceptional-record dataset.
- Migrates v2.0.5 mapping queues in place without modifying committed shards.
- Adds complete mapping-exception metrics and adversarial resume, timeout and token-boundary tests.

# REACTS Product Two v2.0.5 indexed route assignment

## Fixed

- Creates route and step source indexes before any identity assignment query.
- Assigns all nonduplicated source routes through a narrow indexed SQL join rather than updating the 600+ MB staged step table.
- Restricts route-text matching to the 64 duplicated source-route groups and records the expensive-match count.
- Persists a narrow `route_instance_assignment` table and an atomic `.work/identity_assignment_manifest.json` checkpoint.
- Resumes completed identity assignment without restaging Product One steps, resumes interrupted assignment from the staged SQLite database, and adopts a validated complete v2.0.4 pre-assignment database.
- Joins instance identity into contextual rows at read time, preserving immutable Product One staged step records.
- Preserves duplicate source step identifiers by deriving instance IDs from the repaired Product One `source_step_id` suffix.

# REACTS Product Two v2.0.4 route identity and evidence correction

## Fixed

- Preserves every Product One route row with a unique `route_instance_id` while retaining `source_route_id`.
- Reconstructs all step-to-route-instance assignments deterministically from Product One route text for duplicated source-route groups.
- Uses instance identity throughout contextual tables, mapping queues, derivation, retrieval, and scientific validation.
- Prevents cross-variant edges and duplicate step-instance identifiers.
- Reclassifies symbolic labels as logical evidence unless an observed structural anchor exists.
- Replaces generic quarantine reasons with unresolved-symbolic and deterministic parse-failure taxonomy.
- Adds strict identity and evidence metrics to the contextual manifest.

# REACTS Product Two v2.0.2 staged structural pipeline

## Changed

- Removed inline atom mapping and reaction-centre derivation from contextual canonicalization.
- Added a persistent, checkpointed context-only build with a disk-backed molecule catalogue.
- Added strict mapper preflight, batched RXNMapper, resilient batch splitting, and fail-fast explicit backend selection.
- Added a persistent SQLite mapping queue with immediate interruption recovery.
- Added atomic 5,000-row mapping shards and segregated RXNMapper, MCS fallback, and rejected outputs.
- Restricted MCS to RXNMapper failures and excluded MCS mappings from strict structural training by default.
- Added an independent resumable reaction-centre derivation queue and final scientific-view materialization.
- Added batch-size benchmarking, memory/latency reporting, staged CLI commands, and asynchronous staged API jobs.
- Added strict acceptance checks for completed stage manifests, queues, backend provenance, and mapping-output segregation.
- Pinned `rxnmapper==0.4.3` with the verified `setuptools==81.0.0` compatibility environment.

# REACTS Product Two v2.0.1 hotfix

## Fixed

- Enforced a type-stable `condition_evidence` Parquet schema: the generic normalized value is textual, while numeric and categorical values are also exposed through dedicated typed columns.
- Normalized numpy/Arrow list cells before SQLite staging so empty and multi-valued solvent/agent lists are preserved instead of becoming quoted pseudo-values such as `"[]"`.
- Explicitly closed SQLite cursors, connections, and route generators so temporary `route_context.sqlite3` files are releasable on Windows after both success and failure.
- Added regression coverage for mixed numeric/categorical condition evidence, Parquet list materialization, and generator closure during exceptions.

# REACTS Product Two v2.0.0 patch

## Added

- Immutable Product One `v1.0.0-baseline` freeze and task-stage correction.
- Task-specific promotion gates with permitted-use declarations.
- Calibration, reliability curves, portable paths, and reproducibility hashes.
- Patent, reaction-signature, and scaffold leakage reporting.
- Conservative symbolic-intermediate resolution and quarantine.
- Canonical dataset v2 normalized chemistry and provenance tables.
- Optional RXNMapper integration and confidence-qualified MCS fallback.
- Bond-change, reaction-centre, template, and structural-family derivation.
- Multilabel solvent/agent/catalyst-family training.
- Time and temperature interval-regression training.
- Contextual reaction and route evidence indexes.
- Retrieval-augmented inference with family priors, applicability, and abstention.
- Deterministic parse-repair, condition-anomaly, and route-quality services.
- Strict scientific acceptance and immutable Product Two release lock.

## Preserved

- Product One APIs, stored canonical data, registered model artifacts, and evidence index.
- Complete-corpus processing policy.
- Explicit scientific non-claims for yield and experimental validation.

## Migration

This distribution is a code-only overlay. Product One data and model artifacts are intentionally excluded.
