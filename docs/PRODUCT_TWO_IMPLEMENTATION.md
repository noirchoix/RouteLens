# Product Two implementation

## Module map

```text
reacts.science.baseline       freezes Product One and reclassifies its model stages
reacts.science.promotion      task-specific promotion policies and decisions
reacts.science.calibration    temperature calibration, ECE, reliability curves
reacts.science.hashing        portable reproducibility hashes
reacts.science.release        strict Product Two release lock
reacts.context.route_resolution conservative symbolic-intermediate reconstruction
reacts.chemistry.repair       deterministic repair candidates and strict validation
reacts.mapping.preflight      strict backend resolution before queue mutation
reacts.mapping.queue          persistent SQLite mapping queue and restart state
reacts.mapping.runner         batched RXNMapper with atomic segregated shards
reacts.mapping.derivation     independent reaction-centre/family derivation
reacts.chemistry.mapping      mapped-reaction validation and deterministic MCS fallback
reacts.chemistry.taxonomy     solvent, agent, and catalyst family normalization
reacts.data.canonical_v2      resumable context-only canonical materialization
reacts.retrieval.contextual_index reaction-centre-aware evidence retrieval
reacts.retrieval.route_index  route aggregate embeddings and similarity
reacts.ml.specialists         multilabel and interval-regression trainers
reacts.ml.anomaly             family-conditional anomaly and route-quality models
reacts.validation.leakage     patent/signature/scaffold leakage audit
reacts.validation.acceptance  strict scientific acceptance suite
```

## Product One preservation

Product Two never rebuilds Product One unless an operator explicitly invokes a Product One command. `freeze-product-one` snapshots the existing dataset/index manifests, calculates hashes, reclassifies stored model stages, and registers an immutable `v1.0.0-baseline` release.

## Contextual build

The contextual builder stages Product One steps in a persistent SQLite work database so routes can be processed in deterministic order without loading the complete corpus into memory. For each route it:

1. orders steps;
2. identifies explicit symbolic anchors;
3. creates route edges;
4. resolves only unique adjacent references;
5. validates reconstructed reactions;
6. emits accepted and quarantined resolution records;
7. normalizes molecules and roles through a disk-backed catalogue;
8. materializes typed condition evidence;
9. generates deterministic repair candidates;
10. emits mapping candidates and eligibility reasons;
11. checkpoints committed routes and output parts;
12. computes decomposable route/step quality without invoking an atom mapper.

## Mapping and structural derivation

Atom mapping is a separate persistent stage. Explicit RXNMapper requests fail before queue mutation when initialization is unavailable. A single initialized model maps bounded batches, writes checksum-addressed atomic shards, then commits queue state. Restart recovers interrupted `running` rows immediately and does not repeat completed mappings.

RXNMapper, MCS fallback, and rejected records remain in separate datasets. MCS is invoked only for RXNMapper failures, never as an automatic replacement for low-confidence transformer output. By default, only validated confidence-qualified RXNMapper mappings are eligible for the independent reaction-centre derivation stage. The final `data/canonical_v2/` scientific view is materialized only after derivation is complete.

## Specialist training

Multilabel models use training-derived label vocabularies and preserve rare labels in retrieval even when they are below the supervised support threshold. Time and temperature use transformed regression with family-conditioned residual intervals. Their inference service abstains outside the evidence domain.

## Acceptance

A release cannot be locked merely because training completed. `ScientificAcceptanceValidator` combines dataset integrity, split leakage, model deserialization, checksum agreement, API/batch consistency, and retrieval benchmark gates. The release-lock function refuses any report without `strict_pass=true`.
