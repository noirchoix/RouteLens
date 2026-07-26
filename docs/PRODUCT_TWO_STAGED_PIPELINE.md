# Product Two v2.0.9 staged structural pipeline

Product Two separates route contextualization, atom mapping, reaction-centre derivation, indexing, training, and validation. Each expensive stage has its own persistent state and can be resumed without rebuilding an earlier stage.

## Stage directories

```text
data/canonical_v2_context/                 context-only canonical data
data/state/product_two_mapping.sqlite3     persistent mapping queue
data/mapping_v2/                           segregated mapping products
data/state/product_two_derivation.sqlite3  persistent derivation queue
data/derivation_v2/                        bond-change and family products
data/canonical_v2/                         final qualified scientific view
```

The archived `data/canonical_v2_mcs_partial_20260723/` directory is an engineering baseline only. It is never selected automatically or merged into the scientific dataset.

## 1. Build contextual data

```bash
reacts --project-root . build-contextual-v2 --resume
```

This stage performs route ordering, conservative symbolic-intermediate resolution, molecule normalization, route-edge construction, condition evidence, repair candidates, and mapping eligibility. It does not initialize an atom mapper.

A checkpoint is committed every 5,000 routes. On resume, any data parts beyond the last committed checkpoint are removed before processing continues. The source staging database and molecule catalogue are persistent and explicitly closed on all exit paths.

## 2. Benchmark RXNMapper

```bash
reacts --project-root . benchmark-mapper \
  --backend rxnmapper \
  --batch-sizes 8 16 32 64 \
  --sample-size 512
```

The benchmark records throughput, median and p95 batch latency, process memory, mapping outcomes, confidence, and a recommended batch size. The mapper is initialized once and reused.

RXNMapper 0.4.3 requires the pinned compatibility environment:

```text
rxnmapper==0.4.3
setuptools==81.0.0
```

## 3. Run resumable mapping

```bash
reacts --project-root . map-reactions \
  --backend rxnmapper \
  --fallback-backend mcs \
  --batch-size 16 \
  --workers 1 \
  --prefetch-batches 2 \
  --shard-size 5000 \
  --resume
```

Backend preflight occurs before queue mutation. Explicit RXNMapper mode fails immediately if initialization fails. `auto` mode is strict by default and also fails rather than silently creating an MCS run; `--allow-auto-fallback` is required for an explicitly labelled fallback baseline.

The queue statuses are:

```text
pending
running
mapped
low_confidence
failed
not_eligible
rejected
```

On resume, rows left in `running` by an interrupted process are immediately returned to `pending`. Completed rows are not repeated. Mapping outputs are separated into:

```text
data/mapping_v2/reaction_mappings_rxnmapper/
data/mapping_v2/reaction_mappings_mcs_fallback/
data/mapping_v2/reaction_mappings_rejected/
```

RXNMapper is primary. MCS is attempted only for RXNMapper failures, not for low-confidence mappings. MCS output is never scientifically eligible by default.

## 4. Derive structural chemistry

```bash
reacts --project-root . derive-reaction-centres --resume
```

The default population is validated, confidence-qualified RXNMapper mappings. `--include-mcs` is an explicit opt-in for auxiliary analysis and prevents the run from claiming a strict RXNMapper-only training population.

The stage derives formed, broken, and changed bonds, atom-environment changes, reaction-centre fingerprints, templates, and structural families. It writes atomic shards and maintains a separate queue. The final `data/canonical_v2/` scientific view is materialized only after the derivation queue is complete.

Context tables are hard-linked where supported and copied otherwise, avoiding unnecessary duplication on the same filesystem.

## 5. Rebuild governed splits

```bash
reacts --project-root . rebuild-product-two-splits --seed 42
```

This rewrites split metadata only in the final scientific view and invalidates stale Product Two runtime models. Contextualization, mapping, and derivation are not rerun.

## 6. Continue downstream work

```bash
reacts --project-root . build-product-two-indexes --resume
reacts --project-root . train-product-two --request-promotion
reacts --project-root . validate-product-two
```

Strict validation requires:

- a completed contextual manifest;
- a completed mapping queue;
- RXNMapper as the primary scientific backend;
- segregated mapping outputs;
- a completed derivation queue;
- a materialized final canonical dataset;
- zero patent and exact-reaction split overlap;
- split-bound reaction and route indexes;
- lifecycle-filtered model loading with no archived deserialization;
- exact scikit-learn 1.9.0 model/runtime compatibility;
- existing API and retrieval acceptance checks.

Only after `strict_pass` is true:

```bash
reacts --project-root . lock-product-two --release-id v2.0.9
```

## Staged wrapper

The wrapper remains available but delegates to the independent stages:

```bash
reacts --project-root . product-two \
  --from-stage mapping \
  --stop-after derivation \
  --mapping-backend rxnmapper \
  --batch-size 16 \
  --resume
```

The supported stage names are `context`, `mapping`, `derivation`, `splits`, `indexes`, `training`, and `validation`.


## Indexed route-instance assignment

The context stage creates source-route and source-step indexes before assignment. Nonduplicated source routes are written to the narrow `route_instance_assignment` table with one indexed SQL join. Only duplicated source-route groups enter exact route-text matching. The atomic `.work/identity_assignment_manifest.json` records assignment counts, timing, matching scope, and completion state. Contextual route iteration joins this table to immutable staged Product One steps.
