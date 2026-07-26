# Product Two v2.0.6 migration

This code-only overlay applies over v2.0.5. It migrates `data/state/product_two_mapping.sqlite3` in place and preserves all committed mapping shards. New queue columns record RXNMapper token eligibility, fallback status, bounded fallback attempts and deterministic exceptional reasons.

An interrupted v2.0.5 run should be resumed with the same command. `running` rows are recovered; completed rows and `part-00000` onward remain unchanged. Oversized records are routed directly to bounded fallback and are never retried through RXNMapper.

See `docs/BOUNDED_MAPPING_EXCEPTIONS.md`.

# Product Two v2.0.5 migration

This code-only patch applies over REACTS Product Two v2.0.1. It does not modify or copy `data/`, `reports/`, model artifacts, indexes, the frozen Product One baseline, or the archived partial MCS run.

## Preserved artifacts

```text
data/canonical/
data/models/
data/indexes/
data/baselines/v1.0.0-baseline/
data/canonical_v2_mcs_partial_20260723/   # when present
reports/partial_runs/                    # when present
```

## New stage paths

```text
data/canonical_v2_context/
data/state/product_two_mapping.sqlite3
data/mapping_v2/
data/state/product_two_derivation.sqlite3
data/derivation_v2/
data/canonical_v2/
```

## Install the verified mapper environment

```bash
python -m pip install -e ".[dev,mapping]"
```

The mapping extra pins:

```text
rxnmapper==0.4.3
setuptools==81.0.0
```

## Run the staged pipeline

```bash
reacts --project-root . build-contextual-v2 --resume

reacts --project-root . benchmark-mapper \
  --backend rxnmapper \
  --batch-sizes 8 16 32 64 \
  --sample-size 512

reacts --project-root . map-reactions \
  --backend rxnmapper \
  --fallback-backend mcs \
  --batch-size 16 \
  --workers 1 \
  --shard-size 5000 \
  --resume

reacts --project-root . derive-reaction-centres --resume
reacts --project-root . build-product-two-indexes --resume
reacts --project-root . train-product-two --request-promotion
reacts --project-root . validate-product-two
```

Only after strict acceptance:

```bash
reacts --project-root . lock-product-two --release-id v2.0.5
```

The prior monolithic `product-two` command remains as a stage orchestrator, but it delegates to the same independently resumable stages. Do not rerun the v2.0.1 inline-mapping workflow.


## v2.0.5 identity-assignment migration

The context schema version changes to `2.0.5-indexed-route-assignment-v1`. A complete v2.0.4 pre-assignment staging database is adopted when SQLite integrity, required schemas, and the 235,265/326,787 source row counts match the Product One manifest. A v2.0.5 database whose `stage_state` is `source_staged` or `assignment_started` is resumed directly. Both paths recreate only the narrow assignment table and do not reread Product One source shards.
