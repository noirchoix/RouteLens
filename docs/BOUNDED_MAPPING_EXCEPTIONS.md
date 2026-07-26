# Bounded mapping exceptions — v2.0.6

Product Two atom mapping now treats model-length and fallback-complexity limits as explicit scientific evidence rather than process failures.

## RXNMapper eligibility guard

Before transformer inference, the runner canonicalizes the reaction and counts tokens with the mapper tokenizer. The effective limit is the smallest supported configured/model/tokenizer limit, defaulting to 512 for RXNMapper 0.4.3.

```text
token_count <= token_limit  → RXNMapper batch
token_count > token_limit   → rxnmapper_sequence_too_long
```

Oversized records are never truncated and never sent to RXNMapper. Their original reaction text is preserved. Token count, token limit and eligibility are persisted in the mapping queue so resume does not repeat an invalid transformer attempt.

## Bounded MCS fallback

Every fallback record runs in an isolated spawned process. RDKit retains its internal per-comparison timeout, while the parent runner enforces a separate hard per-record process timeout, defaulting to 30 seconds. The child is terminated when the deadline is exceeded.

Deterministic fallback outcomes are:

```text
mapped
low_confidence
timeout
failed
```

A timeout or terminal fallback failure is written with:

```text
mapping_status = failed
validation_status = quarantined
scientific_eligibility = false
```

Quarantined deterministic exceptions are not automatically reclaimed by later `--resume` runs.

## Atomic exceptional lane

Normal completed mappings are flushed before an exceptional record is committed. Exceptional records are also written to:

```text
data/mapping_v2/reaction_mapping_exceptions/
```

This prevents one pathological organometallic or oversized reaction from holding thousands of normal results in an uncommitted in-memory shard.

## Resuming a v2.0.5 queue

The queue schema is migrated in place. Existing committed shards and statuses are preserved. Interrupted `running` rows are recovered to `pending`; persisted token metadata is reused when present.

```bash
reacts --project-root . map-reactions \
  --backend rxnmapper \
  --fallback-backend mcs \
  --batch-size 8 \
  --workers 1 \
  --shard-size 5000 \
  --rxnmapper-token-limit 512 \
  --fallback-timeout-seconds 30 \
  --resume
```

The mapping manifest reports:

```text
rxnmapper_token_eligible
rxnmapper_sequence_too_long
mcs_fallback_attempted
mcs_fallback_mapped
mcs_fallback_low_confidence
mcs_fallback_timeout
mcs_fallback_failed
mapping_exception_quarantined
```
