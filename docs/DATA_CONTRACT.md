# Canonical data contract

## `steps`

Primary key: `step_id`.

Required semantic groups:

- Identity: `dataset_version`, `step_id`, `route_id`, `patent_document_id`, `split`, `step_index`.
- Chemistry: `reaction_smiles`, `canonical_reaction_smiles`, `reactants`, `products`.
- Structural validity: `parse_ok`, side validity, `parse_failure_class`, symbolic status.
- Conditions: typed solvent and agent lists; observed, legacy and cleaned temperature/time values; buckets; extraction method and confidence.
- Quality: `condition_status`, `quality_issues`, `quality_score`.
- Eligibility: task-specific booleans for parsing, conditions and retrieval.

Observed source values are never overwritten. Clean columns are nullable views suitable for supervised learning.

## `routes`

Primary key: `route_uid`.

`route_id` preserves the source identifier. Exact duplicate routes are collapsed. Conflicting content sharing one source ID is retained using `::v<content-hash>`.

## `quality_events`

Every material correction or exclusion is represented by an event containing entity type, entity ID, rule code, severity, observed value, message and dataset version.

## Split contract

```text
split = SHA1(patent_document_id) mod 10,000
train: 0–7,999
val:   8,000–8,999
test:  9,000–9,999
```

This produces stable 80/10/10 assignment without patent overlap and without requiring row-order state.

## Formats

Parquet is the production format. If `pyarrow` is not installed, Product One emits compressed CSV as an explicit compatibility fallback and records `storage_format=csv.gz` in the manifest.
