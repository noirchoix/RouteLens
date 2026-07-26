# REACTS Product One 1.0.0

## Included

- complete-corpus canonicalization for 326,787 reaction steps and 235,265 retained routes;
- deterministic duplicate repair and patent-document-grouped splits;
- repaired and confidence-scored temperature/time extraction;
- five trained, registered production models;
- 156,082-reaction sharded Morgan evidence index;
- FastAPI service, batch inference, route inspection, jobs, metrics, and embedded operator UI;
- publication-ready Hugging Face export workflow;
- tests, data-contract validation, model leakage guards, model cards, and full-corpus reports.

## Important validation correction

An early condition-model run was rejected because the legacy working-reaction string contained the condition middle field. All shipped condition models were retrained using canonical structure-only `reactants>>products` inputs. Only corrected metrics and model artifacts are included.

## External inputs intentionally omitted

The original USPTO artifact, EDA notebook, and legacy producer ZIP remain user-owned inputs. Their expected paths and SHA-256 checksums are listed in `reports/external_artifacts.json`.

## Storage note

The validated sandbox run used compressed CSV because no Parquet engine was installed. The application is Parquet-first and declares `pyarrow` as a production dependency; rebuilding after normal installation writes Parquet/Zstandard under the same data contract.
