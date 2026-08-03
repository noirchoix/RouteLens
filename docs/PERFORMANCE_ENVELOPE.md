# Product Two v2.1.0 Performance Envelope

The service must be measured on the actual deployment hardware. The values below are initial acceptance targets, not claims about every environment.

## Initial CPU targets

| Measurement | Initial target |
| --- | ---: |
| Cached startup to readiness | < 60 seconds |
| Single non-retrieval inference p95 | < 500 ms |
| Reaction or route retrieval p95 | < 2 seconds |
| Registry writes during startup/inference | 0 |
| Missing/corrupt artifact behavior | fail closed |
| Batch size | explicitly bounded by configuration |
| Request body | explicitly bounded by configuration |
| Route warm-up allocation | one mapped row, not the complete matrix |
| Route-search working set | bounded by `search_chunk_rows` (default 2,048) |

The expanded artifact directory contains an uncompressed memory-mapped route matrix of approximately 2.38 GiB. Capacity planning must include the source bundle, staging directory, installed cache, and container layer where applicable. The ZIP distribution remains compressed, but extraction requires sufficient free disk space.

## Measurements

Record separately:

- artifact download size and duration;
- artifact verification duration;
- cache installation duration;
- model deserialization and warm-up duration;
- idle resident memory;
- warmed resident memory;
- single-request p50/p95/max;
- batch latency by batch size;
- reaction retrieval p50/p95/max;
- route retrieval p50/p95/max;
- throughput at each tested concurrency;
- timeout, rejection, and error rate.

Use `scripts/benchmark_artifact_runtime.py` against a ready staging instance. Store the JSON result with the release evidence. Do not compare performance across unlike hardware without identifying CPU, memory, operating system, Python, NumPy, SciPy, scikit-learn, joblib, and RDKit versions.
