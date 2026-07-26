# External source artifacts

Large unchanged source artifacts are intentionally excluded from the REACTS release archive.

Place either of the following at this location before running the full product pipeline:

```text
data/source_artifacts/uspto_llm_multistep_only.zip
```

or point `REACTS_SOURCE_ARTIFACT` to an extracted directory containing:

```text
uspto_llm_multistep_only/
├── multistep_csv/
│   ├── step_table.csv
│   ├── route_summary.csv
│   └── cleaned_full.csv
├── qc/
│   ├── multistep_csv_qc.json
│   └── multistep_csv_condition_qc.json
├── artifact_manifest.json
└── run_summary.json
```

The uploaded source ZIP used during development had SHA-256 recorded in the generated canonical manifest. The source is read-only and is never rewritten.
