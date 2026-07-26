# Hugging Face publication workflow

REACTS does not require the public dataset to be embedded in the source-code repository. Build the complete canonical dataset locally, then create a publication-ready directory:

```bash
reacts --project-root . build-canonical
reacts --project-root . export-hf --destination dist/reacts-uspto-multistep
```

Add `--include-models` only when preparing a combined private staging bundle. Public model repositories should normally remain separate so each model card can identify its task, dataset version, split version, metrics, limitations, and immutable artifact.

The generated directory includes:

- canonical route, step, and quality-event datasets;
- the canonical dataset manifest;
- full-corpus validation reports;
- a generated Hugging Face dataset card;
- Git LFS patterns for Parquet, model, and vector-index artifacts;
- a compact release manifest linking the publication to its source checksum.

The publication command performs local packaging only. Authentication and network upload remain explicit operator actions.
