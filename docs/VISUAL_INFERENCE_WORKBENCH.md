# RouteLens Visual Inference Workbench

## Purpose

The visual workbench is the plain-language frontend for Product Two analysis. It helps users answer reaction questions without requiring them to understand model lifecycle terminology, registry structure, artifact hashes, or JSON response contracts.

The interface follows one rule:

> **Show the function, the answer, and the supporting context first. Put implementation and governance details behind progressive disclosure.**

The Python CLI remains the operational interface for building datasets, mapping, derivation, training, validation, artifact publication, and release locking. Those operations are intentionally not reproduced as browser controls.

## What users can do

| User-facing workflow | Product Two capability | Endpoint | Availability |
| --- | --- | --- | --- |
| Analyze one reaction | Contextual inference | `POST /api/v2/inference/contextual` | Runtime-dependent |
| Analyze several reactions | Batch contextual inference | `POST /api/v2/inference/batch` | Runtime-dependent |
| Find similar records | Reaction/route retrieval | `POST /api/v2/retrieval/reactions`, `/routes` | Index-dependent |
| Open a route | Route detail | `GET /api/v2/routes/{route_id}` | Index-dependent |
| Check a broken reaction | Deterministic parse repair | `POST /api/v2/inference/repair` | Available |
| Check conditions | Corpus-relative condition comparison | `POST /api/v2/inference/anomaly` | Requires condition statistics |
| Score route quality | Transparent route-quality score | `POST /api/v2/inference/route-quality` | Available |
| Inspect system details | Runtime/artifact/model/dataset inspection | `/health`, `/ready`, `/api/v2/artifacts`, `/models`, `/datasets`, `/capabilities` | Available |

`GET /api/v2/capabilities` is the frontend's authoritative availability contract. It distinguishes:

- **available** — the workflow can run now;
- **setup required** — the backend supports it but a prerequisite artifact is absent;
- **CLI only** — operational work that intentionally stays outside the analysis UI;
- **unavailable** — the current runtime cannot provide the capability.

The frontend should not discover a known missing prerequisite only after the user submits a form.

## Result presentation

The main result surface is deliberately not a raw backend response viewer.

For analysis it shows:

1. whether the input was understood;
2. the useful answer or a clear statement that the evidence is insufficient;
3. an understandable confidence/support description;
4. similar patent records or other supporting corpus context;
5. model/provenance details only when the user opens **Model details** or **Complete response**.

For batch analysis, each reaction is summarized in a compact card. Full per-reaction output and the complete batch JSON open in a large dialog rather than occupying the narrow result column.

For retrieval, records are presented as search results with route/patent identity, similarity, record quality, and available reaction-family information. Retrieval results are never labelled as batch inference results.

For deterministic repair, an empty candidate list is a valid outcome: it means RouteLens found no conservative correction it could justify. The optional evidence-supported replacement field is explained with complete examples instead of assuming knowledge of internal route-context terminology.

For condition comparison, the UI checks capability status before enabling submission. If condition reference statistics are absent, the workflow explains what is missing and exposes the maintainer CLI command under a technical-details disclosure instead of returning an avoidable `409` after submission.

## Condition-comparison setup

The current `product-two-artifacts-v2.0.12-r1` release does not contain the optional condition-anomaly statistics artifact. The UI therefore reports **Setup required** for condition comparison while all other validated read-only Product Two workflows remain usable.

A maintainer can create the statistics in the source workspace:

```bash
reacts --project-root . build-anomaly-model
```

When `data/models/condition_anomaly/robust_family_stats.json` exists, the artifact publisher includes it as an optional checksummed auxiliary artifact. Publish a new exact artifact release rather than overwriting `r1`:

```bash
reacts --project-root . package-product-two-artifacts \
  --release product-two-artifacts-v2.0.12-r2 \
  --destination dist/artifacts

reacts --project-root . validate-artifact-bundle \
  --bundle dist/artifacts/product-two-artifacts-v2.0.12-r2 \
  --service-version 2.1.6
```

Serving `r2` then makes condition comparison available if validation succeeds. This process does not retrain the eight inference models or alter the governed split/index science.

## Frontend/backend separation

The maintainable Svelte 5/SvelteKit frontend lives in `ui/`. Development uses two independent processes.

Terminal 1 — backend from the repository root:

```bash
reacts --project-root . serve \
  --artifact-uri dist/artifacts \
  --artifact-release product-two-artifacts-v2.0.12-r1 \
  --artifact-cache-dir dist/artifacts \
  --require-artifacts \
  --port 8000
```

Terminal 2 — frontend:

```bash
cd ui
npm run check
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

Vite proxies only `/api`, `/health`, and `/ready` to `http://127.0.0.1:8000`. The frontend and backend therefore remain separate development services without requiring local CORS changes.

`0.0.0.0:8000` is only Uvicorn's bind address. It is not the browser URL.

## Type and accessibility contract

API responses are untrusted JSON at the frontend boundary and are narrowed into domain-specific view models before their fields are rendered. The UI does not use `any`, blanket type assertions, or non-null assertions to bypass strict checking.

The interface uses persistent labels, explicit loading/error/empty states, keyboard-visible focus, semantic controls, colour-independent status text, and `prefers-reduced-motion` handling. Technical JSON remains available for debugging and provenance inspection, but it is secondary to the user-facing interpretation.

## CLI-only operations

The UI's System details view lists, but does not execute, operational commands such as:

- contextual dataset construction;
- atom mapping and derivation;
- split/index rebuilding;
- model training and promotion evaluation;
- scientific validation and release locking;
- artifact publication and validation;
- Hugging Face export.

This separation keeps the visual workbench useful to less-technical users without reducing or hiding the backend's technical capability.
