# RouteLens visual workbench source

This directory is the standalone Svelte 5/SvelteKit TypeScript frontend for Product Two. It is deliberately separate from the Python CLI/backend runtime.

## Local development: two terminals

Start the backend from the repository root:

```bash
reacts --project-root . serve \
  --artifact-uri dist/artifacts \
  --artifact-release product-two-artifacts-v2.0.12-r1 \
  --artifact-cache-dir dist/artifacts \
  --require-artifacts \
  --port 8000
```

Then start the frontend independently:

```bash
cd ui
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

The Vite development server owns port `5173` and proxies `/api`, `/health`, and `/ready` to the backend at `http://127.0.0.1:8000`. The browser therefore talks to the frontend origin during development; Vite forwards only the RouteLens API paths. No combined frontend/backend start command is required.

When Uvicorn reports `http://0.0.0.0:8000`, that is a backend bind address, not the frontend browser address.

## Type and build gates

```bash
npm run check
npm run build
npm run preview
```

`npm run check` is a release gate. API responses enter the client as `unknown`/JSON and are narrowed into typed UI view models before components read domain fields.

## Optional embedded fallback

The Python package still carries committed fallback assets under `../src/reacts/ui/static` for environments that want a single packaged artifact. This is not the frontend development path.

To refresh that fallback deliberately:

```bash
npm run build:embed
```

`build:embed` stages the SvelteKit build first and promotes it only after a successful build, so a compiler failure cannot delete the last working embedded page.

Do not add secrets or private backend SDKs to this client application. API keys are user-supplied at runtime and held in browser `sessionStorage` only.

## Presentation contract

The frontend is answer-first rather than backend-vocabulary-first. Primary surfaces show the requested function, the result, and understandable supporting corpus context. Model lifecycle, permitted-use strings, hashes, provenance, and complete JSON responses are available under detail disclosures or large dialogs.

Batch and retrieval results use dedicated compact summaries. They are not rendered as long JSON blocks in the main result column, and retrieval records are not labelled as batch reactions.

Workflow availability comes from `GET /api/v2/capabilities`. A supported feature whose optional data artifact is missing is shown as **Setup required** before submission, with the maintainer command hidden under technical details. This prevents predictable backend `409` responses from being presented as user input mistakes.
