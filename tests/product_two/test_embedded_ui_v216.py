from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from reacts.api.main import create_app
from reacts.settings import Settings


def test_visual_workbench_is_embedded_without_changing_cli_runtime(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(project_root=tmp_path))) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "RouteLens · Visual Inference Workbench" in home.text
        assert 'data-ui-version="2.1.6"' in home.text
        assert "Read-only inference mode" in home.text

        assets = {
            "api": client.get("/static/api.js"),
            "render": client.get("/static/render.js"),
            "app": client.get("/static/app.js"),
            "styles": client.get("/static/styles.css"),
        }
        assert all(response.status_code == 200 for response in assets.values())

    javascript = "\n".join(response.text for name, response in assets.items() if name != "styles")
    for endpoint in (
        "/api/v2/inference/contextual",
        "/api/v2/inference/batch",
        "/api/v2/inference/repair",
        "/api/v2/inference/anomaly",
        "/api/v2/inference/route-quality",
        "/api/v2/retrieval/reactions",
        "/api/v2/retrieval/routes",
        "/api/v2/routes/",
    ):
        assert endpoint in javascript

    assert "sessionStorage" in javascript
    assert "X-API-Key" in javascript
    assert "X-REACTS-Allow-Experimental" in javascript
    assert "prefers-reduced-motion" in assets["styles"].text
    assert ":focus-visible" in assets["styles"].text


def test_svelte_source_uses_svelte_five_typescript_contract() -> None:
    root = Path(__file__).parents[2]
    page = (root / "ui" / "src" / "routes" / "+page.svelte").read_text(encoding="utf-8")
    component = (root / "ui" / "src" / "lib" / "components" / "ContextualPanel.svelte").read_text(
        encoding="utf-8"
    )
    client = (root / "ui" / "src" / "lib" / "api" / "client.ts").read_text(encoding="utf-8")

    assert '<script lang="ts">' in page
    assert "$state" in page
    assert "$derived" in page
    assert "onclick=" in page
    assert "on:click" not in page
    assert "$props()" in component
    assert "any" not in client


def test_svelte_embed_build_uses_current_paths_contract_and_staged_promotion() -> None:
    root = Path(__file__).parents[2]
    config = (root / "ui" / "svelte.config.js").read_text(encoding="utf-8")
    package = (root / "ui" / "package.json").read_text(encoding="utf-8")
    build_script = (root / "ui" / "scripts" / "build-embedded.mjs").read_text(encoding="utf-8")

    assert "assets: '/static'" not in config
    assert "bundleStrategy: 'inline'" in config
    assert "ROUTELENS_UI_OUTPUT_DIR" in config
    assert '"build:embed": "node scripts/build-embedded.mjs"' in package
    assert "spawnSync" in build_script
    assert ".embedded-build" in build_script
    assert "copyFile" in build_script
    assert "SvelteKit did not emit index.html" in build_script


def test_svelte_dev_server_is_separate_from_python_backend() -> None:
    root = Path(__file__).parents[2]
    package = (root / "ui" / "package.json").read_text(encoding="utf-8")
    vite = (root / "ui" / "vite.config.ts").read_text(encoding="utf-8")
    readme = (root / "ui" / "README.md").read_text(encoding="utf-8")

    assert '"dev": "vite dev"' in package
    assert "port: 5173" in vite
    assert "strictPort: true" in vite
    assert "http://127.0.0.1:8000" in vite
    for path in ("/api", "/health", "/ready"):
        assert path in vite
    assert "npm run dev" in readme
    assert "No combined frontend/backend start command" in readme


def test_svelte_result_contracts_are_narrowed_before_rendering() -> None:
    root = Path(__file__).parents[2]
    contracts = (root / "ui" / "src" / "lib" / "api" / "contracts.ts").read_text(encoding="utf-8")
    result_panel = (root / "ui" / "src" / "lib" / "components" / "ResultPanel.svelte").read_text(encoding="utf-8")
    retrieval_panel = (root / "ui" / "src" / "lib" / "components" / "RetrievalPanel.svelte").read_text(encoding="utf-8")

    assert "taskResultViews" in contracts
    assert "evidenceViews" in contracts
    assert "batchResultViews" in contracts
    assert "jsonObjectOrNull" in result_panel
    assert "taskResultViews(record?.tasks)" in result_panel
    assert "RetrievalSubmission" in retrieval_panel
    assert "kind: 'lookup'" in retrieval_panel
    assert "kind: 'routes'" in retrieval_panel
    assert "kind: 'reactions'" in retrieval_panel


def test_condition_setup_state_does_not_render_dead_form_or_result_column() -> None:
    root = Path(__file__).parents[2]
    anomaly = (root / "ui" / "src" / "lib" / "components" / "AnomalyPanel.svelte").read_text(encoding="utf-8")
    page = (root / "ui" / "src" / "routes" / "+page.svelte").read_text(encoding="utf-8")
    styles = (root / "ui" / "src" / "app.css").read_text(encoding="utf-8")

    assert "const checking = $derived(capability === null)" in anomaly
    assert "{:else if unavailable}" in anomaly
    assert "Nothing is wrong with your reaction input" in anomaly
    assert "anomalyStatusOnly" in page
    assert "{#if !anomalyStatusOnly}<ResultPanel" in page
    assert ".workspace-layout--single" in styles


def test_reaction_tools_navigation_scrolls_independently_on_desktop() -> None:
    root = Path(__file__).parents[2]
    styles = (root / "ui" / "src" / "app.css").read_text(encoding="utf-8")

    assert "overflow-y: auto" in styles
    assert "overscroll-behavior: contain" in styles
    assert "scrollbar-gutter: stable" in styles
    assert "@media (max-width: 820px)" in styles
