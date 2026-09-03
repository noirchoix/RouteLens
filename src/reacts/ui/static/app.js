(function startRouteLensWorkbench(global) {
  "use strict";

  const workspaceCopy = {
    contextual: {
      eyebrow: "Single reaction · contextual inference",
      title: "Analyze a reaction with every governed model",
      description: "Submit reaction SMILES, choose the tasks that matter, and review calibrated outputs beside their supporting patent evidence.",
    },
    batch: {
      eyebrow: "Multiple reactions · governed batch",
      title: "Run the same inference contract across a batch",
      description: "Analyze one reaction per line without losing model lifecycle, warning, applicability, evidence, or provenance fields.",
    },
    retrieval: {
      eyebrow: "Similarity search · corpus evidence",
      title: "Find related reactions and complete routes",
      description: "Search the immutable reaction and route indexes, then inspect a route returned by the evidence layer.",
    },
    repair: {
      eyebrow: "Parse intelligence · deterministic repair",
      title: "Evaluate a malformed or symbolic reaction",
      description: "Rank conservative repair candidates and preserve ambiguity rather than silently altering the source chemistry.",
    },
    anomaly: {
      eyebrow: "Condition intelligence · corpus-relative scoring",
      title: "Check whether observed conditions are unusual",
      description: "Compare reaction temperature and time against transparent family-conditional statistics.",
    },
    quality: {
      eyebrow: "Route governance · decomposable score",
      title: "Build a transparent route-quality assessment",
      description: "Adjust each auditable component and review the service-calculated aggregate without hiding the inputs.",
    },
    system: {
      eyebrow: "Runtime contract · exact release",
      title: "Inspect the service before trusting a result",
      description: "Review readiness, artifact identity, model lifecycle, dataset version, and memory-mapped route storage.",
    },
  };

  const state = {
    activeWorkspace: "contextual",
    retrievalMode: "reactions",
    apiKey: sessionStorage.getItem("routelens-api-key") || "",
    bootstrap: { health: null, ready: null, artifacts: null, models: null, datasets: null },
    rawResult: null,
    history: [],
  };

  const api = global.RouteLensApi.createClient(() => state.apiKey);
  const render = global.RouteLensRender;

  const byId = (id) => document.getElementById(id);
  const asInput = (id) => byId(id);
  const resultContent = byId("result-content");
  const copyButton = byId("copy-result");

  function toast(message, tone = "info") {
    const region = byId("toast-region");
    const node = document.createElement("div");
    node.className = "toast";
    node.dataset.tone = tone;
    node.textContent = message;
    region.append(node);
    requestAnimationFrame(() => node.classList.add("is-visible"));
    setTimeout(() => {
      node.classList.remove("is-visible");
      setTimeout(() => node.remove(), 250);
    }, 3200);
  }

  function setLoading(button, loading, label) {
    button.disabled = loading;
    button.classList.toggle("is-loading", loading);
    const text = button.querySelector("span");
    if (text) {
      if (!button.dataset.idleLabel) button.dataset.idleLabel = text.textContent || "Run";
      text.textContent = loading ? label : button.dataset.idleLabel;
    }
  }

  function setResult(html, raw) {
    state.rawResult = raw;
    copyButton.disabled = raw == null;
    resultContent.classList.remove("is-updating");
    void resultContent.offsetWidth;
    resultContent.innerHTML = html;
    resultContent.classList.add("is-updating");
  }

  function showPending(label) {
    setResult(`<div class="loading-state"><span class="loading-orbit" aria-hidden="true"><i></i><i></i><i></i></span><h3>${render.escapeHTML(label)}</h3><p>The runtime is validating the request and preserving its inference contract.</p></div>`, null);
  }

  function showError(error) {
    const status = error && typeof error.status === "number" ? error.status : 0;
    const recovery = status === 401 ? "Open Access, enter the configured API key, then reconnect." : status === 503 ? "Check the runtime status and artifact readiness before retrying." : "Review the fields and retry the request.";
    setResult(`<div class="error-state"><span class="error-state__mark" aria-hidden="true">!</span><p class="eyebrow">Request not completed</p><h3>${render.escapeHTML(error instanceof Error ? error.message : "Unexpected error")}</h3><p>${render.escapeHTML(recovery)}</p></div>`, error && error.payload ? error.payload : { error: String(error) });
    toast("Request failed. Review the result panel.", "danger");
  }

  function addHistory(workflow, summary, payload) {
    state.history.unshift({ workflow, summary, payload, timestamp: new Date() });
    state.history = state.history.slice(0, 6);
    const container = byId("request-history");
    container.innerHTML = state.history.map((item, index) => `<button type="button" class="history-item" data-history-index="${index}"><span>${render.escapeHTML(render.humanize(item.workflow))}</span><strong>${render.escapeHTML(item.summary)}</strong><time>${render.escapeHTML(item.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }))}</time></button>`).join("");
    container.querySelectorAll("[data-history-index]").forEach((button) => {
      button.addEventListener("click", () => {
        const item = state.history[Number(button.dataset.historyIndex)];
        if (item) setResult(render.renderGeneric(item.payload, `${render.humanize(item.workflow)} response`), item.payload);
      });
    });
  }

  function selectedTasks(containerId) {
    return [...document.querySelectorAll(`#${containerId} input[type="checkbox"]:checked`)].map((input) => input.value);
  }

  function renderTaskPickers() {
    const response = state.bootstrap.models;
    const models = response && Array.isArray(response.models) ? response.models : [];
    ["contextual-tasks", "batch-tasks"].forEach((containerId) => {
      const container = byId(containerId);
      if (!models.length) {
        container.innerHTML = `<p class="empty-inline">No runtime models are available.</p>`;
        return;
      }
      container.innerHTML = models.map((model, index) => `<label class="task-option" title="${render.escapeHTML(model.permitted_use || "")}">
        <input type="checkbox" value="${render.escapeHTML(model.task)}" ${model.enabled_by_default !== false && (containerId === "contextual-tasks" || index < 3) ? "checked" : ""} />
        <span><strong>${render.escapeHTML(render.humanize(model.task))}</strong><small>${render.escapeHTML(model.stage || model.lifecycle_state || "unknown")}</small></span>
      </label>`).join("");
    });
  }

  function updateRuntimeProof() {
    const health = state.bootstrap.health || {};
    const ready = state.bootstrap.ready || {};
    const artifact = state.bootstrap.artifacts || {};
    const storage = ready.warmup && ready.warmup.route_index_storage ? ready.warmup.route_index_storage : {};
    const proof = byId("runtime-proof");
    proof.innerHTML = `<div><span>Artifact</span><strong title="${render.escapeHTML(artifact.artifact_release || ready.artifact_release || "")}">${render.escapeHTML(artifact.artifact_release || ready.artifact_release || "Unavailable")}</strong></div>
      <div><span>Models</span><strong>${render.escapeHTML(ready.runtime_model_count ?? health.models ?? "—")}</strong></div>
      <div><span>Route vectors</span><strong>${storage.memory_mapped ? "Memory mapped" : "Unavailable"}</strong></div>`;

    const pill = byId("service-pill");
    const isReady = ready.ready === true;
    pill.dataset.state = isReady ? "ready" : ready.reason_code ? "blocked" : "checking";
    pill.querySelector("span:last-child").textContent = isReady ? `Ready · v${health.version || ready.version || ""}` : ready.reason_code ? render.humanize(ready.reason_code) : "Checking runtime";
  }

  async function bootstrap() {
    const labels = ["health", "ready", "artifacts", "models", "datasets"];
    const calls = [api.health(), api.ready(), api.artifacts(), api.models(), api.datasets()];
    const results = await Promise.allSettled(calls);
    results.forEach((result, index) => {
      state.bootstrap[labels[index]] = result.status === "fulfilled" ? result.value : { error: result.reason instanceof Error ? result.reason.message : String(result.reason) };
    });
    renderTaskPickers();
    updateRuntimeProof();
    if (state.activeWorkspace === "system") showSystem();
    const modelError = state.bootstrap.models && state.bootstrap.models.error;
    if (modelError) toast(modelError, "warning");
  }

  function switchWorkspace(name) {
    state.activeWorkspace = name;
    document.querySelectorAll("[data-workspace]").forEach((button) => {
      const active = button.dataset.workspace === name;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    });
    document.querySelectorAll("[data-workspace-panel]").forEach((panel) => {
      const active = panel.dataset.workspacePanel === name;
      panel.hidden = !active;
      panel.classList.toggle("is-active", active);
    });
    const copy = workspaceCopy[name];
    byId("workspace-eyebrow").textContent = copy.eyebrow;
    byId("workspace-title").textContent = copy.title;
    byId("workspace-description").textContent = copy.description;
    byId("workspace-main").focus({ preventScroll: true });
    if (name === "system") showSystem();
  }

  function showSystem() {
    const html = render.renderSystem(state.bootstrap);
    byId("system-overview").innerHTML = html;
    setResult(html, state.bootstrap);
  }

  async function submitWorkflow(event, config) {
    event.preventDefault();
    const button = event.currentTarget.querySelector('button[type="submit"]');
    const payload = config.payload();
    if (!payload) return;
    setLoading(button, true, config.loadingLabel);
    showPending(config.loadingLabel);
    try {
      const response = await config.request(payload);
      setResult(config.render(response), response);
      addHistory(config.historyName, config.summary(response, payload), response);
      toast("Request completed.", "success");
    } catch (error) {
      showError(error);
    } finally {
      setLoading(button, false, config.loadingLabel);
    }
  }

  document.querySelectorAll("[data-workspace]").forEach((button) => button.addEventListener("click", () => switchWorkspace(button.dataset.workspace)));

  document.querySelectorAll('input[type="range"]').forEach((input) => {
    const output = input.closest("label")?.querySelector("output") || document.querySelector(`output[for="${input.id}"]`);
    const sync = () => { if (output) output.value = input.step && Number(input.step) < 1 ? Number(input.value).toFixed(2) : input.value; };
    input.addEventListener("input", sync);
    sync();
  });

  document.querySelectorAll("[data-retrieval-mode]").forEach((button) => button.addEventListener("click", () => {
    state.retrievalMode = button.dataset.retrievalMode;
    document.querySelectorAll("[data-retrieval-mode]").forEach((item) => item.setAttribute("aria-selected", String(item === button)));
    const lookup = state.retrievalMode === "lookup";
    byId("retrieval-query-fields").hidden = lookup;
    byId("route-lookup-fields").hidden = !lookup;
    byId("minimum-quality-field").hidden = state.retrievalMode !== "reactions";
  }));

  byId("contextual-form").addEventListener("submit", (event) => submitWorkflow(event, {
    loadingLabel: "Analyzing reaction…",
    historyName: "contextual inference",
    payload: () => {
      const reaction_smiles = asInput("contextual-reaction").value.trim();
      const tasks = selectedTasks("contextual-tasks");
      if (!reaction_smiles) { toast("Enter a reaction SMILES.", "warning"); return null; }
      if (!tasks.length) { toast("Select at least one model task.", "warning"); return null; }
      return { reaction_smiles, tasks, include_evidence: asInput("contextual-evidence").checked, evidence_k: Number(asInput("contextual-evidence-k").value), allow_experimental: asInput("contextual-experimental").checked };
    },
    request: (payload) => api.contextual(payload),
    render: render.renderInference,
    summary: (_response, payload) => payload.reaction_smiles,
  }));

  byId("batch-form").addEventListener("submit", (event) => submitWorkflow(event, {
    loadingLabel: "Analyzing batch…",
    historyName: "batch inference",
    payload: () => {
      const reactions = asInput("batch-reactions").value.split(/\r?\n/).map((value) => value.trim()).filter(Boolean);
      const tasks = selectedTasks("batch-tasks");
      if (!reactions.length) { toast("Enter at least one reaction.", "warning"); return null; }
      if (!tasks.length) { toast("Select at least one model task.", "warning"); return null; }
      return { reactions, tasks, include_evidence: asInput("batch-evidence").checked, evidence_k: Number(asInput("batch-evidence-k").value), allow_experimental: asInput("batch-experimental").checked };
    },
    request: (payload) => api.batch(payload),
    render: render.renderBatch,
    summary: (response) => `${response.rows || 0} reactions`,
  }));

  byId("retrieval-form").addEventListener("submit", (event) => submitWorkflow(event, {
    loadingLabel: "Searching evidence…",
    historyName: `${state.retrievalMode} retrieval`,
    payload: () => {
      if (state.retrievalMode === "lookup") {
        const routeId = asInput("route-id").value.trim();
        if (!routeId) { toast("Enter a route identifier.", "warning"); return null; }
        return { routeId };
      }
      const reaction_smiles = asInput("retrieval-reaction").value.trim();
      if (!reaction_smiles) { toast("Enter a query reaction.", "warning"); return null; }
      return { reaction_smiles, k: Number(asInput("retrieval-k").value), minimum_quality: Number(asInput("retrieval-quality").value) };
    },
    request: (payload) => state.retrievalMode === "lookup" ? api.route(payload.routeId) : state.retrievalMode === "routes" ? api.retrieveRoutes({ reaction_smiles: payload.reaction_smiles, k: payload.k }) : api.retrieveReactions(payload),
    render: (response) => state.retrievalMode === "lookup" ? render.renderGeneric(response, "Route detail") : render.renderRetrieval(response, state.retrievalMode),
    summary: (_response, payload) => payload.routeId || payload.reaction_smiles,
  }));

  byId("repair-form").addEventListener("submit", (event) => submitWorkflow(event, {
    loadingLabel: "Evaluating repair…",
    historyName: "parse repair",
    payload: () => {
      const reaction_smiles = asInput("repair-reaction").value.trim();
      if (!reaction_smiles) { toast("Enter a reaction to repair.", "warning"); return null; }
      const contextual_candidate = asInput("repair-candidate").value.trim();
      return { reaction_smiles, contextual_candidate: contextual_candidate || null, route_continuity_score: Number(asInput("repair-continuity").value) };
    },
    request: (payload) => api.repair(payload),
    render: (response) => render.renderGeneric(response, "Deterministic repair evaluation"),
    summary: (_response, payload) => payload.reaction_smiles,
  }));

  byId("anomaly-form").addEventListener("submit", (event) => submitWorkflow(event, {
    loadingLabel: "Scoring conditions…",
    historyName: "condition anomaly",
    payload: () => {
      const reaction_smiles = asInput("anomaly-reaction").value.trim();
      const temperature = asInput("anomaly-temperature").value;
      const time = asInput("anomaly-time").value;
      if (!reaction_smiles) { toast("Enter a reaction SMILES.", "warning"); return null; }
      if (temperature === "" && time === "") { toast("Provide temperature, time, or both.", "warning"); return null; }
      return { reaction_smiles, temperature_c: temperature === "" ? null : Number(temperature), time_h: time === "" ? null : Number(time) };
    },
    request: (payload) => api.anomaly(payload),
    render: (response) => render.renderGeneric(response, "Condition anomaly score"),
    summary: (_response, payload) => payload.reaction_smiles,
  }));

  byId("quality-form").addEventListener("submit", (event) => submitWorkflow(event, {
    loadingLabel: "Calculating quality…",
    historyName: "route quality",
    payload: () => Object.fromEntries([...document.querySelectorAll("[data-quality-key]")].map((input) => [input.dataset.qualityKey, Number(input.value)])),
    request: (payload) => api.routeQuality(payload),
    render: (response) => render.renderGeneric(response, "Route quality assessment"),
    summary: (response) => `score ${response.score ?? response.overall_score ?? "returned"}`,
  }));

  byId("copy-result").addEventListener("click", async () => {
    if (state.rawResult == null) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(state.rawResult, null, 2));
      toast("Raw response copied.", "success");
    } catch (_error) {
      toast("Clipboard access was unavailable.", "warning");
    }
  });

  byId("api-key").value = state.apiKey;
  byId("save-api-key").addEventListener("click", async () => {
    state.apiKey = byId("api-key").value.trim();
    if (state.apiKey) sessionStorage.setItem("routelens-api-key", state.apiKey); else sessionStorage.removeItem("routelens-api-key");
    await bootstrap();
    toast("Access settings applied.", "success");
  });

  byId("refresh-system").addEventListener("click", async () => {
    await bootstrap();
    showSystem();
    toast("Runtime state refreshed.", "success");
  });

  const themeButton = byId("theme-toggle");
  const preferredTheme = localStorage.getItem("routelens-theme") || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  document.documentElement.dataset.theme = preferredTheme;
  themeButton.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("routelens-theme", next);
    toast(`${render.humanize(next)} theme active.`);
  });

  bootstrap();
})(window);
