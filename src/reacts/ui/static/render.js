(function attachRouteLensRender(global) {
  "use strict";

  function escapeHTML(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function humanize(value) {
    return String(value ?? "")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function number(value, digits = 3) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(digits) : "—";
  }

  function percent(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : "—";
  }

  function compactHash(value) {
    const text = String(value || "");
    return text.length > 18 ? `${text.slice(0, 10)}…${text.slice(-6)}` : text || "—";
  }

  function statusBadge(label, tone = "neutral") {
    return `<span class="data-badge" data-tone="${escapeHTML(tone)}">${escapeHTML(label)}</span>`;
  }

  function renderPredictions(task) {
    const predictions = Array.isArray(task.predictions) ? task.predictions : [];
    if (!predictions.length && task.point_estimate == null) {
      return `<p class="empty-inline">No ranked prediction was returned.</p>`;
    }
    const estimate = task.point_estimate == null ? "" : `
      <div class="estimate-block">
        <strong>${escapeHTML(number(task.point_estimate, 2))} ${escapeHTML(task.units || "")}</strong>
        ${Array.isArray(task.interval) ? `<span>Interval ${escapeHTML(number(task.interval[0], 2))}–${escapeHTML(number(task.interval[1], 2))}</span>` : ""}
      </div>`;
    const bars = predictions.slice(0, 8).map((item) => {
      const score = Number(item.probability ?? item.combined_score ?? 0);
      const width = Math.max(0, Math.min(100, score * 100));
      return `<div class="prediction-row">
        <div class="prediction-row__label"><strong>${escapeHTML(humanize(item.label))}</strong><span>${escapeHTML(percent(score))}</span></div>
        <div class="score-track" aria-label="${escapeHTML(item.label)} probability ${escapeHTML(percent(score))}"><span style="--score:${width}%"></span></div>
      </div>`;
    }).join("");
    return `${estimate}${bars}`;
  }

  function renderTask(task) {
    const stage = String(task.model_stage || task.lifecycle_state || "unknown");
    const tone = task.abstained ? "warning" : stage === "production" || stage === "validated" ? "success" : "info";
    const warnings = Array.isArray(task.warnings) ? task.warnings : [];
    return `<article class="task-result">
      <header>
        <div><p class="task-result__task">${escapeHTML(humanize(task.task))}</p><h3>${task.abstained ? "Governed abstention" : "Ranked output"}</h3></div>
        <div class="badge-row">${statusBadge(stage, tone)}${task.applicability ? statusBadge(humanize(task.applicability), "neutral") : ""}</div>
      </header>
      ${task.reason ? `<p class="task-reason">${escapeHTML(task.reason)}</p>` : ""}
      ${renderPredictions(task)}
      <dl class="task-meta">
        <div><dt>Model</dt><dd>${escapeHTML(task.model_id || "—")}</dd></div>
        <div><dt>Neighbour support</dt><dd>${escapeHTML(task.neighbour_support ?? 0)}</dd></div>
        <div><dt>Permitted use</dt><dd>${escapeHTML(task.permitted_use || "Not declared")}</dd></div>
      </dl>
      ${warnings.map((warning) => `<div class="inline-warning"><span aria-hidden="true">!</span><p>${escapeHTML(warning)}</p></div>`).join("")}
    </article>`;
  }

  function renderEvidence(evidence) {
    if (!Array.isArray(evidence) || !evidence.length) return "";
    return `<section class="evidence-section"><div class="section-heading"><div><p class="eyebrow">Retrieved corpus evidence</p><h3>${evidence.length} nearest record${evidence.length === 1 ? "" : "s"}</h3></div></div>
      <div class="evidence-list">${evidence.map((item, index) => `
        <article class="evidence-item">
          <div class="evidence-item__rank">${String(index + 1).padStart(2, "0")}</div>
          <div class="evidence-item__body">
            <div class="evidence-item__top"><code>${escapeHTML(item.reaction_smiles || "")}</code>${statusBadge(`score ${number(item.score, 3)}`, "neutral")}</div>
            <dl>
              <div><dt>Route</dt><dd>${escapeHTML(item.route_id || "—")}</dd></div>
              <div><dt>Patent</dt><dd>${escapeHTML(item.patent_document_id || "—")}</dd></div>
              <div><dt>Temperature</dt><dd>${escapeHTML(item.temperature_bucket || "—")}</dd></div>
              <div><dt>Time</dt><dd>${escapeHTML(item.time_bucket || "—")}</dd></div>
            </dl>
          </div>
        </article>`).join("")}</div></section>`;
  }

  function renderInference(payload) {
    const tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
    const tone = payload.parse_ok === false ? "danger" : payload.applicability === "in_domain" ? "success" : "warning";
    return `<div class="result-stack">
      <section class="result-summary">
        <div class="result-summary__main">
          <div class="reaction-chip"><span>Input</span><code>${escapeHTML(payload.input_reaction || "")}</code></div>
          ${payload.canonical_reaction && payload.canonical_reaction !== payload.input_reaction ? `<div class="reaction-chip"><span>Canonical</span><code>${escapeHTML(payload.canonical_reaction)}</code></div>` : ""}
        </div>
        <div class="badge-row">${statusBadge(payload.parse_ok ? "Parse valid" : humanize(payload.parse_failure_class || "parse failure"), tone)}${statusBadge(humanize(payload.applicability || "unknown"), "neutral")}</div>
      </section>
      <section class="task-results">${tasks.map(renderTask).join("") || `<p class="empty-inline">No task results were returned.</p>`}</section>
      ${renderEvidence(payload.evidence)}
      <details class="raw-result"><summary>Raw response and provenance</summary><pre>${escapeHTML(JSON.stringify(payload, null, 2))}</pre></details>
    </div>`;
  }

  function metricEntries(payload) {
    if (!global.RouteLensApi.isRecord(payload)) return [];
    return Object.entries(payload).filter(([, value]) => ["string", "number", "boolean"].includes(typeof value)).slice(0, 16);
  }

  function renderGeneric(payload, title = "Response") {
    const entries = metricEntries(payload);
    const metricGrid = entries.length ? `<dl class="metric-grid">${entries.map(([key, value]) => `<div><dt>${escapeHTML(humanize(key))}</dt><dd>${escapeHTML(typeof value === "number" ? number(value, 4) : value)}</dd></div>`).join("")}</dl>` : "";
    return `<div class="result-stack"><section class="generic-result"><p class="eyebrow">${escapeHTML(title)}</p>${metricGrid}<details class="raw-result" open><summary>Structured response</summary><pre>${escapeHTML(JSON.stringify(payload, null, 2))}</pre></details></section></div>`;
  }

  function renderBatch(payload) {
    const results = Array.isArray(payload.results) ? payload.results : Array.isArray(payload.predictions) ? payload.predictions : [];
    return `<div class="result-stack"><section class="result-summary"><div><p class="eyebrow">Batch complete</p><h3>${escapeHTML(payload.rows ?? results.length)} reaction${Number(payload.rows ?? results.length) === 1 ? "" : "s"}</h3></div>${statusBadge(payload.artifact_release || "artifact runtime", "info")}</section>
      <div class="batch-results">${results.map((result, index) => `<details ${index === 0 ? "open" : ""}><summary><span>Reaction ${index + 1}</span><code>${escapeHTML(result.input_reaction || "")}</code></summary>${renderInference(result)}</details>`).join("")}</div>
      <details class="raw-result"><summary>Raw batch response</summary><pre>${escapeHTML(JSON.stringify(payload, null, 2))}</pre></details></div>`;
  }

  function renderRetrieval(payload, mode) {
    const candidates = Array.isArray(payload.results) ? payload.results : Array.isArray(payload.routes) ? payload.routes : Array.isArray(payload.evidence) ? payload.evidence : [];
    if (!candidates.length) return renderGeneric(payload, `${humanize(mode)} retrieval`);
    return `<div class="result-stack"><section class="result-summary"><div><p class="eyebrow">${escapeHTML(humanize(mode))} retrieval</p><h3>${candidates.length} nearest result${candidates.length === 1 ? "" : "s"}</h3></div></section>
      <div class="retrieval-results">${candidates.map((item, index) => `<article class="retrieval-card"><span class="retrieval-card__rank">${String(index + 1).padStart(2, "0")}</span><div><code>${escapeHTML(item.reaction_smiles || item.route_id || item.route_instance_id || "Result")}</code><dl>${Object.entries(item).filter(([key, value]) => key !== "reaction_smiles" && ["string", "number", "boolean"].includes(typeof value)).slice(0, 8).map(([key, value]) => `<div><dt>${escapeHTML(humanize(key))}</dt><dd>${escapeHTML(typeof value === "number" ? number(value, 4) : value)}</dd></div>`).join("")}</dl></div></article>`).join("")}</div>
      <details class="raw-result"><summary>Raw retrieval response</summary><pre>${escapeHTML(JSON.stringify(payload, null, 2))}</pre></details></div>`;
  }

  function renderSystem(bootstrap) {
    const health = bootstrap.health || {};
    const ready = bootstrap.ready || {};
    const artifact = bootstrap.artifacts || {};
    const models = bootstrap.models && Array.isArray(bootstrap.models.models) ? bootstrap.models.models : [];
    const datasets = bootstrap.datasets || {};
    const storage = ready.warmup && ready.warmup.route_index_storage ? ready.warmup.route_index_storage : {};
    return `<div class="system-grid">
      <section class="system-block"><p class="eyebrow">Runtime</p><h3>${ready.ready ? "Ready for inference" : "Not ready"}</h3><dl class="metric-grid"><div><dt>Service version</dt><dd>${escapeHTML(health.version || ready.version || "—")}</dd></div><div><dt>Artifact release</dt><dd>${escapeHTML(artifact.artifact_release || ready.artifact_release || "—")}</dd></div><div><dt>Validation</dt><dd>${escapeHTML(ready.validation_pass === true ? "Pass" : ready.validation_pass === false ? "Fail" : "—")}</dd></div><div><dt>Cache hit</dt><dd>${escapeHTML(ready.cache_hit ?? "—")}</dd></div></dl></section>
      <section class="system-block"><p class="eyebrow">Memory-safe retrieval</p><h3>${escapeHTML(storage.vectors_format || "Route index")}</h3><dl class="metric-grid"><div><dt>Rows</dt><dd>${escapeHTML(storage.rows ?? "—")}</dd></div><div><dt>Dimensions</dt><dd>${escapeHTML(storage.dimensions ?? "—")}</dd></div><div><dt>Chunk rows</dt><dd>${escapeHTML(storage.search_chunk_rows ?? "—")}</dd></div><div><dt>Memory mapped</dt><dd>${escapeHTML(storage.memory_mapped ?? "—")}</dd></div></dl></section>
      <section class="system-block system-block--wide"><p class="eyebrow">Model capabilities</p><h3>${models.length} runtime model${models.length === 1 ? "" : "s"}</h3><div class="model-ledger">${models.map((model) => `<article><div><strong>${escapeHTML(humanize(model.task))}</strong>${statusBadge(model.stage || model.lifecycle_state || "unknown", model.release_approved ? "success" : "info")}</div><p>${escapeHTML(model.permitted_use || "No permitted-use declaration.")}</p>${model.warning ? `<span class="ledger-warning">${escapeHTML(model.warning)}</span>` : ""}</article>`).join("") || `<p class="empty-inline">No model capabilities available.</p>`}</div></section>
      <section class="system-block system-block--wide"><p class="eyebrow">Dataset contract</p><h3>${escapeHTML((datasets.contextual && datasets.contextual.dataset_version) || artifact.manifest?.dataset_version || "Contextual dataset")}</h3><details class="raw-result"><summary>Dataset and artifact details</summary><pre>${escapeHTML(JSON.stringify({ datasets, artifacts: artifact }, null, 2))}</pre></details></section>
    </div>`;
  }

  global.RouteLensRender = {
    escapeHTML,
    humanize,
    compactHash,
    renderInference,
    renderGeneric,
    renderBatch,
    renderRetrieval,
    renderSystem,
  };
})(window);
