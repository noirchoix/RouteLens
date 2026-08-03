async function getJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
    ...options,
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function metrics(obj) {
  if (!obj) return '<p class="muted">Unavailable.</p>';
  return Object.entries(obj)
    .filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
    .slice(0, 12)
    .map(([key, value]) => `<div class="metric"><span>${key}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderModels(models) {
  const registry = document.querySelector("#models");
  const controls = document.querySelector("#task-controls");
  if (!models.length) {
    registry.innerHTML = '<p class="muted">No runtime models are available.</p>';
    controls.innerHTML = '<p class="muted">Inference is unavailable.</p>';
    return;
  }
  registry.innerHTML = models.map((model) => `
    <div class="model-card">
      <div><strong>${model.task}</strong><span class="badge">${model.stage}</span></div>
      <p>${model.permitted_use || "No permitted-use declaration."}</p>
      ${model.warning ? `<p class="warning">${model.warning}</p>` : ""}
    </div>`).join("");
  controls.innerHTML = models
    .filter((model) => model.enabled_by_default)
    .map((model, index) => `
      <label title="${model.permitted_use || ""}">
        <input type="checkbox" value="${model.task}" ${index < 4 ? "checked" : ""} />
        ${model.task}
      </label>`).join("");
}

async function load() {
  const healthNode = document.querySelector("#health");
  try {
    const health = await getJSON("/health");
    const ready = await getJSON("/ready");
    healthNode.textContent = `${health.status} · ${ready.ready ? "ready" : "not ready"} · v${health.version}`;
    healthNode.classList.toggle("not-ready", !ready.ready);
  } catch (error) {
    healthNode.textContent = error.message;
    healthNode.classList.add("not-ready");
  }

  try {
    const artifact = await getJSON("/api/v2/artifacts");
    document.querySelector("#artifact").innerHTML = metrics({
      release: artifact.artifact_release || "local unmanaged",
      mode: artifact.mode,
      ready: artifact.ready,
      cache_hit: artifact.cache_hit,
      split: artifact.training_split_sha256 ? artifact.training_split_sha256.slice(0, 16) : "n/a",
      models: artifact.runtime_model_count || 0,
    });
  } catch (error) {
    document.querySelector("#artifact").textContent = error.message;
  }

  try {
    const response = await getJSON("/api/v2/models");
    renderModels(response.models || []);
  } catch (error) {
    document.querySelector("#models").textContent = error.message;
    document.querySelector("#task-controls").textContent = error.message;
  }

  try {
    const datasets = await getJSON("/api/v2/datasets");
    document.querySelector("#dataset").innerHTML = metrics(datasets.contextual);
  } catch (error) {
    document.querySelector("#dataset").textContent = error.message;
  }
}

document.querySelector("#predict").addEventListener("click", async () => {
  const tasks = [...document.querySelectorAll("#task-controls input:checked")].map((item) => item.value);
  const reaction_smiles = document.querySelector("#reaction").value.trim();
  const output = document.querySelector("#prediction");
  output.textContent = "Running…";
  try {
    output.textContent = JSON.stringify(await getJSON("/api/v2/inference/contextual", {
      method: "POST",
      body: JSON.stringify({reaction_smiles, tasks, include_evidence: true, evidence_k: 5}),
    }), null, 2);
  } catch (error) {
    output.textContent = error.message;
  }
});

load();
