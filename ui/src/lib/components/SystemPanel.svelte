<script lang="ts">
  import { stageLabel, taskLabel } from '$lib/domain/presentation';
  import type { BootstrapState, CapabilityItem } from '$lib/api/types';

  type Props = { bootstrap: BootstrapState; onRefresh: () => void };
  let { bootstrap, onRefresh }: Props = $props();

  const storage = $derived(bootstrap.ready?.warmup?.route_index_storage);
  const release = $derived(String(bootstrap.artifacts?.artifact_release ?? bootstrap.ready?.artifact_release ?? 'not reported'));
  const version = $derived(String(bootstrap.health?.version ?? bootstrap.ready?.version ?? 'not reported'));
  const modelCount = $derived(bootstrap.ready?.runtime_model_count ?? bootstrap.models?.models.length ?? 0);

  function capabilityStatus(capability: CapabilityItem): string {
    if (capability.available) return 'Available';
    if (capability.state === 'setup_required') return 'Setup required';
    if (capability.state === 'cli_only') return 'CLI only';
    return 'Unavailable';
  }
</script>

<div class="system-overview">
  <header class="system-lead">
    <div>
      <p class="eyebrow">Current runtime</p>
      <h3>{bootstrap.ready?.ready ? 'RouteLens is ready for analysis.' : 'RouteLens is not ready for analysis.'}</h3>
      <p>
        Service version {version}, using artifact release <code>{release}</code>. {modelCount} analysis model{modelCount === 1 ? '' : 's'} are loaded.
        {#if storage?.memory_mapped} Route search is using the memory-mapped index.{/if}
      </p>
    </div>
    <button class="button button--secondary" type="button" onclick={onRefresh}>Refresh status</button>
  </header>

  <dl class="system-facts" aria-label="Runtime facts">
    <div><dt>Artifact validation</dt><dd>{bootstrap.ready?.validation_pass === true ? 'Passed' : bootstrap.ready?.validation_pass === false ? 'Failed' : 'Not reported'}</dd></div>
    <div><dt>Models loaded</dt><dd>{modelCount}</dd></div>
    <div><dt>Routes indexed</dt><dd>{String(storage?.rows ?? 'Not reported')}</dd></div>
    <div><dt>Route-vector size</dt><dd>{String(storage?.dimensions ?? 'Not reported')}</dd></div>
  </dl>

  <section class="system-section" aria-labelledby="available-tools-title">
    <div class="system-section__heading">
      <div><p class="eyebrow">Available now</p><h3 id="available-tools-title">Analysis tools in this release</h3></div>
      <p>These are the functions the browser can use without changing datasets, models or release state.</p>
    </div>
    <ul class="system-list">
      {#each bootstrap.capabilities?.workflows ?? [] as capability}
        <li>
          <div><strong>{capability.label}</strong>{#if capability.reason}<p>{capability.reason}</p>{/if}</div>
          <span class="system-status" data-state={capability.state}>{capabilityStatus(capability)}</span>
          {#if capability.setup_command}
            <details class="system-command"><summary>Maintainer setup</summary><code>{capability.setup_command}</code></details>
          {/if}
        </li>
      {/each}
    </ul>
  </section>

  <section class="system-section" aria-labelledby="models-title">
    <div class="system-section__heading">
      <div><p class="eyebrow">Models</p><h3 id="models-title">What powers the reaction analysis</h3></div>
      <p>Preview status is shown because some models are intentionally research-facing rather than unrestricted production models.</p>
    </div>
    <div class="system-model-list">
      {#each bootstrap.models?.models ?? [] as model}
        <div class="system-model-row">
          <div><strong>{taskLabel(model.task)}</strong><p>{model.permitted_use ?? 'No use statement was provided.'}</p></div>
          <span>{stageLabel(model.stage ?? model.lifecycle_state ?? null)}</span>
          {#if model.warning}<details><summary>Why this is marked as preview</summary><p>{model.warning}</p></details>{/if}
        </div>
      {/each}
    </div>
  </section>

  <section class="system-section" aria-labelledby="cli-title">
    <div class="system-section__heading">
      <div><p class="eyebrow">Maintainer operations</p><h3 id="cli-title">What intentionally stays in the CLI</h3></div>
      <p>These operations change data, model or release state, so the read-only workbench does not expose them as browser buttons.</p>
    </div>
    <ul class="system-command-list">
      {#each bootstrap.capabilities?.cli_only ?? [] as capability}
        <li><strong>{capability.label}</strong>{#if capability.reason}<span>{capability.reason}</span>{/if}{#if capability.setup_command}<code>{capability.setup_command}</code>{/if}</li>
      {/each}
    </ul>
  </section>

  <details class="technical-disclosure system-technical"><summary>Dataset and artifact JSON</summary><pre>{JSON.stringify({ datasets: bootstrap.datasets, artifacts: bootstrap.artifacts }, null, 2)}</pre></details>
</div>
