<script lang="ts">
  import { browser } from '$app/environment';
  import { onMount } from 'svelte';
  import '../app.css';
  import RouteField from '$lib/components/RouteField.svelte';
  import WorkflowNav from '$lib/components/WorkflowNav.svelte';
  import ContextualPanel from '$lib/components/ContextualPanel.svelte';
  import BatchPanel from '$lib/components/BatchPanel.svelte';
  import RetrievalPanel from '$lib/components/RetrievalPanel.svelte';
  import RepairPanel from '$lib/components/RepairPanel.svelte';
  import AnomalyPanel from '$lib/components/AnomalyPanel.svelte';
  import QualityPanel from '$lib/components/QualityPanel.svelte';
  import ResultPanel from '$lib/components/ResultPanel.svelte';
  import SystemPanel from '$lib/components/SystemPanel.svelte';
  import { ApiError, RouteLensClient } from '$lib/api/client';
  import { asJsonValue } from '$lib/api/contracts';
  import type { BatchRequest, BootstrapState, CapabilityItem, ContextualRequest, HistoryItem, JsonValue, RetrievalSubmission, Workspace } from '$lib/api/types';

  const copy: Record<Workspace, { eyebrow: string; title: string; description: string }> = {
    contextual: { eyebrow: 'One reaction', title: 'What does this reaction look like?', description: 'Choose the questions you want answered. RouteLens shows the result, how much supporting context it found, and the closest patent records.' },
    batch: { eyebrow: 'Several reactions', title: 'Run the same checks across a list', description: 'Paste one reaction per line. The page shows compact summaries; each full response opens separately so large batches do not crowd the workspace.' },
    retrieval: { eyebrow: 'Similarity search', title: 'Find related reactions and routes', description: 'Search the stored patent corpus for reactions or routes that look similar, or open a route directly when you already have its ID.' },
    repair: { eyebrow: 'Reaction text check', title: 'Check a malformed or symbolic reaction', description: 'RouteLens tests conservative repairs and keeps the original text unchanged. Use the examples below to see when each optional field is appropriate.' },
    anomaly: { eyebrow: 'Condition comparison', title: 'Are these conditions unusual in the dataset?', description: 'Compare an observed temperature or reaction time with the stored reference statistics when that capability is included in the current release.' },
    quality: { eyebrow: 'Route quality', title: 'Combine six transparent quality inputs', description: 'Set the six evidence-quality components and see the weighted score. Every input and weight remains visible.' },
    system: { eyebrow: 'System details', title: 'What is available right now?', description: 'Check service readiness, available analysis tools, loaded models, the current artifact release, and the operations that remain CLI-only.' }
  };

  let workspace = $state<Workspace>('contextual');
  let toolsOpen = $state(false);
  let resultsOpen = $state(false);
  let apiKey = $state('');
  let theme = $state<'dark' | 'light'>('dark');
  let busy = $state(false);
  let error = $state('');
  let result = $state<JsonValue | null>(null);
  let bootstrap = $state<BootstrapState>({ health: null, ready: null, artifacts: null, models: null, datasets: null, capabilities: null, errors: [] });
  let history = $state<HistoryItem[]>([]);
  const client = new RouteLensClient(() => apiKey);
  const activeCopy = $derived(copy[workspace]);
  const storage = $derived(bootstrap.ready?.warmup?.route_index_storage);
  const anomalyCapability = $derived<CapabilityItem | null>(bootstrap.capabilities?.workflows.find((item) => item.id === 'anomaly') ?? null);
  const resultAvailable = $derived(busy || error.length > 0 || result !== null);

  async function loadBootstrap(): Promise<void> {
    const settled = await Promise.allSettled([client.health(), client.ready(), client.artifacts(), client.models(), client.datasets(), client.capabilities()]);
    const errors: string[] = [];
    for (const item of settled) if (item.status === 'rejected') errors.push(item.reason instanceof Error ? item.reason.message : String(item.reason));
    bootstrap = {
      health: settled[0]?.status === 'fulfilled' ? settled[0].value : null,
      ready: settled[1]?.status === 'fulfilled' ? settled[1].value : null,
      artifacts: settled[2]?.status === 'fulfilled' ? settled[2].value : null,
      models: settled[3]?.status === 'fulfilled' ? settled[3].value : null,
      datasets: settled[4]?.status === 'fulfilled' ? settled[4].value : null,
      capabilities: settled[5]?.status === 'fulfilled' ? settled[5].value : null,
      errors
    };
  }

  function recordHistory(activeWorkspace: Workspace, name: string, summary: string, payload: JsonValue): void {
    history = [{ id: crypto.randomUUID(), workspace: activeWorkspace, workflow: name, summary, payload, createdAt: new Date() }, ...history].slice(0, 8);
  }

  async function execute(activeWorkspace: Workspace, name: string, summary: string, action: () => Promise<JsonValue>): Promise<void> {
    busy = true;
    error = '';
    result = null;
    resultsOpen = true;
    try {
      result = asJsonValue(await action());
      recordHistory(activeWorkspace, name, summary, result);
    } catch (caught) {
      error = caught instanceof ApiError || caught instanceof Error ? caught.message : 'Unexpected request failure.';
    } finally {
      busy = false;
    }
  }

  async function submitContextual(payload: ContextualRequest): Promise<void> {
    await execute('contextual', 'reaction analysis', payload.reaction_smiles, () => client.contextual(payload));
  }

  async function submitBatch(payload: BatchRequest): Promise<void> {
    await execute('batch', 'batch analysis', `${payload.reactions.length} reactions`, () => client.batch(payload));
  }

  async function submitRetrieval(submission: RetrievalSubmission): Promise<void> {
    if (submission.kind === 'lookup') {
      await execute('retrieval', 'route lookup', submission.routeId, () => client.route(submission.routeId));
      return;
    }
    if (submission.kind === 'routes') {
      const request = submission.request;
      await execute('retrieval', 'similar routes', request.reaction_smiles, () => client.post('/api/v2/retrieval/routes', request));
      return;
    }
    const request = submission.request;
    await execute('retrieval', 'similar reactions', request.reaction_smiles, () => client.post('/api/v2/retrieval/reactions', request));
  }

  async function copyResult(): Promise<void> {
    if (!result) return;
    await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
  }

  function restoreHistory(item: HistoryItem): void {
    workspace = item.workspace;
    result = item.payload;
    error = '';
    resultsOpen = true;
  }

  function applyAccess(): void {
    if (!browser) return;
    if (apiKey) sessionStorage.setItem('routelens-api-key', apiKey); else sessionStorage.removeItem('routelens-api-key');
    void loadBootstrap();
  }

  function toggleTheme(): void {
    theme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('routelens-theme', theme);
  }

  onMount(() => {
    apiKey = sessionStorage.getItem('routelens-api-key') ?? '';
    theme = (localStorage.getItem('routelens-theme') as 'dark' | 'light' | null) ?? (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    document.documentElement.dataset.theme = theme;
    void loadBootstrap();
  });
</script>

<svelte:head><title>RouteLens · Reaction Analysis Workbench</title></svelte:head>
<a class="skip-link" href="#workspace-main">Skip to workbench</a>
<RouteField />
<header class="topbar">
  <a class="brand" href="/" aria-label="RouteLens workbench home"><span class="brand__mark" aria-hidden="true"><span></span><span></span><span></span></span><span><strong>RouteLens</strong><small>Reaction analysis workbench</small></span></a>
  <div class="topbar__actions">
    <button class="toolbar-button" type="button" aria-haspopup="dialog" aria-expanded={toolsOpen} onclick={() => (toolsOpen = true)}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" /></svg>
      <span>Tools</span>
    </button>
    <button class="toolbar-button" type="button" aria-haspopup="dialog" aria-expanded={resultsOpen} disabled={!resultAvailable} onclick={() => (resultsOpen = true)}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h14v14H5zM8 9h8M8 12h8M8 15h5" /></svg>
      <span>Results</span>
      {#if resultAvailable}<i class="toolbar-button__signal" aria-hidden="true"></i>{/if}
    </button>
    <div class="status-pill" data-state={bootstrap.ready?.ready ? 'ready' : bootstrap.ready?.reason_code ? 'blocked' : 'checking'} role="status" aria-live="polite"><span class="status-pill__dot" aria-hidden="true"></span><span>{bootstrap.ready?.ready ? `Service ready · v${String(bootstrap.health?.version ?? bootstrap.ready?.version ?? '')}` : bootstrap.ready?.reason_code?.replaceAll('_', ' ') ?? 'Checking service'}</span></div>
    <button class="icon-button" type="button" aria-label="Switch colour theme" onclick={toggleTheme}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a9 9 0 1 0 9 9c0-.5-.04-1-.12-1.47A7 7 0 0 1 12 3Z" /></svg></button>
    <details class="access-menu"><summary class="button button--secondary">API access</summary><div class="access-menu__panel"><label for="svelte-api-key">API key <span class="optional">optional</span></label><input id="svelte-api-key" type="password" bind:value={apiKey} autocomplete="off" /><p class="field-hint">Stored only for this browser session.</p><button class="button button--primary" type="button" onclick={applyAccess}>Reconnect</button></div></details>
  </div>
</header>
<WorkflowNav
  active={workspace}
  open={toolsOpen}
  onClose={() => (toolsOpen = false)}
  onSelect={(next) => { workspace = next; result = null; error = ''; resultsOpen = false; }}
/>
<ResultPanel {workspace} {result} {busy} {error} open={resultsOpen} onClose={() => (resultsOpen = false)} onCopy={copyResult} />
<div class="app-shell">
  <main id="workspace-main" class="workspace" tabindex="-1">
    <section class="workspace-hero" aria-labelledby="workspace-title"><div><p class="eyebrow">{activeCopy.eyebrow}</p><h1 id="workspace-title">{activeCopy.title}</h1><p>{activeCopy.description}</p></div><div class="proof-strip"><div><span>Current release</span><strong>{String(bootstrap.artifacts?.artifact_release ?? bootstrap.ready?.artifact_release ?? 'Unavailable')}</strong></div><div><span>Loaded models</span><strong>{String(bootstrap.ready?.runtime_model_count ?? bootstrap.models?.models.length ?? '—')}</strong></div><div><span>Route search</span><strong>{storage?.memory_mapped ? 'Ready' : 'Unavailable'}</strong></div></div></section>
    <section class="workspace-layout">
      <div class="control-surface" class:control-surface--plain={workspace === 'system'}>
        {#if workspace === 'contextual'}<ContextualPanel models={bootstrap.models?.models ?? []} {busy} onSubmit={submitContextual} />
        {:else if workspace === 'batch'}<BatchPanel models={bootstrap.models?.models ?? []} {busy} onSubmit={submitBatch} />
        {:else if workspace === 'retrieval'}<RetrievalPanel {busy} onSubmit={submitRetrieval} />
        {:else if workspace === 'repair'}<RepairPanel {busy} onSubmit={(payload) => execute('repair', 'repair check', String(payload.reaction_smiles), () => client.post('/api/v2/inference/repair', payload))} />
        {:else if workspace === 'anomaly'}<AnomalyPanel {busy} capability={anomalyCapability} onSubmit={(payload) => execute('anomaly', 'condition comparison', String(payload.reaction_smiles), () => client.post('/api/v2/inference/anomaly', payload))} />
        {:else if workspace === 'quality'}<QualityPanel {busy} onSubmit={(payload) => execute('quality', 'route quality', 'six-component quality score', () => client.post('/api/v2/inference/route-quality', payload))} />
        {:else}<SystemPanel {bootstrap} onRefresh={loadBootstrap} />{/if}
      </div>
    </section>
    <section class="history-section" aria-labelledby="history-title"><div><p class="eyebrow">This session</p><h2 id="history-title">Recent requests</h2></div><div class="request-history">{#if history.length === 0}<p class="empty-inline">No requests in this browser session.</p>{:else}{#each history as item}<button type="button" class="history-item" onclick={() => restoreHistory(item)}><span>{item.workflow}</span><strong>{item.summary}</strong><time>{item.createdAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></button>{/each}{/if}</div></section>
  </main>
</div>
