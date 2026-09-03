<script lang="ts">
  import DetailDialog from './DetailDialog.svelte';
  import {
    batchResultViews,
    distributionViews,
    evidenceViews,
    isJsonObject,
    jsonObjectOrNull,
    retrievalResultViews,
    taskResultViews
  } from '$lib/api/contracts';
  import {
    humanize,
    plainAnomalyReason,
    plainReason,
    stageLabel,
    stageNote,
    supportLabel,
    taskDescription,
    taskLabel,
    taskSummary
  } from '$lib/domain/presentation';
  import type { DistributionView, JsonObject, JsonValue, Workspace } from '$lib/api/types';

  type Props = {
    workspace: Workspace;
    result: JsonValue | null;
    busy: boolean;
    error: string;
    open: boolean;
    onClose: () => void;
    onCopy: () => void;
  };

  let { workspace, result, busy, error, open, onClose, onCopy }: Props = $props();
  let dialog: HTMLDialogElement;

  const record = $derived(jsonObjectOrNull(result));
  const tasks = $derived(taskResultViews(record?.tasks));
  const evidence = $derived(evidenceViews(record?.evidence));
  const distributions = $derived(distributionViews(record?.neighbour_label_distributions));
  const batchResults = $derived(batchResultViews(record?.results));
  const retrievalResults = $derived(retrievalResultViews(record?.results));

  let detailOpen = $state(false);
  let detailTitle = $state('Technical details');
  let detailSubtitle = $state('');
  let detailPayload = $state<JsonValue | null>(null);

  $effect(() => {
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  });

  function closePanel(): void {
    if (dialog?.open) dialog.close();
  }

  function score(value: number): number {
    return Math.max(0, Math.min(1, value));
  }

  function text(value: JsonValue | undefined, fallback = ''): string {
    return typeof value === 'string' ? value : fallback;
  }

  function numberValue(value: JsonValue | undefined): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function objectValue(value: JsonValue | undefined): JsonObject | null {
    return value !== undefined && isJsonObject(value) ? value : null;
  }

  function objectNumber(value: JsonObject | null, key: string): number | null {
    const candidate = value?.[key];
    return typeof candidate === 'number' && Number.isFinite(candidate) ? candidate : null;
  }

  function stringList(value: JsonValue | undefined): string[] {
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
  }

  function openDetails(title: string, payload: JsonValue, subtitle = ''): void {
    detailTitle = title;
    detailSubtitle = subtitle;
    detailPayload = payload;
    detailOpen = true;
  }

  function formatUnits(units: string | null): string {
    if (units === 'degC') return '°C';
    if (units === 'h') return 'h';
    return units ?? '';
  }

  function capabilityErrorHint(message: string): string {
    if (message.toLowerCase().includes('condition anomaly model has not been built')) {
      return 'Condition comparison is not included in this artifact release. Open “System details” to see the setup status.';
    }
    return 'Check the input and the system status, then try again.';
  }

  function distributionLabel(field: string): string {
    if (field === 'temperature_bucket') return 'Temperatures in similar records';
    if (field === 'time_bucket') return 'Reaction times in similar records';
    if (field === 'solvent_primary') return 'Primary solvents in similar records';
    if (field === 'solvents') return 'Solvents in similar records';
    if (field === 'agents') return 'Reagents in similar records';
    if (field === 'reaction_family') return 'Reaction types in similar records';
    return humanize(field);
  }

  function distributionItemLabel(distribution: DistributionView, label: string): string {
    if (distribution.field === 'temperature_bucket') return `${label} °C`;
    return humanize(label);
  }

  function percentage(value: number): string {
    return `${Math.round(score(value) * 100)}%`;
  }
</script>

<dialog
  class="result-drawer"
  bind:this={dialog}
  aria-labelledby="result-title"
  onclose={onClose}
  onclick={(event) => {
    if (event.target === dialog) closePanel();
  }}
>
  <aside class="result-surface" aria-labelledby="result-title">
    <div class="result-surface__header">
      <div>
        <p class="eyebrow">Output</p>
        <h2 id="result-title">Result</h2>
      </div>
      <div class="result-surface__actions">
        <button class="icon-button" type="button" aria-label="Copy complete JSON response" title="Copy complete JSON response" disabled={!result} onclick={onCopy}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 8V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-3M5 8h9a2 2 0 0 1 2 2v9a2 2 0 0 1 2 2H5a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2Z" /></svg>
        </button>
        <button class="icon-button" type="button" aria-label="Close results" onclick={closePanel}>×</button>
      </div>
    </div>

  <div class="result-content" aria-live="polite">
    {#if busy}
      <div class="loading-state">
        <span class="loading-orbit" aria-hidden="true"><i></i><i></i><i></i></span>
        <h3>Working on your request…</h3>
        <p>RouteLens is running the selected analysis and collecting the supporting records.</p>
      </div>
    {:else if error}
      <div class="error-state">
        <span class="error-state__mark" aria-hidden="true">!</span>
        <p class="eyebrow">Could not complete this request</p>
        <h3>{error}</h3>
        <p>{capabilityErrorHint(error)}</p>
      </div>
    {:else if !result}
      <div class="empty-state">
        <div class="empty-state__diagram" aria-hidden="true"><span></span><i></i><span></span><i></i><span></span></div>
        <h3>Results will appear here</h3>
        <p>Run the active tool. RouteLens shows the useful interpretation and supporting evidence here; raw service fields stay available separately.</p>
      </div>
    {:else if workspace === 'batch' && batchResults.length > 0}
      <div class="result-stack">
        <section class="result-summary result-summary--actions">
          <div><p class="eyebrow">Batch finished</p><h3>{String(record?.rows ?? batchResults.length)} reactions analyzed</h3><p>Every selected task is summarized below. Supporting records can be expanded per reaction; the raw batch remains available separately.</p></div>
          <button class="button button--secondary" type="button" onclick={() => openDetails('Raw batch response', result, `${batchResults.length} reactions`)}>Raw batch</button>
        </section>
        <div class="batch-card-list">
          {#each batchResults as batch, index}
            <article class="batch-card">
              <header><span>{String(index + 1).padStart(2, '0')}</span><code>{batch.inputReaction}</code></header>
              <div class="badge-row">
                <span class="data-badge" data-tone={batch.parseOk === true ? 'success' : batch.parseOk === false ? 'danger' : 'info'}>{batch.parseOk === true ? 'Input recognized' : batch.parseOk === false ? 'Input needs repair' : 'Input status unavailable'}</span>
                {#if batch.applicability}<span class="data-badge">{supportLabel(batch.applicability)}</span>{/if}
                {#if batch.reactionFamily}<span class="data-badge" data-tone="info">{humanize(batch.reactionFamily)}</span>{/if}
                {#if batch.evidenceCount > 0}<span class="data-badge">{batch.evidenceCount} supporting record{batch.evidenceCount === 1 ? '' : 's'}</span>{/if}
              </div>
              <div class="batch-card__summary" aria-label={`Task results for reaction ${index + 1}`}>
                {#each batch.tasks as task}
                  <div><strong>{taskLabel(task.task)}</strong><span>{taskSummary(task).replace(`${taskLabel(task.task)}: `, '')}</span></div>
                {/each}
              </div>
              {#if batch.evidence.length > 0}
                <details class="batch-evidence">
                  <summary>Show supporting records ({batch.evidence.length})</summary>
                  <div class="batch-evidence-list">
                    {#each batch.evidence as item}
                      <div><code>{item.reactionSmiles}</code><span>{percentage(item.score)} similar · Patent {item.patentDocumentId ?? 'not reported'}</span></div>
                    {/each}
                  </div>
                </details>
              {/if}
              <button class="text-button" type="button" onclick={() => openDetails(`Raw response · reaction ${index + 1}`, batch.raw, batch.inputReaction)}>Raw response →</button>
            </article>
          {/each}
        </div>
      </div>
    {:else if workspace === 'retrieval' && retrievalResults.length > 0}
      <div class="result-stack">
        <section class="result-summary result-summary--actions">
          <div><p class="eyebrow">Search finished</p><h3>{retrievalResults.length} similar record{retrievalResults.length === 1 ? '' : 's'}</h3><p>Higher similarity means a closer match in this index. Record quality and route context are shown separately so similarity is not mistaken for experimental validation.</p></div>
          <button class="button button--secondary" type="button" onclick={() => openDetails('Raw search response', result)}>Raw response</button>
        </section>
        <div class="retrieval-list">
          {#each retrievalResults as item, index}
            <article class="retrieval-card">
              <div class="retrieval-card__rank">{String(index + 1).padStart(2, '0')}</div>
              <div class="retrieval-card__body">
                <header><div><strong>{item.routeId}</strong>{#if item.patentDocumentId}<small>Patent {item.patentDocumentId}</small>{/if}</div><span class="similarity-score">{percentage(item.score)} similar</span></header>
                {#if item.reactionSmiles}<code>{item.reactionSmiles}</code>{/if}
                <div class="retrieval-card__meta">
                  {#if item.qualityScore !== null}<span>Record quality {percentage(item.qualityScore)}</span>{/if}
                  {#if item.stepCount !== null}<span>{item.stepCount} step{item.stepCount === 1 ? '' : 's'}</span>{/if}
                  {#if item.reactionFamilies.length}<span>Reaction type: {item.reactionFamilies.map(humanize).join(', ')}</span>{/if}
                </div>
                <button class="text-button" type="button" onclick={() => openDetails(`Technical record · result ${index + 1}`, item.raw, item.routeId)}>Technical record →</button>
              </div>
            </article>
          {/each}
        </div>
      </div>
    {:else if workspace === 'retrieval' && record && text(record.route_id)}
      {@const routeFamilies = stringList(record.reaction_families)}
      {@const routeQuality = numberValue(record.quality_score)}
      {@const routeSteps = numberValue(record.step_count)}
      <div class="result-stack">
        <section class="result-summary"><div><p class="eyebrow">Route opened</p><h3>{text(record.route_id)}</h3><p>{text(record.patent_document_id) ? `Patent ${text(record.patent_document_id)}` : 'Patent identifier not reported.'}</p></div></section>
        <dl class="metric-grid"><div><dt>Steps</dt><dd>{routeSteps ?? '—'}</dd></div><div><dt>Record quality</dt><dd>{routeQuality === null ? '—' : percentage(routeQuality)}</dd></div><div><dt>Dataset split</dt><dd>{text(record.split, '—')}</dd></div><div><dt>Reaction types</dt><dd>{routeFamilies.length ? routeFamilies.map(humanize).join(', ') : 'Not labeled'}</dd></div></dl>
        <p class="support-note">This is the complete plain-language route summary available from the route index. Internal identifiers and split-component fields remain in the technical record.</p>
        <button class="button button--secondary" type="button" onclick={() => openDetails('Technical route record', result, text(record.route_id))}>Technical record</button>
      </div>
    {:else if workspace === 'contextual' && tasks.length > 0}
      <div class="result-stack">
        <section class="result-summary">
          <div class="result-summary__main">
            <div class="reaction-chip"><span>Reaction</span><code>{String(record?.input_reaction ?? '')}</code></div>
            {#if record?.canonical_reaction && record.canonical_reaction !== record.input_reaction}<div class="reaction-chip"><span>Normalized</span><code>{String(record.canonical_reaction)}</code></div>{/if}
          </div>
          <div class="badge-row">
            <span class="data-badge" data-tone={record?.parse_ok === true ? 'success' : 'danger'}>{record?.parse_ok === true ? 'Input recognized' : 'Input needs repair'}</span>
            <span class="data-badge">{supportLabel(record?.applicability)}</span>
          </div>
        </section>

        <div class="result-overview-grid">
          <div><span>Reaction type</span><strong>{text(record?.reaction_family) ? humanize(text(record?.reaction_family)) : 'Not confidently labeled'}</strong></div>
          <div><span>Supporting records</span><strong>{evidence.length}</strong></div>
          <div><span>Questions answered</span><strong>{tasks.filter((task) => !task.abstained).length} of {tasks.length}</strong></div>
        </div>

        <section class="task-results">
          {#each tasks as task}
            {@const note = stageNote(task.modelStage)}
            <article class="task-result">
              <header>
                <div><p class="task-result__task">{taskLabel(task.task)}</p><h3>{task.abstained ? 'No reliable answer' : taskDescription(task.task)}</h3></div>
                {#if task.modelStage}<span class="data-badge task-result__stage" data-tone={task.modelStage === 'production' ? 'success' : 'info'}>{stageLabel(task.modelStage)}</span>{/if}
              </header>

              {#if task.reason}<p class="task-reason">{plainReason(task.reason)}</p>{/if}

              {#if task.pointEstimate !== null}
                <div class="estimate-block"><strong>{task.pointEstimate.toFixed(1)} {formatUnits(task.units)}</strong>{#if task.interval}<span>Likely range {task.interval[0].toFixed(1)}–{task.interval[1].toFixed(1)} {formatUnits(task.units)}</span>{/if}</div>
              {/if}

              {#if task.abstained && task.predictions.length > 0}<p class="result-subtitle">Low-support signals — shown for context, not as a recommended answer.</p>{/if}
              {#each task.predictions as prediction}
                <div class="prediction-row"><div class="prediction-row__label"><strong>{humanize(prediction.label)}</strong><span>{percentage(prediction.probability)}</span></div><div class="score-track"><span style={`--score:${score(prediction.probability) * 100}%`}></span></div></div>
              {/each}

              <div class="task-support-row">
                <span>{task.neighbourSupport} similar record{task.neighbourSupport === 1 ? '' : 's'} used for support</span>
                {#if task.reactionFamilyAgreement !== null}<span>{percentage(task.reactionFamilyAgreement)} reaction-type agreement</span>{/if}
              </div>
              {#if note}<p class="result-caution">{note}</p>{/if}
              <details class="technical-disclosure">
                <summary>Model and provenance details</summary>
                <dl class="task-meta"><div><dt>Status</dt><dd>{stageLabel(task.modelStage ?? task.lifecycleState)}</dd></div><div><dt>Model ID</dt><dd>{task.modelId ?? '—'}</dd></div><div><dt>Intended use</dt><dd>{task.permittedUse ?? 'Not declared'}</dd></div></dl>
                {#each task.warnings as warning}<div class="inline-warning"><span aria-hidden="true">!</span><p>{warning}</p></div>{/each}
              </details>
            </article>
          {/each}
        </section>

        {#if distributions.length > 0}
          <section class="evidence-snapshot">
            <div class="section-heading"><p class="eyebrow">What the closest records contain</p><h3>Reference pattern</h3><p>This is a weighted summary of the retrieved patent records, not a prediction by itself.</p></div>
            <div class="evidence-snapshot__grid">
              {#each distributions as distribution}
                <div><strong>{distributionLabel(distribution.field)}</strong><p>{distribution.items.slice(0, 3).map((item) => `${distributionItemLabel(distribution, item.label)} ${percentage(item.probability)}`).join(' · ')}</p></div>
              {/each}
            </div>
          </section>
        {/if}

        {#if evidence.length > 0}
          <section class="evidence-section">
            <div class="section-heading"><p class="eyebrow">Supporting context</p><h3>Similar patent records</h3><p>These records explain the reference context available to the models. They are not experimental proof for your reaction.</p></div>
            <div class="evidence-list">
              {#each evidence as item, index}
                <article class="evidence-item">
                  <div class="evidence-item__rank">{String(index + 1).padStart(2, '0')}</div>
                  <div class="evidence-item__body">
                    <div class="evidence-item__top"><code>{item.reactionSmiles}</code><span class="data-badge">{percentage(item.score)} similar</span></div>
                    <dl>
                      <div><dt>Route</dt><dd>{item.routeId}</dd></div>
                      <div><dt>Patent</dt><dd>{item.patentDocumentId ?? '—'}</dd></div>
                      {#if item.qualityScore !== null}<div><dt>Record quality</dt><dd>{percentage(item.qualityScore)}</dd></div>{/if}
                      {#if item.reactionFamily}<div><dt>Reaction type</dt><dd>{humanize(item.reactionFamily)}</dd></div>{/if}
                      {#if item.temperatureBucket}<div><dt>Temperature</dt><dd>{item.temperatureBucket} °C</dd></div>{/if}
                      {#if item.timeBucket}<div><dt>Time</dt><dd>{item.timeBucket}</dd></div>{/if}
                      {#if item.solventPrimary}<div><dt>Primary solvent</dt><dd><code>{item.solventPrimary}</code></dd></div>{/if}
                      {#if item.agents.length}<div><dt>Reported agents</dt><dd>{item.agents.join(', ')}</dd></div>{/if}
                    </dl>
                  </div>
                </article>
              {/each}
            </div>
          </section>
        {/if}

        <div class="result-detail-actions"><button class="button button--secondary" type="button" onclick={() => openDetails('Technical analysis response', result, 'Model identifiers, hashes, provenance and every raw service field.')}>Technical details</button></div>
      </div>
    {:else if workspace === 'repair'}
      {@const candidates = Array.isArray(record?.candidates) ? record.candidates.filter(isJsonObject) : []}
      {@const accepted = objectValue(record?.accepted_candidate)}
      {@const contract = objectValue(record?.contract)}
      <div class="result-stack">
        <section class="result-summary"><div><p class="eyebrow">Repair check</p><h3>{accepted ? 'A candidate passed the strict checks' : candidates.length ? 'Candidates found, but none passed all checks' : 'No repair candidate was produced'}</h3><p>{candidates.length ? `${candidates.length} candidate${candidates.length === 1 ? '' : 's'} evaluated.` : 'The service preserved the original reaction rather than inventing a replacement.'}</p></div></section>
        {#if accepted}<div class="repair-success"><span>Suggested repaired reaction</span><code>{text(accepted.candidate_reaction_smiles)}</code></div>{/if}
        {#if candidates.length > 0}<div class="candidate-list">{#each candidates as candidate, index}<article><header><strong>Candidate {index + 1}</strong><span class="data-badge" data-tone={candidate.accepted === true ? 'success' : 'warning'}>{candidate.accepted === true ? 'Passed checks' : 'Not accepted'}</span></header><code>{text(candidate.candidate_reaction_smiles)}</code><p>{candidate.accepted === true ? 'This candidate passed parsing and the configured continuity threshold.' : text(candidate.rejection_reason, 'This candidate did not meet the strict acceptance checks.')}</p></article>{/each}</div>{/if}
        {#if contract}<div class="contract-summary"><span>{contract.deterministic_only === true ? 'Deterministic repair only' : 'Repair policy not reported'}</span><span>{contract.strict_post_repair_validation === true ? 'Strict validation after repair' : 'Validation policy not reported'}</span><span>{contract.original_preserved === true ? 'Original reaction preserved' : 'Original-preservation status not reported'}</span></div>{/if}
        <button class="button button--secondary" type="button" onclick={() => openDetails('Technical repair response', result)}>Technical details</button>
      </div>
    {:else if workspace === 'quality'}
      {@const total = numberValue(record?.score)}
      {@const components = objectValue(record?.components)}
      {@const weights = objectValue(record?.weights)}
      <div class="result-stack">
        <section class="result-summary"><div><p class="eyebrow">Route quality</p><h3>{total === null ? 'Score unavailable' : `${Math.round(total * 100)} / 100`}</h3><p>This is a weighted summary of the six values you supplied; it is not an experimental success probability.</p></div></section>
        {#if components}
          <div class="quality-breakdown" role="table" aria-label="Route quality breakdown">
            <div class="quality-breakdown__head" role="row"><span>Quality factor</span><span>Input</span><span>Weight</span><span>Contribution</span></div>
            {#each Object.entries(components) as [key, value]}
              {@const component = typeof value === 'number' ? value : null}
              {@const weightValue = objectNumber(weights, key)}
              <div class="quality-breakdown__row" role="row"><strong>{humanize(key)}</strong><span>{component === null ? '—' : percentage(component)}</span><span>{weightValue === null ? '—' : percentage(weightValue)}</span><span>{component === null || weightValue === null ? '—' : `${Math.round(component * weightValue * 100)} pts`}</span></div>
            {/each}
          </div>
        {/if}
        <button class="button button--secondary" type="button" onclick={() => openDetails('Technical route-quality response', result)}>Raw response</button>
      </div>
    {:else if workspace === 'anomaly'}
      {@const anomalyScore = numberValue(record?.anomaly_score)}
      {@const reasons = stringList(record?.reasons)}
      {@const componentScores = objectValue(record?.component_scores)}
      {@const temperatureScore = objectNumber(componentScores, 'temperature_c')}
      {@const timeScore = objectNumber(componentScores, 'time_h')}
      <div class="result-stack">
        <section class="result-summary"><div><p class="eyebrow">Condition comparison</p><h3>{anomalyScore === null ? 'Score unavailable' : anomalyScore < 0.25 ? 'Conditions look typical for the reference data' : anomalyScore < 0.6 ? 'Somewhat unusual conditions' : 'Unusual conditions in the reference data'}</h3><p>{anomalyScore === null ? 'No score was returned.' : `Overall unusualness: ${Math.round(anomalyScore * 100)} / 100.`}</p></div></section>
        <div class="result-overview-grid">
          <div><span>Reaction type</span><strong>{text(record?.reaction_family) ? humanize(text(record?.reaction_family)) : 'Not labeled'}</strong></div>
          <div><span>Reference group</span><strong>{text(record?.reference_family) ? humanize(text(record?.reference_family)) : 'Global reference'}</strong></div>
          {#if temperatureScore !== null}<div><span>Temperature unusualness</span><strong>{percentage(temperatureScore)}</strong></div>{/if}
          {#if timeScore !== null}<div><span>Time unusualness</span><strong>{percentage(timeScore)}</strong></div>{/if}
        </div>
        {#if reasons.length}<div class="interpretation-list"><strong>Why it was flagged</strong><ul>{#each reasons as reason}<li>{plainAnomalyReason(reason)}</li>{/each}</ul></div>{:else}<p class="support-note">Neither supplied condition triggered an outlier warning in the reference statistics.</p>{/if}
        <p class="support-note">This compares your values with the stored dataset. It does not determine chemical feasibility.</p>
        <button class="button button--secondary" type="button" onclick={() => openDetails('Technical condition-comparison response', result)}>Raw response</button>
      </div>
    {:else}
      <div class="result-stack">
        <section class="result-summary"><div><p class="eyebrow">Completed</p><h3>Response received</h3><p>This response does not have a dedicated summary view yet. Open the complete response below.</p></div></section>
        <button class="button button--secondary" type="button" onclick={() => openDetails('Complete response', result)}>Raw response</button>
      </div>
    {/if}
    </div>
  </aside>
</dialog>

<DetailDialog open={detailOpen} title={detailTitle} subtitle={detailSubtitle} payload={detailPayload} onClose={() => (detailOpen = false)} />
