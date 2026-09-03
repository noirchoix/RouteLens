<script lang="ts">
  import TaskPicker from './TaskPicker.svelte';
  import type { BatchRequest, ModelCapability } from '$lib/api/types';

  type Props = { models: ModelCapability[]; busy: boolean; onSubmit: (payload: BatchRequest) => void };
  let { models, busy, onSubmit }: Props = $props();
  let source = $state('CCO>>CC=O\nCCBr>>CCO');
  let tasks = $state<string[]>([]);
  let evidenceK = $state(2);
  let includeEvidence = $state(true);
  let error = $state('');
  let initialized = $state(false);
  const reactions = $derived(source.split(/\r?\n/).map((value) => value.trim()).filter(Boolean));
  const needsExperimental = $derived(
    tasks.some((task) => models.find((model) => model.task === task)?.enabled_by_default === false)
  );

  const examples = [
    'CCO>>CC=O\nCCBr>>CCO',
    'CC(=O)O>>CC(=O)Cl\nCCN>>CC=N\nCCO>>CC=O',
    'CCBr>>CCO\nCCCBr>>CCCO\nCCCCBr>>CCCCO'
  ] as const;

  $effect(() => {
    if (!initialized && models.length > 0) {
      tasks = models.slice(0, 3).map((model) => model.task);
      initialized = true;
    }
  });

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    error = reactions.length ? (tasks.length ? '' : 'Choose at least one question to answer.') : 'Enter at least one reaction.';
    if (error) return;
    onSubmit({ reactions, tasks, include_evidence: includeEvidence, evidence_k: evidenceK, allow_experimental: needsExperimental });
  }
</script>

<form onsubmit={submit} novalidate>
  <fieldset>
    <legend>Reactions</legend>
    <label for="svelte-batch-reactions">One reaction SMILES per line</label>
    <textarea id="svelte-batch-reactions" bind:value={source} rows="9" spellcheck="false"></textarea>
    <p class="field-hint">{reactions.length} reaction{reactions.length === 1 ? '' : 's'} ready to analyze.</p>
    <div class="example-row"><span>Load a sample batch:</span>{#each examples as example, index}<button type="button" class="example-chip" onclick={() => (source = example)}>Batch {index + 1}</button>{/each}</div>
  </fieldset>
  <fieldset><legend>Questions to answer for every reaction</legend><TaskPicker {models} selected={tasks} onChange={(next) => (tasks = next)} /></fieldset>
  <div class="form-grid form-grid--two">
    <label class="range-field" for="svelte-batch-k"><span>Similar records per reaction <output>{evidenceK}</output></span><input id="svelte-batch-k" type="range" min="0" max="25" bind:value={evidenceK} /></label>
    <div class="toggle-stack"><label class="toggle"><input type="checkbox" bind:checked={includeEvidence} /><span>Show similar patent records</span></label>{#if needsExperimental}<p class="field-hint">This batch includes a research-only model.</p>{/if}</div>
  </div>
  {#if error}<p class="task-reason" role="alert">{error}</p>{/if}
  <div class="form-action"><p>Results are summarized on the page. Open any reaction for the complete technical response.</p><button class="button button--primary button--run" type="submit" disabled={busy}><span>{busy ? 'Analyzing…' : 'Analyze batch'}</span><span aria-hidden="true">→</span></button></div>
</form>
