<script lang="ts">
  import TaskPicker from './TaskPicker.svelte';
  import type { ContextualRequest, ModelCapability } from '$lib/api/types';

  type Props = {
    models: ModelCapability[];
    busy: boolean;
    onSubmit: (payload: ContextualRequest) => void;
  };

  let { models, busy, onSubmit }: Props = $props();
  let reaction = $state('CCO>>CC=O');
  let tasks = $state<string[]>([]);
  let evidenceK = $state(5);
  let includeEvidence = $state(true);
  let error = $state('');
  let initialized = $state(false);

  const examples = [
    { label: 'Alcohol oxidation', value: 'CCO>>CC=O' },
    { label: 'Bromoethane substitution', value: 'CCBr>>CCO' },
    { label: 'Acid activation', value: 'CC(=O)O>>CC(=O)Cl' },
    { label: 'Amine unsaturation', value: 'CCN>>CC=N' },
    { label: 'Agent field example', value: 'CCO>O=C=O>CCOC(=O)O' }
  ] as const;

  const needsExperimental = $derived(
    tasks.some((task) => models.find((model) => model.task === task)?.enabled_by_default === false)
  );

  $effect(() => {
    if (!initialized && models.length > 0) {
      tasks = models.filter((model) => model.enabled_by_default !== false).map((model) => model.task);
      initialized = true;
    }
  });

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    error = '';
    if (!reaction.trim()) error = 'Enter a reaction SMILES.';
    else if (tasks.length === 0) error = 'Choose at least one question to answer.';
    if (error) return;
    onSubmit({
      reaction_smiles: reaction.trim(),
      tasks,
      include_evidence: includeEvidence,
      evidence_k: evidenceK,
      allow_experimental: needsExperimental
    });
  }
</script>

<form onsubmit={submit} novalidate>
  <fieldset>
    <legend>Reaction</legend>
    <label for="svelte-contextual-reaction">Reaction SMILES</label>
    <textarea id="svelte-contextual-reaction" bind:value={reaction} rows="5" spellcheck="false" aria-describedby="svelte-contextual-hint"></textarea>
    <p id="svelte-contextual-hint" class="field-hint">Use <code>reactants&gt;agents&gt;products</code> or <code>reactants&gt;&gt;products</code>.</p>
    <div class="example-row" aria-label="Reaction examples">
      <span>Try an example:</span>
      {#each examples as example}
        <button type="button" class="example-chip" onclick={() => (reaction = example.value)}>{example.label}</button>
      {/each}
    </div>
  </fieldset>

  <fieldset>
    <legend>What should RouteLens estimate?</legend>
    <TaskPicker {models} selected={tasks} onChange={(next) => (tasks = next)} />
  </fieldset>

  <div class="form-grid form-grid--two">
    <label class="range-field" for="svelte-contextual-k">
      <span>Similar records to show <output>{evidenceK}</output></span>
      <input id="svelte-contextual-k" type="range" min="0" max="25" bind:value={evidenceK} />
    </label>
    <div class="toggle-stack">
      <label class="toggle"><input type="checkbox" bind:checked={includeEvidence} /><span>Show similar patent records with the result</span></label>
      {#if needsExperimental}<p class="field-hint">Your selection includes a research-only model. RouteLens will request access automatically.</p>{/if}
    </div>
  </div>

  {#if error}<p class="task-reason" role="alert">{error}</p>{/if}
  <div class="form-action">
    <p>If the available evidence is too weak, RouteLens will say so instead of forcing an answer.</p>
    <button class="button button--primary button--run" type="submit" disabled={busy}><span>{busy ? 'Analyzing…' : 'Analyze reaction'}</span><span aria-hidden="true">→</span></button>
  </div>
</form>
