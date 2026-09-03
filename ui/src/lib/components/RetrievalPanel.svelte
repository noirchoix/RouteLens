<script lang="ts">
  import type { RetrievalMode, RetrievalSubmission } from '$lib/api/types';

  type Props = {
    busy: boolean;
    onSubmit: (submission: RetrievalSubmission) => void;
  };

  let { busy, onSubmit }: Props = $props();
  let mode = $state<RetrievalMode>('reactions');
  let reaction = $state('CCO>>CC=O');
  let routeId = $state('');
  let k = $state(10);
  let minimumQuality = $state(0.35);
  let error = $state('');

  const modes: readonly RetrievalMode[] = ['reactions', 'routes', 'lookup'];
  const examples = [
    { label: 'Alcohol oxidation', value: 'CCO>>CC=O' },
    { label: 'Bromoethane substitution', value: 'CCBr>>CCO' },
    { label: 'Acid activation', value: 'CC(=O)O>>CC(=O)Cl' },
    { label: 'Amine unsaturation', value: 'CCN>>CC=N' },
    { label: 'With an agent field', value: 'CCO>O=C=O>CCOC(=O)O' }
  ] as const;

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    error = '';

    if (mode === 'lookup') {
      if (!routeId.trim()) {
        error = 'Enter a route ID returned by a previous search.';
        return;
      }
      onSubmit({ kind: 'lookup', routeId: routeId.trim() });
      return;
    }

    if (!reaction.trim()) {
      error = 'Enter a reaction to search for.';
      return;
    }

    if (mode === 'routes') {
      onSubmit({ kind: 'routes', request: { reaction_smiles: reaction.trim(), k } });
      return;
    }

    onSubmit({
      kind: 'reactions',
      request: { reaction_smiles: reaction.trim(), k, minimum_quality: minimumQuality }
    });
  }
</script>

<div class="segmented-control" role="tablist" aria-label="Search type">
  {#each modes as option}
    <button type="button" role="tab" aria-selected={mode === option} onclick={() => (mode = option)}>
      {option === 'reactions' ? 'Similar reactions' : option === 'routes' ? 'Similar routes' : 'Open route by ID'}
    </button>
  {/each}
</div>

<form onsubmit={submit} novalidate>
  {#if mode === 'lookup'}
    <fieldset>
      <legend>Route to open</legend>
      <label for="svelte-route-id">Route ID</label>
      <input id="svelte-route-id" type="text" bind:value={routeId} spellcheck="false" placeholder="20140225-US08658646B2-0299" />
      <p class="field-hint">Copy a route ID from a reaction or route search result.</p>
    </fieldset>
  {:else}
    <fieldset>
      <legend>Reaction to compare</legend>
      <label for="svelte-retrieval-reaction">Reaction SMILES</label>
      <textarea id="svelte-retrieval-reaction" bind:value={reaction} rows="5" spellcheck="false"></textarea>
      <div class="example-row"><span>Try an example:</span>{#each examples as example}<button type="button" class="example-chip" onclick={() => (reaction = example.value)}>{example.label}</button>{/each}</div>
    </fieldset>
    <div class="form-grid form-grid--two">
      <label class="range-field" for="svelte-retrieval-k"><span>Results to return <output>{k}</output></span><input id="svelte-retrieval-k" type="range" min="1" max="50" bind:value={k} /></label>
      {#if mode === 'reactions'}
        <label class="range-field" for="svelte-retrieval-quality"><span>Minimum record quality <output>{minimumQuality.toFixed(2)}</output></span><input id="svelte-retrieval-quality" type="range" min="0" max="1" step="0.05" bind:value={minimumQuality} /></label>
      {/if}
    </div>
  {/if}

  {#if error}<p class="task-reason" role="alert">{error}</p>{/if}

  <div class="form-action">
    <p>Similarity means “looks related in this patent corpus.” It is context for review, not proof that a reaction will work.</p>
    <button class="button button--primary button--run" type="submit" disabled={busy}><span>{busy ? 'Searching…' : mode === 'lookup' ? 'Open route' : 'Find similar records'}</span><span aria-hidden="true">→</span></button>
  </div>
</form>
