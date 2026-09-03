<script lang="ts">
  import type { JsonObject } from '$lib/api/types';

  type Props = { busy: boolean; onSubmit: (payload: JsonObject) => void };
  let { busy, onSubmit }: Props = $props();

  type RepairExample = {
    label: string;
    explanation: string;
    reaction: string;
    candidate: string;
    continuity: number;
  };

  const examples: RepairExample[] = [
    {
      label: 'Extra middle field',
      explanation: 'The text has three fields. RouteLens can test whether the middle field should be removed.',
      reaction: 'CCO>O>CC=O',
      candidate: '',
      continuity: 0
    },
    {
      label: 'Duplicate separator',
      explanation: 'The reaction contains more than one reaction separator.',
      reaction: 'CCO>>>>CC=O',
      candidate: '',
      continuity: 0
    },
    {
      label: 'Serialization spaces',
      explanation: 'Whitespace may have been introduced while the reaction was copied or exported.',
      reaction: 'CCO >> CC=O',
      candidate: '',
      continuity: 0
    },
    {
      label: 'Symbolic intermediate + route evidence',
      explanation: 'A symbolic intermediate is present and a neighbouring route step supports a concrete replacement.',
      reaction: 'CCO.<INTERMEDIATE>>>CC=O',
      candidate: 'CCO>>CC=O',
      continuity: 0.85
    },
    {
      label: 'Second contextual example',
      explanation: 'A route-derived candidate is supplied only because the surrounding route supports the replacement.',
      reaction: '<INTERMEDIATE>.CCBr>>CCO',
      candidate: 'CCBr>>CCO',
      continuity: 0.9
    }
  ];

  let reaction = $state(examples[0]?.reaction ?? 'CCO>O>CC=O');
  let candidate = $state(examples[0]?.candidate ?? '');
  let continuity = $state(examples[0]?.continuity ?? 0);
  let error = $state('');

  function loadExample(example: RepairExample): void {
    reaction = example.reaction;
    candidate = example.candidate;
    continuity = example.continuity;
  }

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    error = reaction.trim() ? '' : 'Enter a reaction to check.';
    if (error) return;
    onSubmit({ reaction_smiles: reaction.trim(), contextual_candidate: candidate.trim() || null, route_continuity_score: continuity });
  }
</script>

<form onsubmit={submit} novalidate>
  <fieldset>
    <legend>Reaction to check</legend>
    <label for="svelte-repair-reaction">Malformed or symbolic reaction</label>
    <textarea id="svelte-repair-reaction" bind:value={reaction} rows="5" spellcheck="false"></textarea>

    <label for="svelte-repair-candidate">Evidence-supported replacement <span class="optional">optional</span></label>
    <textarea id="svelte-repair-candidate" bind:value={candidate} rows="3" spellcheck="false" placeholder="Leave blank unless another step in the same route supports a concrete replacement."></textarea>
    <p class="field-hint">This is not a guess field. Leave it blank unless you have route evidence for the replacement.</p>

    <label class="range-field" for="svelte-repair-continuity">
      <span>Route support for that replacement <output>{continuity.toFixed(2)}</output></span>
      <input id="svelte-repair-continuity" type="range" min="0" max="1" step="0.05" bind:value={continuity} disabled={!candidate.trim()} />
    </label>
    <p class="field-hint">0 means no route support; 1 means the surrounding route strongly supports the supplied replacement.</p>
  </fieldset>

  <section class="example-library" aria-labelledby="repair-examples-title">
    <div><p class="eyebrow">Examples</p><h3 id="repair-examples-title">Five complete repair inputs</h3><p>Select one to see how each field is used.</p></div>
    <div class="example-library__grid">
      {#each examples as example, index}
        <button type="button" class="example-card" onclick={() => loadExample(example)}>
          <span>{String(index + 1).padStart(2, '0')}</span><strong>{example.label}</strong><small>{example.explanation}</small>
        </button>
      {/each}
    </div>
  </section>

  <div class="callout callout--info"><strong>What RouteLens will do</strong><p>It will rank conservative text-level repair candidates. It will not silently replace ambiguous chemistry.</p></div>
  {#if error}<p class="task-reason" role="alert">{error}</p>{/if}
  <div class="form-action"><p>If no candidate meets the strict checks, an empty result is a valid outcome.</p><button class="button button--primary button--run" type="submit" disabled={busy}><span>{busy ? 'Checking…' : 'Check reaction'}</span><span aria-hidden="true">→</span></button></div>
</form>
