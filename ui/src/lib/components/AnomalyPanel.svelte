<script lang="ts">
  import type { CapabilityItem, JsonObject } from '$lib/api/types';

  type Props = {
    busy: boolean;
    capability: CapabilityItem | null;
    onSubmit: (payload: JsonObject) => void;
  };

  let { busy, capability, onSubmit }: Props = $props();
  let reaction = $state('CCO>>CC=O');
  let temperature = $state<number | undefined>(25);
  let time = $state<number | undefined>(2);
  let error = $state('');

  const checking = $derived(capability === null);
  const unavailable = $derived(capability !== null && !capability.available);

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    if (checking || unavailable) return;
    error = !reaction.trim()
      ? 'Enter a reaction SMILES.'
      : temperature === undefined && time === undefined
        ? 'Provide temperature, time, or both.'
        : '';
    if (error) return;
    onSubmit({
      reaction_smiles: reaction.trim(),
      temperature_c: temperature ?? null,
      time_h: time ?? null
    });
  }
</script>

{#if checking}
  <section class="capability-blocked capability-blocked--checking" role="status" aria-live="polite">
    <p class="eyebrow">Checking availability</p>
    <h3>Checking whether condition comparison is included in this release.</h3>
    <p>The controls will appear automatically when the backend confirms the required reference statistics are available.</p>
  </section>
{:else if unavailable}
  <section class="capability-blocked" role="status">
    <p class="eyebrow">Setup required</p>
    <h3>Condition comparison is not included in this data release.</h3>
    <p>Nothing is wrong with your reaction input. This release was packaged without the reference statistics used to compare temperature and reaction time.</p>
    {#if capability?.setup_command}
      <details class="technical-disclosure">
        <summary>Maintainer setup</summary>
        <p>Build the condition statistics, publish a new artifact release, then restart the read-only service with that release.</p>
        <code>{capability.setup_command}</code>
      </details>
    {/if}
  </section>
{:else}
  <form onsubmit={submit} novalidate>
    <fieldset>
      <legend>Observed conditions</legend>
      <label for="svelte-anomaly-reaction">Reaction SMILES</label>
      <textarea id="svelte-anomaly-reaction" bind:value={reaction} rows="5" spellcheck="false"></textarea>
      <div class="form-grid form-grid--two">
        <label for="svelte-anomaly-temperature">Temperature °C <span class="optional">optional</span><input id="svelte-anomaly-temperature" type="number" step="0.1" bind:value={temperature} /></label>
        <label for="svelte-anomaly-time">Time in hours <span class="optional">optional</span><input id="svelte-anomaly-time" type="number" min="0" step="0.1" bind:value={time} /></label>
      </div>
    </fieldset>
    <div class="callout callout--warning"><strong>What the score means</strong><p>It tells you whether the entered temperature or time is unusual compared with similar records in this dataset. It does not say the chemistry is impossible.</p></div>
    {#if error}<p class="task-reason" role="alert">{error}</p>{/if}
    <div class="form-action"><p>Enter either temperature, time, or both.</p><button class="button button--primary button--run" type="submit" disabled={busy}><span>{busy ? 'Comparing…' : 'Compare conditions'}</span><span aria-hidden="true">→</span></button></div>
  </form>
{/if}
