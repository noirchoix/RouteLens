<script lang="ts">
  import type { JsonObject } from '$lib/api/types';

  type Props = { busy: boolean; onSubmit: (payload: JsonObject) => void };
  let { busy, onSubmit }: Props = $props();

  type QualityPreset = { label: string; values: [number, number, number, number, number, number] };
  const presets: QualityPreset[] = [
    { label: 'Strong record', values: [1, 0.95, 0.95, 0.9, 0.9, 0.95] },
    { label: 'Mixed evidence', values: [0.9, 0.7, 0.65, 0.6, 0.75, 0.7] },
    { label: 'Weak record', values: [0.6, 0.4, 0.35, 0.4, 0.45, 0.35] }
  ];

  let parse = $state(0.9);
  let resolution = $state(0.8);
  let routeContinuity = $state(0.8);
  let conditionCompleteness = $state(0.7);
  let conditionPlausibility = $state(0.8);
  let mapping = $state(0.75);

  function loadPreset(preset: QualityPreset): void {
    [parse, resolution, routeContinuity, conditionCompleteness, conditionPlausibility, mapping] = preset.values;
  }

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    onSubmit({ parse, resolution, route_continuity: routeContinuity, condition_completeness: conditionCompleteness, condition_plausibility: conditionPlausibility, mapping });
  }
</script>

<form onsubmit={submit} novalidate>
  <fieldset>
    <legend>Quality inputs</legend>
    <p class="field-hint">Each value runs from 0 (poor or missing) to 1 (strong). RouteLens combines them with fixed published weights.</p>
    <div class="example-row"><span>Load a preset:</span>{#each presets as preset}<button type="button" class="example-chip" onclick={() => loadPreset(preset)}>{preset.label}</button>{/each}</div>
    <div class="quality-grid">
      <label class="range-field" for="svelte-quality-parse"><span>Reaction text is valid <output>{parse.toFixed(2)}</output></span><small>Can the reaction be parsed cleanly?</small><input id="svelte-quality-parse" type="range" min="0" max="1" step="0.05" bind:value={parse} /></label>
      <label class="range-field" for="svelte-quality-resolution"><span>Intermediate resolution <output>{resolution.toFixed(2)}</output></span><small>How confidently were symbolic or unresolved intermediates handled?</small><input id="svelte-quality-resolution" type="range" min="0" max="1" step="0.05" bind:value={resolution} /></label>
      <label class="range-field" for="svelte-quality-continuity"><span>Route continuity <output>{routeContinuity.toFixed(2)}</output></span><small>Do adjacent steps connect coherently?</small><input id="svelte-quality-continuity" type="range" min="0" max="1" step="0.05" bind:value={routeContinuity} /></label>
      <label class="range-field" for="svelte-quality-completeness"><span>Conditions are complete <output>{conditionCompleteness.toFixed(2)}</output></span><small>How much of the expected reaction-condition information is present?</small><input id="svelte-quality-completeness" type="range" min="0" max="1" step="0.05" bind:value={conditionCompleteness} /></label>
      <label class="range-field" for="svelte-quality-plausibility"><span>Conditions are typical <output>{conditionPlausibility.toFixed(2)}</output></span><small>Are the recorded conditions plausible relative to the dataset?</small><input id="svelte-quality-plausibility" type="range" min="0" max="1" step="0.05" bind:value={conditionPlausibility} /></label>
      <label class="range-field" for="svelte-quality-mapping"><span>Atom mapping confidence <output>{mapping.toFixed(2)}</output></span><small>How confident is the structural atom correspondence?</small><input id="svelte-quality-mapping" type="range" min="0" max="1" step="0.05" bind:value={mapping} /></label>
    </div>
  </fieldset>
  <div class="form-action"><p>The returned score keeps every component and weight visible.</p><button class="button button--primary button--run" type="submit" disabled={busy}><span>{busy ? 'Calculating…' : 'Calculate quality score'}</span><span aria-hidden="true">→</span></button></div>
</form>
