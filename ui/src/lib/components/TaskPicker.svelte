<script lang="ts">
  import { stageLabel, taskDescription, taskLabel } from '$lib/domain/presentation';
  import type { ModelCapability } from '$lib/api/types';

  type Props = {
    models: ModelCapability[];
    selected: string[];
    onChange: (tasks: string[]) => void;
  };

  let { models, selected, onChange }: Props = $props();

  function toggle(task: string, checked: boolean): void {
    const next = checked ? [...new Set([...selected, task])] : selected.filter((value) => value !== task);
    onChange(next);
  }
</script>

<div class="task-grid" aria-live="polite">
  {#if models.length === 0}
    <p class="empty-inline">No analysis models are available.</p>
  {:else}
    {#each models as model}
      <label class="task-option">
        <input
          type="checkbox"
          value={model.task}
          checked={selected.includes(model.task)}
          onchange={(event) => toggle(model.task, event.currentTarget.checked)}
        />
        <span class="task-option__body">
          <span class="task-option__heading">
            <strong>{taskLabel(model.task)}</strong>
            {#if model.stage && model.stage !== 'production'}
              <span class="task-option__status" title={model.warning ?? model.permitted_use ?? ''}>{stageLabel(model.stage)}</span>
            {/if}
          </span>
          <small>{taskDescription(model.task)}</small>
        </span>
      </label>
    {/each}
  {/if}
</div>
