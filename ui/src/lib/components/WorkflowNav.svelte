<script lang="ts">
  import type { Workspace } from '$lib/api/types';

  type Props = {
    active: Workspace;
    open: boolean;
    onSelect: (workspace: Workspace) => void;
    onClose: () => void;
  };

  let { active, open, onSelect, onClose }: Props = $props();
  let dialog: HTMLDialogElement;

  const items: Array<{ id: Workspace; title: string; description: string }> = [
    { id: 'contextual', title: 'Analyze one reaction', description: 'Predictions with supporting patent records' },
    { id: 'batch', title: 'Analyze several reactions', description: 'Run the same checks across a list' },
    { id: 'retrieval', title: 'Find similar records', description: 'Related reactions and routes' },
    { id: 'repair', title: 'Check a broken reaction', description: 'Conservative repair candidates' },
    { id: 'anomaly', title: 'Check conditions', description: 'Compare temperature and reaction time' },
    { id: 'quality', title: 'Score route quality', description: 'Six-factor route-quality score' },
    { id: 'system', title: 'System details', description: 'Availability, models and current data release' }
  ];

  $effect(() => {
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  });

  function close(): void {
    if (dialog?.open) dialog.close();
  }

  function select(workspace: Workspace): void {
    onSelect(workspace);
    close();
  }
</script>

<dialog
  class="workflow-drawer"
  bind:this={dialog}
  aria-labelledby="workflow-drawer-title"
  onclose={onClose}
  onclick={(event) => {
    if (event.target === dialog) close();
  }}
>
  <div class="workflow-drawer__surface">
    <header class="workflow-drawer__header">
      <div>
        <p class="eyebrow">Reaction tools</p>
        <h2 id="workflow-drawer-title">Choose a tool</h2>
        <p>Switch tools without reducing the reaction workspace.</p>
      </div>
      <button class="icon-button" type="button" aria-label="Close reaction tools" onclick={close}>×</button>
    </header>

    <nav class="workflow-nav" aria-label="Analysis workflows">
      {#each items as item, index}
        <button
          type="button"
          class:is-active={active === item.id}
          class="workflow-nav__item"
          aria-current={active === item.id ? 'page' : undefined}
          aria-label={`${item.title}. ${item.description}`}
          title={item.description}
          onclick={() => select(item.id)}
        >
          <span class="workflow-nav__index">{String(index + 1).padStart(2, '0')}</span>
          <strong>{item.title}</strong>
        </button>
      {/each}
    </nav>

    <footer class="workflow-rail__footer">
      <span class="rail-signal" aria-hidden="true"></span>
      <p><strong>Analysis workspace</strong><br />Data building, model training and release management stay in the CLI.</p>
    </footer>
  </div>
</dialog>
