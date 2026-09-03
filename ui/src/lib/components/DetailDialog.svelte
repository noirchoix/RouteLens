<script lang="ts">
  import type { JsonValue } from '$lib/api/types';

  type Props = {
    open: boolean;
    title: string;
    subtitle?: string;
    payload: JsonValue | null;
    onClose: () => void;
  };

  let { open, title, subtitle = '', payload, onClose }: Props = $props();
  let dialog: HTMLDialogElement;

  $effect(() => {
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  });

  function close(): void {
    if (dialog?.open) dialog.close();
    onClose();
  }
</script>

<dialog
  class="detail-dialog"
  bind:this={dialog}
  aria-labelledby="detail-dialog-title"
  onclose={onClose}
  onclick={(event) => {
    if (event.target === dialog) close();
  }}
>
  <div class="detail-dialog__surface">
    <header>
      <div>
        <p class="eyebrow">Technical details</p>
        <h2 id="detail-dialog-title">{title}</h2>
        {#if subtitle}<p>{subtitle}</p>{/if}
      </div>
      <button class="icon-button" type="button" aria-label="Close details" onclick={close}>×</button>
    </header>
    <div class="detail-dialog__body">
      <pre>{JSON.stringify(payload, null, 2)}</pre>
    </div>
  </div>
</dialog>
