import type { TaskResultView } from '$lib/api/types';

const TASK_COPY: Record<string, { label: string; description: string }> = {
  catalyst_family_multilabel: {
    label: 'Catalyst type',
    description: 'Which broad catalyst family is most consistent with this reaction.'
  },
  agent_family_multilabel: {
    label: 'Reagent / agent type',
    description: 'Which broad reagent families are most consistent with this reaction.'
  },
  temperature_regression: {
    label: 'Typical temperature',
    description: 'An estimated reaction temperature with an uncertainty range.'
  },
  time_regression: {
    label: 'Typical duration',
    description: 'An estimated reaction time with an uncertainty range.'
  },
  solvent_family_multilabel: {
    label: 'Solvent family',
    description: 'Which broad solvent families are most consistent with this reaction.'
  },
  solvent_multilabel: {
    label: 'Likely solvent',
    description: 'Specific solvent suggestions supported by the trained corpus.'
  },
  reaction_family: {
    label: 'Reaction type',
    description: 'The broad transformation family that best matches the reaction.'
  },
  parse_failure_class: {
    label: 'Input check',
    description: 'Whether the reaction text can be parsed and, if not, what failed.'
  }
};

export function taskLabel(task: string): string {
  return TASK_COPY[task]?.label ?? humanize(task);
}

export function taskDescription(task: string): string {
  return TASK_COPY[task]?.description ?? 'Model output for this reaction.';
}

export function humanize(value: unknown): string {
  return String(value ?? '').replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase());
}

export function supportLabel(value: unknown): string {
  if (value === 'in_domain') return 'Strong reference support';
  if (value === 'weakly_supported') return 'Limited reference support';
  if (value === 'out_of_domain') return 'Outside the supported range';
  if (value === 'invalid') return 'Input could not be evaluated';
  return value ? humanize(value) : 'Support not reported';
}

export function stageLabel(stage: string | null): string {
  if (stage === 'production') return 'Production';
  if (stage === 'staging') return 'Preview';
  if (stage === 'candidate') return 'Research preview';
  if (stage === 'screening') return 'Screening';
  if (stage === 'baseline') return 'Baseline';
  if (stage === 'experimental' || stage === 'validated') return 'Experimental';
  return stage ? humanize(stage) : 'Model details';
}

export function stageNote(stage: string | null): string {
  if (stage === 'candidate') return 'Research-preview result. Treat it as guidance rather than a production decision.';
  if (stage === 'staging') return 'Preview result. It is available for evaluation but is not an unrestricted production model.';
  if (stage === 'experimental') return 'Experimental result. Use it for exploration only.';
  return '';
}

export function plainReason(reason: string | null): string {
  if (!reason) return '';
  if (reason.toLowerCase().includes('insufficient calibrated/evidence support')) {
    return 'The service found too little reliable support to present this as an answer.';
  }
  return reason;
}

export function plainAnomalyReason(reason: string): string {
  return reason
    .replace(/^temperature_c\s+/i, 'Temperature ')
    .replace(/^time_h\s+/i, 'Reaction time ')
    .replace(/robust z-score/gi, 'distance from the typical range')
    .replace(/family 1st-99th percentile range/gi, 'usual range for the reference group');
}

export function taskSummary(task: TaskResultView): string {
  const label = taskLabel(task.task);
  if (task.abstained) return `${label}: no reliable answer`;
  if (task.pointEstimate !== null) {
    const units = task.units === 'degC' ? '°C' : task.units === 'h' ? 'h' : task.units ?? '';
    const interval = task.interval
      ? `; likely range ${task.interval[0].toFixed(1)}–${task.interval[1].toFixed(1)}${units ? ` ${units}` : ''}`
      : '';
    return `${label}: ${task.pointEstimate.toFixed(1)}${units ? ` ${units}` : ''}${interval}`;
  }
  if (task.predictions.length) {
    const predictions = task.predictions
      .slice(0, 3)
      .map((prediction) => `${humanize(prediction.label)} ${Math.round(prediction.probability * 100)}%`)
      .join(', ');
    return `${label}: ${predictions}`;
  }
  return `${label}: completed`;
}
