import type {
  BatchResultView,
  EvidenceView,
  JsonObject,
  JsonValue,
  ModelCapability,
  ModelsResponse,
  PredictionView,
  ReadyResponse,
  TaskResultView,
  CapabilitiesResponse,
  CapabilityItem,
  CapabilityState,
  RetrievalResultView,
  DistributionView
} from './types';

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

export function isJsonObject(value: JsonValue | undefined): value is JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function jsonObjectOrNull(value: JsonValue | null): JsonObject | null {
  return value !== null && isJsonObject(value) ? value : null;
}

export function asJsonValue(value: unknown, label = 'response'): JsonValue {
  if (value === null) return null;
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.map((item, index) => asJsonValue(item, `${label}[${index}]`));
  if (isRecord(value)) {
    const result: JsonObject = {};
    for (const [key, item] of Object.entries(value)) {
      result[key] = asJsonValue(item, `${label}.${key}`);
    }
    return result;
  }
  throw new Error(`${label} contains a non-serializable value.`);
}

export function asJsonObject(value: unknown, label = 'response'): JsonObject {
  const parsed = asJsonValue(value, label);
  if (!isJsonObject(parsed)) throw new Error(`${label} must be a JSON object.`);
  return parsed;
}

function optionalString(value: JsonValue | undefined): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function optionalBoolean(value: JsonValue | undefined): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

function optionalFiniteNumber(value: JsonValue | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function jsonObjects(value: JsonValue | undefined): JsonObject[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is JsonObject => isJsonObject(item));
}

function stringArray(value: JsonValue | undefined): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function interval(value: JsonValue | undefined): readonly [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  const lower = value[0];
  const upper = value[1];
  if (typeof lower !== 'number' || !Number.isFinite(lower)) return null;
  if (typeof upper !== 'number' || !Number.isFinite(upper)) return null;
  return [lower, upper] as const;
}

export function taskResultViews(value: JsonValue | undefined): TaskResultView[] {
  return jsonObjects(value).flatMap((task) => {
    const taskName = optionalString(task.task);
    if (!taskName) return [];

    const predictions: PredictionView[] = jsonObjects(task.predictions).flatMap((prediction) => {
      const label = optionalString(prediction.label);
      const probability = optionalFiniteNumber(prediction.probability);
      return label && probability !== undefined ? [{ label, probability }] : [];
    });

    return [
      {
        task: taskName,
        abstained: optionalBoolean(task.abstained) ?? false,
        applicability: optionalString(task.applicability) ?? null,
        reactionFamilyAgreement: optionalFiniteNumber(task.reaction_family_agreement) ?? null,
        modelStage: optionalString(task.model_stage) ?? null,
        lifecycleState: optionalString(task.lifecycle_state) ?? null,
        reason: optionalString(task.reason) ?? null,
        pointEstimate: optionalFiniteNumber(task.point_estimate) ?? null,
        units: optionalString(task.units) ?? null,
        interval: interval(task.interval),
        predictions,
        modelId: optionalString(task.model_id) ?? null,
        neighbourSupport: optionalFiniteNumber(task.neighbour_support) ?? 0,
        permittedUse: optionalString(task.permitted_use) ?? null,
        warnings: stringArray(task.warnings),
        raw: task
      }
    ];
  });
}

export function evidenceViews(value: JsonValue | undefined): EvidenceView[] {
  return jsonObjects(value).map((item) => ({
    reactionSmiles: optionalString(item.reaction_smiles) ?? '',
    score: optionalFiniteNumber(item.score) ?? 0,
    qualityScore: optionalFiniteNumber(item.quality_score) ?? null,
    routeId: optionalString(item.route_id) ?? '—',
    patentDocumentId: optionalString(item.patent_document_id) ?? null,
    timeBucket: optionalString(item.time_bucket) ?? null,
    temperatureBucket: optionalString(item.temperature_bucket) ?? null,
    solventPrimary: optionalString(item.solvent_primary) ?? null,
    solvents: stringArray(item.solvents),
    agents: stringArray(item.agents),
    reactionFamily: optionalString(item.reaction_family) ?? null,
    resolutionStatus: optionalString(item.resolution_status) ?? null,
    raw: item
  }));
}

export function batchResultViews(value: JsonValue | undefined): BatchResultView[] {
  return jsonObjects(value).map((item) => {
    const evidence = evidenceViews(item.evidence);
    return {
      inputReaction: optionalString(item.input_reaction) ?? '',
      parseOk: optionalBoolean(item.parse_ok) ?? null,
      applicability: optionalString(item.applicability) ?? null,
      reactionFamily: optionalString(item.reaction_family) ?? null,
      tasks: taskResultViews(item.tasks),
      evidence,
      evidenceCount: evidence.length,
      raw: item
    };
  });
}

export function distributionViews(value: JsonValue | undefined): DistributionView[] {
  if (!isJsonObject(value)) return [];
  return Object.entries(value).flatMap(([field, distribution]) => {
    if (!isJsonObject(distribution)) return [];
    const items = Object.entries(distribution)
      .flatMap(([label, probability]) =>
        typeof probability === 'number' && Number.isFinite(probability) ? [{ label, probability }] : []
      )
      .sort((left, right) => right.probability - left.probability);
    return items.length ? [{ field, items }] : [];
  });
}

export function retrievalResultViews(value: JsonValue | undefined): RetrievalResultView[] {
  return jsonObjects(value).map((item) => ({
    routeId: optionalString(item.route_id) ?? optionalString(item.route_instance_id) ?? '—',
    patentDocumentId: optionalString(item.patent_document_id) ?? null,
    reactionSmiles: optionalString(item.reaction_smiles) ?? null,
    score: optionalFiniteNumber(item.score) ?? 0,
    qualityScore: optionalFiniteNumber(item.quality_score) ?? null,
    stepCount: optionalFiniteNumber(item.step_count) ?? null,
    split: optionalString(item.split) ?? null,
    reactionFamilies: stringArray(item.reaction_families),
    raw: item
  }));
}

export function asReadyResponse(value: unknown): ReadyResponse {
  const record = asJsonObject(value, 'readiness response');
  const response: ReadyResponse = {};

  const version = optionalString(record.version);
  const ready = optionalBoolean(record.ready);
  const warmedUp = optionalBoolean(record.warmed_up);
  const artifactRelease = record.artifact_release === null ? null : optionalString(record.artifact_release);
  const runtimeModelCount = record.runtime_model_count === null ? null : optionalFiniteNumber(record.runtime_model_count);
  const validationPass = record.validation_pass === null ? null : optionalBoolean(record.validation_pass);
  const cacheHit = optionalBoolean(record.cache_hit);

  if (version !== undefined) response.version = version;
  if (ready !== undefined) response.ready = ready;
  if (warmedUp !== undefined) response.warmed_up = warmedUp;
  if (typeof record.reason_code === 'string' || record.reason_code === null) response.reason_code = record.reason_code;
  if (artifactRelease !== undefined || record.artifact_release === null) response.artifact_release = artifactRelease ?? null;
  if (runtimeModelCount !== undefined || record.runtime_model_count === null) response.runtime_model_count = runtimeModelCount ?? null;
  if (validationPass !== undefined || record.validation_pass === null) response.validation_pass = validationPass ?? null;
  if (cacheHit !== undefined) response.cache_hit = cacheHit;

  if (isJsonObject(record.warmup)) {
    const warmup: NonNullable<ReadyResponse['warmup']> = {};
    const modelsLoaded = optionalFiniteNumber(record.warmup.models_loaded);
    if (modelsLoaded !== undefined) warmup.models_loaded = modelsLoaded;

    if (isJsonObject(record.warmup.route_index_storage)) {
      const source = record.warmup.route_index_storage;
      const storage: NonNullable<NonNullable<ReadyResponse['warmup']>['route_index_storage']> = {};
      const vectors = optionalString(source.vectors);
      const vectorsFormat = optionalString(source.vectors_format);
      const memoryMapped = optionalBoolean(source.memory_mapped);
      const rows = optionalFiniteNumber(source.rows);
      const dimensions = optionalFiniteNumber(source.dimensions);
      const dtype = optionalString(source.dtype);
      const searchChunkRows = optionalFiniteNumber(source.search_chunk_rows);
      if (vectors !== undefined) storage.vectors = vectors;
      if (vectorsFormat !== undefined) storage.vectors_format = vectorsFormat;
      if (memoryMapped !== undefined) storage.memory_mapped = memoryMapped;
      if (rows !== undefined) storage.rows = rows;
      if (dimensions !== undefined) storage.dimensions = dimensions;
      if (dtype !== undefined) storage.dtype = dtype;
      if (searchChunkRows !== undefined) storage.search_chunk_rows = searchChunkRows;
      warmup.route_index_storage = storage;
    }

    response.warmup = warmup;
  }

  return response;
}

export function asModelsResponse(value: unknown): ModelsResponse {
  const record = asJsonObject(value, 'models response');
  if (!Array.isArray(record.models)) throw new Error('models response must contain a models array.');
  const models = record.models.map((item, index) => {
    if (!isJsonObject(item) || typeof item.task !== 'string') {
      throw new Error(`models[${index}] must contain a task string.`);
    }
    const model: ModelCapability = { task: item.task };
    if (typeof item.model_id === 'string') model.model_id = item.model_id;
    if (typeof item.stage === 'string') model.stage = item.stage;
    if (typeof item.lifecycle_state === 'string') model.lifecycle_state = item.lifecycle_state;
    if (typeof item.permitted_use === 'string') model.permitted_use = item.permitted_use;
    if (typeof item.warning === 'string') model.warning = item.warning;
    if (typeof item.enabled_by_default === 'boolean') model.enabled_by_default = item.enabled_by_default;
    if (typeof item.release_approved === 'boolean') model.release_approved = item.release_approved;
    return model;
  });
  return {
    artifact_release: typeof record.artifact_release === 'string' ? record.artifact_release : undefined,
    models
  };
}

function capabilityState(value: JsonValue | undefined): CapabilityState {
  return value === 'available' || value === 'setup_required' || value === 'cli_only' || value === 'unavailable'
    ? value
    : 'unavailable';
}

function capabilityItems(value: JsonValue | undefined): CapabilityItem[] {
  return jsonObjects(value).flatMap((item) => {
    const id = optionalString(item.id);
    const label = optionalString(item.label);
    if (!id || !label) return [];
    const capability: CapabilityItem = {
      id,
      label,
      state: capabilityState(item.state),
      available: optionalBoolean(item.available) ?? false
    };
    const reason = optionalString(item.reason);
    const setupCommand = optionalString(item.setup_command);
    if (reason) capability.reason = reason;
    if (setupCommand) capability.setup_command = setupCommand;
    return [capability];
  });
}

export function asCapabilitiesResponse(value: unknown): CapabilitiesResponse {
  const record = asJsonObject(value, 'capabilities response');
  return {
    mode: optionalString(record.mode) ?? 'unknown',
    read_only: optionalBoolean(record.read_only) ?? false,
    workflows: capabilityItems(record.workflows),
    cli_only: capabilityItems(record.cli_only)
  };
}
