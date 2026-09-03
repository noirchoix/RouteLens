export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

export type Workspace =
  | 'contextual'
  | 'batch'
  | 'retrieval'
  | 'repair'
  | 'anomaly'
  | 'quality'
  | 'system';

export type RetrievalMode = 'reactions' | 'routes' | 'lookup';

export type ModelCapability = {
  task: string;
  model_id?: string;
  stage?: string;
  lifecycle_state?: string;
  permitted_use?: string;
  warning?: string;
  enabled_by_default?: boolean;
  release_approved?: boolean;
};

export type ModelsResponse = {
  artifact_release?: string;
  models: ModelCapability[];
};

export type RuntimeStorage = {
  vectors?: string;
  vectors_format?: string;
  memory_mapped?: boolean;
  rows?: number;
  dimensions?: number;
  dtype?: string;
  search_chunk_rows?: number;
};

export type ReadyResponse = {
  version?: string;
  ready?: boolean;
  warmed_up?: boolean;
  reason_code?: string | null;
  artifact_release?: string | null;
  runtime_model_count?: number | null;
  validation_pass?: boolean | null;
  cache_hit?: boolean;
  warmup?: {
    models_loaded?: number;
    route_index_storage?: RuntimeStorage;
  };
};

export type CapabilityState = 'available' | 'setup_required' | 'cli_only' | 'unavailable';

export type CapabilityItem = {
  id: string;
  label: string;
  state: CapabilityState;
  available: boolean;
  reason?: string;
  setup_command?: string;
};

export type CapabilitiesResponse = {
  mode: string;
  read_only: boolean;
  workflows: CapabilityItem[];
  cli_only: CapabilityItem[];
};

export type BootstrapState = {
  health: JsonObject | null;
  ready: ReadyResponse | null;
  artifacts: JsonObject | null;
  models: ModelsResponse | null;
  datasets: JsonObject | null;
  capabilities: CapabilitiesResponse | null;
  errors: string[];
};

export type ContextualRequest = {
  reaction_smiles: string;
  tasks: string[];
  include_evidence: boolean;
  evidence_k: number;
  allow_experimental: boolean;
};

export type BatchRequest = {
  reactions: string[];
  tasks: string[];
  include_evidence: boolean;
  evidence_k: number;
  allow_experimental: boolean;
};

export type ReactionRetrievalRequest = {
  reaction_smiles: string;
  k: number;
  minimum_quality: number;
};

export type RouteRetrievalRequest = {
  reaction_smiles: string;
  k: number;
};

export type RetrievalSubmission =
  | { kind: 'reactions'; request: ReactionRetrievalRequest }
  | { kind: 'routes'; request: RouteRetrievalRequest }
  | { kind: 'lookup'; routeId: string };

export type PredictionView = Readonly<{
  label: string;
  probability: number;
}>;

export type TaskResultView = Readonly<{
  task: string;
  abstained: boolean;
  applicability: string | null;
  reactionFamilyAgreement: number | null;
  modelStage: string | null;
  lifecycleState: string | null;
  reason: string | null;
  pointEstimate: number | null;
  units: string | null;
  interval: readonly [number, number] | null;
  predictions: readonly PredictionView[];
  modelId: string | null;
  neighbourSupport: number;
  permittedUse: string | null;
  warnings: readonly string[];
  raw: JsonObject;
}>;

export type EvidenceView = Readonly<{
  reactionSmiles: string;
  score: number;
  qualityScore: number | null;
  routeId: string;
  patentDocumentId: string | null;
  timeBucket: string | null;
  temperatureBucket: string | null;
  solventPrimary: string | null;
  solvents: readonly string[];
  agents: readonly string[];
  reactionFamily: string | null;
  resolutionStatus: string | null;
  raw: JsonObject;
}>;

export type DistributionItemView = Readonly<{
  label: string;
  probability: number;
}>;

export type DistributionView = Readonly<{
  field: string;
  items: readonly DistributionItemView[];
}>;

export type BatchResultView = Readonly<{
  inputReaction: string;
  parseOk: boolean | null;
  applicability: string | null;
  reactionFamily: string | null;
  tasks: readonly TaskResultView[];
  evidence: readonly EvidenceView[];
  evidenceCount: number;
  raw: JsonObject;
}>;

export type RetrievalResultView = Readonly<{
  routeId: string;
  patentDocumentId: string | null;
  reactionSmiles: string | null;
  score: number;
  qualityScore: number | null;
  stepCount: number | null;
  split: string | null;
  reactionFamilies: readonly string[];
  raw: JsonObject;
}>;

export type HistoryItem = {
  id: string;
  workspace: Workspace;
  workflow: string;
  summary: string;
  payload: JsonValue;
  createdAt: Date;
};
