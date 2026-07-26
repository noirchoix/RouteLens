# Product One ML task matrix

| Task | Product One state | Training population | Output |
|---|---|---|---|
| Parse validity | Implemented | All steps | Valid/invalid probability |
| Parse failure taxonomy | Deterministic labels implemented; learned multiclass extension ready | All steps | Failure class |
| Repairability audit | Deterministic only; supervised training disabled until the parse-invalid population contains both accepted-repair and non-repair classes | Parse-invalid steps | Repair-candidate audit, no model artifact |
| Primary solvent | Implemented baseline | Parse-valid solvent-bearing steps | Ranked solvent probabilities |
| Time bucket | Implemented baseline | Parse-valid steps with plausibility-valid time | Ranked time buckets |
| Agent presence | Implemented baseline | Parse-valid steps | Binary probability |
| Temperature bucket | Implemented but experimental | Confidence-qualified repaired temperatures | Ranked temperature buckets |
| Reaction retrieval | Implemented | All parse-valid eligible steps | Weighted reactant/product Tanimoto evidence |
| Route quality | Deterministic decomposable score implemented | All steps/routes | Quality score and issues |
| Condition anomaly | Rule layer implemented | All condition-bearing steps | Quality events and cleaned labels |
| Structural reaction family | Deferred to atom mapping | Mapping-qualified steps | Derived taxonomy |
| Yield prediction | Unsupported | No labels | None |

## Baseline release gate

A trained classifier is marked `validated` only when validation accuracy exceeds the validation majority-class baseline. Promotion to `production` is explicit.

## Feature strategy

The first full-corpus models use stateless character n-gram hashing over reaction SMILES. This avoids vocabulary fitting, supports incremental full-corpus training, and gives a strong deterministic baseline for syntax- and motif-sensitive tasks. Morgan evidence retrieval supplies chemical-neighbour support independently of the classifiers.

The architecture deliberately permits later replacement with reaction transformers, graph neural networks, atom-mapped difference fingerprints, or multitask encoders without changing the inference response.

## Leakage controls

Condition models never consume the legacy three-field working reaction because its middle field contains the labels being predicted. Their feature input is the canonical structure-only representation `reactants>>products`. Parse-validity classification may consume the raw working reaction because syntax and symbolic route tokens are part of that task. Class vocabularies, support thresholds, and class weights are derived from the training partition only.

## Repairability governance

A parse-valid reaction is never labelled as repairable. Alternative canonical representations are normalization evidence, not repairs. The `repairability` task remains queryable for audit compatibility, but training produces a task-audit record and no model artifact unless the task is deliberately redesigned with a valid two-class labelled population.

## v2.0.9 class-support and registry contract

Every estimator is screened before fitting. Binary and multiclass tasks require at least two evaluable classes; each retained multilabel target requires positive and negative support in train, validation, and test. Empty multilabel sets are retained as legitimate negative examples. A task with no valid target produces a `task_audit.json` record and no serialized model.

Product Two models are governed by `data/models/model_registry.json`, mirrored from the SQLite registry. Runtime loading is limited to lifecycle `active` or `candidate` records with `runtime_load_required=true`, an artifact hash match, the current connected-component split hash, and an exact scikit-learn 1.9.0 training/runtime match. Superseded and archived artifacts remain available for provenance but are never deserialized.
