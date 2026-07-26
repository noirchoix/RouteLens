# Scientific governance

## Model stages

- `baseline`: retained Product One reference model.
- `candidate`: completed training but failed or has not passed a release gate.
- `experimental`: research-only model; excluded from default inference and release locks.
- `validated`: passed evaluation but awaits an explicit permitted-use promotion.
- `screening`: high-recall or triage use only.
- `staging`: evidence-backed limited product use.
- `production`: qualified for the task-specific declared use.
- `deprecated` and `rejected`: unavailable for normal inference.

## Promotion contract

Promotion evaluates the held-out test metrics against a task policy. A request for promotion cannot change the policy or stage. The registry requires a serialized release decision for any guarded promotion.

The release decision records:

- metrics and thresholds;
- approval status;
- reasons for rejection;
- requested stage;
- permitted use.

## Calibration

Classification models fit a temperature scaler on validation predictions only. Model cards contain raw and calibrated log loss, expected calibration error, Brier score for binary tasks, and reliability bins. Test data is never used to fit calibration.

## Leakage

The strict report measures overlap across train, validation, and test for:

- patent document IDs;
- canonical reaction signatures;
- product Bemis–Murcko scaffolds.

Patent overlap is a hard failure. Signature and scaffold overlap are reported separately because exact reaction duplication and chemistry-family generalization answer different scientific questions.

## Reproducibility

Dataset manifests, split assignments, feature configurations, model artifacts, registry state, index manifests, and release manifests receive SHA-256 hashes. Paths stored in registries and reports are project-relative.

## Release lock

A locked Product Two release requires:

1. strict acceptance pass;
2. canonical-v2 and index manifests;
3. at least one qualified Product Two model;
4. no candidate or experimental model in the release manifest;
5. immutable release snapshot in the registry.
