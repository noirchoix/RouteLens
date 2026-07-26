from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

from reacts.contracts import ModelStage
from reacts.features.text import reaction_text_vectorizer
from reacts.ml.environment import runtime_environment
from reacts.ml.prefit import validate_classification_support
from reacts.ml.registry import Registry
from reacts.ml.tasks import TASKS, TaskSpec
from reacts.science.calibration import TemperatureCalibrator, calibration_report, fit_temperature_scaler
from reacts.science.hashing import canonical_json_hash, hash_dataset_columns, portable_path, sha256_file
from reacts.science.promotion import decide_promotion
from reacts.storage.tabular import iter_dataset

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    canonical_dir: Path
    model_dir: Path
    dataset_version: str = "uspto_multistep_canonical_v1"
    random_seed: int = 42
    n_features: int = 2**18
    alpha: float = 1e-5
    epochs: int = 2
    max_rows: int | None = None


class Trainer:
    def __init__(self, config: TrainingConfig, registry: Registry):
        self.config = config
        self.registry = registry
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_hash: str | None = None
        self._split_hash: str | None = None

    @staticmethod
    def _labels(df: pd.DataFrame, spec: TaskSpec) -> pd.Series:
        labels = df[spec.target_column]
        if spec.target_column in {"parse_ok", "agent_present", "repairable"}:
            labels = labels.fillna(False).astype(bool).astype(str)
        else:
            labels = labels.astype(str)
        return labels

    def _population_audit(self, spec: TaskSpec) -> dict[str, Any]:
        columns = sorted({spec.target_column, *spec.required_columns, "split"})
        split_counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter(), "test": Counter()}
        eligible_rows: Counter[str] = Counter()
        for chunk in iter_dataset(self.config.canonical_dir, "steps", columns=columns):
            eligible_mask = spec.eligibility(chunk)
            for split in split_counts:
                subset = chunk.loc[(chunk["split"] == split) & eligible_mask]
                labels = self._labels(subset, spec)
                split_counts[split].update(labels.tolist())
                eligible_rows[split] += len(labels)
        return {
            "task": spec.name,
            "training_mode": spec.training_mode,
            "target_column": spec.target_column,
            "input_column": spec.input_column,
            "eligible_rows": dict(eligible_rows),
            "class_counts": {split: dict(counts) for split, counts in split_counts.items()},
            "governance_note": spec.governance_note,
        }

    def _complete_non_model_task(
        self,
        run_id: str,
        spec: TaskSpec,
        config_dict: dict[str, Any],
        *,
        promote_validated: bool,
    ) -> dict[str, Any]:
        audit = self._population_audit(spec)
        train_classes = [label for label, count in audit["class_counts"]["train"].items() if count > 0]
        reason = spec.governance_note or "Task is not configured as a supervised classifier."
        metrics = {
            "status": "skipped",
            "reason_code": "deterministic_audit_not_trainable",
            "reason": reason,
            "model_created": False,
            "promotion_requested": bool(promote_validated),
            "promotion_approved": False,
            "eligible_training_classes": sorted(train_classes),
            "population_audit": audit,
        }
        task_dir = self.config.model_dir / spec.name
        task_dir.mkdir(parents=True, exist_ok=True)
        audit_path = task_dir / f"{run_id}.task_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "task": spec.name,
                    "dataset_version": self.config.dataset_version,
                    "config": config_dict,
                    "metrics": metrics,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        reproducibility = self._reproducibility_hashes(spec)
        audit_record = self.registry.register_task_audit(
            task=spec.name,
            dataset_version=self.config.dataset_version,
            audit_path=audit_path,
            reason_code="deterministic_audit_not_trainable",
            metrics=metrics,
            split_sha256=reproducibility.get("split_sha256"),
        )
        self.registry.finish_run(run_id, metrics, reproducibility=reproducibility)
        return {
            "run_id": run_id,
            "status": "skipped",
            "task": spec.name,
            "model": None,
            "task_audit_path": portable_path(audit_path, self.registry.project_root),
            "registry_audit": audit_record,
            "metrics": metrics,
            "release_decision": {
                "approved": False,
                "stage": "not_applicable",
                "reasons": [reason],
            },
            "reproducibility": reproducibility,
        }

    def _complete_prefit_skip(
        self,
        run_id: str,
        spec: TaskSpec,
        config_dict: dict[str, Any],
        *,
        population_audit: dict[str, Any],
        prefit_support: dict[str, Any],
        promote_validated: bool,
    ) -> dict[str, Any]:
        reason_code = str(prefit_support.get("reason_code") or "insufficient_class_support")
        reasons = list(prefit_support.get("reasons") or ["Pre-fit class support requirements were not met."])
        metrics = {
            "status": "skipped",
            "reason_code": reason_code,
            "reason": " ".join(reasons),
            "model_created": False,
            "promotion_requested": bool(promote_validated),
            "promotion_approved": False,
            "prefit_support": prefit_support,
            "population_audit": population_audit,
        }
        task_dir = self.config.model_dir / spec.name
        task_dir.mkdir(parents=True, exist_ok=True)
        audit_path = task_dir / f"{run_id}.task_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "task": spec.name,
                    "dataset_version": self.config.dataset_version,
                    "config": config_dict,
                    "metrics": metrics,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        reproducibility = self._reproducibility_hashes(spec)
        audit_record = self.registry.register_task_audit(
            task=spec.name,
            dataset_version=self.config.dataset_version,
            audit_path=audit_path,
            reason_code=reason_code,
            metrics=metrics,
            split_sha256=reproducibility.get("split_sha256"),
        )
        self.registry.finish_run(run_id, metrics, reproducibility=reproducibility)
        return {
            "run_id": run_id,
            "status": "skipped",
            "task": spec.name,
            "model": None,
            "task_audit_path": portable_path(audit_path, self.registry.project_root),
            "registry_audit": audit_record,
            "metrics": metrics,
            "release_decision": {
                "approved": False,
                "stage": "not_applicable",
                "reasons": reasons,
            },
            "reproducibility": reproducibility,
        }

    def _require_split_governance(self) -> None:
        if self.config.dataset_version != "uspto_multistep_contextual_v2":
            return
        dataset_manifest = self.config.canonical_dir / "dataset_manifest.json"
        if not dataset_manifest.exists():
            return
        split_manifest = self.config.canonical_dir / "split_manifest.json"
        if not split_manifest.exists():
            raise RuntimeError(
                "Product Two split governance is missing. Run `reacts --project-root . "
                "rebuild-product-two-splits` before retraining."
            )
        payload = json.loads(split_manifest.read_text(encoding="utf-8"))
        if not payload.get("invariants", {}).get("strict_pass"):
            raise RuntimeError("Product Two connected-component split invariants did not pass.")

    def _discover_classes(self, spec: TaskSpec) -> tuple[np.ndarray, dict[str, int]]:
        counts: Counter[str] = Counter()
        observed = 0
        columns = sorted({spec.target_column, *spec.required_columns})
        for chunk in iter_dataset(self.config.canonical_dir, "steps", columns=columns + ["split"]):
            eligible = chunk.loc[(chunk["split"] == "train") & spec.eligibility(chunk)]
            labels = self._labels(eligible, spec)
            counts.update(labels.tolist())
            observed += len(labels)
            if self.config.max_rows is not None and observed >= self.config.max_rows:
                break
        if spec.fixed_classes is not None:
            classes = [label for label in spec.fixed_classes if counts.get(label, 0) >= spec.min_class_count]
        else:
            classes = sorted(
                label
                for label, count in counts.items()
                if count >= spec.min_class_count and label not in {"None", "nan", ""}
            )
        return np.array(classes, dtype=object), dict(counts)

    def _iter_task_split(self, spec: TaskSpec, split: str, allowed_classes: set[str]):
        consumed = 0
        columns = sorted({spec.input_column, "split", spec.target_column, *spec.required_columns})
        for chunk in iter_dataset(self.config.canonical_dir, "steps", columns=columns):
            subset = chunk.loc[(chunk["split"] == split) & spec.eligibility(chunk)].copy()
            if subset.empty:
                continue
            labels = self._labels(subset, spec)
            keep = labels.isin(allowed_classes)
            subset, labels = subset.loc[keep], labels.loc[keep]
            if subset.empty:
                continue
            if self.config.max_rows is not None:
                remaining = self.config.max_rows - consumed
                if remaining <= 0:
                    break
                subset, labels = subset.iloc[:remaining], labels.iloc[:remaining]
            consumed += len(subset)
            yield subset[spec.input_column].fillna("").astype(str).tolist(), labels.to_numpy(dtype=object)

    def _prediction_arrays(
        self,
        model: SGDClassifier,
        vectorizer,
        spec: TaskSpec,
        split: str,
        allowed: set[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        for texts, y in self._iter_task_split(spec, split, allowed):
            X = vectorizer.transform(texts)
            labels.append(y)
            probabilities.append(model.predict_proba(X))
        if not labels:
            return np.array([], dtype=object), np.empty((0, len(model.classes_)), dtype=np.float64)
        return np.concatenate(labels), np.vstack(probabilities)

    @staticmethod
    def _metrics_from_probabilities(
        y_true: np.ndarray,
        probabilities: np.ndarray,
        classes: np.ndarray,
        calibrator: TemperatureCalibrator,
    ) -> dict[str, Any]:
        if len(y_true) == 0:
            return {"rows": 0}
        calibrated = calibrator.transform(probabilities)
        y_pred = classes[calibrated.argmax(axis=1)]
        metrics: dict[str, Any] = {
            "rows": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            "log_loss": float(
                -np.mean(
                    np.log(
                        np.clip(
                            calibrated[
                                np.arange(len(y_true)),
                                np.asarray([{str(label): index for index, label in enumerate(classes)}[str(value)] for value in y_true]),
                            ],
                            1e-15,
                            1.0,
                        )
                    )
                )
            ),
        }
        label_counts = Counter(y_true.tolist())
        majority_label, majority_count = label_counts.most_common(1)[0]
        majority_predictions = np.full(len(y_true), majority_label, dtype=object)
        metrics["majority_accuracy"] = float(majority_count / len(y_true))
        metrics["majority_macro_f1"] = float(
            f1_score(y_true, majority_predictions, average="macro", zero_division=0)
        )
        metrics["beats_majority_macro_f1"] = bool(
            metrics["macro_f1"] > metrics["majority_macro_f1"] + 0.01
        )
        return metrics

    def _reproducibility_hashes(self, spec: TaskSpec) -> dict[str, str | None]:
        if self._dataset_hash is None:
            manifest = self.config.canonical_dir / "dataset_manifest.json"
            self._dataset_hash = sha256_file(manifest) if manifest.exists() else canonical_json_hash(
                {"canonical_dir": portable_path(self.config.canonical_dir, self.registry.project_root)}
            )
        if self._split_hash is None:
            try:
                self._split_hash = hash_dataset_columns(
                    self.config.canonical_dir, "steps",
                    ["step_id", "patent_document_id", "reaction_signature", "split_component_id", "split"]
                )
            except Exception:
                self._split_hash = None
        feature_hash = canonical_json_hash(
            {
                "input_column": spec.input_column,
                "n_features": self.config.n_features,
                "vectorizer": "HashingVectorizer(char_wb,3-5)",
                "task": spec.name,
            }
        )
        return {
            "dataset_sha256": self._dataset_hash,
            "split_sha256": self._split_hash,
            "feature_sha256": feature_hash,
        }

    def train_task(self, task: str, promote_validated: bool = False) -> dict[str, Any]:
        if task not in TASKS:
            raise KeyError(f"Unknown task: {task}")
        spec = TASKS[task]
        config_dict = asdict(self.config) | {"task": task}
        config_dict = {
            key: portable_path(value, self.registry.project_root) if isinstance(value, Path) else value
            for key, value in config_dict.items()
        }
        self._require_split_governance()
        run_id = self.registry.start_run(task, self.config.dataset_version, config_dict)
        try:
            if spec.training_mode != "classifier":
                return self._complete_non_model_task(
                    run_id,
                    spec,
                    config_dict,
                    promote_validated=promote_validated,
                )
            classes, class_counts = self._discover_classes(spec)
            population_audit = self._population_audit(spec)
            task_kind = "binary" if spec.fixed_classes is not None and len(spec.fixed_classes) == 2 else "multiclass"
            prefit = validate_classification_support(
                population_audit["class_counts"],
                retained_classes=classes.tolist(),
                task_kind=task_kind,
            )
            if not prefit.trainable:
                return self._complete_prefit_skip(
                    run_id,
                    spec,
                    config_dict,
                    population_audit=population_audit,
                    prefit_support=prefit.to_dict(),
                    promote_validated=promote_validated,
                )
            allowed = set(classes.tolist())
            vectorizer = reaction_text_vectorizer(self.config.n_features)
            model = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=self.config.alpha,
                l1_ratio=0.05,
                class_weight=None,
                random_state=self.config.random_seed,
                average=True,
            )
            examples_seen = 0
            unique_training_rows = 0
            initialized = False
            eligible_count = sum(class_counts.get(str(label), 0) for label in classes)
            class_weights = {
                str(label): eligible_count / (len(classes) * class_counts.get(str(label), 1))
                for label in classes
            }
            for epoch in range(self.config.epochs):
                epoch_rows = 0
                for texts, labels in self._iter_task_split(spec, "train", allowed):
                    X = vectorizer.transform(texts)
                    sample_weight = np.array([class_weights[str(label)] for label in labels], dtype=np.float64)
                    if not initialized:
                        model.partial_fit(X, labels, classes=classes, sample_weight=sample_weight)
                        initialized = True
                    else:
                        model.partial_fit(X, labels, sample_weight=sample_weight)
                    examples_seen += len(labels)
                    epoch_rows += len(labels)
                if epoch == 0:
                    unique_training_rows = epoch_rows
                LOGGER.info("Task %s epoch %s rows=%s", task, epoch + 1, epoch_rows)
            if not initialized:
                raise ValueError(f"No training rows for task {task}")

            y_val, p_val = self._prediction_arrays(model, vectorizer, spec, "val", allowed)
            calibrator = fit_temperature_scaler(y_val, p_val, classes) if len(y_val) else TemperatureCalibrator()
            validation = self._metrics_from_probabilities(y_val, p_val, classes, calibrator)
            validation["calibration"] = calibration_report(y_val, p_val, classes, calibrator) if len(y_val) else {}
            y_test, p_test = self._prediction_arrays(model, vectorizer, spec, "test", allowed)
            test = self._metrics_from_probabilities(y_test, p_test, classes, calibrator)
            test["calibration"] = calibration_report(y_test, p_test, classes, calibrator) if len(y_test) else {}
            metrics = {
                "unique_training_rows": unique_training_rows,
                "epochs": self.config.epochs,
                "training_examples_seen": examples_seen,
                "classes": classes.tolist(),
                "source_class_counts": class_counts,
                "prefit_support": prefit.to_dict(),
                "validation": validation,
                "test": test,
            }
            calibration_error = validation.get("calibration", {}).get("calibrated_ece")
            decision = decide_promotion(task, metrics, calibration_error=calibration_error)
            reproducibility = self._reproducibility_hashes(spec)
            task_dir = self.config.model_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            artifact = task_dir / f"{run_id}.joblib"
            training_environment = runtime_environment()
            joblib.dump(
                {
                    "task": task,
                    "dataset_version": self.config.dataset_version,
                    "vectorizer": vectorizer,
                    "model": model,
                    "calibrator": calibrator.to_dict(),
                    "classes": classes.tolist(),
                    "input_column": spec.input_column,
                    "metrics": metrics,
                    "config": config_dict,
                    "reproducibility": reproducibility,
                    "training_environment": training_environment,
                },
                artifact,
                compress=3,
            )
            initial_stage = ModelStage.VALIDATED if decision.approved else decision.stage
            if decision.rule.requested_stage == ModelStage.EXPERIMENTAL:
                initial_stage = ModelStage.EXPERIMENTAL
            model_record = self.registry.register_model(
                task=task,
                artifact_path=artifact,
                dataset_version=self.config.dataset_version,
                metrics=metrics,
                config=config_dict,
                stage=initial_stage,
                dataset_sha256=reproducibility["dataset_sha256"],
                feature_sha256=reproducibility["feature_sha256"],
                split_sha256=reproducibility["split_sha256"],
                release_decision=decision.to_dict(),
                training_environment=training_environment,
            )
            if promote_validated and decision.approved:
                self.registry.promote(
                    model_record["model_id"], decision.stage, release_decision=decision.to_dict()
                )
                model_record["stage"] = decision.stage.value
            card = artifact.with_suffix(".model_card.json")
            card.write_text(
                json.dumps(
                    {
                        "model": model_record,
                        "metrics": metrics,
                        "release_decision": decision.to_dict(),
                        "reproducibility": reproducibility,
                        "training_environment": training_environment,
                        "limitations": [
                            "Patent-derived labels are noisy.",
                            "Predictions are applicability-gated and are not experimental validation.",
                            "Task-specific promotion rules prohibit blanket production promotion.",
                        ],
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            self.registry.finish_run(run_id, metrics, reproducibility=reproducibility)
            return {
                "run_id": run_id,
                "model": model_record,
                "metrics": metrics,
                "release_decision": decision.to_dict(),
                "reproducibility": reproducibility,
            }
        except Exception as exc:
            self.registry.finish_run(run_id, {}, error=str(exc))
            raise

    def train_many(self, tasks: list[str], promote_validated: bool = False) -> dict[str, Any]:
        return {
            task: self.train_task(task, promote_validated=promote_validated)
            for task in tasks
        }
