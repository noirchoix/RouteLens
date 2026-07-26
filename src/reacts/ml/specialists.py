from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    f1_score,
    label_ranking_average_precision_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from reacts.contracts import ModelStage
from reacts.data.parsing import parse_list
from reacts.features.text import reaction_text_vectorizer
from reacts.ml.environment import runtime_environment
from reacts.ml.prefit import validate_multilabel_support
from reacts.ml.registry import Registry
from reacts.science.calibration import TemperatureCalibrator, calibration_report, fit_temperature_scaler
from reacts.science.hashing import canonical_json_hash, hash_dataset_columns, portable_path, sha256_file
from reacts.storage.tabular import iter_dataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiLabelTask:
    name: str
    label_column: str
    input_column: str = "canonical_resolved_reaction_smiles"
    eligibility_column: str = "eligible_contextual_models"
    minimum_label_count: int = 200
    maximum_labels: int = 64
    top_k: int = 5


@dataclass(frozen=True)
class RegressionTask:
    name: str
    target_column: str
    units: str
    transform: str = "identity"
    input_column: str = "canonical_resolved_reaction_smiles"
    eligibility_column: str = "eligible_contextual_models"
    family_column: str = "reaction_family"


MULTILABEL_TASKS = {
    "solvent_multilabel": MultiLabelTask("solvent_multilabel", "solvents", minimum_label_count=100, maximum_labels=48, top_k=5),
    "solvent_family_multilabel": MultiLabelTask("solvent_family_multilabel", "solvent_families", minimum_label_count=50, maximum_labels=24, top_k=3),
    "agent_multilabel": MultiLabelTask("agent_multilabel", "agents", minimum_label_count=200, maximum_labels=64, top_k=5),
    "agent_family_multilabel": MultiLabelTask("agent_family_multilabel", "agent_families", minimum_label_count=50, maximum_labels=24, top_k=3),
    "catalyst_family_multilabel": MultiLabelTask("catalyst_family_multilabel", "catalyst_families", minimum_label_count=20, maximum_labels=16, top_k=3),
}

REGRESSION_TASKS = {
    "time_regression": RegressionTask("time_regression", "time_h", "h", transform="log1p"),
    "temperature_regression": RegressionTask("temperature_regression", "temperature_c", "degC", transform="identity"),
}


class IncrementalMultiLabelModel:
    def __init__(self, labels: list[str], n_features: int, random_seed: int, alpha: float):
        self.labels = labels
        self.vectorizer = reaction_text_vectorizer(n_features)
        self.models = {
            label: SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=alpha,
                l1_ratio=0.05,
                random_state=random_seed,
                average=True,
            )
            for label in labels
        }
        self.initialized = {label: False for label in labels}

    def partial_fit(self, texts: list[str], label_sets: list[set[str]]) -> None:
        X = self.vectorizer.transform(texts)
        for label, model in self.models.items():
            y = np.array([int(label in values) for values in label_sets], dtype=np.int8)
            positives = max(int(y.sum()), 1)
            negatives = max(len(y) - positives, 1)
            weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))
            if not self.initialized[label]:
                model.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=weights)
                self.initialized[label] = True
            else:
                model.partial_fit(X, y, sample_weight=weights)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return np.column_stack([self.models[label].predict_proba(X)[:, 1] for label in self.labels])


class SpecialistTrainer:
    def __init__(
        self,
        *,
        canonical_dir: Path,
        model_dir: Path,
        dataset_version: str,
        registry: Registry,
        n_features: int = 2**18,
        alpha: float = 1e-5,
        epochs: int = 2,
        random_seed: int = 42,
        max_rows: int | None = None,
    ):
        self.canonical_dir = Path(canonical_dir)
        self.model_dir = Path(model_dir)
        self.dataset_version = dataset_version
        self.registry = registry
        self.n_features = n_features
        self.alpha = alpha
        self.epochs = epochs
        self.random_seed = random_seed
        self.max_rows = max_rows
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _iter_multilabel(self, task: MultiLabelTask, split: str):
        columns = [task.input_column, task.label_column, task.eligibility_column, "split"]
        consumed = 0
        for chunk in iter_dataset(self.canonical_dir, "steps", columns=columns):
            subset = chunk.loc[
                (chunk["split"] == split)
                & chunk[task.eligibility_column].fillna(False).astype(bool)
            ].copy()
            if subset.empty:
                continue
            labels = [set(parse_list(value)) for value in subset[task.label_column]]
            if self.max_rows is not None:
                remaining = self.max_rows - consumed
                if remaining <= 0:
                    break
                subset = subset.iloc[:remaining]
                labels = labels[:remaining]
            consumed += len(subset)
            yield subset[task.input_column].fillna("").astype(str).tolist(), labels

    def _discover_multilabel_classes(
        self, task: MultiLabelTask
    ) -> tuple[list[str], dict[str, int], dict[str, Any], dict[str, Any]]:
        split_totals: Counter[str] = Counter()
        positive_counts: dict[str, Counter[str]] = {
            "train": Counter(), "val": Counter(), "test": Counter()
        }
        for split in ("train", "val", "test"):
            for _, labels in self._iter_multilabel(task, split):
                split_totals[split] += len(labels)
                for values in labels:
                    positive_counts[split].update(values)
        candidate_labels = [
            label
            for label, count in positive_counts["train"].most_common(task.maximum_labels)
            if count >= task.minimum_label_count
        ]
        prefit = validate_multilabel_support(
            dict(split_totals),
            {split: dict(counts) for split, counts in positive_counts.items()},
            candidate_labels=candidate_labels,
            minimum_positive_train=task.minimum_label_count,
            minimum_negative_train=task.minimum_label_count,
        )
        audit = {
            "task": task.name,
            "task_type": "multilabel_binary_targets",
            "label_column": task.label_column,
            "split_totals": dict(split_totals),
            "positive_counts": {split: dict(counts) for split, counts in positive_counts.items()},
            "minimum_positive_train": task.minimum_label_count,
            "minimum_negative_train": task.minimum_label_count,
            "maximum_labels": task.maximum_labels,
        }
        return (
            list(prefit.retained_labels),
            dict(positive_counts["train"]),
            prefit.to_dict(),
            audit,
        )

    def _complete_multilabel_skip(
        self,
        *,
        run_id: str,
        task: MultiLabelTask,
        request_promotion: bool,
        prefit_support: dict[str, Any],
        population_audit: dict[str, Any],
    ) -> dict[str, Any]:
        reason_code = str(
            prefit_support.get("reason_code") or "multilabel_no_valid_binary_targets"
        )
        reasons = list(
            prefit_support.get("reasons")
            or ["No multilabel target has valid positive and negative support."]
        )
        metrics = {
            "status": "skipped",
            "reason_code": reason_code,
            "reason": " ".join(reasons),
            "model_created": False,
            "promotion_requested": bool(request_promotion),
            "promotion_approved": False,
            "prefit_support": prefit_support,
            "population_audit": population_audit,
        }
        task_dir = self.model_dir / task.name
        task_dir.mkdir(parents=True, exist_ok=True)
        audit_path = task_dir / f"{run_id}.task_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "task": task.name,
                    "dataset_version": self.dataset_version,
                    "config": asdict(task),
                    "metrics": metrics,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        reproducibility = self._reproducibility_hashes(
            task=task.name, input_column=task.input_column
        )
        audit_record = self.registry.register_task_audit(
            task=task.name,
            dataset_version=self.dataset_version,
            audit_path=audit_path,
            reason_code=reason_code,
            metrics=metrics,
            split_sha256=reproducibility.get("split_sha256"),
        )
        self.registry.finish_run(run_id, metrics, reproducibility=reproducibility)
        return {
            "run_id": run_id,
            "status": "skipped",
            "task": task.name,
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

    @staticmethod
    def _multilabel_metrics(
        probabilities: np.ndarray,
        label_sets: list[set[str]],
        classes: list[str],
        top_k: int,
    ) -> dict[str, Any]:
        mlb = MultiLabelBinarizer(classes=classes)
        mlb.fit([classes])
        y_true = mlb.transform([sorted(values & set(classes)) for values in label_sets])
        if probabilities.shape != y_true.shape:
            raise ValueError("Prediction and target matrices have different shapes.")
        ranked = np.argsort(probabilities, axis=1)[:, ::-1]
        y_pred = np.zeros_like(y_true)
        for row_index, indices in enumerate(ranked):
            selected = [
                int(index) for index in indices
                if probabilities[row_index, int(index)] >= 0.5
            ][: min(top_k, len(classes))]
            if selected:
                y_pred[row_index, selected] = 1
        positive_row_mask = y_true.sum(axis=1) > 0
        positive_rows = int(positive_row_mask.sum())
        negative_only_rows = int(len(y_true) - positive_rows)
        lrap = (
            float(
                label_ranking_average_precision_score(
                    y_true[positive_row_mask], probabilities[positive_row_mask]
                )
            )
            if positive_rows
            else 0.0
        )
        return {
            "rows": len(label_sets),
            "positive_label_rows": positive_rows,
            "negative_only_rows": negative_only_rows,
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            "lrap": lrap,
            "lrap_population": "rows_with_at_least_one_positive_label",
            "top_k": top_k,
        }

    def _collect_multilabel_split(
        self, task: MultiLabelTask, model: IncrementalMultiLabelModel, split: str
    ) -> tuple[np.ndarray, list[set[str]]]:
        all_probabilities: list[np.ndarray] = []
        all_labels: list[set[str]] = []
        for texts, labels in self._iter_multilabel(task, split):
            all_probabilities.append(model.predict_proba(texts))
            all_labels.extend(labels)
        probabilities = (
            np.vstack(all_probabilities)
            if all_probabilities
            else np.zeros((0, len(model.labels)), dtype=float)
        )
        return probabilities, all_labels

    @staticmethod
    def _fit_multilabel_calibrators(
        probabilities: np.ndarray, label_sets: list[set[str]], classes: list[str]
    ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        calibrators: dict[str, dict[str, float]] = {}
        reports: dict[str, Any] = {}
        for index, label in enumerate(classes):
            y_true = np.asarray([int(label in values) for values in label_sets], dtype=int)
            binary_probabilities = np.column_stack([1.0 - probabilities[:, index], probabilities[:, index]])
            calibrator = fit_temperature_scaler(y_true, binary_probabilities, np.asarray([0, 1]))
            calibrators[label] = calibrator.to_dict()
            reports[label] = calibration_report(
                y_true, binary_probabilities, np.asarray([0, 1]), calibrator
            )
        reports["mean_calibrated_ece"] = float(
            np.mean([item["calibrated_ece"] for item in reports.values() if isinstance(item, dict)])
        ) if classes else None
        return calibrators, reports

    @staticmethod
    def _apply_multilabel_calibrators(
        probabilities: np.ndarray, classes: list[str], calibrators: dict[str, dict[str, float]]
    ) -> np.ndarray:
        calibrated = np.zeros_like(probabilities, dtype=float)
        for index, label in enumerate(classes):
            binary = np.column_stack([1.0 - probabilities[:, index], probabilities[:, index]])
            transformed = TemperatureCalibrator.from_dict(calibrators.get(label)).transform(binary)
            calibrated[:, index] = transformed[:, 1]
        return calibrated

    def _reproducibility_hashes(self, *, task: str, input_column: str) -> dict[str, str | None]:
        manifest = self.canonical_dir / "dataset_manifest.json"
        try:
            split_sha = hash_dataset_columns(
                self.canonical_dir, "steps",
                ["step_id", "patent_document_id", "reaction_signature", "split_component_id", "split"]
            )
        except Exception:
            split_sha = None
        return {
            "dataset_sha256": sha256_file(manifest) if manifest.exists() else None,
            "feature_sha256": canonical_json_hash(
                {"task": task, "input_column": input_column, "n_features": self.n_features}
            ),
            "split_sha256": split_sha,
        }

    def _require_split_governance(self) -> None:
        dataset_manifest = self.canonical_dir / "dataset_manifest.json"
        if not dataset_manifest.exists():
            return
        split_manifest = self.canonical_dir / "split_manifest.json"
        if not split_manifest.exists():
            raise RuntimeError(
                "Product Two split governance is missing. Run `reacts --project-root . "
                "rebuild-product-two-splits` before retraining."
            )
        payload = json.loads(split_manifest.read_text(encoding="utf-8"))
        if not payload.get("invariants", {}).get("strict_pass"):
            raise RuntimeError("Product Two connected-component split invariants did not pass.")

    def train_multilabel(self, task_name: str, request_promotion: bool = False) -> dict[str, Any]:
        task = MULTILABEL_TASKS[task_name]
        self._require_split_governance()
        run_id = self.registry.start_run(task.name, self.dataset_version, asdict(task))
        try:
            classes, source_counts, prefit_support, population_audit = self._discover_multilabel_classes(task)
            if not prefit_support.get("trainable"):
                return self._complete_multilabel_skip(
                    run_id=run_id,
                    task=task,
                    request_promotion=request_promotion,
                    prefit_support=prefit_support,
                    population_audit=population_audit,
                )
            model = IncrementalMultiLabelModel(classes, self.n_features, self.random_seed, self.alpha)
            unique_rows = 0
            examples_seen = 0
            for epoch in range(self.epochs):
                epoch_rows = 0
                for texts, labels in self._iter_multilabel(task, "train"):
                    model.partial_fit(texts, labels)
                    epoch_rows += len(texts)
                    examples_seen += len(texts)
                if epoch == 0:
                    unique_rows = epoch_rows
                LOGGER.info("Task %s epoch %s rows=%s", task.name, epoch + 1, epoch_rows)

            validation_probabilities, validation_labels = self._collect_multilabel_split(task, model, "val")
            calibrators, calibration = (
                self._fit_multilabel_calibrators(validation_probabilities, validation_labels, classes)
                if validation_labels
                else ({}, {})
            )
            evaluations: dict[str, Any] = {}
            for split, probabilities, labels in [
                ("val", validation_probabilities, validation_labels),
                ("test", *self._collect_multilabel_split(task, model, "test")),
            ]:
                if labels:
                    calibrated = self._apply_multilabel_calibrators(probabilities, classes, calibrators)
                    evaluations[split] = self._multilabel_metrics(calibrated, labels, classes, task.top_k)
                else:
                    evaluations[split] = {"rows": 0}
            evaluations["val"]["calibration"] = calibration

            metrics = {
                "unique_training_rows": unique_rows,
                "epochs": self.epochs,
                "training_examples_seen": examples_seen,
                "classes": classes,
                "source_class_counts": source_counts,
                "prefit_support": prefit_support,
                "population_audit": population_audit,
                "validation": evaluations["val"],
                "test": evaluations["test"],
            }
            test = metrics["test"]
            threshold_failures: list[str] = []
            if test.get("rows", 0) < 500:
                threshold_failures.append(
                    f"test_rows={test.get('rows', 0)} is below minimum_rows=500"
                )
            if test.get("micro_f1", 0.0) < 0.40:
                threshold_failures.append(
                    f"micro_f1={test.get('micro_f1', 0.0):.6f} is below minimum_micro_f1=0.40"
                )
            if test.get("lrap", 0.0) < 0.45:
                threshold_failures.append(
                    f"lrap={test.get('lrap', 0.0):.6f} is below minimum_lrap=0.45"
                )
            approved = not threshold_failures
            target_stage = ModelStage.STAGING
            initial_stage = ModelStage.VALIDATED if approved else ModelStage.CANDIDATE
            decision = {
                "approved": approved,
                "stage": target_stage.value if approved else initial_stage.value,
                "reasons": threshold_failures,
                "permitted_use": "top-k retrieval-backed suggestions",
                "thresholds": {"minimum_rows": 500, "minimum_micro_f1": 0.40, "minimum_lrap": 0.45},
            }
            reproducibility = self._reproducibility_hashes(task=task.name, input_column=task.input_column)
            task_dir = self.model_dir / task.name
            task_dir.mkdir(parents=True, exist_ok=True)
            artifact = task_dir / f"{run_id}.joblib"
            training_environment = runtime_environment()
            joblib.dump(
                {
                    "task": task.name,
                    "task_type": "multilabel",
                    "dataset_version": self.dataset_version,
                    "model": model,
                    "classes": classes,
                    "input_column": task.input_column,
                    "top_k": task.top_k,
                    "calibrators": calibrators,
                    "metrics": metrics,
                    "reproducibility": reproducibility,
                    "training_environment": training_environment,
                },
                artifact,
                compress=3,
            )
            record = self.registry.register_model(
                task=task.name,
                artifact_path=artifact,
                dataset_version=self.dataset_version,
                metrics=metrics,
                config=asdict(task),
                stage=initial_stage,
                dataset_sha256=reproducibility["dataset_sha256"],
                feature_sha256=reproducibility["feature_sha256"],
                split_sha256=reproducibility["split_sha256"],
                release_decision=decision,
                training_environment=training_environment,
            )
            if approved and request_promotion:
                self.registry.promote(record["model_id"], target_stage, release_decision=decision)
                record["stage"] = target_stage.value
            card = artifact.with_suffix(".model_card.json")
            card.write_text(json.dumps({
                "model": record,
                "metrics": metrics,
                "release_decision": decision,
                "reproducibility": reproducibility,
                "training_environment": training_environment,
                "limitations": [
                    "Supervised labels below the support threshold are preserved only through retrieval.",
                    "Top-k suggestions require evidence and applicability support.",
                ],
            }, indent=2, default=str), encoding="utf-8")
            self.registry.finish_run(run_id, metrics, reproducibility={**reproducibility, "artifact_sha256": record["artifact_sha256"]})
            return {"run_id": run_id, "model": record, "metrics": metrics, "release_decision": decision, "reproducibility": reproducibility}
        except Exception as exc:
            self.registry.finish_run(run_id, {}, error=str(exc))
            raise

    def _iter_regression(self, task: RegressionTask, split: str):
        columns = [task.input_column, task.target_column, task.eligibility_column, task.family_column, "split"]
        consumed = 0
        for chunk in iter_dataset(self.canonical_dir, "steps", columns=columns):
            subset = chunk.loc[
                (chunk["split"] == split)
                & chunk[task.eligibility_column].fillna(False).astype(bool)
                & pd.to_numeric(chunk[task.target_column], errors="coerce").notna()
            ].copy()
            if subset.empty:
                continue
            subset[task.target_column] = pd.to_numeric(subset[task.target_column], errors="coerce")
            if self.max_rows is not None:
                remaining = self.max_rows - consumed
                if remaining <= 0:
                    break
                subset = subset.iloc[:remaining]
            consumed += len(subset)
            yield (
                subset[task.input_column].fillna("").astype(str).tolist(),
                subset[task.target_column].to_numpy(dtype=np.float64),
                subset[task.family_column].fillna("unknown").astype(str).tolist(),
            )

    @staticmethod
    def _forward(values: np.ndarray, transform: str) -> np.ndarray:
        return np.log1p(np.maximum(values, 0.0)) if transform == "log1p" else values

    @staticmethod
    def _inverse(values: np.ndarray, transform: str) -> np.ndarray:
        return np.expm1(values) if transform == "log1p" else values

    def train_regression(self, task_name: str, request_promotion: bool = False) -> dict[str, Any]:
        task = REGRESSION_TASKS[task_name]
        self._require_split_governance()
        run_id = self.registry.start_run(task.name, self.dataset_version, asdict(task))
        try:
            vectorizer = reaction_text_vectorizer(self.n_features)
            model = SGDRegressor(
                loss="huber",
                penalty="elasticnet",
                alpha=self.alpha,
                l1_ratio=0.05,
                random_state=self.random_seed,
                average=True,
            )
            unique_rows = 0
            examples_seen = 0
            for epoch in range(self.epochs):
                epoch_rows = 0
                for texts, values, _ in self._iter_regression(task, "train"):
                    model.partial_fit(vectorizer.transform(texts), self._forward(values, task.transform))
                    epoch_rows += len(values)
                    examples_seen += len(values)
                if epoch == 0:
                    unique_rows = epoch_rows
                LOGGER.info("Task %s epoch %s rows=%s", task.name, epoch + 1, epoch_rows)

            residuals_by_family: dict[str, list[float]] = defaultdict(list)
            evaluations: dict[str, Any] = {}
            global_interval = (-1.0, 1.0)
            for split in ["val", "test"]:
                y_true: list[float] = []
                y_pred: list[float] = []
                families: list[str] = []
                for texts, values, family_values in self._iter_regression(task, split):
                    transformed = model.predict(vectorizer.transform(texts))
                    predictions = self._inverse(transformed, task.transform)
                    y_true.extend(values.tolist())
                    y_pred.extend(predictions.tolist())
                    families.extend(family_values)
                truth = np.asarray(y_true, dtype=float)
                prediction = np.asarray(y_pred, dtype=float)
                if not len(truth):
                    evaluations[split] = {"rows": 0}
                    continue
                residuals = truth - prediction
                if split == "val":
                    global_interval = (float(np.quantile(residuals, 0.10)), float(np.quantile(residuals, 0.90)))
                    for family, residual in zip(families, residuals):
                        residuals_by_family[family].append(float(residual))
                low = prediction + global_interval[0]
                high = prediction + global_interval[1]
                evaluations[split] = {
                    "rows": len(truth),
                    "mae": float(mean_absolute_error(truth, prediction)),
                    "median_absolute_error": float(median_absolute_error(truth, prediction)),
                    "r2": float(r2_score(truth, prediction)),
                    "interval_80_coverage": float(np.mean((truth >= low) & (truth <= high))),
                    "interval_width_mean": float(np.mean(high - low)),
                }
            family_intervals = {
                family: [float(np.quantile(values, 0.10)), float(np.quantile(values, 0.90))]
                for family, values in residuals_by_family.items()
                if len(values) >= 30
            }
            metrics = {
                "unique_training_rows": unique_rows,
                "epochs": self.epochs,
                "training_examples_seen": examples_seen,
                "validation": evaluations["val"],
                "test": evaluations["test"],
                "global_residual_interval_80": list(global_interval),
                "family_interval_count": len(family_intervals),
            }
            test = metrics["test"]
            approved = bool(test.get("rows", 0) >= 500 and test.get("interval_80_coverage", 0.0) >= 0.70)
            target_stage = ModelStage.STAGING
            initial_stage = ModelStage.VALIDATED if approved else ModelStage.EXPERIMENTAL
            decision = {
                "approved": approved,
                "stage": target_stage.value if approved else initial_stage.value,
                "reasons": [] if approved else ["Regression interval release thresholds were not met."],
                "permitted_use": "confidence-qualified point estimates and intervals",
                "thresholds": {"minimum_rows": 500, "minimum_interval_80_coverage": 0.70},
            }
            reproducibility = self._reproducibility_hashes(task=task.name, input_column=task.input_column)
            task_dir = self.model_dir / task.name
            task_dir.mkdir(parents=True, exist_ok=True)
            artifact = task_dir / f"{run_id}.joblib"
            training_environment = runtime_environment()
            joblib.dump(
                {
                    "task": task.name,
                    "task_type": "regression_interval",
                    "dataset_version": self.dataset_version,
                    "vectorizer": vectorizer,
                    "model": model,
                    "input_column": task.input_column,
                    "units": task.units,
                    "transform": task.transform,
                    "global_residual_interval_80": list(global_interval),
                    "family_residual_intervals_80": family_intervals,
                    "metrics": metrics,
                    "reproducibility": reproducibility,
                    "training_environment": training_environment,
                },
                artifact,
                compress=3,
            )
            record = self.registry.register_model(
                task=task.name,
                artifact_path=artifact,
                dataset_version=self.dataset_version,
                metrics=metrics,
                config=asdict(task),
                stage=initial_stage,
                dataset_sha256=reproducibility["dataset_sha256"],
                feature_sha256=reproducibility["feature_sha256"],
                split_sha256=reproducibility["split_sha256"],
                release_decision=decision,
                training_environment=training_environment,
            )
            if approved and request_promotion:
                self.registry.promote(record["model_id"], target_stage, release_decision=decision)
                record["stage"] = target_stage.value
            card = artifact.with_suffix(".model_card.json")
            card.write_text(json.dumps({
                "model": record,
                "metrics": metrics,
                "release_decision": decision,
                "reproducibility": reproducibility,
                "training_environment": training_environment,
                "limitations": [
                    "Intervals are corpus-conditional and not experimental operating guarantees.",
                    "Inference abstains outside the evidence applicability domain.",
                ],
            }, indent=2, default=str), encoding="utf-8")
            self.registry.finish_run(run_id, metrics, reproducibility={**reproducibility, "artifact_sha256": record["artifact_sha256"]})
            return {"run_id": run_id, "model": record, "metrics": metrics, "release_decision": decision, "reproducibility": reproducibility}
        except Exception as exc:
            self.registry.finish_run(run_id, {}, error=str(exc))
            raise

    def train(self, task_name: str, request_promotion: bool = False) -> dict[str, Any]:
        if task_name in MULTILABEL_TASKS:
            return self.train_multilabel(task_name, request_promotion=request_promotion)
        if task_name in REGRESSION_TASKS:
            return self.train_regression(task_name, request_promotion=request_promotion)
        raise KeyError(task_name)
