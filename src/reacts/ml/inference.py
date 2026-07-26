from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from reacts.chemistry.mapping import AtomMappingEngine, derive_reaction_centre
from reacts.chemistry.reactions import canonicalize_reaction, parse_reaction
from reacts.contracts import (
    Applicability,
    EvidenceItem,
    InferenceResponse,
    ModelStage,
    PredictionItem,
    TaskPrediction,
)
from reacts.ml.environment import validate_runtime_environment
from reacts.ml.registry import Registry
from reacts.retrieval.contextual_index import (
    ContextualFingerprintIndex,
    neighbour_distributions,
)
from reacts.retrieval.fingerprint_index import FingerprintIndex
from reacts.science.calibration import TemperatureCalibrator


class InferenceService:
    """Product One compatibility runtime."""

    def __init__(self, registry: Registry, index_dir: Path | None = None):
        self.registry = registry
        self.index_dir = Path(index_dir) if index_dir else None
        self._models: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._index: FingerprintIndex | None = None

    def _load_model(self, task: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if task in self._models:
            return self._models[task]
        record = self.registry.model_for_task(
            task,
            allowed_stages=(
                ModelStage.PRODUCTION,
                ModelStage.BASELINE,
                ModelStage.STAGING,
                ModelStage.SCREENING,
            ),
        )
        if record is None:
            return None
        training_environment = record.get("training_environment") or {}
        if (
            training_environment
            and training_environment.get("status") != "unknown_legacy"
            and not validate_runtime_environment(training_environment)["pass"]
        ):
            return None
        bundle = joblib.load(self.registry.resolve_artifact_path(record["artifact_path"]))
        self._models[task] = (record, bundle)
        return record, bundle

    def _load_index(self) -> FingerprintIndex | None:
        if self._index is not None:
            return self._index
        if self.index_dir and (self.index_dir / "index_manifest.json").exists():
            self._index = FingerprintIndex(self.index_dir)
        return self._index

    @staticmethod
    def _task_prediction(task: str, record: dict[str, Any], bundle: dict[str, Any], reaction: str) -> TaskPrediction:
        model_input = reaction
        if bundle.get("input_column") == "canonical_reaction_smiles":
            canonical = canonicalize_reaction(reaction)
            if canonical is None:
                return TaskPrediction(
                    task=task,
                    predictions=[],
                    abstained=True,
                    reason="Canonical structural reaction could not be generated.",
                    model_version=record["version"],
                    model_stage=record["stage"],
                )
            model_input = canonical
        X = bundle["vectorizer"].transform([model_input])
        model = bundle["model"]
        probabilities = model.predict_proba(X)
        calibrator = TemperatureCalibrator.from_dict(bundle.get("calibrator"))
        probabilities = calibrator.transform(probabilities)[0]
        ranked = np.argsort(probabilities)[::-1][: min(5, len(probabilities))]
        return TaskPrediction(
            task=task,
            predictions=[
                PredictionItem(
                    label=str(model.classes_[index]),
                    probability=float(probabilities[index]),
                    calibrated_probability=float(probabilities[index]),
                    model_probability=float(probabilities[index]),
                )
                for index in ranked
            ],
            model_version=record["version"],
            model_stage=record["stage"],
        )

    def predict(
        self,
        reaction_smiles: str,
        tasks: list[str],
        include_evidence: bool = True,
        evidence_k: int = 5,
    ) -> InferenceResponse:
        parsed = parse_reaction(reaction_smiles)
        warnings: list[str] = []
        task_results: list[TaskPrediction] = []
        provenance: dict[str, Any] = {"models": {}}
        for task in tasks:
            loaded = self._load_model(task)
            if loaded is None:
                task_results.append(TaskPrediction(task=task, predictions=[], abstained=True, reason="No registered model is available."))
                continue
            record, bundle = loaded
            if task != "parse_validity" and not parsed.parse_ok:
                task_results.append(
                    TaskPrediction(
                        task=task,
                        predictions=[],
                        abstained=True,
                        reason=f"Reaction is not chemically complete: {parsed.failure_class.value}.",
                    )
                )
                continue
            result = self._task_prediction(task, record, bundle, reaction_smiles)
            task_results.append(result)
            provenance["models"][task] = {
                "version": record["version"],
                "stage": record["stage"],
                "dataset_version": record["dataset_version"],
            }

        evidence: list[EvidenceItem] = []
        applicability = Applicability.INVALID if not parsed.parse_ok else Applicability.WEAKLY_SUPPORTED
        if parsed.parse_ok and include_evidence:
            index = self._load_index()
            if index:
                found = index.search(reaction_smiles, k=evidence_k)
                evidence = [EvidenceItem(**item) for item in found]
                best = evidence[0].score if evidence else 0.0
                applicability = Applicability.IN_DOMAIN if best >= 0.65 else (
                    Applicability.WEAKLY_SUPPORTED if best >= 0.35 else Applicability.OUT_OF_DOMAIN
                )
                provenance["retrieval_index"] = index.manifest.get("index_version")
            else:
                warnings.append("No reaction evidence index is available; applicability is model-confidence-only.")

        return InferenceResponse(
            input_reaction=reaction_smiles,
            canonical_reaction=canonicalize_reaction(reaction_smiles),
            parse_ok=parsed.parse_ok,
            parse_failure_class=parsed.failure_class.value,
            applicability=applicability,
            tasks=task_results,
            evidence=evidence,
            provenance=provenance,
            warnings=warnings,
        )


_TASK_EVIDENCE_FIELDS = {
    "primary_solvent": "solvent_primary",
    "solvent_multilabel": "solvents",
    "solvent_family_multilabel": "solvents",
    "agent_multilabel": "agents",
    "agent_family_multilabel": "agents",
    "catalyst_family_multilabel": "agents",
    "time_bucket": "time_bucket",
    "temperature_bucket": "temperature_bucket",
    "reaction_family": "reaction_family",
}


class ContextualInferenceService:
    def __init__(
        self,
        registry: Registry,
        index_dir: Path,
        *,
        in_domain_threshold: float = 0.65,
        weak_threshold: float = 0.35,
        abstention_threshold: float = 0.35,
    ):
        self.registry = registry
        self.index_dir = Path(index_dir)
        self.in_domain_threshold = in_domain_threshold
        self.weak_threshold = weak_threshold
        self.abstention_threshold = abstention_threshold
        self._models: dict[tuple[str, bool], tuple[dict[str, Any], dict[str, Any]]] = {}
        self._index: ContextualFingerprintIndex | None = None
        self._mapper = AtomMappingEngine("auto")

    def _load_index(self) -> ContextualFingerprintIndex | None:
        if self._index is None and (self.index_dir / "index_manifest.json").exists():
            self._index = ContextualFingerprintIndex(self.index_dir)
        return self._index

    def _load_model(self, task: str, allow_experimental: bool) -> tuple[dict[str, Any], dict[str, Any]] | None:
        key = (task, allow_experimental)
        if key in self._models:
            return self._models[key]
        allowed = [ModelStage.PRODUCTION, ModelStage.STAGING, ModelStage.SCREENING, ModelStage.BASELINE]
        if allow_experimental:
            allowed.extend([ModelStage.VALIDATED, ModelStage.EXPERIMENTAL, ModelStage.CANDIDATE])
        record = self.registry.model_for_task(task, allowed_stages=tuple(allowed))
        if record is None:
            return None
        training_environment = record.get("training_environment") or {}
        if (
            training_environment
            and training_environment.get("status") != "unknown_legacy"
            and not validate_runtime_environment(training_environment)["pass"]
        ):
            return None
        bundle = joblib.load(self.registry.resolve_artifact_path(record["artifact_path"]))
        self._models[key] = (record, bundle)
        return record, bundle

    def _reaction_family(self, canonical: str | None) -> tuple[str | None, str | None]:
        if not canonical:
            return None, None
        mapping = self._mapper.map_reaction(canonical)
        if not mapping.mapped_reaction_smiles or mapping.confidence < 0.5:
            return None, None
        try:
            centre = derive_reaction_centre(mapping.mapped_reaction_smiles)
            return centre.structural_family, centre.fingerprint
        except Exception:
            return None, None

    @staticmethod
    def _applicability(best_score: float, in_domain: float, weak: float) -> Applicability:
        if best_score >= in_domain:
            return Applicability.IN_DOMAIN
        if best_score >= weak:
            return Applicability.WEAKLY_SUPPORTED
        return Applicability.OUT_OF_DOMAIN

    def _classification_prediction(
        self,
        task: str,
        record: dict[str, Any],
        bundle: dict[str, Any],
        original: str,
        canonical: str | None,
        distributions: dict[str, dict[str, float]],
        evidence: list[dict[str, Any]],
        family: str | None,
        applicability: Applicability,
    ) -> TaskPrediction:
        input_column = str(bundle.get("input_column") or "reaction_smiles")
        input_text = original if input_column in {"reaction_smiles", "original_reaction_smiles"} else (canonical or original)
        model = bundle["model"]
        probabilities = model.predict_proba(bundle["vectorizer"].transform([input_text]))
        probabilities = TemperatureCalibrator.from_dict(bundle.get("calibrator")).transform(probabilities)[0]
        evidence_field = _TASK_EVIDENCE_FIELDS.get(task)
        neighbours = distributions.get(evidence_field or "", {})
        family_items = [item for item in evidence if family and item.get("reaction_family") == family]
        family_distribution = neighbour_distributions(family_items, [evidence_field])[evidence_field] if evidence_field and family_items else {}
        applicability_score = {
            Applicability.IN_DOMAIN: 1.0,
            Applicability.WEAKLY_SUPPORTED: 0.6,
            Applicability.OUT_OF_DOMAIN: 0.15,
            Applicability.INVALID: 0.0,
        }[applicability]
        candidates: list[PredictionItem] = []
        for index, label in enumerate(model.classes_):
            model_probability = float(probabilities[index])
            neighbour_probability = float(neighbours.get(str(label), 0.0))
            family_prior = float(family_distribution.get(str(label), neighbour_probability))
            combined = 0.55 * model_probability + 0.25 * neighbour_probability + 0.15 * family_prior + 0.05 * applicability_score
            candidates.append(
                PredictionItem(
                    label=str(label),
                    probability=combined,
                    calibrated_probability=model_probability,
                    model_probability=model_probability,
                    neighbour_probability=neighbour_probability,
                    family_prior=family_prior,
                    combined_score=combined,
                )
            )
        candidates.sort(key=lambda item: item.probability, reverse=True)
        top = candidates[: min(5, len(candidates))]
        confidence = top[0].probability if top else 0.0
        abstained = confidence < self.abstention_threshold or applicability == Applicability.OUT_OF_DOMAIN
        agreement = None
        if top and family_distribution:
            agreement = float(family_distribution.get(top[0].label, 0.0))
        return TaskPrediction(
            task=task,
            predictions=top,
            abstained=abstained,
            reason="Insufficient calibrated/evidence support." if abstained else None,
            model_version=record["version"],
            model_stage=record["stage"],
            applicability=applicability,
            neighbour_support=len(evidence),
            reaction_family_agreement=agreement,
            calibration_error=(bundle.get("metrics", {}).get("validation", {}).get("calibration", {}).get("calibrated_ece")),
        )

    def _multilabel_prediction(
        self,
        task: str,
        record: dict[str, Any],
        bundle: dict[str, Any],
        canonical: str,
        distributions: dict[str, dict[str, float]],
        evidence: list[dict[str, Any]],
        family: str | None,
        applicability: Applicability,
    ) -> TaskPrediction:
        raw_probabilities = bundle["model"].predict_proba([canonical])[0]
        calibrators = bundle.get("calibrators", {})
        probabilities = np.asarray([
            TemperatureCalibrator.from_dict(calibrators.get(str(label))).transform(
                np.asarray([[1.0 - float(probability), float(probability)]])
            )[0, 1]
            for label, probability in zip(bundle["classes"], raw_probabilities)
        ], dtype=float)
        field = _TASK_EVIDENCE_FIELDS.get(task, "agents")
        neighbours = distributions.get(field, {})
        family_items = [item for item in evidence if family and item.get("reaction_family") == family]
        family_prior_distribution = neighbour_distributions(family_items, [field])[field] if family_items else {}
        applicability_score = {
            Applicability.IN_DOMAIN: 1.0,
            Applicability.WEAKLY_SUPPORTED: 0.6,
            Applicability.OUT_OF_DOMAIN: 0.15,
            Applicability.INVALID: 0.0,
        }[applicability]
        candidates: list[PredictionItem] = []
        for label, model_probability in zip(bundle["classes"], probabilities):
            neighbour_probability = float(neighbours.get(str(label), 0.0))
            family_prior = float(family_prior_distribution.get(str(label), neighbour_probability))
            combined = 0.55 * float(model_probability) + 0.25 * neighbour_probability + 0.15 * family_prior + 0.05 * applicability_score
            candidates.append(
                PredictionItem(
                    label=str(label),
                    probability=combined,
                    calibrated_probability=float(model_probability),
                    model_probability=float(model_probability),
                    neighbour_probability=neighbour_probability,
                    family_prior=family_prior,
                    combined_score=combined,
                )
            )
        candidates.sort(key=lambda item: item.probability, reverse=True)
        top = candidates[: int(bundle.get("top_k", 5))]
        confidence = top[0].probability if top else 0.0
        abstained = confidence < self.abstention_threshold or applicability == Applicability.OUT_OF_DOMAIN
        return TaskPrediction(
            task=task,
            predictions=top,
            abstained=abstained,
            reason="Insufficient calibrated/evidence support." if abstained else None,
            model_version=record["version"],
            model_stage=record["stage"],
            applicability=applicability,
            neighbour_support=len(evidence),
        )

    def _regression_prediction(
        self,
        task: str,
        record: dict[str, Any],
        bundle: dict[str, Any],
        canonical: str,
        family: str | None,
        applicability: Applicability,
    ) -> TaskPrediction:
        transformed = float(bundle["model"].predict(bundle["vectorizer"].transform([canonical]))[0])
        point = float(np.expm1(transformed)) if bundle.get("transform") == "log1p" else transformed
        intervals = bundle.get("family_residual_intervals_80", {})
        residual = intervals.get(family) or bundle.get("global_residual_interval_80", [-1.0, 1.0])
        low, high = point + float(residual[0]), point + float(residual[1])
        if bundle.get("transform") == "log1p":
            low = max(0.0, low)
        abstained = applicability == Applicability.OUT_OF_DOMAIN
        return TaskPrediction(
            task=task,
            predictions=[],
            abstained=abstained,
            reason="Reaction is outside the model applicability domain." if abstained else None,
            model_version=record["version"],
            model_stage=record["stage"],
            applicability=applicability,
            point_estimate=point,
            interval=(low, high),
            units=bundle.get("units"),
        )

    def predict(
        self,
        reaction_smiles: str,
        tasks: list[str],
        *,
        include_evidence: bool = True,
        evidence_k: int = 5,
        allow_experimental: bool = False,
    ) -> InferenceResponse:
        parsed = parse_reaction(reaction_smiles)
        canonical = canonicalize_reaction(reaction_smiles) if parsed.reactants_valid and parsed.products_valid else None
        parse_tasks = {"parse_validity", "parse_failure_class", "repairability"}
        requested_parse_tasks = [task for task in tasks if task in parse_tasks]
        blocked_tasks = [task for task in tasks if task not in parse_tasks]
        if not parsed.parse_ok and not requested_parse_tasks:
            return InferenceResponse(
                input_reaction=reaction_smiles,
                canonical_reaction=canonical,
                parse_ok=False,
                parse_failure_class=parsed.failure_class.value,
                applicability=Applicability.INVALID,
                tasks=[
                    TaskPrediction(
                        task=task,
                        predictions=[],
                        abstained=True,
                        reason=f"Reaction is not chemically complete: {parsed.failure_class.value}.",
                    )
                    for task in tasks
                ],
                warnings=["Contextual condition inference requires a chemically complete reaction."],
            )

        family, centre_fingerprint = self._reaction_family(canonical) if parsed.parse_ok else (None, None)
        evidence_raw: list[dict[str, Any]] = []
        warnings: list[str] = []
        index = self._load_index()
        if include_evidence and index and canonical:
            evidence_raw = index.search(
                canonical,
                k=evidence_k,
                reaction_centre_fingerprint=centre_fingerprint,
                minimum_quality=0.35,
            )
        elif include_evidence:
            warnings.append("No Product Two contextual evidence index is available.")
        best_score = float(evidence_raw[0]["score"]) if evidence_raw else 0.0
        applicability = (
            self._applicability(best_score, self.in_domain_threshold, self.weak_threshold)
            if parsed.parse_ok
            else Applicability.INVALID
        )
        distributions = neighbour_distributions(
            evidence_raw,
            ["solvent_primary", "solvents", "agents", "time_bucket", "temperature_bucket", "reaction_family"],
        )
        task_results: list[TaskPrediction] = []
        provenance: dict[str, Any] = {
            "models": {},
            "retrieval_index": index.manifest.get("index_version") if index else None,
            "dataset_version": index.manifest.get("dataset_version") if index else None,
        }
        for task in tasks:
            if not parsed.parse_ok and task in blocked_tasks:
                task_results.append(
                    TaskPrediction(
                        task=task,
                        predictions=[],
                        abstained=True,
                        reason=f"Reaction is not chemically complete: {parsed.failure_class.value}.",
                        applicability=Applicability.INVALID,
                    )
                )
                continue
            loaded = self._load_model(task, allow_experimental)
            if loaded is None:
                task_results.append(
                    TaskPrediction(
                        task=task,
                        predictions=[],
                        abstained=True,
                        reason="No model in a permitted release stage is available.",
                    )
                )
                continue
            record, bundle = loaded
            bundle_type = bundle.get("task_type", "classification")
            if bundle_type == "multilabel":
                result = self._multilabel_prediction(
                    task, record, bundle, canonical or reaction_smiles, distributions, evidence_raw, family, applicability
                )
            elif bundle_type == "regression_interval":
                result = self._regression_prediction(
                    task, record, bundle, canonical or reaction_smiles, family, applicability
                )
            else:
                result = self._classification_prediction(
                    task, record, bundle, reaction_smiles, canonical, distributions, evidence_raw, family, applicability
                )
            task_results.append(result)
            provenance["models"][task] = {
                "model_id": record["model_id"],
                "version": record["version"],
                "stage": record["stage"],
                "dataset_version": record["dataset_version"],
                "artifact_sha256": record.get("artifact_sha256"),
                "feature_sha256": record.get("feature_sha256"),
                "split_sha256": record.get("split_sha256"),
            }

        if not parsed.parse_ok:
            warnings.append("Only parse-intelligence tasks were evaluated; contextual chemistry tasks abstained.")
        evidence = [EvidenceItem(**item) for item in evidence_raw]
        return InferenceResponse(
            input_reaction=reaction_smiles,
            canonical_reaction=canonical,
            parse_ok=parsed.parse_ok,
            parse_failure_class=parsed.failure_class.value,
            applicability=applicability,
            tasks=task_results,
            evidence=evidence,
            neighbour_label_distributions=distributions,
            reaction_family=family,
            provenance=provenance,
            warnings=warnings,
        )
