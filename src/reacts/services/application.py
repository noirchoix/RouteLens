from __future__ import annotations

import json

import joblib
from pathlib import Path
from typing import Any

from reacts import __version__
from reacts.artifacts.bundle import ArtifactBundlePublisher, ArtifactBundleValidator
from reacts.artifacts.runtime import ArtifactRuntimeState, bootstrap_artifact_runtime
from reacts.data.canonical import CanonicalBuildConfig, CanonicalBuilder
from reacts.data.canonical_v2 import ContextualBuildConfig, ContextualCanonicalBuilder
from reacts.data.split_governance import ProductTwoSplitRebuilder
from reacts.data.source import ArtifactSource
from reacts.ml.anomaly import ConditionAnomalyModel, RouteQualityScorer
from reacts.chemistry.repair import deterministic_repair_candidates
from reacts.ml.inference import ContextualInferenceService, InferenceService
from reacts.ml.capabilities import model_capability
from reacts.ml.registry import Registry, UnavailableRegistry
from reacts.ml.specialists import SpecialistTrainer
from reacts.ml.training import Trainer, TrainingConfig
from reacts.mapping.benchmark import MappingBenchmarkConfig, benchmark_mapper
from reacts.mapping.derivation import DerivationConfig, ReactionCentreDeriver
from reacts.mapping.runner import MappingRunConfig, ResumableMappingRunner
from reacts.retrieval.contextual_index import ContextualFingerprintIndexBuilder, ContextualIndexBuildConfig
from reacts.retrieval.fingerprint_index import FingerprintIndexBuilder, IndexBuildConfig
from reacts.retrieval.route_index import RouteEmbeddingIndex, RouteEmbeddingIndexBuilder, RouteIndexBuildConfig
from reacts.science.baseline import freeze_product_one_baseline
from reacts.science.release import lock_product_two_release
from reacts.services.jobs import JobManager
from reacts.services.routes import RouteRepository
from reacts.settings import Settings
from reacts.validation.acceptance import ScientificAcceptanceValidator


class Application:
    def __init__(self, settings: Settings, *, read_only_registry: bool = False):
        original = settings.resolve()
        bound, artifact_runtime = bootstrap_artifact_runtime(original, service_version=__version__)
        self.settings = bound
        self.artifact_runtime = artifact_runtime
        artifact_read_only = artifact_runtime.configured
        if artifact_runtime.configured and artifact_runtime.artifact_root is None:
            self.registry = UnavailableRegistry(artifact_runtime.detail or "artifact startup failed")
        else:
            self.registry = Registry(
                bound.registry_db,
                read_only=read_only_registry or artifact_read_only,
                project_root=artifact_runtime.artifact_root or bound.project_root,
                model_dir=bound.model_dir,
            )
        self.jobs = JobManager(self.registry)
        self.inference = InferenceService(self.registry, bound.index_dir)
        self.contextual_inference = ContextualInferenceService(
            self.registry,
            bound.index_v2_dir,
            in_domain_threshold=bound.evidence_in_domain_threshold,
            weak_threshold=bound.evidence_weak_threshold,
            abstention_threshold=bound.inference_abstention_threshold,
            artifact_release=artifact_runtime.artifact_release,
            training_split_sha256=artifact_runtime.manifest.get("training_split_sha256"),
        )
        self.routes = RouteRepository(bound.canonical_dir)
        self.contextual_routes = RouteRepository(bound.canonical_v2_dir)
        self.route_index: RouteEmbeddingIndex | None = None
        if artifact_runtime.configured and artifact_runtime.artifact_root is not None and bound.artifact_warmup:
            self.initialize_artifact_runtime()


    def close(self) -> None:
        self.jobs.shutdown()

    @property
    def artifact_mode(self) -> bool:
        return self.artifact_runtime.configured

    def ensure_ready(self) -> None:
        if not self.artifact_runtime.ready:
            detail = self.artifact_runtime.detail or "Artifact-backed runtime is not ready."
            raise RuntimeError(f"{self.artifact_runtime.reason_code or 'service_not_ready'}: {detail}")

    def ensure_mutable(self) -> None:
        if self.artifact_mode or getattr(self.registry, "read_only", False):
            raise PermissionError("Artifact-backed inference runtime is read-only; training and build jobs are disabled.")

    def initialize_artifact_runtime(self) -> dict[str, Any]:
        state = self.artifact_runtime
        if not state.configured:
            state.ready = True
            state.warmed_up = True
            return state.public()
        if state.artifact_root is None:
            return state.public()
        loaded_models: list[str] = []
        loading_task: str | None = None
        try:
            records = self.registry.list_models(runtime_only=True)
            expected_tasks = sorted(str(value) for value in state.manifest.get("required_tasks") or [])
            actual_tasks = sorted(str(record.get("task")) for record in records)
            if expected_tasks != actual_tasks:
                raise RuntimeError(f"Runtime task mismatch: expected={expected_tasks}, actual={actual_tasks}")
            for record in records:
                loading_task = str(record.get("task") or "unknown")
                joblib.load(self.registry.resolve_artifact_path(record["artifact_path"]))
                loaded_models.append(str(record["model_id"]))
            index = self.contextual_inference._load_index()
            if index is None:
                raise FileNotFoundError(self.settings.index_v2_dir / "index_manifest.json")
            self.route_index = RouteEmbeddingIndex(self.settings.index_v2_dir)
            state.warmup = {
                "models_loaded": len(loaded_models),
                "model_ids": loaded_models,
                "reaction_index_version": index.manifest.get("index_version"),
                "route_index_version": self.route_index.manifest.get("index_version"),
            }
            state.warmed_up = True
            state.ready = True
            state.reason_code = None
            state.detail = None
        except Exception as exc:
            state.ready = False
            state.warmed_up = False
            state.reason_code = "artifact_warmup_failed"
            state.detail = str(exc)
            state.warmup = {
                "models_loaded": len(loaded_models),
                "model_ids": loaded_models,
                "failed_task": loading_task,
            }
        return state.public()

    def readiness(self) -> dict[str, Any]:
        if not self.artifact_runtime.configured:
            return {
                "ready": True,
                "mode": "local_unmanaged",
                "reason_code": None,
                "version": __version__,
            }
        return {"version": __version__, **self.artifact_runtime.public()}

    def artifact_info(self) -> dict[str, Any]:
        return {"service_version": __version__, **self.artifact_runtime.public(), "manifest": self.artifact_runtime.manifest}

    def model_capabilities(self) -> list[dict[str, Any]]:
        return [model_capability(record) for record in self.registry.list_models(runtime_only=True)]

    def retrieve_reactions(
        self,
        reaction_smiles: str,
        *,
        k: int = 10,
        minimum_quality: float | None = 0.35,
    ) -> dict[str, Any]:
        self.ensure_ready()
        index = self.contextual_inference._load_index()
        if index is None:
            raise FileNotFoundError("Product Two reaction index is unavailable.")
        results = index.search(reaction_smiles, k=k, minimum_quality=minimum_quality)
        return {
            "artifact_release": self.artifact_runtime.artifact_release,
            "training_split_sha256": self.artifact_runtime.manifest.get("training_split_sha256"),
            "index_version": index.manifest.get("index_version"),
            "count": len(results),
            "results": results,
        }

    def get_contextual_route(self, route_id: str) -> dict[str, Any] | None:
        if self.artifact_mode:
            self.ensure_ready()
            if self.route_index is None:
                self.route_index = RouteEmbeddingIndex(self.settings.index_v2_dir)
            return self.route_index.get_route(route_id)
        return self.contextual_routes.get_route(route_id)

    def retrieve_routes(self, reaction_smiles: str, *, k: int = 10) -> dict[str, Any]:
        self.ensure_ready()
        if self.route_index is None:
            self.route_index = RouteEmbeddingIndex(self.settings.index_v2_dir)
        results = self.route_index.search_reaction(reaction_smiles, k=k)
        return {
            "artifact_release": self.artifact_runtime.artifact_release,
            "training_split_sha256": self.artifact_runtime.manifest.get("training_split_sha256"),
            "index_version": self.route_index.manifest.get("index_version"),
            "count": len(results),
            "results": results,
        }

    def package_product_two_artifacts(
        self,
        *,
        release: str,
        destination: Path,
        compatible_service_version: str = ">=2.1.0,<2.2.0",
        archive: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        self.ensure_mutable()
        return ArtifactBundlePublisher(self.settings).package(
            release=release,
            destination=destination,
            compatible_service_version=compatible_service_version,
            archive=archive,
            overwrite=overwrite,
        )

    @staticmethod
    def validate_artifact_bundle(bundle: Path, *, service_version: str = __version__) -> dict[str, Any]:
        return ArtifactBundleValidator(bundle).validate(service_version=service_version)

    def dataset_manifest(self, version: str = "v1") -> dict[str, Any] | None:
        root = self.settings.canonical_v2_dir if version == "v2" else self.settings.canonical_dir
        path = root / "dataset_manifest.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def freeze_product_one(self) -> dict[str, Any]:
        return freeze_product_one_baseline(
            registry=self.registry,
            canonical_dir=self.settings.canonical_dir,
            model_dir=self.settings.model_dir,
            index_dir=self.settings.index_dir,
            baseline_dir=self.settings.baseline_dir,
        )

    def build_canonical(self, prefer_parquet: bool = True) -> dict[str, Any]:
        builder = CanonicalBuilder(
            ArtifactSource(self.settings.source_artifact),
            self.settings.canonical_dir,
            CanonicalBuildConfig(
                prefer_parquet=prefer_parquet,
                temperature_min_c=self.settings.temperature_min_c,
                temperature_max_c=self.settings.temperature_max_c,
                time_min_h=self.settings.time_min_h,
                time_max_h=self.settings.time_max_h,
            ),
        )
        result = builder.build()
        manifest = self.settings.canonical_dir / "dataset_manifest.json"
        self.registry.register_dataset(result["dataset_version"], manifest, result)
        return result

    def build_contextual_canonical(
        self,
        *,
        prefer_parquet: bool = True,
        resume: bool = False,
        clean: bool = False,
        **deprecated_mapping_options: Any,
    ) -> dict[str, Any]:
        if any(value not in {None, False, "auto"} for value in deprecated_mapping_options.values()):
            raise ValueError(
                "Atom mapping is no longer executed during contextualization. "
                "Run build-contextual-v2 followed by map-reactions."
            )
        builder = ContextualCanonicalBuilder(
            self.settings.canonical_dir,
            self.settings.canonical_v2_context_dir,
            ContextualBuildConfig(
                prefer_parquet=prefer_parquet,
                resume=resume,
                clean=clean,
                map_reactions=False,
            ),
        )
        result = builder.build()
        manifest = self.settings.canonical_v2_context_dir / "dataset_manifest.json"
        self.registry.register_dataset(result["dataset_version"], manifest, result)
        return result

    def benchmark_mapper(
        self,
        *,
        backend: str = "rxnmapper",
        batch_sizes: tuple[int, ...] = (8, 16, 32, 64),
        sample_size: int = 512,
    ) -> dict[str, Any]:
        return benchmark_mapper(
            MappingBenchmarkConfig(
                context_dir=self.settings.canonical_v2_context_dir,
                report_path=self.settings.reports_dir / "mapping" / "rxnmapper_batch_benchmark.json",
                backend=backend,
                batch_sizes=batch_sizes,
                sample_size=sample_size,
                min_confidence=0.50,
                min_coverage=self.settings.mapping_min_coverage,
                mcs_timeout_seconds=self.settings.mapping_timeout_seconds,
            )
        )

    def map_reactions(
        self,
        *,
        backend: str = "rxnmapper",
        fallback_backend: str | None = "mcs",
        allow_auto_fallback: bool = False,
        batch_size: int | None = None,
        workers: int = 1,
        prefetch_batches: int | None = None,
        shard_size: int | None = None,
        rxnmapper_token_limit: int = 512,
        fallback_process_timeout_seconds: int = 30,
        resume: bool = False,
        max_rows: int | None = None,
        prefer_parquet: bool = True,
    ) -> dict[str, Any]:
        return ResumableMappingRunner(
            MappingRunConfig(
                context_dir=self.settings.canonical_v2_context_dir,
                output_dir=self.settings.mapping_v2_dir,
                queue_db=self.settings.mapping_queue_db,
                reports_dir=self.settings.reports_dir / "mapping",
                backend=backend,
                fallback_backend=fallback_backend,
                allow_auto_fallback=allow_auto_fallback,
                batch_size=batch_size or self.settings.mapping_batch_size,
                workers=workers,
                prefetch_batches=prefetch_batches or self.settings.mapping_prefetch_batches,
                shard_size=shard_size or self.settings.mapping_shard_size,
                min_coverage=self.settings.mapping_min_coverage,
                mcs_timeout_seconds=self.settings.mapping_timeout_seconds,
                fallback_process_timeout_seconds=fallback_process_timeout_seconds,
                rxnmapper_token_limit=rxnmapper_token_limit,
                stale_after_minutes=self.settings.mapping_stale_after_minutes,
                resume=resume,
                max_rows=max_rows,
                prefer_parquet=prefer_parquet,
            )
        ).run()

    def derive_reaction_centres(
        self,
        *,
        resume: bool = False,
        include_mcs: bool = False,
        min_confidence: float = 0.50,
        max_rows: int | None = None,
        prefer_parquet: bool = True,
    ) -> dict[str, Any]:
        result = ReactionCentreDeriver(
            DerivationConfig(
                context_dir=self.settings.canonical_v2_context_dir,
                mapping_dir=self.settings.mapping_v2_dir,
                derivation_dir=self.settings.derivation_v2_dir,
                final_canonical_dir=self.settings.canonical_v2_dir,
                queue_db=self.settings.derivation_queue_db,
                min_confidence=min_confidence,
                include_mcs=include_mcs,
                resume=resume,
                max_rows=max_rows,
                prefer_parquet=prefer_parquet,
            )
        ).run()
        final_manifest = self.settings.canonical_v2_dir / "dataset_manifest.json"
        if final_manifest.exists():
            final = json.loads(final_manifest.read_text(encoding="utf-8"))
            self.registry.register_dataset(final["dataset_version"], final_manifest, final)
        return result

    def rebuild_product_two_splits(
        self,
        *,
        prefer_parquet: bool = True,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        result = ProductTwoSplitRebuilder(
            self.settings.canonical_v2_dir,
            random_seed=self.settings.random_seed if random_seed is None else random_seed,
            prefer_parquet=prefer_parquet,
        ).run()
        split_sha = str(result["training_split_sha256"])
        result["training_split_sha256"] = split_sha
        result["superseded_runtime_models"] = self.registry.invalidate_models_for_split(
            "uspto_multistep_contextual_v2", split_sha
        )
        manifest_path = self.settings.canonical_v2_dir / "dataset_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.registry.register_dataset(
                manifest.get("dataset_version", "uspto_multistep_contextual_v2"),
                manifest_path,
                manifest,
            )
        self.registry.sync_json()
        return result

    def trainer(self, dataset_version: str, max_rows: int | None = None) -> Trainer:
        contextual = dataset_version == "uspto_multistep_contextual_v2"
        return Trainer(
            TrainingConfig(
                canonical_dir=self.settings.canonical_v2_dir if contextual else self.settings.canonical_dir,
                model_dir=self.settings.model_dir,
                dataset_version=dataset_version,
                random_seed=self.settings.random_seed,
                max_rows=max_rows,
            ),
            self.registry,
        )

    def specialist_trainer(self, max_rows: int | None = None) -> SpecialistTrainer:
        return SpecialistTrainer(
            canonical_dir=self.settings.canonical_v2_dir,
            model_dir=self.settings.model_dir,
            dataset_version="uspto_multistep_contextual_v2",
            registry=self.registry,
            random_seed=self.settings.random_seed,
            max_rows=max_rows,
        )

    def build_index(self, max_rows: int | None = None) -> dict[str, Any]:
        return FingerprintIndexBuilder(
            IndexBuildConfig(
                canonical_dir=self.settings.canonical_dir,
                index_dir=self.settings.index_dir,
                max_rows=max_rows,
            )
        ).build()

    def build_contextual_index(self, max_rows: int | None = None) -> dict[str, Any]:
        reaction_index = ContextualFingerprintIndexBuilder(
            ContextualIndexBuildConfig(
                canonical_dir=self.settings.canonical_v2_dir,
                index_dir=self.settings.index_v2_dir,
                max_rows=max_rows,
            )
        ).build()
        route_index = RouteEmbeddingIndexBuilder(
            RouteIndexBuildConfig(
                canonical_dir=self.settings.canonical_v2_dir,
                index_dir=self.settings.index_v2_dir,
            )
        ).build()
        return {"reaction_index": reaction_index, "route_index": route_index}

    def build_anomaly_model(self) -> dict[str, Any]:
        model = ConditionAnomalyModel.fit(self.settings.canonical_v2_dir)
        path = self.settings.model_dir / "condition_anomaly" / "robust_family_stats.json"
        model.save(path)
        return {"artifact_path": path.relative_to(self.settings.project_root).as_posix(), "families": len(model.statistics)}

    def repair_reaction(
        self,
        reaction_smiles: str,
        *,
        contextual_candidate: str | None = None,
        route_continuity_score: float = 0.0,
    ) -> dict[str, Any]:
        candidates = deterministic_repair_candidates(
            reaction_smiles,
            contextual_candidate=contextual_candidate,
            route_continuity_score=route_continuity_score,
        )
        return {
            "input_reaction": reaction_smiles,
            "candidates": [candidate.__dict__ for candidate in candidates],
            "accepted_candidate": next(
                (candidate.__dict__ for candidate in candidates if candidate.accepted),
                None,
            ),
            "contract": {
                "deterministic_only": True,
                "strict_post_repair_validation": True,
                "original_preserved": True,
            },
        }

    def score_condition_anomaly(
        self,
        reaction_smiles: str,
        *,
        temperature_c: float | None = None,
        time_h: float | None = None,
    ) -> dict[str, Any]:
        artifact = self.settings.model_dir / "condition_anomaly" / "robust_family_stats.json"
        if not artifact.exists():
            raise FileNotFoundError("Condition anomaly model has not been built.")
        family, _ = self.contextual_inference._reaction_family(reaction_smiles)
        result = ConditionAnomalyModel.load(artifact).score(
            reaction_family=family,
            temperature_c=temperature_c,
            time_h=time_h,
        )
        return {
            "reaction_smiles": reaction_smiles,
            "reaction_family": family,
            **result,
            "interpretation": "Corpus-relative anomaly signal; not an experimental feasibility judgment.",
        }

    @staticmethod
    def score_route_quality(components: dict[str, float]) -> dict[str, Any]:
        return RouteQualityScorer.score(components)

    def validate_product_two(self) -> dict[str, Any]:
        report_path = self.settings.reports_dir / "product_two_scientific_acceptance.json"
        return ScientificAcceptanceValidator(self.settings).write(report_path)

    def lock_product_two(self, release_id: str = "v2.0.0") -> dict[str, Any]:
        report_path = self.settings.reports_dir / "product_two_scientific_acceptance.json"
        if not report_path.exists():
            acceptance = self.validate_product_two()
        else:
            acceptance = json.loads(report_path.read_text(encoding="utf-8"))
        return lock_product_two_release(
            self.settings,
            self.registry,
            acceptance,
            release_id=release_id,
        )
