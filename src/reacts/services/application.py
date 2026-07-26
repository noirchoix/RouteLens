from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reacts.data.canonical import CanonicalBuildConfig, CanonicalBuilder
from reacts.data.canonical_v2 import ContextualBuildConfig, ContextualCanonicalBuilder
from reacts.data.split_governance import ProductTwoSplitRebuilder
from reacts.data.source import ArtifactSource
from reacts.ml.anomaly import ConditionAnomalyModel, RouteQualityScorer
from reacts.chemistry.repair import deterministic_repair_candidates
from reacts.ml.inference import ContextualInferenceService, InferenceService
from reacts.ml.registry import Registry
from reacts.ml.specialists import SpecialistTrainer
from reacts.ml.training import Trainer, TrainingConfig
from reacts.mapping.benchmark import MappingBenchmarkConfig, benchmark_mapper
from reacts.mapping.derivation import DerivationConfig, ReactionCentreDeriver
from reacts.mapping.runner import MappingRunConfig, ResumableMappingRunner
from reacts.retrieval.contextual_index import ContextualFingerprintIndexBuilder, ContextualIndexBuildConfig
from reacts.retrieval.fingerprint_index import FingerprintIndexBuilder, IndexBuildConfig
from reacts.retrieval.route_index import RouteEmbeddingIndexBuilder, RouteIndexBuildConfig
from reacts.science.baseline import freeze_product_one_baseline
from reacts.science.release import lock_product_two_release
from reacts.services.jobs import JobManager
from reacts.services.routes import RouteRepository
from reacts.settings import Settings
from reacts.validation.acceptance import ScientificAcceptanceValidator


class Application:
    def __init__(self, settings: Settings, *, read_only_registry: bool = False):
        self.settings = settings
        self.registry = Registry(settings.registry_db, read_only=read_only_registry)
        self.jobs = JobManager(self.registry)
        self.inference = InferenceService(self.registry, settings.index_dir)
        self.contextual_inference = ContextualInferenceService(
            self.registry,
            settings.index_v2_dir,
            in_domain_threshold=settings.evidence_in_domain_threshold,
            weak_threshold=settings.evidence_weak_threshold,
            abstention_threshold=settings.inference_abstention_threshold,
        )
        self.routes = RouteRepository(settings.canonical_dir)
        self.contextual_routes = RouteRepository(settings.canonical_v2_dir)

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
