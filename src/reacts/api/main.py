from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from reacts import __version__
from reacts.api.dependencies import require_api_key
from reacts.api.observability import (
    INFERENCE_FAILURES,
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS,
    RETRIEVAL_LATENCY,
    RequestBodyLimitMiddleware,
    metrics_response,
    record_runtime_metrics,
    telemetry_middleware,
)
from reacts.contracts import (
    BatchPredictionRequest,
    ConditionPredictionRequest,
    ConditionAnomalyRequest,
    InferenceResponse,
    JobResponse,
    ProductTwoBuildRequest,
    MappingRunRequest,
    DerivationRunRequest,
    ReactionInput,
    ReactionRetrievalRequest,
    RouteRetrievalRequest,
    ReleaseLockRequest,
    RepairRequest,
    RouteQualityRequest,
    TrainingRequest,
)
from reacts.services.application import Application
from reacts.settings import Settings


def create_app(
    settings: Settings | None = None,
    *,
    read_only_registry: bool = False,
) -> FastAPI:
    cfg = (settings or Settings()).resolve()
    application = Application(cfg, read_only_registry=read_only_registry)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            application.close()

    app = FastAPI(
        title="REACTS Product Two",
        version=__version__,
        description="Contextual reaction intelligence with scientific release governance and evidence-grounded inference.",
        lifespan=lifespan,
    )
    app.state.application = application
    app.state.request_semaphore = asyncio.Semaphore(max(1, cfg.max_concurrent_requests))
    app.middleware("http")(telemetry_middleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=cfg.trusted_host_list)
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=cfg.max_request_bytes)
    if cfg.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origin_list,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "X-REACTS-Allow-Experimental"],
        )
    record_runtime_metrics(application)

    static_dir = Path(__file__).resolve().parents[1] / "ui" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def require_ready() -> None:
        try:
            application.ensure_ready()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": application.artifact_runtime.reason_code or "service_not_ready",
                    "message": str(exc),
                    "artifact_release": application.artifact_runtime.artifact_release,
                },
            ) from exc

    def require_mutable_runtime() -> None:
        try:
            application.ensure_mutable()
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail={"code": "runtime_read_only", "message": str(exc)}) from exc

    def _truthy(value: str | None) -> bool:
        return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})

    def experimental_access(requested: bool, header: str | None) -> bool:
        if not requested:
            return False
        allowed = cfg.allow_experimental_models or _truthy(header)
        if not allowed:
            raise HTTPException(403, detail={"code": "experimental_access_denied"})
        return True

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return metrics_response()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "product_one": {
                "canonical_ready": (cfg.canonical_dir / "dataset_manifest.json").exists(),
                "index_ready": (cfg.index_dir / "index_manifest.json").exists(),
            },
            "product_two": {
                "context_ready": (cfg.canonical_v2_context_dir / "dataset_manifest.json").exists(),
                "mapping_ready": (cfg.mapping_v2_dir / "mapping_manifest.json").exists(),
                "derivation_ready": (cfg.derivation_v2_dir / "derivation_manifest.json").exists(),
                "canonical_ready": (cfg.canonical_v2_dir / "dataset_manifest.json").exists(),
                "index_ready": (cfg.index_v2_dir / "index_manifest.json").exists(),
                "baseline_frozen": (cfg.baseline_dir / "baseline_manifest.json").exists(),
            },
            "models": len(application.registry.list_models()),
            "runtime": application.readiness(),
        }

    @app.get("/ready")
    def ready() -> JSONResponse:
        payload = application.readiness()
        return JSONResponse(status_code=200 if payload.get("ready") else 503, content=payload)

    @app.get("/api/v2/artifacts", dependencies=[Depends(require_api_key)])
    def artifacts_v2() -> dict[str, Any]:
        return application.artifact_info()

    @app.get("/api/v2/models", dependencies=[Depends(require_api_key)])
    def models_v2() -> dict[str, Any]:
        return {
            "artifact_release": application.artifact_runtime.artifact_release,
            "models": application.model_capabilities(),
        }

    @app.get("/api/v1/datasets", dependencies=[Depends(require_api_key)])
    def datasets() -> dict[str, Any]:
        return {"active": application.dataset_manifest("v1")}

    @app.get("/api/v1/models", dependencies=[Depends(require_api_key)])
    def models() -> list[dict[str, Any]]:
        return application.registry.list_models()

    @app.post("/api/v1/inference/parse-quality", response_model=InferenceResponse, dependencies=[Depends(require_api_key)])
    def parse_quality(payload: ReactionInput) -> InferenceResponse:
        return application.inference.predict(payload.reaction_smiles, ["parse_validity"], payload.include_evidence, payload.evidence_k)

    @app.post("/api/v1/inference/conditions", response_model=InferenceResponse, dependencies=[Depends(require_api_key)])
    def conditions(payload: ConditionPredictionRequest) -> InferenceResponse:
        return application.inference.predict(payload.reaction_smiles, payload.tasks, payload.include_evidence, payload.evidence_k)

    @app.post("/api/v1/inference/batch", dependencies=[Depends(require_api_key)])
    def batch_v1(payload: BatchPredictionRequest) -> dict[str, Any]:
        if len(payload.reactions) > cfg.max_batch_rows:
            raise HTTPException(413, f"Batch exceeds max_batch_rows={cfg.max_batch_rows}")
        return {
            "rows": len(payload.reactions),
            "predictions": [
                application.inference.predict(reaction, payload.tasks, include_evidence=False).model_dump()
                for reaction in payload.reactions
            ],
        }

    @app.get("/api/v1/routes/{route_id}", dependencies=[Depends(require_api_key)])
    def route_v1(route_id: str) -> dict[str, Any]:
        result = application.routes.get_route(route_id)
        if result is None:
            raise HTTPException(404, "Route not found")
        return result

    @app.get("/api/v1/search", dependencies=[Depends(require_api_key)])
    def search_v1(
        patent_document_id: str | None = None,
        parse_class: str | None = None,
        limit: int = Query(default=25, ge=1, le=250),
    ) -> list[dict[str, Any]]:
        return application.routes.search(patent_document_id=patent_document_id, parse_class=parse_class, limit=limit)

    @app.post("/api/v1/jobs/train", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def train_v1(payload: TrainingRequest) -> JobResponse:
        require_mutable_runtime()
        trainer = application.trainer(payload.dataset_version, payload.max_rows)
        job_id = application.jobs.submit(
            "train",
            payload.model_dump(),
            lambda: trainer.train_many(payload.tasks, promote_validated=payload.request_promotion),
        )
        return JobResponse(job_id=job_id, status="queued", detail=payload.model_dump())

    @app.get("/api/v2/health")
    def health_v2() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "dataset_version": "uspto_multistep_contextual_v2",
            "baseline": "v1.0.0-baseline",
            "context_ready": (cfg.canonical_v2_context_dir / "dataset_manifest.json").exists(),
            "mapping_ready": (cfg.mapping_v2_dir / "mapping_manifest.json").exists(),
            "derivation_ready": (cfg.derivation_v2_dir / "derivation_manifest.json").exists(),
            "canonical_ready": (cfg.canonical_v2_dir / "dataset_manifest.json").exists(),
            "index_ready": (application.settings.index_v2_dir / "index_manifest.json").exists(),
            "runtime": application.readiness(),
        }

    @app.get("/api/v2/datasets", dependencies=[Depends(require_api_key)])
    def datasets_v2() -> dict[str, Any]:
        return {
            "baseline": application.dataset_manifest("v1"),
            "contextual": application.dataset_manifest("v2"),
        }

    @app.post("/api/v2/inference/contextual", response_model=InferenceResponse, dependencies=[Depends(require_api_key)])
    def contextual_inference(
        payload: ConditionPredictionRequest,
        x_reacts_allow_experimental: str | None = Header(default=None),
    ) -> InferenceResponse:
        require_ready()
        endpoint = "contextual"
        INFERENCE_REQUESTS.labels(endpoint).inc()
        started = time.perf_counter()
        try:
            return application.contextual_inference.predict(
                payload.reaction_smiles,
                payload.tasks,
                include_evidence=payload.include_evidence,
                evidence_k=payload.evidence_k,
                allow_experimental=experimental_access(
                    payload.allow_experimental, x_reacts_allow_experimental
                ),
            )
        except Exception:
            INFERENCE_FAILURES.labels(endpoint, "inference_failed").inc()
            raise
        finally:
            INFERENCE_LATENCY.labels(endpoint).observe(time.perf_counter() - started)

    @app.post("/api/v2/inference/repair", dependencies=[Depends(require_api_key)])
    def repair_v2(payload: RepairRequest) -> dict[str, Any]:
        return application.repair_reaction(
            payload.reaction_smiles,
            contextual_candidate=payload.contextual_candidate,
            route_continuity_score=payload.route_continuity_score,
        )

    @app.post("/api/v2/inference/anomaly", dependencies=[Depends(require_api_key)])
    def anomaly_v2(payload: ConditionAnomalyRequest) -> dict[str, Any]:
        try:
            return application.score_condition_anomaly(
                payload.reaction_smiles,
                temperature_c=payload.temperature_c,
                time_h=payload.time_h,
            )
        except FileNotFoundError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/v2/inference/route-quality", dependencies=[Depends(require_api_key)])
    def route_quality_v2(payload: RouteQualityRequest) -> dict[str, Any]:
        return application.score_route_quality(payload.model_dump())

    @app.post("/api/v2/inference/batch", dependencies=[Depends(require_api_key)])
    def contextual_batch(
        payload: BatchPredictionRequest,
        x_reacts_allow_experimental: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_ready()
        if len(payload.reactions) > cfg.inference_max_batch_rows:
            raise HTTPException(413, f"Batch exceeds inference_max_batch_rows={cfg.inference_max_batch_rows}")
        allow_experimental = experimental_access(payload.allow_experimental, x_reacts_allow_experimental)
        endpoint = "batch"
        INFERENCE_REQUESTS.labels(endpoint).inc()
        started = time.perf_counter()
        try:
            results = [
                application.contextual_inference.predict(
                    reaction,
                    payload.tasks,
                    include_evidence=payload.include_evidence,
                    evidence_k=payload.evidence_k,
                    allow_experimental=allow_experimental,
                ).model_dump()
                for reaction in payload.reactions
            ]
            return {
                "rows": len(results),
                "artifact_release": application.artifact_runtime.artifact_release,
                "results": results,
            }
        except Exception:
            INFERENCE_FAILURES.labels(endpoint, "inference_failed").inc()
            raise
        finally:
            INFERENCE_LATENCY.labels(endpoint).observe(time.perf_counter() - started)

    @app.post("/api/v2/retrieval/reactions", dependencies=[Depends(require_api_key)])
    def retrieve_reactions_v2(payload: ReactionRetrievalRequest) -> dict[str, Any]:
        require_ready()
        started = time.perf_counter()
        try:
            return application.retrieve_reactions(
                payload.reaction_smiles,
                k=payload.k,
                minimum_quality=payload.minimum_quality,
            )
        finally:
            RETRIEVAL_LATENCY.labels("reactions").observe(time.perf_counter() - started)

    @app.post("/api/v2/retrieval/routes", dependencies=[Depends(require_api_key)])
    def retrieve_routes_v2(payload: RouteRetrievalRequest) -> dict[str, Any]:
        require_ready()
        started = time.perf_counter()
        try:
            return application.retrieve_routes(payload.reaction_smiles, k=payload.k)
        finally:
            RETRIEVAL_LATENCY.labels("routes").observe(time.perf_counter() - started)

    @app.get("/api/v2/routes/{route_id}", dependencies=[Depends(require_api_key)])
    def route_v2(route_id: str) -> dict[str, Any]:
        result = application.get_contextual_route(route_id)
        if result is None:
            raise HTTPException(404, "Contextual route not found")
        return result

    @app.post("/api/v2/jobs/freeze-baseline", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def freeze_baseline() -> JobResponse:
        require_mutable_runtime()
        job_id = application.jobs.submit("freeze_baseline", {}, application.freeze_product_one)
        return JobResponse(job_id=job_id, status="queued", detail={"release_id": "v1.0.0-baseline"})

    @app.post("/api/v2/jobs/build-contextual", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def build_contextual(payload: ProductTwoBuildRequest) -> JobResponse:
        require_mutable_runtime()
        detail = payload.model_dump()
        job_id = application.jobs.submit(
            "build_contextual",
            detail,
            lambda: application.build_contextual_canonical(
                prefer_parquet=payload.prefer_parquet,
                resume=payload.resume,
                clean=payload.clean,
            ),
        )
        return JobResponse(job_id=job_id, status="queued", detail=detail)


    @app.post("/api/v2/jobs/map-reactions", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def map_reactions_v2(payload: MappingRunRequest) -> JobResponse:
        require_mutable_runtime()
        detail = payload.model_dump()
        job_id = application.jobs.submit(
            "map_reactions",
            detail,
            lambda: application.map_reactions(
                backend=payload.backend,
                fallback_backend=payload.fallback_backend,
                allow_auto_fallback=payload.allow_auto_fallback,
                batch_size=payload.batch_size,
                workers=payload.workers,
                prefetch_batches=payload.prefetch_batches,
                shard_size=payload.shard_size,
                rxnmapper_token_limit=payload.rxnmapper_token_limit,
                fallback_process_timeout_seconds=payload.fallback_process_timeout_seconds,
                resume=payload.resume,
                max_rows=payload.max_rows,
                prefer_parquet=payload.prefer_parquet,
            ),
        )
        return JobResponse(job_id=job_id, status="queued", detail=detail)

    @app.post("/api/v2/jobs/derive-reaction-centres", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def derive_reaction_centres_v2(payload: DerivationRunRequest) -> JobResponse:
        require_mutable_runtime()
        detail = payload.model_dump()
        job_id = application.jobs.submit(
            "derive_reaction_centres",
            detail,
            lambda: application.derive_reaction_centres(
                resume=payload.resume,
                include_mcs=payload.include_mcs,
                min_confidence=payload.min_confidence,
                max_rows=payload.max_rows,
                prefer_parquet=payload.prefer_parquet,
            ),
        )
        return JobResponse(job_id=job_id, status="queued", detail=detail)

    @app.post("/api/v2/jobs/build-index", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def build_contextual_index(max_rows: int | None = None) -> JobResponse:
        require_mutable_runtime()
        job_id = application.jobs.submit(
            "build_contextual_index",
            {"max_rows": max_rows},
            lambda: application.build_contextual_index(max_rows),
        )
        return JobResponse(job_id=job_id, status="queued", detail={"max_rows": max_rows})

    @app.post("/api/v2/jobs/train", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def train_v2(payload: TrainingRequest) -> JobResponse:
        require_mutable_runtime()
        classification = [task for task in payload.tasks if task in {"parse_failure_class", "repairability", "reaction_family"}]
        specialist = [task for task in payload.tasks if task not in classification]

        def run_training() -> dict[str, Any]:
            result: dict[str, Any] = {}
            if classification:
                result.update(
                    application.trainer(payload.dataset_version, payload.max_rows).train_many(
                        classification,
                        promote_validated=payload.request_promotion,
                    )
                )
            trainer = application.specialist_trainer(payload.max_rows)
            for task in specialist:
                result[task] = trainer.train(task, request_promotion=payload.request_promotion)
            return result

        job_id = application.jobs.submit("train_product_two", payload.model_dump(), run_training)
        return JobResponse(job_id=job_id, status="queued", detail=payload.model_dump())

    @app.post("/api/v2/jobs/validate", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def validate_v2() -> JobResponse:
        require_mutable_runtime()
        job_id = application.jobs.submit("validate_product_two", {}, application.validate_product_two)
        return JobResponse(job_id=job_id, status="queued", detail={})

    @app.post("/api/v2/jobs/lock-release", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def lock_release_v2(payload: ReleaseLockRequest) -> JobResponse:
        require_mutable_runtime()
        detail = payload.model_dump()
        job_id = application.jobs.submit(
            "lock_product_two_release",
            detail,
            lambda: application.lock_product_two(payload.release_id),
        )
        return JobResponse(job_id=job_id, status="queued", detail=detail)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    @app.get("/api/v2/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
    def job(job_id: str) -> JobResponse:
        result = application.registry.get_job(job_id)
        if result is None:
            raise HTTPException(404, "Job not found")
        return JobResponse(job_id=job_id, status=result["status"], detail=result["detail"])

    return app


def _module_registry_read_only() -> bool:
    value = os.environ.get("REACTS_API_READ_ONLY_REGISTRY", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


app = create_app(read_only_registry=_module_registry_read_only())
