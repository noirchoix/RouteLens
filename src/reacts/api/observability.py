from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter("reacts_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("reacts_http_request_duration_seconds", "HTTP request latency", ["method", "path"])
INFERENCE_REQUESTS = Counter("inference_requests_total", "Inference requests", ["endpoint"])
INFERENCE_FAILURES = Counter("inference_failures_total", "Inference failures", ["endpoint", "error_code"])
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Inference latency", ["endpoint"])
RETRIEVAL_LATENCY = Histogram("retrieval_latency_seconds", "Retrieval latency", ["endpoint"])
ARTIFACT_VERIFICATION_FAILURES = Counter(
    "artifact_verification_failures_total", "Artifact verification failures", ["reason_code"]
)
MODEL_LOAD_FAILURES = Counter("model_load_failures_total", "Model load failures", ["task"])
READINESS_STATE = Gauge("readiness_state", "Service readiness: 1 ready, 0 not ready")
ACTIVE_MODEL_INFO = Gauge("active_model_info", "Active model information", ["task", "model_id", "stage"])

_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


class _RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Enforce the configured body limit even for streamed/chunked requests."""

    def __init__(self, app: Any, *, max_bytes: int):
        self.app = app
        self.max_bytes = max(1, int(max_bytes))

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        total = 0
        response_started = False

        async def limited_receive():
            nonlocal total
            message = await receive()
            if message.get("type") == "http.request":
                total += len(message.get("body") or b"")
                if total > self.max_bytes:
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            headers = {
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in scope.get("headers") or []
            }
            request_id = headers.get("x-request-id") or uuid.uuid4().hex
            response = _error(
                413,
                "request_too_large",
                "Request body exceeds the configured limit.",
                request_id,
            )
            await response(scope, receive, send)


def _error(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={"X-Request-ID": request_id},
    )


async def telemetry_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    application = request.app.state.application
    settings = application.settings

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_request_bytes:
                response = _error(413, "request_too_large", "Request body exceeds the configured limit.", request_id)
                REQUESTS.labels(request.method, request.url.path, "413").inc()
                return response
        except ValueError:
            response = _error(400, "invalid_content_length", "Invalid Content-Length header.", request_id)
            REQUESTS.labels(request.method, request.url.path, "400").inc()
            return response

    remote = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _RATE_BUCKETS[remote]
    while bucket and now - bucket[0] >= 60.0:
        bucket.popleft()
    if settings.rate_limit_requests_per_minute > 0 and len(bucket) >= settings.rate_limit_requests_per_minute:
        response = _error(429, "rate_limit_exceeded", "Request rate limit exceeded.", request_id)
        REQUESTS.labels(request.method, request.url.path, "429").inc()
        return response
    bucket.append(now)

    semaphore: asyncio.Semaphore = request.app.state.request_semaphore
    try:
        async with semaphore:
            response = await asyncio.wait_for(call_next(request), timeout=settings.request_timeout_seconds)
    except TimeoutError:
        response = _error(504, "request_timeout", "Request exceeded the configured timeout.", request_id)
    except Exception:
        REQUESTS.labels(request.method, request.url.path, "500").inc()
        LATENCY.labels(request.method, request.url.path).observe(time.perf_counter() - started)
        raise
    REQUESTS.labels(request.method, request.url.path, str(response.status_code)).inc()
    LATENCY.labels(request.method, request.url.path).observe(time.perf_counter() - started)
    response.headers["X-Request-ID"] = request_id
    return response


def record_runtime_metrics(application: Any) -> None:
    readiness = application.readiness()
    READINESS_STATE.set(1 if readiness.get("ready") else 0)
    ACTIVE_MODEL_INFO.clear()
    for capability in application.model_capabilities():
        ACTIVE_MODEL_INFO.labels(
            str(capability.get("task")),
            str(capability.get("model_id")),
            str(capability.get("stage")),
        ).set(1)
    if application.artifact_runtime.configured and not application.artifact_runtime.validation.get("pass", False):
        ARTIFACT_VERIFICATION_FAILURES.labels(
            application.artifact_runtime.reason_code or "unknown"
        ).inc()
    if application.artifact_runtime.reason_code == "artifact_warmup_failed":
        MODEL_LOAD_FAILURES.labels(
            str(application.artifact_runtime.warmup.get("failed_task") or "unknown")
        ).inc()


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
