from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter("reacts_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("reacts_http_request_duration_seconds", "HTTP request latency", ["method", "path"])


async def telemetry_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        REQUESTS.labels(request.method, request.url.path, "500").inc()
        LATENCY.labels(request.method, request.url.path).observe(time.perf_counter() - started)
        raise
    REQUESTS.labels(request.method, request.url.path, str(response.status_code)).inc()
    LATENCY.labels(request.method, request.url.path).observe(time.perf_counter() - started)
    response.headers["X-Request-ID"] = request_id
    return response


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
