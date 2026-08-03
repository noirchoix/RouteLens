from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any


def _request(url: str, *, payload: dict[str, Any] | None = None, api_key: str | None = None) -> tuple[int, float, dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, time.perf_counter() - started, body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw": raw}
        return exc.code, time.perf_counter() - started, body


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[position]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a ready Product Two artifact-backed runtime.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--reaction", default="CCO>>CC=O")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--api-key")
    parser.add_argument("--output")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    ready_status, ready_latency, ready = _request(f"{base}/ready", api_key=args.api_key)
    if ready_status != 200 or not ready.get("ready"):
        print(json.dumps({"error": "service_not_ready", "status": ready_status, "payload": ready}, indent=2))
        return 2

    payload = {
        "reaction_smiles": args.reaction,
        "tasks": [],
        "include_evidence": True,
        "evidence_k": 5,
    }

    def run_once(_: int) -> tuple[int, float]:
        status, latency, _ = _request(
            f"{base}/api/v2/inference/contextual",
            payload=payload,
            api_key=args.api_key,
        )
        return status, latency

    results: list[tuple[int, float]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(run_once, index) for index in range(max(1, args.iterations))]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started

    latencies = [latency for status, latency in results if status == 200]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base,
        "artifact_release": ready.get("artifact_release"),
        "iterations": len(results),
        "concurrency": args.concurrency,
        "successes": sum(status == 200 for status, _ in results),
        "failures": sum(status != 200 for status, _ in results),
        "wall_seconds": elapsed,
        "throughput_requests_per_second": len(results) / elapsed if elapsed else 0.0,
        "ready_latency_ms": ready_latency * 1000,
        "latency_ms": {
            "mean": statistics.fmean(latencies) * 1000 if latencies else 0.0,
            "p50": _percentile(latencies, 0.50) * 1000,
            "p95": _percentile(latencies, 0.95) * 1000,
            "max": max(latencies) * 1000 if latencies else 0.0,
        },
        "statuses": {str(code): sum(status == code for status, _ in results) for code in sorted({s for s, _ in results})},
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
