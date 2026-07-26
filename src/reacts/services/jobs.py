from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from reacts.ml.registry import Registry


class JobManager:
    def __init__(self, registry: Registry, max_workers: int = 2):
        self.registry = registry
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="reacts-job")

    def submit(self, job_type: str, detail: dict[str, Any], fn: Callable[[], dict[str, Any]]) -> str:
        job_id = self.registry.create_job(job_type, detail)

        def runner() -> None:
            self.registry.update_job(job_id, "running", detail)
            try:
                result = fn()
                self.registry.update_job(job_id, "completed", result)
            except Exception as exc:  # pragma: no cover - integration behavior
                self.registry.update_job(
                    job_id,
                    "failed",
                    {"error": str(exc), "traceback": traceback.format_exc(limit=20)},
                )

        self.executor.submit(runner)
        return job_id
