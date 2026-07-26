from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any

SCIKIT_LEARN_PIN = "1.9.0"


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "scikit_learn": _version("scikit-learn"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "joblib": _version("joblib"),
        "rdkit": _version("rdkit"),
    }


def validate_runtime_environment(training_environment: dict[str, Any] | None) -> dict[str, Any]:
    runtime = runtime_environment()
    expected = (training_environment or {}).get("scikit_learn")
    actual = runtime.get("scikit_learn")
    return {
        "expected_scikit_learn": expected,
        "runtime_scikit_learn": actual,
        "project_pin": SCIKIT_LEARN_PIN,
        "training_environment_present": bool(training_environment),
        "pin_satisfied": actual == SCIKIT_LEARN_PIN,
        "training_runtime_match": bool(expected) and expected == actual,
        "pass": actual == SCIKIT_LEARN_PIN and bool(expected) and expected == actual,
    }
