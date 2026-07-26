from __future__ import annotations

import importlib.metadata
import logging
import platform
from dataclasses import dataclass, asdict
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MappingPreflight:
    backend_requested: str
    primary_backend: str
    fallback_backend: str | None
    rxnmapper_available: bool
    rxnmapper_version: str | None
    setuptools_version: str | None
    device: str
    initialization_status: str
    initialization_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_mapping_backend(
    requested: str,
    *,
    fallback_backend: str | None = "mcs",
    allow_auto_fallback: bool = False,
) -> tuple[MappingPreflight, Any | None]:
    requested = requested.lower().strip()
    if requested not in {"auto", "rxnmapper", "mcs", "mcs_fallback"}:
        raise ValueError(f"Unsupported mapping backend: {requested}")

    rxnmapper_version = _version("rxnmapper")
    setuptools_version = _version("setuptools")
    device = "cpu"
    mapper = None
    init_error: str | None = None
    if requested in {"auto", "rxnmapper"}:
        try:
            from rxnmapper import RXNMapper  # type: ignore

            mapper = RXNMapper()
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        except Exception as exc:  # pragma: no cover - exercised with dependency-specific environments
            init_error = f"{type(exc).__name__}: {exc}"

    if mapper is not None:
        primary = "rxnmapper"
        status = "successful"
    elif requested == "rxnmapper":
        raise RuntimeError(f"RXNMapper initialization failed: {init_error or 'unknown error'}")
    elif requested == "auto":
        if not allow_auto_fallback:
            raise RuntimeError(
                "RXNMapper is unavailable in strict auto mode. "
                f"Initialization error: {init_error or 'unknown error'}. "
                "Use --allow-auto-fallback to run an explicitly labelled MCS baseline."
            )
        primary = "mcs_fallback"
        fallback_backend = None
        status = "fallback_selected"
    else:
        primary = "mcs_fallback"
        fallback_backend = None
        status = "explicit_mcs"

    report = MappingPreflight(
        backend_requested=requested,
        primary_backend=primary,
        fallback_backend=("mcs_fallback" if fallback_backend in {"mcs", "mcs_fallback"} and primary == "rxnmapper" else None),
        rxnmapper_available=mapper is not None,
        rxnmapper_version=rxnmapper_version,
        setuptools_version=setuptools_version,
        device=device,
        initialization_status=status,
        initialization_error=init_error,
    )
    LOGGER.info("Requested backend: %s", report.backend_requested)
    LOGGER.info("Resolved primary backend: %s", report.primary_backend)
    LOGGER.info("Fallback backend: %s", report.fallback_backend or "none")
    LOGGER.info("RXNMapper initialization: %s", report.initialization_status)
    LOGGER.info("Device: %s", report.device)
    return report, mapper
