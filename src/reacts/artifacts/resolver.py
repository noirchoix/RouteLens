from __future__ import annotations

import ntpath
import os
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from reacts.artifacts.bundle import ArtifactBundleValidator, extract_zip_safely
from reacts.artifacts.errors import (
    ArtifactCompatibilityError,
    ArtifactContractError,
    ArtifactIntegrityError,
    ArtifactUnavailableError,
)


class ArtifactResolver:
    """Resolve an exact artifact release into a verified immutable cache directory."""

    def __init__(
        self,
        *,
        uri: str | None,
        release: str,
        cache_dir: Path,
        verify_sha256: bool = True,
        offline_mode: bool = False,
        service_version: str = "2.1.0",
        lock_timeout_seconds: int = 120,
    ):
        self.uri = uri
        self.release = release
        self.cache_dir = Path(cache_dir).resolve()
        self.verify_sha256 = bool(verify_sha256)
        self.offline_mode = bool(offline_mode)
        self.service_version = service_version
        self.lock_timeout_seconds = lock_timeout_seconds

    @property
    def installed_root(self) -> Path:
        return self.cache_dir / self.release

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / f".{self.release}.lock"
        deadline = time.monotonic() + self.lock_timeout_seconds
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ArtifactUnavailableError(
                        f"Timed out waiting for artifact cache lock: {lock_path}",
                        reason_code="artifact_cache_lock_timeout",
                    )
                time.sleep(0.1)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            descriptor = None
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    def _validate(self, root: Path) -> dict:
        validation = ArtifactBundleValidator(root).validate(service_version=self.service_version)
        if validation["pass"]:
            return validation
        failures = [str(value) for value in validation.get("failures") or []]
        message = f"Artifact release {self.release} failed verification: {failures}"
        lowered = " ".join(failures).lower()
        if any(token in lowered for token in ("checksum mismatch", "hash mismatch", "sha256sums")):
            raise ArtifactIntegrityError(message)
        if any(
            token in lowered
            for token in (
                "outside",
                "scikit-learn",
                "environment mismatch",
                "training split",
                "not bound",
                "service compatibility",
            )
        ):
            raise ArtifactCompatibilityError(message)
        raise ArtifactContractError(message)

    def _source_value(self) -> str:
        if not self.uri:
            raise ArtifactUnavailableError(
                "REACTS_ARTIFACT_URI is required when the exact release is not cached.",
                reason_code="artifact_uri_missing",
            )
        return self.uri.format(release=self.release)

    @staticmethod
    def _single_bundle_root(extracted: Path, release: str) -> Path:
        direct = extracted / release
        if direct.is_dir():
            return direct
        if (extracted / "artifact_manifest.json").is_file():
            return extracted
        children = [path for path in extracted.iterdir() if path.is_dir()]
        if len(children) == 1 and (children[0] / "artifact_manifest.json").is_file():
            return children[0]
        raise ArtifactContractError("Downloaded artifact does not contain one unambiguous bundle root.")

    @staticmethod
    def _is_windows_absolute_path(value: str) -> bool:
        """Return True for absolute Windows drive or UNC filesystem paths.

        ``urllib.parse.urlparse`` treats the drive letter in ``C:\\path`` or
        ``C:/path`` as a URI scheme. Detect filesystem paths before URI parsing
        so local artifact directories work consistently on Windows.
        """
        drive, tail = ntpath.splitdrive(value)
        if drive and tail.startswith(("\\", "/")):
            return True
        return value.startswith(("\\\\", "//"))

    def _materialize_source(self, temporary: Path) -> Path:
        source_value = self._source_value()
        if self._is_windows_absolute_path(source_value):
            source = Path(source_value)
            parsed = None
        else:
            parsed = urllib.parse.urlparse(source_value)
            source = None

        if parsed is not None and parsed.scheme in {"http", "https"}:
            archive = temporary / f"{self.release}.zip"
            try:
                with urllib.request.urlopen(source_value, timeout=60) as response, archive.open("wb") as output:
                    shutil.copyfileobj(response, output)
            except Exception as exc:
                raise ArtifactUnavailableError(
                    f"Unable to download exact artifact release from {source_value}: {exc}"
                ) from exc
            extracted = temporary / "extracted"
            extracted.mkdir()
            extract_zip_safely(archive, extracted)
            return self._single_bundle_root(extracted, self.release)

        if parsed is not None:
            if parsed.scheme == "file":
                source = Path(urllib.request.url2pathname(parsed.path))
            elif parsed.scheme:
                raise ArtifactUnavailableError(
                    f"Unsupported artifact URI scheme: {parsed.scheme}",
                    reason_code="artifact_uri_scheme_unsupported",
                )
            else:
                source = Path(source_value)
        assert source is not None
        source = source.expanduser().resolve()
        if source.is_dir():
            if (source / "artifact_manifest.json").is_file():
                return source
            candidate = source / self.release
            if candidate.is_dir():
                return candidate
            archive_candidate = source / f"{self.release}.zip"
            if archive_candidate.is_file():
                source = archive_candidate
            else:
                raise ArtifactUnavailableError(f"Exact artifact release was not found under {source}.")
        if source.is_file() and source.suffix.lower() == ".zip":
            extracted = temporary / "extracted"
            extracted.mkdir()
            extract_zip_safely(source, extracted)
            return self._single_bundle_root(extracted, self.release)
        raise ArtifactUnavailableError(f"Artifact source is not a bundle directory or ZIP archive: {source}")

    def resolve(self) -> tuple[Path, dict, bool]:
        installed = self.installed_root
        if installed.is_dir():
            validation = self._validate(installed)
            return installed, validation, True
        if self.offline_mode:
            raise ArtifactUnavailableError(
                f"Offline mode requires the exact cached release at {installed}.",
                reason_code="artifact_not_cached_offline",
            )

        with self._lock():
            if installed.is_dir():
                validation = self._validate(installed)
                return installed, validation, True
            with tempfile.TemporaryDirectory(prefix=f".{self.release}-", dir=self.cache_dir) as raw_temp:
                temporary = Path(raw_temp)
                source_root = self._materialize_source(temporary)
                validation = self._validate(source_root)
                staging = self.cache_dir / f".{self.release}.installing-{os.getpid()}"
                if staging.exists():
                    shutil.rmtree(staging)
                try:
                    shutil.copytree(source_root, staging)
                    self._validate(staging)
                    try:
                        os.chmod(staging, 0o750)
                        for path in staging.rglob("*"):
                            if path.is_file():
                                os.chmod(path, 0o440)
                            elif path.is_dir():
                                os.chmod(path, 0o750)
                    except OSError:
                        # Windows permissions are best-effort; immutability is still enforced by read-only registry use.
                        pass
                    os.replace(staging, installed)
                finally:
                    if staging.exists():
                        try:
                            for path in staging.rglob("*"):
                                if path.is_file():
                                    os.chmod(path, 0o600)
                                elif path.is_dir():
                                    os.chmod(path, 0o700)
                            os.chmod(staging, 0o700)
                        except OSError:
                            pass
                        shutil.rmtree(staging, ignore_errors=True)
                return installed, validation, False
