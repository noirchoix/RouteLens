from __future__ import annotations

import urllib.parse

import pytest

from reacts.artifacts.resolver import ArtifactResolver


@pytest.mark.parametrize(
    "value",
    [
        r"C:\\artifacts\\product-two-artifacts-v2.0.12",
        "C:/artifacts/product-two-artifacts-v2.0.12",
        r"\\server\share\product-two-artifacts-v2.0.12",
        "//server/share/product-two-artifacts-v2.0.12",
    ],
)
def test_absolute_windows_artifact_paths_are_detected_before_uri_parsing(value: str) -> None:
    assert ArtifactResolver._is_windows_absolute_path(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "C:relative-artifacts",
        "artifacts/product-two-artifacts-v2.0.12",
        "file:///C:/artifacts/product-two-artifacts-v2.0.12",
        "https://artifacts.example/product-two-artifacts-v2.0.12.zip",
        "s3://bucket/product-two-artifacts-v2.0.12.zip",
    ],
)
def test_relative_paths_and_real_uris_are_not_classified_as_windows_absolute_paths(value: str) -> None:
    assert ArtifactResolver._is_windows_absolute_path(value) is False


def test_windows_drive_letter_would_otherwise_be_misclassified_as_uri_scheme() -> None:
    value = r"C:\\artifacts\\product-two-artifacts-v2.0.12"
    assert urllib.parse.urlparse(value).scheme.lower() == "c"
    assert ArtifactResolver._is_windows_absolute_path(value) is True
