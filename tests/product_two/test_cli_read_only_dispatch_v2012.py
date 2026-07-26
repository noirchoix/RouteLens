from __future__ import annotations

from pathlib import Path

from reacts import cli


def test_validate_product_two_constructs_read_only_application(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    constructed: list[bool] = []

    class FakeApplication:
        def __init__(self, settings, *, read_only_registry: bool = False):
            constructed.append(read_only_registry)

        def validate_product_two(self):
            return {"strict_pass": True}

    monkeypatch.setattr(cli, "Application", FakeApplication)

    exit_code = cli.main([
        "--project-root", str(tmp_path), "validate-product-two"
    ])

    assert exit_code == 0
    assert constructed == [True]
    assert '"strict_pass": true' in capsys.readouterr().out


def test_mutating_command_keeps_writable_application(
    tmp_path: Path, monkeypatch
) -> None:
    constructed: list[bool] = []

    class FakeApplication:
        def __init__(self, settings, *, read_only_registry: bool = False):
            constructed.append(read_only_registry)

        def build_anomaly_model(self):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "Application", FakeApplication)

    exit_code = cli.main([
        "--project-root", str(tmp_path), "build-anomaly-model"
    ])

    assert exit_code == 0
    assert constructed == [False]


def test_release_defaults_match_v2012() -> None:
    parser = cli.build_parser()

    lock_args = parser.parse_args(["lock-product-two"])
    workflow_args = parser.parse_args(["product-two"])

    assert lock_args.release_id == "v2.0.12"
    assert workflow_args.release_id == "v2.0.12"
