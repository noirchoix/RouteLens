from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_log_path(project_root: Path, stamp: str) -> Path:
    return project_root / "reports" / "test_runs" / f"pytest_{stamp}.txt"


def _build_command(pytest_args: Sequence[str]) -> list[str]:
    # Ignore pyproject addopts=-q so the saved report always contains full test names,
    # long tracebacks, captured output, warnings, and the complete short summary.
    return [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-vv",
        "-ra",
        "--tb=long",
        "--show-capture=all",
        "--color=no",
        *pytest_args,
    ]


def _format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def run_logged_pytest(
    *,
    project_root: Path,
    log_path: Path,
    pytest_args: Sequence[str],
    write_latest: bool = True,
) -> int:
    project_root = project_root.resolve()
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(pytest_args)
    started_at = datetime.now(timezone.utc)

    environment = os.environ.copy()
    source_dir = project_root / "src"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_dir)
        if not existing_pythonpath
        else str(source_dir) + os.pathsep + existing_pythonpath
    )
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment["PY_COLORS"] = "0"
    environment["NO_COLOR"] = "1"

    header = [
        "RouteLens pytest full-output capture",
        f"started_utc: {started_at.isoformat()}",
        f"project_root: {project_root}",
        f"python_executable: {sys.executable}",
        f"python_version: {platform.python_version()}",
        f"platform: {platform.platform()}",
        f"command: {_format_command(command)}",
        "=" * 100,
        "",
    ]

    return_code = 1
    process: subprocess.Popen[str] | None = None
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        for line in header:
            print(line)
            log.write(line + "\n")
        log.flush()

        try:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for output_line in process.stdout:
                print(output_line, end="", flush=True)
                log.write(output_line)
                log.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            return_code = 130
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            message = "\nTest run interrupted by user.\n"
            print(message, end="", flush=True)
            log.write(message)
        finally:
            finished_at = datetime.now(timezone.utc)
            footer = [
                "",
                "=" * 100,
                f"finished_utc: {finished_at.isoformat()}",
                f"elapsed_seconds: {(finished_at - started_at).total_seconds():.3f}",
                f"pytest_exit_code: {return_code}",
                f"full_log: {log_path}",
            ]
            for line in footer:
                print(line)
                log.write(line + "\n")
            log.flush()

    if write_latest:
        latest = log_path.parent / "pytest_latest.txt"
        if latest.resolve() != log_path:
            shutil.copyfile(log_path, latest)
        print(f"latest_log: {latest.resolve()}")

    return return_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pytest while streaming and saving the complete terminal output to a text file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output file. Defaults to reports/test_runs/pytest_<UTC timestamp>.txt.",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not update reports/test_runs/pytest_latest.txt.",
    )
    args, pytest_args = parser.parse_known_args(argv)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    project_root = _project_root()
    stamp = _timestamp()
    log_path = args.output or _default_log_path(project_root, stamp)
    if not log_path.is_absolute():
        log_path = project_root / log_path

    return run_logged_pytest(
        project_root=project_root,
        log_path=log_path,
        pytest_args=pytest_args,
        write_latest=not args.no_latest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
