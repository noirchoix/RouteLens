# Full pytest output capture

Use the repository logging wrapper when terminal output is too long or truncated:

```bash
python scripts/run_pytest_logged.py
```

The command streams the test run to the terminal and writes the complete output to:

```text
reports/test_runs/pytest_<UTC timestamp>.txt
reports/test_runs/pytest_latest.txt
```

The wrapper deliberately disables the repository's quiet pytest option and runs with verbose test names, long tracebacks, captured output, warnings, and the complete short summary.

Pass normal pytest selectors and options after the wrapper options:

```bash
python scripts/run_pytest_logged.py tests/product_two/test_artifact_runtime_v210.py
python scripts/run_pytest_logged.py tests/product_two/test_artifact_runtime_v210.py -x
python scripts/run_pytest_logged.py -k artifact -vv
```

Write to a specific text file:

```bash
python scripts/run_pytest_logged.py \
  --output reports/test_runs/pytest_windows_full.txt
```

The wrapper exits with pytest's exit code, so it remains suitable for local automation and CI diagnostics.

The equivalent Make target is:

```bash
make test-log
```
