from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reacts.services.application import Application  # noqa: E402
from reacts.settings import Settings  # noqa: E402


def main() -> int:
    report = Application(Settings(project_root=ROOT).resolve()).validate_product_two()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("strict_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
