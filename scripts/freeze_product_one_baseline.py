from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reacts.services.application import Application  # noqa: E402
from reacts.settings import Settings  # noqa: E402


def main() -> int:
    result = Application(Settings(project_root=ROOT).resolve()).freeze_product_one()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
