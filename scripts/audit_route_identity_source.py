from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from reacts.storage.tabular import iter_dataset


def _normalize(value: object) -> str:
    return str(value or "").strip().replace("\r\n", "\n").replace("\r", "\n")


def audit(canonical_dir: Path) -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    for chunk in iter_dataset(
        canonical_dir,
        "routes",
        columns=["route_uid", "route_id", "multistep_reaction_text", "step_count"],
    ):
        routes.extend(chunk.to_dict(orient="records"))
    counts = Counter(str(row["route_id"]) for row in routes)
    duplicated = {route_id for route_id, count in counts.items() if count > 1}

    steps_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in iter_dataset(
        canonical_dir,
        "steps",
        columns=["step_id", "route_id", "step_index", "raw_reaction_text"],
    ):
        for row in chunk.to_dict(orient="records"):
            route_id = str(row["route_id"])
            if route_id in duplicated:
                steps_by_route[route_id].append(row)

    assignment_failures: list[dict[str, Any]] = []
    assigned = 0
    for route_id in sorted(duplicated):
        route_rows = [row for row in routes if str(row["route_id"]) == route_id]
        source_steps = steps_by_route[route_id]
        by_exact: dict[tuple[int, str], deque[dict[str, Any]]] = defaultdict(deque)
        used: set[int] = set()
        for ordinal, row in enumerate(source_steps):
            item = {
                "ordinal": ordinal,
                "step_index": int(row["step_index"]),
                "text": _normalize(row["raw_reaction_text"]),
            }
            by_exact[(item["step_index"], item["text"])].append(item)
        for route in route_rows:
            lines = [line.strip() for line in _normalize(route["multistep_reaction_text"]).split("\n") if line.strip()]
            if len(lines) != int(route["step_count"]):
                assignment_failures.append({"route_id": route_id, "reason": "line_count_mismatch"})
                continue
            for index, line in enumerate(lines):
                queue = by_exact[(index, line)]
                while queue and queue[0]["ordinal"] in used:
                    queue.popleft()
                if not queue:
                    assignment_failures.append({"route_id": route_id, "step_index": index, "reason": "no_exact_step"})
                    continue
                used.add(queue[0]["ordinal"])
                assigned += 1
        if len(used) != len(source_steps):
            assignment_failures.append(
                {"route_id": route_id, "reason": "unassigned_source_steps", "count": len(source_steps) - len(used)}
            )

    route_rows = len(routes)
    unique_sources = len(counts)
    return {
        "routes_total": route_rows,
        "unique_source_route_id": unique_sources,
        "duplicate_source_route_groups": len(duplicated),
        "preserved_conflicting_variant_rows": route_rows - unique_sources,
        "duplicate_group_steps_assigned": assigned,
        "assignment_failures": assignment_failures,
        "pass": not assignment_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", type=Path, default=Path("data/canonical"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.canonical_dir)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
