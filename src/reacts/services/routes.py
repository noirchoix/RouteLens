from __future__ import annotations

from pathlib import Path
from typing import Any

from reacts.storage.tabular import iter_dataset


class RouteRepository:
    def __init__(self, canonical_dir: Path):
        self.canonical_dir = Path(canonical_dir)

    def get_route(self, route_id: str) -> dict[str, Any] | None:
        route: dict[str, Any] | None = None
        for chunk in iter_dataset(self.canonical_dir, "routes"):
            mask = chunk["route_id"].astype(str) == route_id
            if "route_uid" in chunk.columns:
                mask = mask | (chunk["route_uid"].astype(str) == route_id)
            found = chunk.loc[mask]
            if not found.empty:
                route = found.iloc[0].to_dict()
                break
        if route is None:
            return None
        steps: list[dict[str, Any]] = []
        for chunk in iter_dataset(self.canonical_dir, "steps"):
            found = chunk.loc[chunk["route_id"].astype(str) == str(route["route_id"])].sort_values("step_index")
            if not found.empty:
                steps.extend(found.to_dict(orient="records"))
        route["steps"] = steps
        route_edges: list[dict[str, Any]] = []
        try:
            for chunk in iter_dataset(self.canonical_dir, "route_edges"):
                found = chunk.loc[chunk["route_id"].astype(str) == str(route["route_id"])]
                if not found.empty:
                    route_edges.extend(found.to_dict(orient="records"))
        except FileNotFoundError:
            pass
        if route_edges:
            route["route_edges"] = route_edges
        return route

    def search(
        self,
        patent_document_id: str | None = None,
        parse_class: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if patent_document_id:
            for chunk in iter_dataset(self.canonical_dir, "routes"):
                subset = chunk.loc[chunk["patent_document_id"].astype(str) == patent_document_id]
                results.extend(subset.head(limit - len(results)).to_dict(orient="records"))
                if len(results) >= limit:
                    break
            return results
        for chunk in iter_dataset(self.canonical_dir, "steps"):
            subset = chunk
            if parse_class:
                column = "contextual_parse_failure_class" if "contextual_parse_failure_class" in subset.columns else "parse_failure_class"
                subset = subset.loc[subset[column].astype(str) == parse_class]
            results.extend(subset.head(limit - len(results)).to_dict(orient="records"))
            if len(results) >= limit:
                break
        return results
