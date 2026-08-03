from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reacts.chemistry.reactions import reaction_fingerprint
from reacts.science.hashing import hash_dataset_columns, sha256_file
from reacts.storage.tabular import iter_dataset


@dataclass
class RouteIndexBuildConfig:
    canonical_dir: Path
    index_dir: Path
    n_bits: int = 2048


class RouteEmbeddingIndexBuilder:
    """Build deterministic route embeddings by aggregating contextual step fingerprints."""

    def __init__(self, config: RouteIndexBuildConfig):
        self.config = config

    def build(self) -> dict[str, Any]:
        split_manifest_path = self.config.canonical_dir / "split_manifest.json"
        if not split_manifest_path.exists():
            raise RuntimeError(
                "Product Two split governance is missing. Run `reacts --project-root . "
                "rebuild-product-two-splits` before rebuilding indexes."
            )
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if not split_manifest.get("invariants", {}).get("strict_pass"):
            raise RuntimeError("Product Two split invariants did not pass.")
        training_split_sha = hash_dataset_columns(
            self.config.canonical_dir,
            "steps",
            ["step_id", "patent_document_id", "reaction_signature", "split_component_id", "split"],
        )

        route_vectors: dict[str, np.ndarray] = {}
        route_meta: dict[str, dict[str, Any]] = {}
        columns = [
            "route_id",
            "route_instance_id",
            "source_route_id",
            "canonical_resolved_reaction_smiles",
            "eligible_retrieval_v2",
            "reaction_family",
            "contextual_quality_score",
            "patent_document_id",
            "split",
            "split_component_id",
        ]
        for chunk in iter_dataset(self.config.canonical_dir, "steps", columns=columns):
            eligible = chunk.loc[chunk["eligible_retrieval_v2"].fillna(False).astype(bool)]
            for row in eligible.to_dict(orient="records"):
                route_id = str(row.get("route_instance_id") or row["route_id"])
                rfp, pfp = reaction_fingerprint(
                    str(row["canonical_resolved_reaction_smiles"]), self.config.n_bits
                )
                vector = np.concatenate([rfp, pfp]).astype(np.float32)
                route_vectors[route_id] = route_vectors.get(
                    route_id, np.zeros_like(vector)
                ) + vector
                meta = route_meta.setdefault(
                    route_id,
                    {
                        "route_id": route_id,
                        "route_instance_id": row.get("route_instance_id", route_id),
                        "source_route_id": row.get("source_route_id", route_id),
                        "patent_document_id": row.get("patent_document_id"),
                        "split": row.get("split"),
                        "split_component_id": row.get("split_component_id"),
                        "step_count": 0,
                        "reaction_families": [],
                        "quality_sum": 0.0,
                    },
                )
                if meta["split"] != row.get("split"):
                    raise RuntimeError(f"Route {route_id} crosses Product Two splits.")
                if meta["split_component_id"] != row.get("split_component_id"):
                    raise RuntimeError(f"Route {route_id} crosses split components.")
                meta["step_count"] += 1
                family = row.get("reaction_family")
                if family and str(family) != "nan":
                    meta["reaction_families"].append(str(family))
                quality = row.get("contextual_quality_score")
                if quality is not None and str(quality) != "nan":
                    meta["quality_sum"] += float(quality)

        route_ids = sorted(route_vectors)
        matrix = (
            np.vstack(
                [
                    route_vectors[route_id] / max(route_meta[route_id]["step_count"], 1)
                    for route_id in route_ids
                ]
            )
            if route_ids
            else np.zeros((0, self.config.n_bits * 2), dtype=np.float32)
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        output_dir = self.config.index_dir / "routes"
        output_dir.mkdir(parents=True, exist_ok=True)
        vector_path = output_dir / "route_embeddings.npz"
        metadata_path = output_dir / "route_metadata.jsonl"
        np.savez_compressed(vector_path, vectors=matrix)
        metadata = []
        for route_id in route_ids:
            meta = route_meta[route_id]
            meta["reaction_families"] = sorted(set(meta["reaction_families"]))
            meta["quality_score"] = meta.pop("quality_sum") / max(meta["step_count"], 1)
            metadata.append(meta)
        metadata_path.write_text(
            "\n".join(json.dumps(item, default=str) for item in metadata)
            + ("\n" if metadata else ""),
            encoding="utf-8",
        )
        manifest = {
            "index_version": "route_aggregate_fingerprint_v2",
            "rows": len(route_ids),
            "dimensions": int(matrix.shape[1]),
            "vectors_sha256": sha256_file(vector_path),
            "metadata_sha256": sha256_file(metadata_path),
            "canonical_manifest_sha256": sha256_file(
                self.config.canonical_dir / "dataset_manifest.json"
            ),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "training_split_sha256": training_split_sha,
            "split_algorithm": split_manifest.get("algorithm"),
            "split_seed": split_manifest.get("seed"),
        }
        (output_dir / "route_index_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest


class RouteEmbeddingIndex:
    def __init__(self, index_dir: Path):
        candidate = Path(index_dir)
        root = candidate if (candidate / "route_index_manifest.json").exists() else candidate / "routes"
        self.root = root
        self.manifest = json.loads((root / "route_index_manifest.json").read_text(encoding="utf-8"))
        self.vectors = np.load(root / "route_embeddings.npz")["vectors"]
        self.metadata = [
            json.loads(line)
            for line in (root / "route_metadata.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]


    def get_route(self, route_id: str) -> dict[str, Any] | None:
        for record in self.metadata:
            if route_id in {
                str(record.get("route_id") or ""),
                str(record.get("route_instance_id") or ""),
                str(record.get("source_route_id") or ""),
            }:
                return {**record, "artifact_backed_summary": True}
        return None

    def search_reaction(self, reaction_smiles: str, k: int = 10) -> list[dict[str, Any]]:
        rfp, pfp = reaction_fingerprint(reaction_smiles, self.manifest.get("dimensions", 4096) // 2)
        vector = np.concatenate([rfp, pfp]).astype(np.float32)
        return self.search_vector(vector, k=k)

    def search_vector(self, vector: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
        norm = np.linalg.norm(vector)
        query = vector / norm if norm else vector
        scores = self.vectors @ query
        local_k = min(k, len(scores))
        if local_k == 0:
            return []
        indices = np.argpartition(scores, -local_k)[-local_k:]
        return [
            {**self.metadata[int(index)], "score": float(scores[int(index)])}
            for index in indices[np.argsort(scores[indices])[::-1]]
        ]
