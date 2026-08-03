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
    """Disk-backed route retrieval over an immutable dense vector matrix.

    Artifact bundles published by Product Two v2.1.5 store the matrix as a
    standalone ``.npy`` file. NumPy can memory-map that format, which avoids
    materialising the complete route matrix during application warm-up or
    retrieval. Older local indexes that still contain ``route_embeddings.npz``
    remain readable for development and migration tests.
    """

    DEFAULT_SEARCH_CHUNK_ROWS = 2_048

    def __init__(self, index_dir: Path, *, preload_vectors: bool = False):
        candidate = Path(index_dir)
        root = candidate if (candidate / "route_index_manifest.json").exists() else candidate / "routes"
        self.root = root
        self.manifest = json.loads((root / "route_index_manifest.json").read_text(encoding="utf-8"))
        self.vector_path = root / str(self.manifest.get("vectors") or "route_embeddings.npz")
        self.metadata_path = root / str(self.manifest.get("metadata") or "route_metadata.jsonl")
        self._vectors: np.ndarray | np.memmap | None = None
        self._npz_handle: Any | None = None
        self._metadata_offsets: np.ndarray | None = None
        if preload_vectors:
            self._ensure_vectors()

    def close(self) -> None:
        vectors = self._vectors
        self._vectors = None
        if isinstance(vectors, np.memmap):
            mmap_handle = getattr(vectors, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        if self._npz_handle is not None:
            self._npz_handle.close()
            self._npz_handle = None

    def _validate_shape(self, vectors: np.ndarray) -> None:
        expected = (
            int(self.manifest.get("rows", 0)),
            int(self.manifest.get("dimensions", 0)),
        )
        actual = tuple(int(value) for value in vectors.shape)
        if actual != expected:
            raise RuntimeError(f"Route vector shape mismatch: expected={expected}, actual={actual}")
        if vectors.dtype != np.dtype(np.float32):
            raise RuntimeError(f"Route vector dtype mismatch: expected=float32, actual={vectors.dtype}")

    def _ensure_vectors(self) -> np.ndarray:
        if self._vectors is not None:
            return self._vectors
        if not self.vector_path.is_file():
            raise FileNotFoundError(self.vector_path)
        if self.vector_path.suffix.lower() == ".npy":
            vectors = np.load(self.vector_path, mmap_mode="r", allow_pickle=False)
        elif self.vector_path.suffix.lower() == ".npz":
            # Compatibility path for pre-v2.1.5 local indexes. Published bundles
            # are converted to mmap-capable NPY storage by the artifact publisher.
            self._npz_handle = np.load(self.vector_path, allow_pickle=False)
            vectors = self._npz_handle["vectors"]
        else:
            raise RuntimeError(f"Unsupported route vector storage: {self.vector_path.name}")
        self._validate_shape(vectors)
        self._vectors = vectors
        return vectors

    def storage_info(self, *, sample: bool = True) -> dict[str, Any]:
        vectors = self._ensure_vectors()
        if sample and len(vectors):
            # Touch one row only. This verifies that the mapped payload can be
            # read without allocating the full matrix.
            np.asarray(vectors[0:1]).sum(dtype=np.float64)
        return {
            "vectors": self.vector_path.name,
            "vectors_format": self.manifest.get("vectors_format")
            or ("npy_memmap_v1" if self.vector_path.suffix.lower() == ".npy" else "npz_dense_legacy"),
            "memory_mapped": isinstance(vectors, np.memmap),
            "rows": int(vectors.shape[0]),
            "dimensions": int(vectors.shape[1]),
            "dtype": str(vectors.dtype),
            "search_chunk_rows": int(
                self.manifest.get("search_chunk_rows") or self.DEFAULT_SEARCH_CHUNK_ROWS
            ),
        }

    def _offsets(self) -> np.ndarray:
        if self._metadata_offsets is not None:
            return self._metadata_offsets
        expected_rows = int(self.manifest.get("rows", 0))
        offsets = np.empty(expected_rows, dtype=np.uint64)
        count = 0
        position = 0
        with self.metadata_path.open("rb") as handle:
            for line in handle:
                if count >= expected_rows:
                    raise RuntimeError("Route metadata contains more rows than its manifest.")
                offsets[count] = position
                position += len(line)
                count += 1
        if count != expected_rows:
            raise RuntimeError(
                f"Route metadata row mismatch: expected={expected_rows}, actual={count}"
            )
        self._metadata_offsets = offsets
        return offsets

    def _metadata_at(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        offsets = self._offsets()
        records: dict[int, dict[str, Any]] = {}
        with self.metadata_path.open("rb") as handle:
            for index in sorted(set(indices)):
                if index < 0 or index >= len(offsets):
                    raise IndexError(index)
                handle.seek(int(offsets[index]))
                line = handle.readline()
                records[index] = json.loads(line.decode("utf-8"))
        return [records[index] for index in indices]

    def get_route(self, route_id: str) -> dict[str, Any] | None:
        # Route-ID lookup remains streaming and bounded; it does not parse the
        # complete metadata corpus into memory.
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
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
        vectors = self._ensure_vectors()
        if vector.ndim != 1 or vector.shape[0] != vectors.shape[1]:
            raise ValueError(
                f"Route query dimension mismatch: expected={vectors.shape[1]}, actual={vector.shape}"
            )
        norm = np.linalg.norm(vector)
        query = (vector / norm if norm else vector).astype(np.float32, copy=False)
        scores = np.empty(vectors.shape[0], dtype=np.float32)
        chunk_rows = max(
            1,
            int(self.manifest.get("search_chunk_rows") or self.DEFAULT_SEARCH_CHUNK_ROWS),
        )
        for start in range(0, vectors.shape[0], chunk_rows):
            stop = min(start + chunk_rows, vectors.shape[0])
            scores[start:stop] = np.asarray(vectors[start:stop]) @ query
        local_k = min(max(int(k), 0), len(scores))
        if local_k == 0:
            return []
        indices = np.argpartition(scores, -local_k)[-local_k:]
        ordered = indices[np.argsort(scores[indices])[::-1]]
        ordered_indices = [int(index) for index in ordered]
        metadata = self._metadata_at(ordered_indices)
        return [
            {**record, "score": float(scores[index])}
            for index, record in zip(ordered_indices, metadata, strict=True)
        ]
