from __future__ import annotations

import heapq
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reacts.chemistry.reactions import parse_reaction, reaction_fingerprint
from reacts.science.hashing import canonical_json_hash, hash_dataset_columns, hash_paths, sha256_file
from reacts.storage.tabular import iter_dataset

POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


@dataclass
class ContextualIndexBuildConfig:
    canonical_dir: Path
    index_dir: Path
    dataset_version: str = "uspto_multistep_contextual_v2"
    n_bits: int = 2048
    shard_rows: int = 10_000
    max_rows: int | None = None
    reactant_weight: float = 0.40
    product_weight: float = 0.45
    centre_weight: float = 0.15


class ContextualFingerprintIndexBuilder:
    def __init__(self, config: ContextualIndexBuildConfig):
        self.config = config
        self.config.index_dir.mkdir(parents=True, exist_ok=True)

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
        for path in self.config.index_dir.glob("shard-*.*"):
            path.unlink()
        shard_meta: list[dict[str, Any]] = []
        reactants: list[np.ndarray] = []
        products: list[np.ndarray] = []
        centres: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        total = 0
        shard_id = 0

        def centre_bits(fingerprint: str | None) -> np.ndarray:
            out = np.zeros((self.config.n_bits,), dtype=np.uint8)
            if not fingerprint or str(fingerprint) == "nan":
                return out
            raw = bytes.fromhex(str(fingerprint)[:64])
            for index, value in enumerate(raw):
                out[(index * 257 + value) % self.config.n_bits] = 1
            return out

        def flush() -> None:
            nonlocal shard_id, reactants, products, centres, metadata
            if not metadata:
                return
            prefix = self.config.index_dir / f"shard-{shard_id:05d}"
            np.savez_compressed(
                prefix.with_suffix(".npz"),
                reactants=np.packbits(np.vstack(reactants), axis=1),
                products=np.packbits(np.vstack(products), axis=1),
                centres=np.packbits(np.vstack(centres), axis=1),
            )
            metadata_path = prefix.with_suffix(".jsonl")
            metadata_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in metadata) + "\n",
                encoding="utf-8",
            )
            shard_meta.append(
                {
                    "shard": shard_id,
                    "rows": len(metadata),
                    "vectors": prefix.with_suffix(".npz").name,
                    "metadata": metadata_path.name,
                    "vectors_sha256": sha256_file(prefix.with_suffix(".npz")),
                    "metadata_sha256": sha256_file(metadata_path),
                }
            )
            shard_id += 1
            reactants, products, centres, metadata = [], [], [], []

        columns = [
            "eligible_retrieval_v2",
            "canonical_resolved_reaction_smiles",
            "step_id",
            "step_instance_id",
            "source_step_id",
            "route_id",
            "route_instance_id",
            "source_route_id",
            "patent_document_id",
            "reaction_signature",
            "split_component_id",
            "solvent_primary",
            "solvents",
            "agents",
            "time_bucket",
            "temperature_bucket",
            "contextual_quality_score",
            "reaction_family",
            "reaction_centre_fingerprint",
            "resolution_status",
            "split",
        ]
        for chunk in iter_dataset(self.config.canonical_dir, "steps", columns=columns):
            subset = chunk.loc[chunk["eligible_retrieval_v2"].fillna(False).astype(bool)]
            for _, row in subset.iterrows():
                if self.config.max_rows is not None and total >= self.config.max_rows:
                    break
                reaction = str(row["canonical_resolved_reaction_smiles"])
                rfp, pfp = reaction_fingerprint(reaction, n_bits=self.config.n_bits)
                reactants.append(rfp)
                products.append(pfp)
                centres.append(centre_bits(row.get("reaction_centre_fingerprint")))
                metadata.append(
                    {
                        "step_id": row["step_id"],
                        "step_instance_id": row.get("step_instance_id", row["step_id"]),
                        "source_step_id": row.get("source_step_id", row["step_id"]),
                        "route_id": row["route_id"],
                        "route_instance_id": row.get("route_instance_id", row["route_id"]),
                        "source_route_id": row.get("source_route_id", row["route_id"]),
                        "patent_document_id": row.get("patent_document_id"),
                        "reaction_signature": row.get("reaction_signature"),
                        "split_component_id": row.get("split_component_id"),
                        "reaction_smiles": reaction,
                        "solvent_primary": row.get("solvent_primary"),
                        "solvents": row.get("solvents") if isinstance(row.get("solvents"), list) else [],
                        "agents": row.get("agents") if isinstance(row.get("agents"), list) else [],
                        "time_bucket": row.get("time_bucket"),
                        "temperature_bucket": row.get("temperature_bucket"),
                        "quality_score": row.get("contextual_quality_score"),
                        "reaction_family": row.get("reaction_family"),
                        "reaction_centre_fingerprint": row.get("reaction_centre_fingerprint"),
                        "resolution_status": row.get("resolution_status"),
                        "split": row.get("split"),
                    }
                )
                total += 1
                if len(metadata) >= self.config.shard_rows:
                    flush()
            if self.config.max_rows is not None and total >= self.config.max_rows:
                break
        flush()
        manifest = {
            "index_version": "reaction_contextual_morgan_v2",
            "dataset_version": self.config.dataset_version,
            "n_bits": self.config.n_bits,
            "rows": total,
            "weights": {
                "reactant": self.config.reactant_weight,
                "product": self.config.product_weight,
                "centre": self.config.centre_weight,
            },
            "shards": shard_meta,
            "canonical_manifest_sha256": sha256_file(self.config.canonical_dir / "dataset_manifest.json"),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "training_split_sha256": training_split_sha,
            "split_algorithm": split_manifest.get("algorithm"),
            "split_seed": split_manifest.get("seed"),
        }
        manifest["configuration_sha256"] = canonical_json_hash(manifest["weights"] | {"n_bits": self.config.n_bits})
        manifest_path = self.config.index_dir / "index_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        manifest["index_tree_sha256"] = hash_paths(
            [self.config.index_dir / item[key] for item in shard_meta for key in ("vectors", "metadata")],
            root=self.config.index_dir,
        ) if shard_meta else None
        return manifest


class ContextualFingerprintIndex:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "index_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.n_bits = int(self.manifest["n_bits"])
        self.weights = self.manifest.get("weights", {"reactant": 0.40, "product": 0.45, "centre": 0.15})
        self._metadata_cache: dict[str, list[dict[str, Any]]] = {}

    @staticmethod
    def _scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        intersection = POPCOUNT[np.bitwise_and(matrix, query)].sum(axis=1).astype(np.float32)
        union = POPCOUNT[np.bitwise_or(matrix, query)].sum(axis=1).astype(np.float32)
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    def _metadata(self, name: str) -> list[dict[str, Any]]:
        if name not in self._metadata_cache:
            with (self.index_dir / name).open("r", encoding="utf-8") as handle:
                self._metadata_cache[name] = [json.loads(line) for line in handle]
        return self._metadata_cache[name]

    def search(
        self,
        reaction_smiles: str,
        k: int = 10,
        *,
        reaction_centre_fingerprint: str | None = None,
        exclude_step_id: str | None = None,
        minimum_quality: float | None = None,
    ) -> list[dict[str, Any]]:
        parsed = parse_reaction(reaction_smiles)
        if not parsed.parse_ok or k <= 0:
            return []
        rfp, pfp = reaction_fingerprint(reaction_smiles, n_bits=self.n_bits)
        rq, pq = np.packbits(rfp), np.packbits(pfp)
        cq_bits = np.zeros((self.n_bits,), dtype=np.uint8)
        if reaction_centre_fingerprint:
            raw = bytes.fromhex(str(reaction_centre_fingerprint)[:64])
            for index, value in enumerate(raw):
                cq_bits[(index * 257 + value) % self.n_bits] = 1
        cq = np.packbits(cq_bits)
        heap: list[tuple[float, int, dict[str, Any]]] = []
        serial = 0
        for shard in self.manifest["shards"]:
            arrays = np.load(self.index_dir / shard["vectors"])
            scores = (
                float(self.weights["reactant"]) * self._scores(rq, arrays["reactants"])
                + float(self.weights["product"]) * self._scores(pq, arrays["products"])
            )
            if reaction_centre_fingerprint and "centres" in arrays:
                scores += float(self.weights["centre"]) * self._scores(cq, arrays["centres"])
            rows = self._metadata(shard["metadata"])
            candidate_count = min(max(k * 3, k), len(scores))
            if candidate_count == 0:
                continue
            indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
            for idx in indices:
                item = dict(rows[int(idx)])
                if exclude_step_id and item.get("step_id") == exclude_step_id:
                    continue
                quality = item.get("quality_score")
                if minimum_quality is not None and quality is not None and float(quality) < minimum_quality:
                    continue
                for key, value in list(item.items()):
                    if isinstance(value, float) and np.isnan(value):
                        item[key] = None
                score = float(scores[int(idx)])
                item["score"] = score
                candidate = (score, serial, item)
                serial += 1
                if len(heap) < k:
                    heapq.heappush(heap, candidate)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, candidate)
        return [item for _, _, item in sorted(heap, reverse=True)]

    def benchmark(self, queries: list[dict[str, Any]], k: int = 10) -> dict[str, Any]:
        latencies_ms: list[float] = []
        self_hits = 0
        family_hits = 0
        family_total = 0
        for query in queries:
            start = time.perf_counter()
            results = self.search(str(query["reaction_smiles"]), k=k)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            self_hits += int(any(item.get("step_id") == query.get("step_id") for item in results))
            family = query.get("reaction_family")
            if family and str(family) != "nan":
                family_total += 1
                family_hits += int(any(item.get("reaction_family") == family for item in results))
        values = np.array(latencies_ms, dtype=float) if latencies_ms else np.array([0.0])
        return {
            "queries": len(queries),
            "k": k,
            "self_recall_at_k": self_hits / max(len(queries), 1),
            "family_recall_at_k": family_hits / max(family_total, 1),
            "latency_ms": {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "max": float(values.max()),
            },
        }


def neighbour_distributions(evidence: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for field in fields:
        counts: Counter[str] = Counter()
        weight_total = 0.0
        for item in evidence:
            value = item.get(field)
            values = value if isinstance(value, list) else [value]
            for label in values:
                if label is None or str(label) == "nan" or str(label) == "":
                    continue
                weight = max(float(item.get("score", 0.0)), 1e-6)
                counts[str(label)] += weight
                weight_total += weight
        output[field] = {label: count / weight_total for label, count in counts.most_common()} if weight_total else {}
    return output
