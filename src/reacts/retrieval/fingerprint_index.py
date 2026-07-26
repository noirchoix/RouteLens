from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from reacts.chemistry.reactions import parse_reaction, reaction_fingerprint
from reacts.storage.tabular import iter_dataset

POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


@dataclass
class IndexBuildConfig:
    canonical_dir: Path
    index_dir: Path
    n_bits: int = 2048
    shard_rows: int = 10_000
    max_rows: int | None = None


class FingerprintIndexBuilder:
    def __init__(self, config: IndexBuildConfig):
        self.config = config
        self.config.index_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> dict[str, Any]:
        shard_meta: list[dict[str, Any]] = []
        reactants: list[np.ndarray] = []
        products: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        total = 0
        shard_id = 0

        def flush() -> None:
            nonlocal shard_id, reactants, products, metadata
            if not metadata:
                return
            prefix = self.config.index_dir / f"shard-{shard_id:05d}"
            r = np.packbits(np.vstack(reactants), axis=1)
            p = np.packbits(np.vstack(products), axis=1)
            np.savez_compressed(prefix.with_suffix(".npz"), reactants=r, products=p)
            prefix.with_suffix(".jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in metadata) + "\n",
                encoding="utf-8",
            )
            shard_meta.append({"shard": shard_id, "rows": len(metadata), "vectors": prefix.with_suffix(".npz").name, "metadata": prefix.with_suffix(".jsonl").name})
            shard_id += 1
            reactants, products, metadata = [], [], []

        columns = [
            "eligible_retrieval", "canonical_reaction_smiles", "step_id", "route_id",
            "patent_document_id", "solvent_primary", "time_bucket", "temperature_bucket", "quality_score",
        ]
        for chunk in iter_dataset(self.config.canonical_dir, "steps", columns=columns):
            subset = chunk.loc[chunk["eligible_retrieval"].fillna(False).astype(bool)]
            for _, row in subset.iterrows():
                if self.config.max_rows is not None and total >= self.config.max_rows:
                    break
                reaction = str(row["canonical_reaction_smiles"])
                rfp, pfp = reaction_fingerprint(reaction, n_bits=self.config.n_bits)
                reactants.append(rfp)
                products.append(pfp)
                metadata.append(
                    {
                        "step_id": row["step_id"],
                        "route_id": row["route_id"],
                        "patent_document_id": row.get("patent_document_id"),
                        "reaction_smiles": reaction,
                        "solvent_primary": row.get("solvent_primary"),
                        "time_bucket": row.get("time_bucket"),
                        "temperature_bucket": row.get("temperature_bucket"),
                        "quality_score": row.get("quality_score"),
                    }
                )
                total += 1
                if len(metadata) >= self.config.shard_rows:
                    flush()
            if self.config.max_rows is not None and total >= self.config.max_rows:
                break
        flush()
        manifest = {"index_version": "reaction_morgan_v1", "n_bits": self.config.n_bits, "rows": total, "shards": shard_meta}
        (self.config.index_dir / "index_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


class FingerprintIndex:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "index_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.n_bits = int(self.manifest["n_bits"])

    @staticmethod
    def _scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        intersection = POPCOUNT[np.bitwise_and(matrix, query)].sum(axis=1).astype(np.float32)
        union = POPCOUNT[np.bitwise_or(matrix, query)].sum(axis=1).astype(np.float32)
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    def search(self, reaction_smiles: str, k: int = 10) -> list[dict[str, Any]]:
        parsed = parse_reaction(reaction_smiles)
        if not parsed.parse_ok:
            return []
        rfp, pfp = reaction_fingerprint(reaction_smiles, n_bits=self.n_bits)
        rq, pq = np.packbits(rfp), np.packbits(pfp)
        heap: list[tuple[float, int, dict[str, Any]]] = []
        serial = 0
        for shard in self.manifest["shards"]:
            arrays = np.load(self.index_dir / shard["vectors"])
            scores = 0.45 * self._scores(rq, arrays["reactants"]) + 0.55 * self._scores(pq, arrays["products"])
            local_k = min(k, len(scores))
            if local_k == 0:
                continue
            indices = np.argpartition(scores, -local_k)[-local_k:]
            with (self.index_dir / shard["metadata"]).open("r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh]
            for idx in indices:
                item = dict(rows[int(idx)])
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
