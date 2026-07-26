from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from reacts.chemistry.reactions import parse_reaction
from reacts.science.hashing import canonical_json_hash
from reacts.storage.tabular import iter_dataset


@dataclass(frozen=True)
class LeakageSummary:
    key_type: str
    unique_keys: int
    overlapping_keys: int
    train_val_overlap: int
    train_test_overlap: int
    val_test_overlap: int
    examples: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_type": self.key_type,
            "unique_keys": self.unique_keys,
            "overlapping_keys": self.overlapping_keys,
            "train_val_overlap": self.train_val_overlap,
            "train_test_overlap": self.train_test_overlap,
            "val_test_overlap": self.val_test_overlap,
            "examples": list(self.examples),
        }


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none", "null"} else text


def _product_scaffold(reaction_smiles: str) -> str | None:
    parsed = parse_reaction(reaction_smiles)
    if not parsed.products_valid:
        return None
    scaffolds: list[str] = []
    for product in parsed.products:
        mol = Chem.MolFromSmiles(product)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold:
            scaffolds.append(scaffold)
    return ".".join(sorted(set(scaffolds))) or None


class LeakageAuditor:
    def __init__(self, canonical_dir: Path, dataset_name: str = "steps"):
        self.canonical_dir = Path(canonical_dir)
        self.dataset_name = dataset_name

    @staticmethod
    def _summarize(
        key_type: str,
        memberships: dict[str, set[str]],
        max_examples: int = 50,
    ) -> LeakageSummary:
        overlapping = {key: splits for key, splits in memberships.items() if len(splits) > 1}
        train_val = sum({"train", "val"}.issubset(splits) for splits in memberships.values())
        train_test = sum({"train", "test"}.issubset(splits) for splits in memberships.values())
        val_test = sum({"val", "test"}.issubset(splits) for splits in memberships.values())
        examples = tuple(
            {"key": key, "splits": sorted(splits)}
            for key, splits in list(sorted(overlapping.items()))[:max_examples]
        )
        return LeakageSummary(
            key_type=key_type,
            unique_keys=len(memberships),
            overlapping_keys=len(overlapping),
            train_val_overlap=int(train_val),
            train_test_overlap=int(train_test),
            val_test_overlap=int(val_test),
            examples=examples,
        )

    def audit(self) -> dict[str, Any]:
        memberships: dict[str, dict[str, set[str]]] = {
            "patent_document_id": defaultdict(set),
            "reaction_signature": defaultdict(set),
            "product_scaffold": defaultdict(set),
        }
        columns = [
            "split",
            "patent_document_id",
            "reaction_signature",
            "canonical_resolved_reaction_smiles",
            "canonical_reaction_smiles",
        ]
        rows = 0
        for chunk in iter_dataset(self.canonical_dir, self.dataset_name):
            for row in chunk.to_dict(orient="records"):
                split = _clean(row.get("split"))
                if split not in {"train", "val", "test"}:
                    continue
                rows += 1
                patent = _clean(row.get("patent_document_id"))
                if patent:
                    memberships["patent_document_id"][patent].add(split)

                reaction = _clean(row.get("canonical_resolved_reaction_smiles")) or _clean(
                    row.get("canonical_reaction_smiles")
                )
                signature = _clean(row.get("reaction_signature"))
                if not signature and reaction:
                    signature = canonical_json_hash(reaction)
                if signature:
                    memberships["reaction_signature"][signature].add(split)
                if reaction:
                    scaffold = _product_scaffold(reaction)
                    if scaffold:
                        memberships["product_scaffold"][canonical_json_hash(scaffold)].add(split)

        summaries = {
            key: self._summarize(key, values).to_dict()
            for key, values in memberships.items()
        }
        strict_pass = all(
            summaries[key]["overlapping_keys"] == 0
            for key in ("patent_document_id", "reaction_signature")
        )
        return {
            "rows_examined": rows,
            "summaries": summaries,
            "strict_invariants": {
                "patent_document_id_overlap_is_zero": summaries["patent_document_id"]["overlapping_keys"] == 0,
                "reaction_signature_overlap_is_zero": summaries["reaction_signature"]["overlapping_keys"] == 0,
                "product_scaffold_is_diagnostic_only": True,
            },
            "strict_pass": strict_pass,
            "interpretation": {
                "patent_document_id": "Must be zero for the grouped benchmark.",
                "reaction_signature": "Must be zero for strict supervised evaluation.",
                "product_scaffold": "Challenge diagnostic only; overlap may remain across connected-component splits.",
            },
        }

    def write(self, path: Path) -> dict[str, Any]:
        report = self.audit()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report
