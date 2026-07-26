from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

from reacts.contracts import ParseFailureClass
from reacts.data.parsing import INTERMEDIATE_RE


@dataclass(frozen=True)
class ReactionParse:
    reaction_smiles: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    reactants_valid: bool
    products_valid: bool
    parse_ok: bool
    failure_class: ParseFailureClass


def _split_side(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.split(".") if token.strip())


def parse_reaction(reaction_smiles: str) -> ReactionParse:
    text = str(reaction_smiles or "").strip()
    if text.count(">>") == 1:
        left, right = text.split(">>", 1)
    elif ">>" not in text and text.count(">") >= 2:
        left, _, right = text.split(">", 2)
    else:
        return ReactionParse(text, (), (), False, False, False, ParseFailureClass.MALFORMED_DELIMITER)

    reactants, products = _split_side(left), _split_side(right)
    if not reactants or not products:
        return ReactionParse(text, reactants, products, bool(reactants), bool(products), False, ParseFailureClass.EMPTY_REACTION_SIDE)

    symbolic = any(INTERMEDIATE_RE.fullmatch(x) for x in (*reactants, *products))
    react_ok = all(INTERMEDIATE_RE.fullmatch(x) or Chem.MolFromSmiles(x) is not None for x in reactants)
    product_ok = all(INTERMEDIATE_RE.fullmatch(x) or Chem.MolFromSmiles(x) is not None for x in products)
    chemically_complete = react_ok and product_ok and not symbolic
    if chemically_complete:
        failure = ParseFailureClass.VALID
    elif symbolic and react_ok and product_ok:
        failure = ParseFailureClass.SYMBOLIC_INTERMEDIATE
    elif not react_ok and not product_ok:
        failure = ParseFailureClass.INVALID_BOTH
    elif not react_ok:
        failure = ParseFailureClass.INVALID_REACTANT
    elif not product_ok:
        failure = ParseFailureClass.INVALID_PRODUCT
    else:
        failure = ParseFailureClass.UNKNOWN
    return ReactionParse(text, reactants, products, react_ok, product_ok, chemically_complete, failure)


def canonicalize_molecule(smiles: str) -> str | None:
    if INTERMEDIATE_RE.fullmatch(smiles):
        return smiles.upper()
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else None


def canonicalize_reaction(reaction_smiles: str) -> str | None:
    parsed = parse_reaction(reaction_smiles)
    if not (parsed.reactants_valid and parsed.products_valid):
        return None
    left = sorted(filter(None, (canonicalize_molecule(x) for x in parsed.reactants)))
    right = sorted(filter(None, (canonicalize_molecule(x) for x in parsed.products)))
    return ".".join(left) + ">>" + ".".join(right)


@lru_cache(maxsize=16)
def _morgan_generator(radius: int, n_bits: int):
    return AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _side_morgan(smiles: Iterable[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    out = np.zeros((n_bits,), dtype=np.uint8)
    generator = _morgan_generator(radius, n_bits)
    for item in smiles:
        if INTERMEDIATE_RE.fullmatch(item):
            continue
        mol = Chem.MolFromSmiles(item)
        if mol is None:
            continue
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
        out |= arr
    return out


def reaction_fingerprint(reaction_smiles: str, n_bits: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Build side fingerprints without repeating the full validation parse.

    Callers that accept external input must validate with ``parse_reaction`` first. Canonical
    index construction already restricts rows to ``eligible_retrieval``.
    """
    text = str(reaction_smiles or "").strip()
    if text.count(">>") == 1:
        left, right = text.split(">>", 1)
    elif ">>" not in text and text.count(">") >= 2:
        left, _, right = text.split(">", 2)
    else:
        return np.zeros((n_bits,), dtype=np.uint8), np.zeros((n_bits,), dtype=np.uint8)
    return _side_morgan(_split_side(left), n_bits=n_bits), _side_morgan(_split_side(right), n_bits=n_bits)


def molecular_descriptor_summary(smiles_list: Iterable[str]) -> dict[str, float]:
    mols = [Chem.MolFromSmiles(x) for x in smiles_list if not INTERMEDIATE_RE.fullmatch(x)]
    mols = [m for m in mols if m is not None]
    if not mols:
        return {"mw": 0.0, "logp": 0.0, "hbd": 0.0, "hba": 0.0, "rings": 0.0}
    return {
        "mw": float(sum(Descriptors.MolWt(m) for m in mols)),
        "logp": float(sum(Descriptors.MolLogP(m) for m in mols)),
        "hbd": float(sum(Descriptors.NumHDonors(m) for m in mols)),
        "hba": float(sum(Descriptors.NumHAcceptors(m) for m in mols)),
        "rings": float(sum(Descriptors.RingCount(m) for m in mols)),
    }
