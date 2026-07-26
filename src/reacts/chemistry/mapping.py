from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdFMCS

from reacts.chemistry.reactions import canonicalize_reaction, parse_reaction
from reacts.contracts import MappingStatus
from reacts.science.hashing import canonical_json_hash


@dataclass(frozen=True)
class MappingResult:
    status: MappingStatus
    mapped_reaction_smiles: str | None
    backend: str
    confidence: float
    atom_coverage: float
    diagnostics: tuple[str, ...] = ()
    error_code: str | None = None
    rxnmapper_token_count: int | None = None
    rxnmapper_token_limit: int | None = None
    rxnmapper_eligible: bool | None = None
    fallback_status: str | None = None


@dataclass(frozen=True)
class ReactionCentre:
    formed_bonds: tuple[dict[str, Any], ...]
    broken_bonds: tuple[dict[str, Any], ...]
    changed_bonds: tuple[dict[str, Any], ...]
    changed_atom_maps: tuple[int, ...]
    atom_environment_changes: tuple[dict[str, Any], ...]
    fingerprint: str
    reaction_template: str
    structural_family: str


def _split_reaction(reaction_smiles: str) -> tuple[list[str], list[str]]:
    canonical = canonicalize_reaction(reaction_smiles)
    if not canonical:
        return [], []
    left, right = canonical.split(">>", 1)
    return [item for item in left.split(".") if item], [item for item in right.split(".") if item]


def _has_maps(molecules: list[Chem.Mol]) -> bool:
    return any(atom.GetAtomMapNum() > 0 for mol in molecules for atom in mol.GetAtoms())


def validate_mapped_reaction(mapped: str) -> tuple[bool, float, list[str]]:
    left, right = mapped.split(">>", 1)
    reactants = [Chem.MolFromSmiles(item) for item in left.split(".") if item]
    products = [Chem.MolFromSmiles(item) for item in right.split(".") if item]
    if any(mol is None for mol in [*reactants, *products]):
        return False, 0.0, ["mapped_smiles_parse_failure"]
    left_maps = [atom.GetAtomMapNum() for mol in reactants for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
    right_maps = [atom.GetAtomMapNum() for mol in products for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
    overlap = set(left_maps) & set(right_maps)
    denominator = max(len(set(left_maps) | set(right_maps)), 1)
    coverage = len(overlap) / denominator
    issues: list[str] = []
    if len(left_maps) != len(set(left_maps)):
        issues.append("duplicate_reactant_atom_maps")
    if len(right_maps) != len(set(right_maps)):
        issues.append("duplicate_product_atom_maps")
    if not overlap:
        issues.append("no_cross_side_atom_map_overlap")
    return not issues, coverage, issues


class AtomMappingEngine:
    def __init__(
        self,
        backend: str = "auto",
        *,
        min_coverage: float = 0.60,
        timeout_seconds: int = 3,
    ):
        self.backend = backend
        self.min_coverage = min_coverage
        self.timeout_seconds = timeout_seconds
        self._rxn_mapper = None
        if backend in {"auto", "rxnmapper"}:
            try:
                from rxnmapper import RXNMapper  # type: ignore

                self._rxn_mapper = RXNMapper()
            except Exception:
                if backend == "rxnmapper":
                    raise

    def map_reaction(self, reaction_smiles: str) -> MappingResult:
        parsed = parse_reaction(reaction_smiles)
        if not parsed.parse_ok:
            return MappingResult(MappingStatus.NOT_ELIGIBLE, None, "none", 0.0, 0.0, (parsed.failure_class.value,))
        canonical = canonicalize_reaction(reaction_smiles)
        if canonical is None:
            return MappingResult(MappingStatus.FAILED, None, "none", 0.0, 0.0, ("canonicalization_failed",))

        left, right = canonical.split(">>", 1)
        molecules = [Chem.MolFromSmiles(item) for item in [*left.split("."), *right.split(".")] if item]
        if all(molecules) and _has_maps([mol for mol in molecules if mol is not None]):
            valid, coverage, issues = validate_mapped_reaction(canonical)
            return MappingResult(
                MappingStatus.EXISTING if valid else MappingStatus.LOW_CONFIDENCE,
                canonical,
                "existing",
                coverage,
                coverage,
                tuple(issues),
            )

        if self._rxn_mapper is not None:
            try:
                result = self._rxn_mapper.get_attention_guided_atom_maps([canonical])[0]
                mapped = str(result["mapped_rxn"])
                confidence = float(result.get("confidence", 0.0))
                valid, coverage, issues = validate_mapped_reaction(mapped)
                status = MappingStatus.MAPPED if valid and confidence >= 0.5 else MappingStatus.LOW_CONFIDENCE
                return MappingResult(status, mapped, "rxnmapper", confidence, coverage, tuple(issues))
            except Exception as exc:
                if self.backend == "rxnmapper":
                    return MappingResult(MappingStatus.FAILED, None, "rxnmapper", 0.0, 0.0, (str(exc),))

        return self._map_with_mcs(canonical)

    def _map_with_mcs(self, canonical: str) -> MappingResult:
        reactant_smiles, product_smiles = _split_reaction(canonical)
        reactants = [Chem.MolFromSmiles(item) for item in reactant_smiles]
        products = [Chem.MolFromSmiles(item) for item in product_smiles]
        if any(mol is None for mol in [*reactants, *products]):
            return MappingResult(MappingStatus.FAILED, None, "mcs_fallback", 0.0, 0.0, ("input_parse_failure",))
        reactants = [Chem.Mol(mol) for mol in reactants if mol is not None]
        products = [Chem.Mol(mol) for mol in products if mol is not None]

        next_map = 1
        used_reactants: set[int] = set()
        matched_atoms = 0
        total_product_atoms = sum(mol.GetNumAtoms() for mol in products)
        diagnostics: list[str] = []

        for product in products:
            best: tuple[int, tuple[int, ...], tuple[int, ...], int] | None = None
            for reactant_index, reactant in enumerate(reactants):
                if reactant_index in used_reactants:
                    continue
                mcs = rdFMCS.FindMCS(
                    [reactant, product],
                    timeout=self.timeout_seconds,
                    ringMatchesRingOnly=True,
                    completeRingsOnly=True,
                    matchValences=True,
                )
                if mcs.canceled or mcs.numAtoms == 0:
                    continue
                query = Chem.MolFromSmarts(mcs.smartsString)
                if query is None:
                    continue
                reactant_matches = reactant.GetSubstructMatches(query, uniquify=True)
                product_matches = product.GetSubstructMatches(query, uniquify=True)
                # Multiple matches are chemically ambiguous; the fallback does not guess.
                if len(reactant_matches) != 1 or len(product_matches) != 1:
                    continue
                candidate = (reactant_index, reactant_matches[0], product_matches[0], mcs.numAtoms)
                if best is None or candidate[3] > best[3]:
                    best = candidate
            if best is None:
                diagnostics.append("unmatched_product_component")
                continue
            reactant_index, reactant_match, product_match, count = best
            used_reactants.add(reactant_index)
            reactant = reactants[reactant_index]
            for reactant_atom_index, product_atom_index in zip(reactant_match, product_match):
                reactant.GetAtomWithIdx(reactant_atom_index).SetAtomMapNum(next_map)
                product.GetAtomWithIdx(product_atom_index).SetAtomMapNum(next_map)
                next_map += 1
                matched_atoms += 1

        for mol in [*reactants, *products]:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    atom.SetAtomMapNum(next_map)
                    next_map += 1

        mapped = ".".join(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in reactants)
        mapped += ">>" + ".".join(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in products)
        valid, cross_coverage, issues = validate_mapped_reaction(mapped)
        product_coverage = matched_atoms / max(total_product_atoms, 1)
        confidence = min(product_coverage, cross_coverage)
        diagnostics.extend(issues)
        status = MappingStatus.MAPPED if valid and confidence >= self.min_coverage else MappingStatus.LOW_CONFIDENCE
        if confidence == 0.0:
            status = MappingStatus.FAILED
        return MappingResult(status, mapped if confidence > 0 else None, "mcs_fallback", confidence, product_coverage, tuple(diagnostics))


def _bond_dict(molecules: list[Chem.Mol]) -> dict[tuple[int, int], dict[str, Any]]:
    bonds: dict[tuple[int, int], dict[str, Any]] = {}
    for mol in molecules:
        for bond in mol.GetBonds():
            a = bond.GetBeginAtom().GetAtomMapNum()
            b = bond.GetEndAtom().GetAtomMapNum()
            if not a or not b:
                continue
            key = tuple(sorted((a, b)))
            bonds[key] = {
                "atoms": list(key),
                "bond_type": str(bond.GetBondType()),
                "bond_order": float(bond.GetBondTypeAsDouble()),
            }
    return bonds


def _atom_environment(molecules: list[Chem.Mol]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for mol in molecules:
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if not map_num:
                continue
            output[map_num] = {
                "element": atom.GetSymbol(),
                "charge": atom.GetFormalCharge(),
                "aromatic": atom.GetIsAromatic(),
                "degree": atom.GetDegree(),
                "neighbours": sorted(
                    neighbour.GetSymbol() for neighbour in atom.GetNeighbors()
                ),
            }
    return output


def classify_structural_family(
    formed: list[dict[str, Any]],
    broken: list[dict[str, Any]],
    changed: list[dict[str, Any]],
    environments_after: dict[int, dict[str, Any]],
) -> str:
    if changed and not formed and not broken:
        delta = sum(item["after_order"] - item["before_order"] for item in changed)
        return "oxidation_or_unsaturation" if delta > 0 else "reduction_or_saturation"
    if formed:
        formed_elements = []
        for item in formed:
            a, b = item["atoms"]
            formed_elements.append(tuple(sorted((environments_after.get(a, {}).get("element"), environments_after.get(b, {}).get("element")))))
        if ("C", "C") in formed_elements:
            return "carbon_carbon_bond_formation"
        if ("C", "N") in formed_elements:
            return "carbon_nitrogen_bond_formation"
        if ("C", "O") in formed_elements:
            return "carbon_oxygen_bond_formation"
        if broken:
            return "substitution_or_exchange"
        return "bond_formation"
    if broken:
        return "cleavage_or_deprotection"
    return "no_detected_core_change"


def derive_reaction_centre(mapped_reaction_smiles: str) -> ReactionCentre:
    left, right = mapped_reaction_smiles.split(">>", 1)
    reactants = [Chem.MolFromSmiles(item) for item in left.split(".") if item]
    products = [Chem.MolFromSmiles(item) for item in right.split(".") if item]
    if any(mol is None for mol in [*reactants, *products]):
        raise ValueError("Mapped reaction could not be parsed.")
    reactants = [mol for mol in reactants if mol is not None]
    products = [mol for mol in products if mol is not None]
    before = _bond_dict(reactants)
    after = _bond_dict(products)
    formed = [after[key] for key in sorted(after.keys() - before.keys())]
    broken = [before[key] for key in sorted(before.keys() - after.keys())]
    changed: list[dict[str, Any]] = []
    for key in sorted(before.keys() & after.keys()):
        if before[key]["bond_order"] != after[key]["bond_order"]:
            changed.append(
                {
                    "atoms": list(key),
                    "before_type": before[key]["bond_type"],
                    "before_order": before[key]["bond_order"],
                    "after_type": after[key]["bond_type"],
                    "after_order": after[key]["bond_order"],
                }
            )
    changed_maps = sorted({atom for item in [*formed, *broken, *changed] for atom in item["atoms"]})
    env_before = _atom_environment(reactants)
    env_after = _atom_environment(products)
    environment_changes = [
        {
            "atom_map": atom_map,
            "before": env_before.get(atom_map),
            "after": env_after.get(atom_map),
        }
        for atom_map in changed_maps
        if env_before.get(atom_map) != env_after.get(atom_map)
    ]
    payload = {"formed": formed, "broken": broken, "changed": changed, "environments": environment_changes}
    fingerprint = canonical_json_hash(payload)
    template = json.dumps(
        {
            "formed": [item["atoms"] + [item["bond_type"]] for item in formed],
            "broken": [item["atoms"] + [item["bond_type"]] for item in broken],
            "changed": [item["atoms"] + [item["before_type"], item["after_type"]] for item in changed],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    family = classify_structural_family(formed, broken, changed, env_after)
    return ReactionCentre(
        formed_bonds=tuple(formed),
        broken_bonds=tuple(broken),
        changed_bonds=tuple(changed),
        changed_atom_maps=tuple(changed_maps),
        atom_environment_changes=tuple(environment_changes),
        fingerprint=fingerprint,
        reaction_template=template,
        structural_family=family,
    )
