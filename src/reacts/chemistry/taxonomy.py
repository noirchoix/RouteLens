from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


SOLVENT_FAMILY_OVERRIDES = {
    "O": "water",
    "CO": "alcohol_protic",
    "CCO": "alcohol_protic",
    "CC(C)O": "alcohol_protic",
    "C1CCOC1": "ether_aprotic",
    "COC": "ether_aprotic",
    "CN(C)C=O": "amide_polar_aprotic",
    "CS(C)=O": "sulfoxide_polar_aprotic",
    "CC#N": "nitrile_polar_aprotic",
    "ClCCl": "halogenated",
    "ClC(Cl)Cl": "halogenated",
    "c1ccccc1": "aromatic_nonpolar",
    "Cc1ccccc1": "aromatic_nonpolar",
    "CCCCCC": "aliphatic_nonpolar",
    "CCCCCCC": "aliphatic_nonpolar",
    "CCOC(C)=O": "ester_medium_polarity",
}


@lru_cache(maxsize=16_384)
def solvent_family(smiles: str) -> str:
    text = str(smiles or "").strip()
    if text in SOLVENT_FAMILY_OVERRIDES:
        return SOLVENT_FAMILY_OVERRIDES[text]
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return "unknown_solvent"
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if elements & {"Cl", "Br", "I", "F"}:
        return "halogenated"
    if "P" in elements or "S" in elements:
        return "heteroatom_polar_aprotic"
    hbd = Lipinski.NumHDonors(mol)
    logp = Crippen.MolLogP(mol)
    rings = Lipinski.RingCount(mol)
    if hbd:
        return "protic"
    if rings and logp > 1.5:
        return "aromatic_nonpolar"
    if logp > 2.0:
        return "aliphatic_nonpolar"
    return "polar_aprotic"


METALS = {"Li", "Na", "K", "Mg", "Ca", "Fe", "Co", "Ni", "Cu", "Zn", "Pd", "Pt", "Rh", "Ru", "Ir", "Ag", "Au", "Al", "Sn", "B"}
TRANSITION_METALS = {"Fe", "Co", "Ni", "Cu", "Pd", "Pt", "Rh", "Ru", "Ir", "Ag", "Au"}


@lru_cache(maxsize=65_536)
def agent_family(smiles: str) -> str:
    text = str(smiles or "").strip()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return "unknown_agent"
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if elements & TRANSITION_METALS:
        return "transition_metal_catalyst"
    if elements & METALS:
        return "metal_reagent_or_salt"
    if "B" in elements:
        return "boron_reagent"
    if "P" in elements:
        return "phosphorus_reagent_or_ligand"
    if "S" in elements:
        return "sulfur_reagent"
    charge = Chem.GetFormalCharge(mol)
    if charge:
        return "ionic_reagent"
    mw = Descriptors.MolWt(mol)
    if mw < 80:
        return "small_molecule_reagent"
    return "organic_reagent"


def catalyst_family(smiles: str) -> str | None:
    family = agent_family(smiles)
    if family == "transition_metal_catalyst":
        return family
    text = str(smiles or "")
    if any(token in text for token in ["[Pd", "[Pt", "[Rh", "[Ru", "[Ni", "[Cu", "[Fe"]):
        return "transition_metal_catalyst"
    return None
