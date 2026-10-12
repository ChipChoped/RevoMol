"""
Systèmes π conjugués d'une molécule, comptés comme _pi_components de
~/Desktop/Stage/Compte_DA_π_arom+annalyse/gap_pi_relation.ipynb : atome π = aromatique ou hybridé
SP2 / SP (RDKit) ; systèmes π = composantes connexes de ces atomes.
  - PI_CHAIN_SIZE : taille du plus grand système π ;
  - PCT_PI_ATOMS : % des atomes lourds qui sont π (NaN sans atome).
sulfur_rule : corrige les atomes π autour du soufre (apply_sulfur_rule).

Le SMILES est relu par RDKit (aromaticité perçue), et non pris du MolecularGraph, qui est kékulisé.
"""

from __future__ import annotations

import math

from rdkit import Chem
from rdkit.Chem import rdchem
from typing_extensions import override

from evomol.evaluation.evaluation import Evaluation, EvaluationError
from evomol.representation import MolecularGraph, Molecule


def apply_sulfur_rule(mol: Chem.Mol, pi_set: set[int]) -> set[int]:
    """Règle du soufre : la conjugaison π passe par un S divalent, pas par un S hypervalent
    (l'hyperconjugaison n'est pas comptée).
      - S hypervalent (non aromatique, valence > 2 : sulfoxyde, sulfone, S=C ylure, S≡S…) : pas π ;
      - hétéroatome dont toutes les liaisons multiples vont vers un S hypervalent (O de S=O, N de S=N…) :
        pas π ; un C dans ce cas (C de S=C) reste π, mais le S ne le relie pas au reste ;
      - S divalent sans liaison multiple (thioéther, thiol), voisin d'un atome π : π (comme RDKit le fait
        déjà pour O et N à doublet).
    =S terminal (thiocétone) et S aromatique (thiophène) : inchangés."""
    def hypervalent(atom: Chem.Atom) -> bool:
        return atom.GetSymbol() == "S" and not atom.GetIsAromatic() and atom.GetTotalValence() > 2

    excluded = {a.GetIdx() for a in mol.GetAtoms() if hypervalent(a)}
    for atom in mol.GetAtoms():
        if atom.GetIdx() in excluded or atom.GetIsAromatic() or atom.GetSymbol() == "C":
            continue
        multiple = [b for b in atom.GetBonds() if b.GetBondTypeAsDouble() >= 2]
        if multiple and all(hypervalent(b.GetOtherAtom(atom)) for b in multiple):
            excluded.add(atom.GetIdx())
    pi_set = pi_set - excluded
    divalent_s = [a for a in mol.GetAtoms() if a.GetSymbol() == "S" and not a.GetIsAromatic()
                  and a.GetTotalValence() == 2 and all(b.GetBondTypeAsDouble() == 1 for b in a.GetBonds())]
    pi_set |= {a.GetIdx() for a in divalent_s
               if any(n.GetIdx() in pi_set and n.GetSymbol() != "S" for n in a.GetNeighbors())}
    return pi_set


def pi_atoms(mol: Chem.Mol, sulfur_rule: bool = True) -> set[int]:
    """Indices des atomes π : aromatiques ou hybridés SP2 / SP, corrigés par la règle du soufre."""
    sp2, sp = rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP
    pi_set = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic() or a.GetHybridization() in (sp2, sp)}
    if sulfur_rule:
        pi_set = apply_sulfur_rule(mol, pi_set)
    return pi_set


def pi_components(mol: Chem.Mol, pi_set: set[int]) -> list[int]:
    """Taille de chaque système π (composante connexe des atomes π)."""
    visited, sizes = set(), []
    for start in pi_set:
        if start in visited:
            continue
        size, stack = 0, [start]
        visited.add(start)
        while stack:
            u = stack.pop()
            size += 1
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                v = nb.GetIdx()
                if v in pi_set and v not in visited:
                    visited.add(v)
                    stack.append(v)
        sizes.append(size)
    return sizes


def pi_systems(smiles: str, sulfur_rule: bool = True) -> dict[str, float]:
    """PI_CHAIN_SIZE et PCT_PI_ATOMS d'un SMILES (NaN si RDKit ne le lit pas)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"PI_CHAIN_SIZE": math.nan, "PCT_PI_ATOMS": math.nan}
    pi_set = pi_atoms(mol, sulfur_rule)
    n_atoms = mol.GetNumAtoms()
    return {"PI_CHAIN_SIZE": max(pi_components(mol, pi_set), default=0),
            "PCT_PI_ATOMS": 100 * len(pi_set) / n_atoms if n_atoms else math.nan}


class PiSystems(Evaluation):
    """Calcule et met en cache PI_CHAIN_SIZE et PCT_PI_ATOMS (dict, pi_systems)."""

    def __init__(self, sulfur_rule: bool = True) -> None:
        super().__init__("pi_systems" if sulfur_rule else "pi_systems_no_sulfur_rule")
        self.sulfur_rule = sulfur_rule

    @override
    def _evaluate(self, molecule: Molecule) -> dict[str, float]:
        return pi_systems(molecule.get_representation(MolecularGraph).canonical_smiles, self.sulfur_rule)


class PiChainSize(Evaluation):
    """Taille du plus grand système π (lit le résultat de PiSystems, règle du soufre comprise)."""

    def __init__(self) -> None:
        super().__init__("pi_chain_size")

    @override
    def _evaluate(self, molecule: Molecule) -> float:
        value = PiSystems().evaluate(molecule)["PI_CHAIN_SIZE"]
        if math.isnan(value):
            raise EvaluationError("PI_CHAIN_SIZE is undefined.")
        return value


class PctPiAtoms(Evaluation):
    """% des atomes lourds qui sont π (lit le résultat de PiSystems, règle du soufre comprise)."""

    def __init__(self) -> None:
        super().__init__("pct_pi_atoms")

    @override
    def _evaluate(self, molecule: Molecule) -> float:
        value = PiSystems().evaluate(molecule)["PCT_PI_ATOMS"]
        if math.isnan(value):
            raise EvaluationError("PCT_PI_ATOMS is undefined.")
        return value
