"""
Structural filters for Hückel-compatible molecule enumeration.
"""

import os
import sys
from typing import Optional

from rdkit.Chem.rdchem import BondType
from typing_extensions import override

# racine du projet (scripts/Lea/ -> ../..), pour importer evomol
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# pylint: disable=wrong-import-position
from evomol import default_parameters as dp
from evomol import evaluation as evaluator
from evomol.evaluation.evaluation import Evaluation, EvaluationError
from evomol.representation import MolecularGraph, Molecule

# Atoms accepted in the Hückel enumeration and random walks.
HUCKEL_ACCEPTED_ATOMS: list[str] = ["C", "N", "O", "S", "F", "Cl", "Br"]

# Non-carbon heavy atoms considered as heteroatoms
HETEROATOMS = {"N", "O", "S", "F", "Cl", "Br", "P", "B"}


class HeteroatomChainFilter(Evaluation):
    """Rejects molecules that violate heteroatom chain rules:

    1. No 3+ consecutive heteroatoms in any acyclic (non-ring) path.
       H-N-N-H (hydrazine) is OK; N-N-N in a chain is not.
    2. No O-O-O (3 consecutive oxygens) anywhere, including in rings.
    3. At most `max_ring_n` nitrogen atoms in ring positions (default 3).
       Triazoles (3 ring-N) are OK; tetrazoles (4 ring-N) are rejected.
    """

    def __init__(self, max_ring_n: int = 3) -> None:
        super().__init__("heteroatom_chain_filter")
        self.max_ring_n = max_ring_n

    @override
    def _evaluate(self, molecule: Molecule) -> bool:
        mol = molecule.get_representation(MolecularGraph).mol
        ring_info = mol.GetRingInfo()

        ring_atom_indices = {
            idx for ring in ring_info.AtomRings() for idx in ring
        }
        ring_bond_indices = {
            idx for ring in ring_info.BondRings() for idx in ring
        }

        # Rule 3: count N atoms in rings
        ring_n_count = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetSymbol() == "N" and atom.GetIdx() in ring_atom_indices
        )
        if ring_n_count > self.max_ring_n:
            raise EvaluationError(
                f"{ring_n_count} N in rings > max allowed ({self.max_ring_n})"
            )

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()

            # Rule 2: no O-O-O anywhere (rings or chains)
            if symbol == "O":
                o_neighbors = [
                    nb for nb in atom.GetNeighbors() if nb.GetSymbol() == "O"
                ]
                for o_nb in o_neighbors:
                    for o_nb2 in o_nb.GetNeighbors():
                        if o_nb2.GetSymbol() == "O" and o_nb2.GetIdx() != atom.GetIdx():
                            raise EvaluationError(
                                f"O-O-O chain detected (atoms "
                                f"{atom.GetIdx()}-{o_nb.GetIdx()}-{o_nb2.GetIdx()})"
                            )

            # Rule 1: no 3+ consecutive heteroatoms via acyclic bonds
            if symbol in HETEROATOMS:
                hetero_via_acyclic = [
                    bond.GetOtherAtom(atom)
                    for bond in atom.GetBonds()
                    if bond.GetIdx() not in ring_bond_indices
                    and bond.GetOtherAtom(atom).GetSymbol() in HETEROATOMS
                ]
                if len(hetero_via_acyclic) >= 2:
                    raise EvaluationError(
                        f"3+ consecutive heteroatoms in acyclic chain at atom "
                        f"{atom.GetIdx()} ({symbol})"
                    )

        return True


_PI_BOND_TYPES = (BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC)


class FilterSP3(Evaluation):
    """Rejects molecules with more than `max_sp3` atoms bearing no π bond
    (double, triple or aromatic), whatever their element.

    The value stored in the molecule is the number of such atoms.
    """

    def __init__(self, max_sp3: int = 0) -> None:
        super().__init__("n_sp3_atoms")
        self.max_sp3 = max_sp3

    @override
    def _evaluate(self, molecule: Molecule) -> int:
        mol = molecule.get_representation(MolecularGraph).mol
        count = sum(
            1
            for atom in mol.GetAtoms()
            if not any(bond.GetBondType() in _PI_BOND_TYPES for bond in atom.GetBonds())
        )
        if count > self.max_sp3:
            raise EvaluationError(f"Too many sp3 atoms: {count} > {self.max_sp3}")
        return count


def setup_huckel_filters(
    huckel: bool = True,
    max_sp3: Optional[int] = None,
    realism: Optional[str] = "chembl_zinc",
    hetero: bool = True,
    max_ring_n: int = 3,
) -> list[evaluator.Evaluation]:
    """Return the validity filters of the Hückel enumeration in the order
    Hückel -> ChEMBL/ZINC -> heteroatoms (measured as the fastest in Phase 0).
    FilterSP3, cheap, is placed right after Hückel.

    Args:
        huckel: HuckelCompatible (every sp3 carbon must be terminal).
        max_sp3: FilterSP3, max number of atoms bearing no π bond.
            None: filter not used.
        realism: GCF + ECFP filters on "chembl_zinc" or "chembl".
            None: no realism filter.
        hetero: HeteroatomChainFilter.
        max_ring_n: max number of ring nitrogens for HeteroatomChainFilter.

    Returns:
        list[evaluator.Evaluation]: ordered filters for is_valid_molecule.
    """
    filters: list[evaluator.Evaluation] = []
    if huckel:
        filters.append(evaluator.HuckelCompatible())
    if max_sp3 is not None:
        filters.append(FilterSP3(max_sp3=max_sp3))
    if realism is not None:
        filters += dp.setup_filters(realism)
    if hetero:
        filters.append(HeteroatomChainFilter(max_ring_n=max_ring_n))
    return filters
