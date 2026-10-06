"""
Change bond for molecular graph representation.
"""

import itertools

from typing_extensions import override

from evomol.action import Action
from evomol.representation import MolecularGraph, Molecule

from .action_molecular_graph import ActionMolGraph


class BreakCycleMG(ActionMolGraph):
    """
    Breaks at least one cycle in the molecular graph by removing a single bond between two atoms.

    """

    avoid_bond_breaking: bool = False
    avoid_bond_forming: bool = False

    def __init__(
        self,
        molecule: Molecule,
        atom1: int,
        atom2: int,
    ) -> None:
        super().__init__(molecule)
        self.atom1: int = atom1
        self.atom2: int = atom2

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BreakCycleMG):
            return False
        return (
            self.molecule == other.molecule
            and self.atom1 == other.atom1
            and self.atom2 == other.atom2
        )

    @override
    def __hash__(self) -> int:
        return hash(self.__repr__())

    @override
    def apply_action(self, new_mol_graph: MolecularGraph) -> None:
        new_mol_graph.set_bond(self.atom1, self.atom2, 0)

    def __repr__(self) -> str:
        return (
            f"BreakCycleMG({self.molecule}, "
            f"{self.atom1}, {self.atom2}, {0})"
        )

    @override
    @classmethod
    def list_actions(cls, molecule: Molecule) -> list[Action]:
        """List possible actions to break a cycle in the molecular graph."""

        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)

        action_list: list[Action] = []

        bridge_bonds_matrix = mol_graph.bridge_bonds_matrix

        charged_or_radical: list[bool] = [
            mol_graph.atom_charged_or_radical(atom)
            for atom in range(mol_graph.nb_atoms)
        ]

        # for each bond
        for atom1, atom2 in itertools.combinations(range(mol_graph.nb_atoms), 2):
            if charged_or_radical[atom1] or charged_or_radical[atom2]:
                continue

            current_bond: int = mol_graph.bond_order(atom1, atom2)

            # determine possible new bonds based on current bond
            if current_bond == 1:
                # Bond decrement
                # only bond that are not bridges can be completely removed.
                # Bonds can be changed only if at least one of the atoms is
                # mutable
                if not bridge_bonds_matrix[atom1][atom2] and not cls.avoid_bond_breaking:
                    action_list.append(BreakCycleMG(molecule, atom1, atom2))

        return action_list
