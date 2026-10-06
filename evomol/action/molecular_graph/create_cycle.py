"""
Change bond for molecular graph representation.
"""

import itertools

from typing_extensions import override

from evomol.action import Action
from evomol.representation import MolecularGraph, Molecule

from .action_molecular_graph import ActionMolGraph


class CreateCycleMG(ActionMolGraph):
    """
    Creates at least one cycle in the molecular graph by adding a single bond between two unbounded atoms.

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
        if not isinstance(other, CreateCycleMG):
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
        new_mol_graph.set_bond(self.atom1, self.atom2, 1)

    def __repr__(self) -> str:
        return (
            f"CreateCycleMG({self.molecule}, "
            f"{self.atom1}, {self.atom2}, {1})"
        )

    @override
    @classmethod
    def list_actions(cls, molecule: Molecule) -> list[Action]:
        """List possible actions to create a cycle in the molecular graph."""

        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)

        action_list: list[Action] = []

        implicit_valences = mol_graph.implicit_valences

        charged_or_radical: list[bool] = [
            mol_graph.atom_charged_or_radical(atom)
            for atom in range(mol_graph.nb_atoms)
        ]

        # for each bond
        for atom1, atom2 in itertools.combinations(range(mol_graph.nb_atoms), 2):
            if charged_or_radical[atom1] or charged_or_radical[atom2]:
                continue

            current_bond: int = mol_graph.bond_order(atom1, atom2)

            # the max bond that can be formed between atom1 and atom2
            # is the minimum of the implicit valence of the two atoms
            # plus the current bond
            max_bond: int = (
                min(implicit_valences[atom1], implicit_valences[atom2]) + current_bond
            )

            # determine possible new bonds based on current bond
            new_possible_bonds: list[int]
            if current_bond == 0:
                # Bond increment
                # Bond can be incremented only if the new bond is less than
                # the maximum bond that can be formed between the two atoms
                if max_bond >= 1:
                    if not (cls.avoid_bond_forming and current_bond == 0):
                        action_list.append(CreateCycleMG(molecule, atom1, atom2))

        return action_list
