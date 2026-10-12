"""
Annelate ("Annel" of the specification): fuse a ring on a bond, with a
fragment of the vocabulary that has two attachment points.

For a bond i-j and a fragment [*:1]...[*:2], the fragment atom bonded to [*:1]
is bonded to i and the one bonded to [*:2] to j, by single bonds: a new ring
is fused on the bond i-j. The two orientations (i <- [*:1], j <- [*:2] and the
reverse) are two actions; the reverse one is not listed when the fragment is
symmetric (it would give the same molecule twice as often).

Valid if the bond exists, if i and j have each at least one hydrogen, if the
molecule stays within Molecule.max_heavy_atoms, if every element of the
fragment is accepted, and if the result respects the matrix rules.
"""

from __future__ import annotations

from typing_extensions import override

from evomol.action import Action
from evomol.action.molecular_graph.action_molecular_graph import ActionMolGraph
from evomol.newmut.fragments import ANNELATION_FRAGMENTS, parse_fragment
from evomol.newmut.rule_filtered import RuleCheckedMixin
from evomol.representation import MolecularGraph, Molecule


class AnnelateMG(RuleCheckedMixin, ActionMolGraph):
    """
    Fuse a ring on a bond of the molecular graph.
    """

    # name -> SMILES with two attachment points [*:1] and [*:2] (set by
    # setup_newmut_action_space)
    fragments: dict[str, str] = dict(ANNELATION_FRAGMENTS)

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        molecule: Molecule,
        atom1: int,
        atom2: int,
        name: str,
        fragment_smiles: str,
    ) -> None:
        """Initialize the action.

        Args:
            molecule: molecule to modify.
            atom1: atom bonded to the attachment atom [*:1] of the fragment.
            atom2: atom bonded to the attachment atom [*:2] of the fragment.
            name: name of the fragment in the vocabulary.
            fragment_smiles: SMILES of the fragment with its attachment points.
        """
        super().__init__(molecule)
        self.atom1: int = atom1
        self.atom2: int = atom2
        self.name: str = name
        self.fragment_smiles: str = fragment_smiles

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnnelateMG):
            return False
        return (
            self.molecule == other.molecule
            and self.atom1 == other.atom1
            and self.atom2 == other.atom2
            and self.fragment_smiles == other.fragment_smiles
        )

    @override
    def __hash__(self) -> int:
        return hash(self.__repr__())

    @override
    def apply_action(self, new_mol_graph: MolecularGraph) -> None:
        fragment = parse_fragment(self.fragment_smiles)
        nb_atoms = new_mol_graph.nb_atoms
        new_mol_graph.add_group(self.atom1, fragment.smiles, fragment.attach[0])
        new_mol_graph.set_bond(self.atom2, nb_atoms + fragment.attach[1], 1)

    def __repr__(self) -> str:
        return (
            f"AnnelateMolGraph({self.molecule}, {self.atom1}, {self.atom2}, "
            f"{self.name})"
        )

    @override
    @classmethod
    def list_unchecked_actions(cls, molecule: Molecule) -> list[Action]:
        """For each bond between uncharged atoms with at least one hydrogen
        each, each fragment that fits and each orientation, before the matrix
        rules."""
        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)
        accepted = set(Molecule.accepted_atoms)
        fragments = [
            (name, smiles, parse_fragment(smiles))
            for name, smiles in cls.fragments.items()
        ]
        fragments = [
            (name, smiles, fragment)
            for name, smiles, fragment in fragments
            if len(fragment.attach) == 2
            and fragment.elements <= accepted
            and mol_graph.nb_atoms + fragment.nb_atoms <= Molecule.max_heavy_atoms
        ]
        if not fragments:
            return []

        implicit_valences = mol_graph.implicit_valences
        free = [
            implicit_valences[atom] >= 1 and not mol_graph.atom_charged_or_radical(atom)
            for atom in range(mol_graph.nb_atoms)
        ]

        action_list: list[Action] = []
        for atom1, atom2 in sorted(
            (min(bond), max(bond)) for bond in mol_graph.bonds
        ):
            if not (free[atom1] and free[atom2]):
                continue
            for name, smiles, fragment in fragments:
                action_list.append(AnnelateMG(molecule, atom1, atom2, name, smiles))
                if not fragment.symmetric:
                    action_list.append(AnnelateMG(molecule, atom2, atom1, name, smiles))
        return action_list
