"""
GraftFragment ("Graft" of the specification): graft a fragment of the
vocabulary on an atom.

Matrix operation: block concatenation [[M, B], [B^T, F]] with a single
connection (anchor, attachment atom of the fragment) = 1. Valid if the anchor
has at least one hydrogen, if the molecule stays within
Molecule.max_heavy_atoms, if every element of the fragment is accepted, and if
the result respects the matrix rules.
"""

from __future__ import annotations

from typing_extensions import override

from evomol.action import Action
from evomol.action.molecular_graph.action_molecular_graph import ActionMolGraph
from evomol.newmut.fragments import GRAFT_FRAGMENTS, parse_fragment
from evomol.newmut.rule_filtered import RuleCheckedMixin
from evomol.representation import MolecularGraph, Molecule


class GraftFragmentMG(RuleCheckedMixin, ActionMolGraph):
    """
    Graft a fragment of the vocabulary on an atom of the molecular graph.
    """

    # name -> SMILES with one attachment point [*:1] (set by
    # setup_newmut_action_space)
    fragments: dict[str, str] = dict(GRAFT_FRAGMENTS)

    def __init__(
        self, molecule: Molecule, anchor: int, name: str, fragment_smiles: str
    ) -> None:
        """Initialize the action.

        Args:
            molecule: molecule to modify.
            anchor: atom of the molecule bonded to the fragment.
            name: name of the fragment in the vocabulary.
            fragment_smiles: SMILES of the fragment with its attachment point.
        """
        super().__init__(molecule)
        self.anchor: int = anchor
        self.name: str = name
        self.fragment_smiles: str = fragment_smiles

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraftFragmentMG):
            return False
        return (
            self.molecule == other.molecule
            and self.anchor == other.anchor
            and self.fragment_smiles == other.fragment_smiles
        )

    @override
    def __hash__(self) -> int:
        return hash(self.__repr__())

    @override
    def apply_action(self, new_mol_graph: MolecularGraph) -> None:
        fragment = parse_fragment(self.fragment_smiles)
        new_mol_graph.add_group(self.anchor, fragment.smiles, fragment.attach[0])

    def __repr__(self) -> str:
        return f"GraftFragmentMolGraph({self.molecule}, {self.anchor}, {self.name})"

    @override
    @classmethod
    def list_unchecked_actions(cls, molecule: Molecule) -> list[Action]:
        """For each uncharged atom with at least one hydrogen and each fragment
        that fits, before the matrix rules."""
        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)
        accepted = set(Molecule.accepted_atoms)
        fragments = [
            (name, smiles, parse_fragment(smiles))
            for name, smiles in cls.fragments.items()
        ]
        fragments = [
            (name, smiles, fragment)
            for name, smiles, fragment in fragments
            if len(fragment.attach) == 1
            and fragment.elements <= accepted
            and mol_graph.nb_atoms + fragment.nb_atoms <= Molecule.max_heavy_atoms
        ]

        implicit_valences = mol_graph.implicit_valences
        return [
            GraftFragmentMG(molecule, anchor, name, smiles)
            for anchor in range(mol_graph.nb_atoms)
            if implicit_valences[anchor] >= 1
            and not mol_graph.atom_charged_or_radical(anchor)
            for name, smiles, _ in fragments
        ]
