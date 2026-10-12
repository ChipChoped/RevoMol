"""
AppendAtomRule ("AppR" of the specification): add an atom, with the sp3-sp3
rule of reference_app.js (appendOrder).

The new atom is bonded to an existing atom by a single bond, except when a
carbon is added onto a carbon that has only single bonds (an "s" carbon):
the new bond is then double if the anchor has at least 2 hydrogens, and the
action does not exist otherwise. The result must also respect the matrix rules.
"""

from typing_extensions import override

from evomol.action import Action
from evomol.action.molecular_graph import AddAtomMG
from evomol.newmut.rule_filtered import RuleCheckedMixin
from evomol.representation import MolecularGraph, Molecule


class AppendAtomRuleMG(RuleCheckedMixin, AddAtomMG):
    """
    Add an atom to the molecular graph, never as an sp3 carbon on an sp3 carbon.
    """

    @override
    def __repr__(self) -> str:
        order = "" if self.bond_order == 1 else f", {self.bond_order}"
        return (
            f"AppendAtomRuleMolGraph({self.molecule}, {self.index_atom}, "
            f"{self.atom_type}{order})"
        )

    @override
    @classmethod
    def list_unchecked_actions(cls, molecule: Molecule) -> list[Action]:
        """List the atoms that can be added, before the matrix rules.

        - molecule size >= Molecule.max_heavy_atoms: no action;
        - empty molecule: any atom;
        - otherwise, for each uncharged atom with at least one hydrogen and
          each accepted atom: single bond, except C on an "s" carbon (double
          bond if the anchor has >= 2 hydrogens, no action otherwise).
        """
        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)

        if mol_graph.nb_atoms >= Molecule.max_heavy_atoms:
            return []

        if mol_graph.nb_atoms == 0:
            return [
                AppendAtomRuleMG(molecule, 0, atom_type)
                for atom_type in Molecule.accepted_atoms
            ]

        implicit_valences = mol_graph.implicit_valences
        action_list: list[Action] = []
        for atom_idx in range(mol_graph.nb_atoms):
            free = implicit_valences[atom_idx]
            if free < 1 or mol_graph.atom_charged_or_radical(atom_idx):
                continue
            s_carbon = mol_graph.atom_type(atom_idx) == "C" and all(
                mol_graph.bond_order(atom_idx, other) == 1
                for other in mol_graph.atoms_bonded_to(atom_idx)
            )
            for atom_type in Molecule.accepted_atoms:
                bond_order = 1
                if atom_type == "C" and s_carbon:
                    # sp3-sp3 rule: the new C-C bond is double
                    if free < 2:
                        continue
                    bond_order = 2
                action_list.append(
                    AppendAtomRuleMG(molecule, atom_idx, atom_type, bond_order)
                )
        return action_list
