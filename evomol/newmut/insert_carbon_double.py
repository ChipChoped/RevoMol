"""
InsertCarbon that keeps a double bond (newmut).

Same actions as InsertCarbonMG, except when the carbon is inserted into a
double bond between two uncharged atoms: instead of two single bonds
(a-C-b, which removes the double bond), two actions keep one double bond on
either side, a=C-b and a-C=b. The other insertions are unchanged.

Used by setup_newmut_action_space(insert_carbon_keep_double=True).
"""

from __future__ import annotations

from typing_extensions import override

from evomol.action import Action
from evomol.action import molecular_graph as mg
from evomol.representation import MolecularGraph, Molecule


class InsertCarbonDoubleMG(mg.InsertCarbonMG):
    """InsertCarbonMG whose insertion into a double bond keeps a double bond."""

    # class of the listed actions (random_walk compares the type of the actions)
    instance_class = mg.InsertCarbonMG

    @override
    @classmethod
    def class_name(cls) -> str:
        """Same name as InsertCarbonMG (same traces)."""
        return mg.InsertCarbonMG.class_name()

    @override
    @classmethod
    def list_actions(cls, molecule: Molecule) -> list[Action]:
        """Actions of InsertCarbonMG; an insertion into a double bond (two single
        bonds to the new carbon) becomes the two insertions a=C-b and a-C=b."""
        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)
        action_list: list[Action] = []
        for action in mg.InsertCarbonMG.list_actions(molecule):
            assert isinstance(action, mg.InsertCarbonMG)
            atom1, atom2 = action.atom1_idx_to_bond, action.atom2_idx_to_bond
            if (
                mol_graph.bond_order(atom1, atom2) == 2
                and (action.bond_to_1, action.bond_to_2) == (1, 1)
            ):
                action_list.append(mg.InsertCarbonMG(molecule, atom1, 2, atom2, 1))
                action_list.append(mg.InsertCarbonMG(molecule, atom1, 1, atom2, 2))
            else:
                action_list.append(action)
        return action_list
