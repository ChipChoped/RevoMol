"""
AddAtom of newmut (formerly AppendAtomRuleMG, "AppR" of the specification):
add an atom, with the sp3-sp3 rule of reference_app.js (appendOrder), and the
result must respect the matrix rules (the action checks them itself).

setup_newmut_action_space puts it in place of the AddAtomMG of evomol (same
name, so the same traces), unless evomol_add_atom=True.

For each uncharged atom with at least one hydrogen and each accepted element,
the new atom is bonded by a single, double or triple bond, as long as the
anchor has enough hydrogens and the element accepts the order (=CH2, #CH, =O,
=NH, #N, on an s carbon as on an s nitrogen…). One exception, the sp3-sp3 rule
of the workshop: a carbon added onto a carbon that has only single bonds (an
"s" carbon) is never bonded by a single bond. Then the matrix rules decide.

The candidates and their check are computed on the bond-order matrix (see
newmut.rule_filtered and newmut.fast_rules).
"""

from typing_extensions import override

from evomol.action import Action
from evomol.action.molecular_graph import AddAtomMG as EvomolAddAtomMG
from evomol.newmut.matrix_rules import max_valence
from evomol.newmut.rule_filtered import RuleCheckedMixin
from evomol.representation import MolecularGraph, Molecule


class AddAtomMG(RuleCheckedMixin, EvomolAddAtomMG):
    """
    Add an atom to the molecular graph, by a single, double or triple bond,
    never as an sp3 carbon on an sp3 carbon, respecting the matrix rules.
    """

    @override
    @classmethod
    def list_unchecked_actions(cls, molecule: Molecule) -> list[Action]:
        """List the atoms that can be added, before the matrix rules.

        - molecule size >= Molecule.max_heavy_atoms: no action;
        - empty molecule: any atom;
        - otherwise, for each uncharged atom with at least one hydrogen and
          each accepted element: bond order 1 to 3, up to the number of
          hydrogens of the anchor and the valence of the element; a C on an
          "s" carbon only by a double or triple bond.
        """
        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)

        if mol_graph.nb_atoms >= Molecule.max_heavy_atoms:
            return []

        if mol_graph.nb_atoms == 0:
            return [AddAtomMG(molecule, 0, atom_type) for atom_type in Molecule.accepted_atoms]

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
                max_order = min(free, max_valence(atom_type), 3)
                # sp3-sp3 rule: no C bonded to an s carbon by a single bond
                min_order = 2 if atom_type == "C" and s_carbon else 1
                for bond_order in range(min_order, max_order + 1):
                    action_list.append(AddAtomMG(molecule, atom_idx, atom_type, bond_order))
        return action_list
