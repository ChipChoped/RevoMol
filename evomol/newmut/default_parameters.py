"""
Action space with the new mutations (newmut).

setup_newmut_action_space first calls default_parameters.setup_default_action_space
with the same arguments, then applies the new options. With all the new
options left to False, the action space is exactly the default one.
"""

from evomol import default_parameters as dp
from evomol.newmut import rule_filtered as rf
from evomol.representation import MolecularGraph


# pylint: disable=too-many-arguments
def setup_newmut_action_space(
    with_add_group: bool = False,
    with_remove_group: bool = False,
    insert_carbon: bool = True,
    avoid_bond_breaking: bool = False,
    avoid_bond_forming: bool = False,
    avoid_sp3_on_sp3: bool = False,
    add_atom_multiple_bonds: bool = False,
    bond_change_by_one: bool = False,
    matrix_rules: bool = False,
) -> None:
    """Set the action space of evomol with the new mutations.

    Args:
        with_add_group, with_remove_group, insert_carbon, avoid_bond_breaking,
        avoid_bond_forming, avoid_sp3_on_sp3, add_atom_multiple_bonds,
        bond_change_by_one: same as default_parameters.setup_default_action_space.
        matrix_rules: every action of the action space only lists the actions
            whose result creates no new violation of the matrix rules (R1 to
            R4, see newmut.matrix_rules). False: original behaviour.
    """
    dp.setup_default_action_space(
        with_add_group=with_add_group,
        with_remove_group=with_remove_group,
        insert_carbon=insert_carbon,
        avoid_bond_breaking=avoid_bond_breaking,
        avoid_bond_forming=avoid_bond_forming,
        avoid_sp3_on_sp3=avoid_sp3_on_sp3,
        add_atom_multiple_bonds=add_atom_multiple_bonds,
        bond_change_by_one=bond_change_by_one,
    )

    # the options of the actions may have changed: cached lists are stale
    rf.clear_cache()

    if matrix_rules:
        MolecularGraph.action_space = [
            rf.rule_filtered(action) for action in MolecularGraph.action_space
        ]
