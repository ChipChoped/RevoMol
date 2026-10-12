"""
Action space with the new mutations (newmut).

setup_newmut_action_space first calls default_parameters.setup_default_action_space
with the same arguments, then applies the new options. With all the new
options left to False, the action space is exactly the default one.
"""

from __future__ import annotations

from evomol import default_parameters as dp
from evomol.action import Action
from evomol.action import molecular_graph as mg
from evomol.newmut import rule_filtered as rf
from evomol.newmut.annelate import AnnelateMG
from evomol.newmut.append_atom_rule import AppendAtomRuleMG
from evomol.newmut.aromatize import AromatizeMG
from evomol.newmut.flip_path import FlipAlternatingPathMG
from evomol.newmut.fragments import ANNELATION_FRAGMENTS, GRAFT_FRAGMENTS, parse_fragment
from evomol.newmut.graft import GraftFragmentMG
from evomol.newmut.substitute_shift import SubstituteShiftMG
from evomol.representation import MolecularGraph


def _replace(
    space: list[type[Action]], old: type[Action], new: type[Action]
) -> list[type[Action]]:
    """Replace old by new at the same position (new appended if old is absent)."""
    if old in space:
        return [new if action is old else action for action in space]
    return [*space, new]


# pylint: disable=too-many-arguments,too-many-locals
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
    append_atom_rule: bool = False,
    substitution_shift: bool = False,
    graft: bool = False,
    annelate: bool = False,
    flip_path: bool = False,
    aromatize: bool = False,
    graft_fragments: dict[str, str] | None = None,
    annelation_fragments: dict[str, str] | None = None,
) -> None:
    """Set the action space of evomol with the new mutations.

    Args:
        with_add_group, with_remove_group, insert_carbon, avoid_bond_breaking,
        avoid_bond_forming, avoid_sp3_on_sp3, add_atom_multiple_bonds,
        bond_change_by_one: same as default_parameters.setup_default_action_space.
        matrix_rules: the existing actions only list the actions whose result
            creates no new violation of the matrix rules (R1 to R4, see
            newmut.matrix_rules). The new actions always apply the rules.
        append_atom_rule: AppendAtomRuleMG replaces AddAtomMG.
        substitution_shift: SubstituteShiftMG replaces SubstituteAtomMG.
        graft: add GraftFragmentMG.
        annelate: add AnnelateMG.
        flip_path: add FlipAlternatingPathMG.
        aromatize: add AromatizeMG.
        graft_fragments: vocabulary of GraftFragmentMG, name -> SMILES with
            one attachment point [*:1]. None: fragments.GRAFT_FRAGMENTS.
        annelation_fragments: vocabulary of AnnelateMG, name -> SMILES with
            two attachment points [*:1] and [*:2]. None:
            fragments.ANNELATION_FRAGMENTS.

    With all the new options left to False, the action space is the one of
    setup_default_action_space.

    Raises:
        ValueError: a fragment SMILES is not readable or has a wrong number of
            attachment points.
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

    GraftFragmentMG.fragments = dict(
        GRAFT_FRAGMENTS if graft_fragments is None else graft_fragments
    )
    AnnelateMG.fragments = dict(
        ANNELATION_FRAGMENTS if annelation_fragments is None else annelation_fragments
    )
    for vocabulary, nb_points in (
        (GraftFragmentMG.fragments, 1),
        (AnnelateMG.fragments, 2),
    ):
        for name, smiles in vocabulary.items():
            if len(parse_fragment(smiles).attach) != nb_points:
                raise ValueError(
                    f"Fragment {name} ({smiles}) must have {nb_points} "
                    "attachment point(s)"
                )

    space: list[type[Action]] = list(MolecularGraph.action_space)
    if append_atom_rule:
        space = _replace(space, mg.AddAtomMG, AppendAtomRuleMG)
    if substitution_shift:
        space = _replace(space, mg.SubstituteAtomMG, SubstituteShiftMG)
    if matrix_rules:
        # the new actions already apply the rules: returned unchanged
        space = [rf.rule_filtered(action) for action in space]  # type: ignore[arg-type]
    for enabled, action in (
        (graft, GraftFragmentMG),
        (annelate, AnnelateMG),
        (flip_path, FlipAlternatingPathMG),
        (aromatize, AromatizeMG),
    ):
        if enabled:
            space.append(action)

    MolecularGraph.action_space = space
