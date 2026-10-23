"""
Non-regression of newmut: with the new options off, evomol behaves exactly as
before; with lazy_rules, the draw is the same as with the filtered lists.
"""

import random
from collections import Counter

import pytest

from evomol import default_parameters as dp
from evomol.action import molecular_graph as mg
from evomol.newmut import rule_filtered as rf
from evomol.newmut.default_parameters import setup_newmut_action_space
from evomol.newmut.evolutionary_algorithm import warn_rule_violations
from evomol.newmut.neighborhood_strategy import (
    NewMutNeighborhoodStrategy,
    draw_allowed_action,
)
from evomol.representation import MolecularGraph, Molecule
from evomol.search.neighborhood_strategy import RandomNeighborhoodStrategy

MOLECULES = ["c1ccccc1C", "CC(=O)Nc1ccc(O)cc1", "C1=CSC=C1", "C=CC=O", "c1ccc2ccccc2c1"]

DEFAULT_OPTIONS = [
    {},
    {"with_add_group": True, "with_remove_group": True},
    {"insert_carbon": False, "avoid_sp3_on_sp3": True},
    {"add_atom_multiple_bonds": True, "bond_change_by_one": True},
    {"avoid_bond_breaking": True, "avoid_bond_forming": True},
]


@pytest.fixture(autouse=True)
def setup_parameters() -> None:
    """set the parameters for the tests."""
    dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "F", "S"], max_heavy_atoms=38)
    yield
    dp.setup_default_action_space()


def snapshot() -> tuple:
    """Action space and options of the actions."""
    return (
        list(MolecularGraph.action_space),
        mg.AddAtomMG.avoid_sp3_on_sp3,
        mg.AddAtomMG.allow_multiple_bonds,
        mg.ChangeBondMG.by_one_step,
        mg.ChangeBondMG.avoid_bond_breaking,
        mg.ChangeBondMG.avoid_bond_forming,
    )


def nb_actions() -> dict[str, list[int]]:
    """Number of actions of each type, for each molecule."""
    return {
        action.class_name(): [len(action.list_actions(Molecule(s))) for s in MOLECULES]
        for action in MolecularGraph.action_space
    }


@pytest.mark.parametrize("options", DEFAULT_OPTIONS)
def test_same_action_space(options: dict) -> None:
    """Without new option (and with the AddAtomMG of evomol): same action space, same number of actions."""
    dp.setup_default_action_space(**options)
    expected = (snapshot(), nb_actions())
    setup_newmut_action_space(**options, evomol_add_atom=True)
    assert (snapshot(), nb_actions()) == expected


def test_all_options() -> None:
    """All options: replaced, wrapped and added actions."""
    setup_newmut_action_space(
        matrix_rules=True,
        substitution_shift=True,
        graft=True,
        annelate=True,
        flip_path=True,
        aromatize=True,
    )
    names = [action.class_name() for action in MolecularGraph.action_space]
    assert names == [
        "AddAtomMG",
        "ChangeBondMG",
        "CutAtomMG",
        "InsertCarbonMG",
        "MoveGroupMG",
        "RemoveAtomMG",
        "SubstituteShiftMG",
        "GraftFragmentMG",
        "AnnelateMG",
        "FlipAlternatingPathMG",
        "AromatizeMG",
    ]
    assert all(getattr(a, "checks_rules", False) for a in MolecularGraph.action_space)


def test_wrong_fragment() -> None:
    """A graft fragment needs one attachment point."""
    with pytest.raises(ValueError):
        setup_newmut_action_space(graft=True, graft_fragments={"bad": "[*:1]CC[*:2]"})


@pytest.mark.parametrize("smiles", MOLECULES)
def test_lazy_off_same_mutation(smiles: str) -> None:
    """lazy_rules=False: same mutations as RandomNeighborhoodStrategy."""
    setup_newmut_action_space(matrix_rules=True)
    results = []
    for strategy in (
        RandomNeighborhoodStrategy(depth=2),
        NewMutNeighborhoodStrategy(depth=2, lazy_rules=False),
    ):
        random.seed(3)
        results.append([str(strategy.random_mutation(Molecule(smiles))) for _ in range(20)])
    assert results[0] == results[1]


def test_lazy_draw_uniform() -> None:
    """lazy_rules=True: each allowed action is drawn, uniformly."""
    setup_newmut_action_space(matrix_rules=True)
    wrapper = rf.rule_filtered(mg.ChangeBondMG)
    molecule = Molecule("Cc1ccccc1")
    allowed = [repr(a) for a in wrapper.list_actions(molecule)]
    random.seed(0)
    nb_draws = 200 * len(allowed)
    counts = Counter(repr(draw_allowed_action(wrapper, molecule)) for _ in range(nb_draws))
    assert set(counts) == set(allowed)
    assert all(120 <= count <= 280 for count in counts.values())


def test_lazy_draw_none() -> None:
    """lazy_rules=True: no allowed action gives None."""
    setup_newmut_action_space(flip_path=True)
    from evomol.newmut.flip_path import FlipAlternatingPathMG  # pylint: disable=import-outside-toplevel

    assert draw_allowed_action(FlipAlternatingPathMG, Molecule("c1ccccc1")) is None


def test_warn_rule_violations() -> None:
    """Initial molecules outside the rules are counted (and logged)."""
    population = [Molecule("c1ccccc1"), Molecule("CCC"), Molecule("C1CCCCC1")]
    assert warn_rule_violations(population) == 2
