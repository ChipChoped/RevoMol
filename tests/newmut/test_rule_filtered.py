"""
Tests for the wrapper that applies the matrix rules to existing actions (newmut).
"""

import pytest

from evomol import default_parameters as dp
from evomol.action import molecular_graph as mg
from evomol.newmut import matrix_rules as mr
from evomol.newmut import rule_filtered as rf
from evomol.newmut.matrix_rules import result_is_allowed
from evomol.representation import MolecularGraph, Molecule

ACTIONS = [
    mg.AddAtomMG,
    mg.ChangeBondMG,
    mg.CutAtomMG,
    mg.InsertCarbonMG,
    mg.MoveGroupMG,
    mg.RemoveAtomMG,
    mg.SubstituteAtomMG,
]

MOLECULES = [
    "",
    "C",
    "CC",
    "C=CC",
    "c1ccccc1",
    "Cc1ccccc1",
    "CC(=O)Nc1ccc(O)cc1",
    "C1=CSC=C1",
    "c1ccc2ccccc2c1",
    "C1CCCCC1",
    "OC(=O)c1ccncc1",
    "c1ccc(cc1)[N+](=O)[O-]",
]


@pytest.fixture(autouse=True, scope="module")
def setup_parameters() -> None:
    """set the parameters for the tests."""
    dp.setup_default_parameters(
        accepted_atoms=["C", "O", "N", "F", "S"],
        max_heavy_atoms=38,
    )
    dp.setup_default_action_space()


@pytest.mark.parametrize("action_class", ACTIONS, ids=lambda c: c.__name__)
def test_same_name(action_class: type) -> None:
    """The wrapper keeps the name of the wrapped action."""
    wrapper = rf.rule_filtered(action_class)
    assert wrapper is not action_class
    assert issubclass(wrapper, action_class)
    assert wrapper.class_name() == action_class.class_name()
    assert wrapper.__name__ == action_class.__name__
    # same wrapper for the same class, a wrapper is not wrapped again
    assert rf.rule_filtered(action_class) is wrapper
    assert rf.rule_filtered(wrapper) is wrapper


def test_change_bond_rule() -> None:
    """ChBR of the specification is ChangeBondMG restricted by the rules."""
    assert rf.ChangeBondRuleMG is rf.rule_filtered(mg.ChangeBondMG)


@pytest.mark.parametrize("smiles", MOLECULES)
@pytest.mark.parametrize("action_class", ACTIONS, ids=lambda c: c.__name__)
def test_listed_actions_respect_rules(action_class: type, smiles: str) -> None:
    """Every listed action is an action of the wrapped class whose result
    (once applied for real) creates no new violation."""
    molecule = Molecule(smiles)
    inner = [repr(action) for action in action_class.list_actions(molecule)]
    filtered = rf.rule_filtered(action_class).list_actions(molecule)
    assert all(repr(action) in inner for action in filtered)
    before = molecule.get_representation(MolecularGraph)
    for action in filtered:
        after = action.apply().get_representation(MolecularGraph)
        assert result_is_allowed(before, after), action


def test_filtering() -> None:
    """Toluene (V1): a C added on the methyl creates two bonded s carbons (R2a),
    an O a Cs-OH (R2b); an O on a ring carbon is allowed."""
    molecule = Molecule("Cc1ccccc1")
    wrapper = rf.rule_filtered(mg.AddAtomMG)
    listed = {(a.index_atom, a.atom_type) for a in wrapper.list_actions(molecule)}
    assert (0, "C") not in listed
    assert (0, "O") not in listed
    assert (2, "O") in listed


def test_filtering_v0() -> None:
    """Toluene (V0): a C added on the methyl creates two bonded s carbons, an O
    is allowed."""
    mr.RULES_VERSION = "V0"
    rf.clear_cache()
    try:
        molecule = Molecule("Cc1ccccc1")
        wrapper = rf.rule_filtered(mg.AddAtomMG)
        listed = {(a.index_atom, a.atom_type) for a in wrapper.list_actions(molecule)}
        assert (0, "C") not in listed
        assert (0, "O") in listed
    finally:
        mr.RULES_VERSION = "V1"
        rf.clear_cache()


def test_cache_same_result() -> None:
    """The cache gives the same actions as a computation without cache."""
    wrapper = rf.rule_filtered(mg.ChangeBondMG)
    try:
        for smiles in MOLECULES:
            rf.set_cache_size(0)
            without_cache = [repr(a) for a in wrapper.list_actions(Molecule(smiles))]
            rf.set_cache_size(10_000)
            first = [repr(a) for a in wrapper.list_actions(Molecule(smiles))]
            second = [repr(a) for a in wrapper.list_actions(Molecule(smiles))]
            assert without_cache == first == second
    finally:
        rf.set_cache_size(10_000)
