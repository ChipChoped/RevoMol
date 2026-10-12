"""
Tests for the new actions (newmut).

For each action: a positive case, a refused case, and the property that every
listed action applies without error and gives a result that creates no new
violation of the matrix rules, on varied molecules.
"""

import pytest

from evomol import default_parameters as dp
from evomol.action import Action
from evomol.newmut.append_atom_rule import AppendAtomRuleMG
from evomol.newmut.aromatize import AromatizeMG
from evomol.newmut.flip_path import FlipAlternatingPathMG
from evomol.newmut.matrix_rules import result_is_allowed
from evomol.newmut.substitute_shift import SubstituteShiftMG
from evomol.representation import MolecularGraph, Molecule

# varied molecules for the property tests
MOLECULES = [
    "",
    "C",
    "CC",
    "C=CC=C",
    "C#CC=C",
    "c1ccccc1",
    "Cc1ccccc1",
    "C=Cc1ccccc1",
    "C=Cc1cccs1",
    "CC1=CC=C(C)S1",
    "C1=CNC=CC1",
    "C1=CC=CC1",
    "c1ccc2ccccc2c1",
    "c1ccncc1",
    "O=C1C=CC(=O)C=C1",
    "Nc1ccc(O)cc1",
    "CC(=O)Nc1ccc(O)cc1",
    "N#Cc1ccc(C=O)s1",
    "C1COc2cscc2O1",
    "c1ccc(cc1)[N+](=O)[O-]",
    "c1ccc2c(c1)ccc1ccccc12",
]

RULE_ACTIONS: list[type[Action]] = [
    AppendAtomRuleMG,
    SubstituteShiftMG,
    FlipAlternatingPathMG,
    AromatizeMG,
]


@pytest.fixture(autouse=True, scope="module")
def setup_parameters() -> None:
    """set the parameters for the tests."""
    dp.setup_default_parameters(
        accepted_atoms=["C", "O", "N", "F", "S"],
        max_heavy_atoms=38,
    )
    dp.setup_default_action_space()


def canonical(smiles: str) -> str:
    """Aromatic canonical SMILES (independent of the Kekulé structure)."""
    return Molecule(smiles).get_representation(MolecularGraph).aromatic_canonical_smiles


def results(action_class: type[Action], smiles: str) -> set[str]:
    """Canonical SMILES of the molecules produced by the listed actions."""
    return {
        canonical(str(action.apply()))
        for action in action_class.list_actions(Molecule(smiles))
    }


########################################
#   Property: listed actions are allowed
########################################
@pytest.mark.parametrize("smiles", MOLECULES)
@pytest.mark.parametrize("action_class", RULE_ACTIONS, ids=lambda c: c.__name__)
def test_listed_actions_respect_rules(action_class: type[Action], smiles: str) -> None:
    """Every listed action applies and creates no new violation."""
    molecule = Molecule(smiles)
    before = molecule.get_representation(MolecularGraph)
    for action in action_class.list_actions(molecule):
        after = action.apply().get_representation(MolecularGraph)
        assert result_is_allowed(before, after), action


@pytest.mark.parametrize("smiles", MOLECULES)
@pytest.mark.parametrize("action_class", RULE_ACTIONS, ids=lambda c: c.__name__)
def test_unchecked_superset(action_class: type[Action], smiles: str) -> None:
    """The listed actions are the allowed candidates (lazy mode relies on it)."""
    molecule = Molecule(smiles)
    candidates = action_class.list_unchecked_actions(molecule)  # type: ignore[attr-defined]
    allowed = [
        repr(action)
        for action in candidates
        if action_class.action_is_allowed(action, molecule)  # type: ignore[attr-defined]
    ]
    assert allowed == [repr(action) for action in action_class.list_actions(molecule)]


########################################
#            AppendAtomRule
########################################
def test_append_atom_rule_empty() -> None:
    """Empty molecule: any accepted atom."""
    assert results(AppendAtomRuleMG, "") == {canonical(s) for s in "CONFS"}


def test_append_atom_rule_double_bond() -> None:
    """A C added on an s carbon with >= 2 H is bonded by a double bond."""
    produced = results(AppendAtomRuleMG, "Cc1ccccc1")
    assert canonical("C=Cc1ccccc1") in produced
    assert canonical("CCc1ccccc1") not in produced


def test_append_atom_rule_single_hydrogen() -> None:
    """An s carbon with a single H receives no carbon."""
    molecule = Molecule("CC(C)(C)c1ccccc1C(C)c1ccccc1")
    mol_graph = molecule.get_representation(MolecularGraph)
    ch_index = next(
        i
        for i in range(mol_graph.nb_atoms)
        if mol_graph.atom_type(i) == "C"
        and mol_graph.implicit_valences[i] == 1
        and all(mol_graph.bond_order(i, j) == 1 for j in mol_graph.atoms_bonded_to(i))
    )
    assert not any(
        action.index_atom == ch_index and action.atom_type == "C"
        for action in AppendAtomRuleMG.list_actions(molecule)
    )


def test_append_atom_rule_max_size() -> None:
    """No atom added beyond Molecule.max_heavy_atoms."""
    try:
        dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "F", "S"], max_heavy_atoms=6)
        assert not AppendAtomRuleMG.list_actions(Molecule("c1ccccc1"))
    finally:
        dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "F", "S"], max_heavy_atoms=38)


########################################
#            SubstituteShift
########################################
@pytest.mark.parametrize(
    ("smiles", "atom", "element", "expected"),
    (
        pytest.param("C=Cc1ccccc1", 0, "O", "O=Cc1ccccc1", id="styrene -> benzaldehyde"),
        pytest.param("c1ccccc1", 0, "O", "C1=CCOC=C1", id="benzene -> 2H-pyran"),
        pytest.param(
            "C=Cc1cccs1", 0, "O", "O=Cc1cccs1", id="vinylthiophene -> carbaldehyde"
        ),
        pytest.param("c1ccccc1", 0, "N", "c1ccncc1", id="benzene -> pyridine"),
    ),
)
def test_substitute_shift(smiles: str, atom: int, element: str, expected: str) -> None:
    """Examples of the specification."""
    actions = [
        action
        for action in SubstituteShiftMG.list_actions(Molecule(smiles))
        if action.atom_idx == atom and action.new_type == element
    ]
    assert len(actions) == 1
    assert canonical(str(actions[0].apply())) == canonical(expected)


def test_substitute_shift_refused() -> None:
    """A ring carbon of benzene cannot become F (valence)."""
    assert not any(
        action.new_type == "F"
        for action in SubstituteShiftMG.list_actions(Molecule("c1ccccc1"))
    )


def test_substitute_shift_moves_hole() -> None:
    """Toluene, ring carbon -> O: the double bonds are shifted so that no
    rule is broken (the gap left by C= is moved along the ring)."""
    molecule = Molecule("Cc1ccccc1")
    produced = {
        canonical(str(action.apply()))
        for action in SubstituteShiftMG.list_actions(molecule)
        if action.new_type == "O"
    }
    before = molecule.get_representation(MolecularGraph)
    for smiles in produced:
        assert result_is_allowed(before, MolecularGraph(smiles))
    assert produced


########################################
#          FlipAlternatingPath
########################################
def test_flip_quinoid() -> None:
    """Example of the specification: quinoid switch of 2,5-dimethylthiophene."""
    assert canonical("C=C1C=CC(=C)S1") in results(FlipAlternatingPathMG, "CC1=CC=C(C)S1")


def test_flip_butene() -> None:
    """CC=CC: the only flip gives butadiene."""
    assert results(FlipAlternatingPathMG, "CC=CC") == {canonical("C=CC=C")}


def test_flip_refused() -> None:
    """Benzene: every flip would give a carbon two double bonds."""
    molecule = Molecule("c1ccccc1")
    assert FlipAlternatingPathMG.list_unchecked_actions(molecule)
    assert not FlipAlternatingPathMG.list_actions(molecule)


def test_flip_path_alternates() -> None:
    """Listed paths alternate 1, 2, ..., 1 and have at least 3 bonds."""
    molecule = Molecule("CC=CC=CC")
    mol_graph = molecule.get_representation(MolecularGraph)
    for action in FlipAlternatingPathMG.list_unchecked_actions(molecule):
        path = action.path
        assert len(path) >= 4
        orders = [mol_graph.bond_order(path[k], path[k + 1]) for k in range(len(path) - 1)]
        assert orders == [1 if k % 2 == 0 else 2 for k in range(len(orders))]


########################################
#               Aromatize
########################################
def test_aromatize_dihydropyridine() -> None:
    """1,4-dihydropyridine -> pyridine."""
    assert results(AromatizeMG, "C1=CNC=CC1") == {canonical("c1ccncc1")}


def test_aromatize_dihydroperylene() -> None:
    """Dihydroperylene -> perylene, with at most 3 bond changes."""
    actions = AromatizeMG.list_actions(Molecule("C(C=C1C=CC=C2C11)C=C1c1cccc3c1c2ccc3"))
    assert len(actions) == 1
    assert len(actions[0].bond_changes) <= 3
    assert canonical(str(actions[0].apply())) == canonical(
        "c1cc2cccc3c4cccc5cccc(c(c1)c23)c54"
    )


@pytest.mark.parametrize(
    "smiles",
    (
        pytest.param("C1=CC=CC1", id="cyclopentadiene"),
        pytest.param("c1ccccc1", id="benzene (already aromatic)"),
        pytest.param("CCCC", id="no ring"),
    ),
)
def test_aromatize_refused(smiles: str) -> None:
    """No Kekulé structure, or nothing to change."""
    assert not AromatizeMG.list_actions(Molecule(smiles))


def test_aromatize_one_action_per_system() -> None:
    """Two separate ring systems give at most one action each."""
    actions = AromatizeMG.list_unchecked_actions(Molecule("C1=CCC=CC1CC1C=CCC=C1"))
    assert len(actions) == 2
    assert len({action.atom_idx for action in actions}) == 2
