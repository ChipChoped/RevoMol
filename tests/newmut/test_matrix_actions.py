"""
Tests for the new actions (newmut).

For each action: a positive case, a refused case, and the property that every
listed action applies without error and gives a result that creates no new
violation of the matrix rules, on varied molecules.
"""

import pytest

from evomol import default_parameters as dp
from evomol.action import Action
from evomol.newmut import matrix_rules as mr
from evomol.newmut import rule_filtered as rf
from evomol.newmut.annelate import AnnelateMG
from evomol.newmut.add_atom import AddAtomMG
from evomol.newmut.aromatize import AromatizeMG
from evomol.newmut.flip_path import FlipAlternatingPathMG
from evomol.newmut.fragments import parse_fragment
from evomol.newmut.graft import GraftFragmentMG
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

# AddAtomMG of newmut (formerly AppendAtomRuleMG): applies the rules itself
ADD_RULE = AddAtomMG

RULE_ACTIONS: list[type[Action]] = [
    ADD_RULE,
    SubstituteShiftMG,
    FlipAlternatingPathMG,
    AromatizeMG,
    GraftFragmentMG,
    AnnelateMG,
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
#   AddAtom (newmut)
########################################
def test_add_atom_empty() -> None:
    """Empty molecule: any accepted atom."""
    assert results(ADD_RULE, "") == {canonical(s) for s in "CONFS"}


def test_add_atom_on_s_carbon() -> None:
    """On the CH3 of toluene: =CH2, #CH, =O, =NH, #N; never a C s on the C s (R2a)."""
    produced = results(ADD_RULE, "Cc1ccccc1")
    assert {canonical(s) for s in ("C=Cc1ccccc1", "C#Cc1ccccc1", "O=Cc1ccccc1", "N=Cc1ccccc1",
                                   "N#Cc1ccccc1")} <= produced
    assert canonical("CCc1ccccc1") not in produced


def test_add_atom_on_s_nitrogen() -> None:
    """On the NH2 of aniline: =CH2, =O, =NH."""
    produced = results(ADD_RULE, "Nc1ccccc1")
    assert {canonical(s) for s in ("C=Nc1ccccc1", "O=Nc1ccccc1", "N=Nc1ccccc1")} <= produced


def test_add_atom_triple_needs_three_hydrogens() -> None:
    """No triple bond on a CH2 (2 hydrogens)."""
    candidates = ADD_RULE.list_unchecked_actions(Molecule("C=CCc1ccccc1"))
    assert not any(a.index_atom == 2 and a.bond_order == 3 for a in candidates)


def test_add_atom_single_hydrogen() -> None:
    """An s carbon with a single H receives no carbon (never a C by a single bond on an s carbon)."""
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
        for action in ADD_RULE.list_actions(molecule)
    )


def test_add_atom_max_size() -> None:
    """No atom added beyond Molecule.max_heavy_atoms."""
    try:
        dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "F", "S"], max_heavy_atoms=6)
        rf.clear_cache()
        assert not ADD_RULE.list_actions(Molecule("c1ccccc1"))
    finally:
        dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "F", "S"], max_heavy_atoms=38)
        rf.clear_cache()


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


def test_substitute_no_shift() -> None:
    """Option shift_bonds = False: benzene -> pyridine stays (valence holds),
    toluene ring carbon -> O disappears (a double bond must be lowered)."""
    SubstituteShiftMG.shift_bonds = False
    try:
        molecule = Molecule("Cc1ccccc1")
        actions = SubstituteShiftMG.list_actions(molecule)
        assert not any(action.new_type == "O" and action.atom_idx != 0 for action in actions)
        # no bond lowered or shifted (a raised bond, raise_bonds, is still possible)
        graph = molecule.get_representation(MolecularGraph)
        assert all(order > graph.bond_order(i, j) for action in actions for i, j, order in action.bond_changes)
        assert any(action.new_type == "N" for action in actions)
    finally:
        SubstituteShiftMG.shift_bonds = True


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


########################################
#               Fragments
########################################
def test_parse_fragment() -> None:
    """Attachment atoms found in the fragment without its dummy atoms."""
    phenyl = parse_fragment("C1=CC=C(C=C1)[*:1]")
    assert phenyl.nb_atoms == 6
    assert len(phenyl.attach) == 1
    thieno = parse_fragment("[*:1]C=CS[*:2]")
    assert thieno.nb_atoms == 3
    assert not thieno.symmetric
    assert parse_fragment("[*:1]C=CC=C[*:2]").symmetric


def test_parse_fragment_errors() -> None:
    """Badly numbered attachment points are refused."""
    with pytest.raises(ValueError):
        parse_fragment("C=C[*:2]")


########################################
#             GraftFragment
########################################
def test_graft_thiophene() -> None:
    """Thiophene -> 2-vinylthiophene (example of reference_app.js)."""
    assert canonical("C=CC1=CC=CS1") in results(GraftFragmentMG, "C1=CSC=C1")


def test_graft_size_limit() -> None:
    """No graft beyond Molecule.max_heavy_atoms."""
    try:
        dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "F", "S"], max_heavy_atoms=7)
        names = {a.name for a in GraftFragmentMG.list_actions(Molecule("c1ccsc1"))}
        assert "vinyl" in names
        assert "phenyl" not in names
    finally:
        dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "F", "S"], max_heavy_atoms=38)


def test_graft_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ethyl graft (other vocabulary) creates two bonded s carbons."""
    monkeypatch.setattr(GraftFragmentMG, "fragments", {"ethyl": "CC[*:1]"})
    molecule = Molecule("c1ccccc1")
    assert GraftFragmentMG.list_unchecked_actions(molecule)
    assert not GraftFragmentMG.list_actions(molecule)


def test_graft_accepted_atoms() -> None:
    """Fragments with an element not accepted are not used."""
    try:
        dp.setup_default_parameters(accepted_atoms=["C"], max_heavy_atoms=38)
        names = {a.name for a in GraftFragmentMG.list_actions(Molecule("c1ccccc1"))}
        assert names == {"vinyl", "ethynyl", "phenyl"}
    finally:
        dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "F", "S"], max_heavy_atoms=38)


########################################
#                Annelate
########################################
def test_annelate_thiophene() -> None:
    """Thiophene -> benzo[b]thiophene and benzo[c]thiophene (examples of
    reference_app.js)."""
    produced = results(AnnelateMG, "C1=CSC=C1")
    assert canonical("C1=CC2=C(C=C1)SC=C2") in produced
    assert canonical("C1=CC2=CSC=C2C=C1") in produced


def test_annelate_orientations() -> None:
    """Both orientations of an asymmetric fragment, one of a symmetric one."""
    actions = AnnelateMG.list_unchecked_actions(Molecule("C=C"))
    benzo = [(a.atom1, a.atom2) for a in actions if a.name == "benzo"]
    thieno = [(a.atom1, a.atom2) for a in actions if a.name == "thieno"]
    assert benzo == [(0, 1)]
    assert sorted(thieno) == [(0, 1), (1, 0)]


def test_annelate_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A propano bridge (other vocabulary) on ethylene gives cyclopentene,
    whose middle CH2 is bonded to two s carbons: refused. An ethano bridge
    gives cyclobutene, allowed (each CH2 has an sp2 neighbour)."""
    monkeypatch.setattr(
        AnnelateMG,
        "fragments",
        {"propano": "[*:1]CCC[*:2]", "ethano": "[*:1]CC[*:2]"},
    )
    molecule = Molecule("C=C")
    assert {a.name for a in AnnelateMG.list_unchecked_actions(molecule)} == {
        "propano",
        "ethano",
    }
    assert results(AnnelateMG, "C=C") == {canonical("C1=CCC1")}


def test_r5_refuses_quaternary_carbon() -> None:
    """R5 (V0): no O added on the CH of cumene (it would carry 4 heavy atoms)."""
    mr.RULES_VERSION = "V0"
    rf.clear_cache()
    try:
        molecule = Molecule("CC(C)c1ccccc1")
        actions = ADD_RULE.list_actions(molecule)
        assert not any(a.index_atom == 1 for a in actions)
        assert any(a.index_atom == 0 for a in actions)
    finally:
        mr.RULES_VERSION = "V1"
        rf.clear_cache()


def test_r5_v1_refuses_third_heavy_atom() -> None:
    """R5 (V1): no atom added by a single bond on the CH2 of allylbenzene (it
    would stay an s carbon with 3 heavy atoms); atoms can still be added on the
    ring. A double bond (=O) makes it a d carbon: allowed."""
    molecule = Molecule("C=CCc1ccccc1")
    actions = ADD_RULE.list_actions(molecule)
    assert not any(a.index_atom == 2 and a.bond_order == 1 for a in actions)
    assert any(a.index_atom == 2 and a.atom_type == "O" and a.bond_order == 2 for a in actions)
    assert any(a.index_atom == 5 for a in actions)



########################################
#      InsertCarbon keeping a double
########################################
def test_insert_carbon_keep_double() -> None:
    """A carbon inserted into C=C keeps a double bond on either side."""
    from evomol.newmut.insert_carbon_double import InsertCarbonDoubleMG  # pylint: disable=import-outside-toplevel

    actions = InsertCarbonDoubleMG.list_actions(Molecule("C=CC=C"))
    orders = sorted((a.bond_to_1, a.bond_to_2) for a in actions if a.bond_to_1 + a.bond_to_2 > 2)
    assert orders == [(1, 2), (1, 2), (2, 1), (2, 1)]
    produced = {canonical(str(a.apply())) for a in actions}
    assert canonical("C=CC=CC") in produced and canonical("C=CCC=C") in produced
    # the single bonds keep the original insertion (two single bonds)
    single = [a for a in actions if (a.atom1_idx_to_bond, a.atom2_idx_to_bond) in ((1, 2), (2, 1))]
    assert [(a.bond_to_1, a.bond_to_2) for a in single] == [(1, 1)]
    assert InsertCarbonDoubleMG.class_name() == "InsertCarbonMG"


def test_insert_carbon_keep_double_benzene() -> None:
    """Benzene: inserting into a double bond gives cycloheptatriene, allowed by the rules."""
    from evomol.newmut import rule_filtered as rf  # pylint: disable=import-outside-toplevel
    from evomol.newmut.insert_carbon_double import InsertCarbonDoubleMG  # pylint: disable=import-outside-toplevel

    produced = results(rf.rule_filtered(InsertCarbonDoubleMG), "c1ccccc1")
    assert canonical("C1=CC=CCC=C1") in produced


def test_substitute_raise_bond() -> None:
    """Anisole: O replaced by N or C, the bond to the CH3 goes up (C=N-Ph, C=C-Ph,
    C#C-Ph); not with raise_bonds = False."""
    produced = results(SubstituteShiftMG, "COc1ccccc1")
    expected = {canonical(s) for s in ("C=Nc1ccccc1", "C=Cc1ccccc1", "C#Cc1ccccc1")}
    assert expected <= produced
    try:
        SubstituteShiftMG.raise_bonds = False
        rf.clear_cache()
        assert not expected & results(SubstituteShiftMG, "COc1ccccc1")
    finally:
        SubstituteShiftMG.raise_bonds = True
        rf.clear_cache()


def test_substitute_raise_needs_hydrogens() -> None:
    """No raise towards an atom without hydrogen (the aromatic C of anisole)."""
    actions = SubstituteShiftMG.list_unchecked_actions(Molecule("COc1ccccc1"))
    raised = [a for a in actions if a.atom_idx == 1 and a.bond_changes]
    assert raised and all(0 in pair[:2] for a in raised for pair in a.bond_changes)
