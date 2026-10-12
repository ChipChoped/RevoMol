"""
Tests for the naming of the operations (newmut.matrix_diff), on the examples
of reference_app.js and on the new actions.

The alignments of mcs were checked against reference_core.js (node) on the
same graphs when the module was written: same maps, same operations.
"""

import pytest

from evomol import default_parameters as dp
from evomol.newmut.annelate import AnnelateMG
from evomol.newmut.append_atom_rule import AppendAtomRuleMG
from evomol.newmut.flip_path import FlipAlternatingPathMG
from evomol.newmut.graft import GraftFragmentMG
from evomol.newmut.matrix_diff import (
    classify_action,
    classify_mutation,
    matrix_mol,
    mcs,
    operation_names,
)
from evomol.representation import MolecularGraph, Molecule


@pytest.fixture(autouse=True, scope="module")
def setup_parameters() -> None:
    """set the parameters for the tests."""
    dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "F", "S"], max_heavy_atoms=38)
    dp.setup_default_action_space()


def kinds(before: str, after: str) -> list[str]:
    """Kinds of the operations that change the molecule."""
    return [op["kind"] for op in classify_mutation(before, after) if op["cost"] != 0]


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    (
        pytest.param("C1=CSC=C1", "C1=CSC=N1", ["subst"], id="thiophene -> thiazole"),
        pytest.param("C1=CSC=C1", "C=CC1=CC=CS1", ["graft"], id="2-vinylthiophene"),
        pytest.param("C1=CSC=C1", "C1=CC2=C(C=C1)SC=C2", ["annel"], id="benzo[b]thiophene"),
        pytest.param("C1=CSC=C1", "C1=CC2=CSC=C2C=C1", ["annel"], id="benzo[c]thiophene"),
        pytest.param("CC1=CC=C(C)S1", "C=C1C=CC(=C)S1", ["flip"], id="quinoid switch"),
        pytest.param(
            "C1=CSC=C1", "CN(C)C1=CC=C(C=C(C#N)C#N)S1", ["graft", "graft"], id="D-pi-A"
        ),
        pytest.param(
            "C1=CSC(=C1)C1=CC=CS1", "C1=CSC(=C1)C=CC1=CC=CS1", ["insert"], id="insertion"
        ),
    ),
)
def test_reference_examples(before: str, after: str, expected: list[str]) -> None:
    """The seven examples of the workshop."""
    assert kinds(before, after) == expected


def test_same_molecule() -> None:
    """Two Kekulé structures of benzene: no operation that changes the molecule."""
    assert kinds("c1ccccc1", "C1=CC=CC=C1") == []


def test_mcs_identity() -> None:
    """A molecule is aligned with itself atom by atom."""
    mol = matrix_mol(MolecularGraph("CC(=O)Nc1ccc(O)cc1"))
    assert mcs(mol, mol) == {i: i for i in mol.ids}


@pytest.mark.parametrize(
    ("action_class", "smiles", "expected"),
    (
        pytest.param(AppendAtomRuleMG, "Cc1ccccc1", "append", id="AppendAtomRule"),
        pytest.param(GraftFragmentMG, "C1=CSC=C1", "graft", id="GraftFragment"),
        pytest.param(AnnelateMG, "C1=CSC=C1", "annel", id="Annelate"),
        pytest.param(FlipAlternatingPathMG, "CC1=CC=C(C)S1", "flip", id="FlipAlternatingPath"),
    ),
)
def test_classify_action(action_class: type, smiles: str, expected: str) -> None:
    """The new actions are named as in the workshop."""
    actions = action_class.list_actions(Molecule(smiles))
    assert actions
    found = {
        op["kind"] for action in actions for op in classify_action(action) if op["cost"] != 0
    }
    assert expected in found


def test_operation_names() -> None:
    """Titles in French, Kekulé rewritings left out."""
    names = operation_names(classify_mutation("C1=CSC=C1", "C1=CSC=N1"))
    assert names == ["Substitution C → N sur l'atome 4 (cycle aromatique)"]
