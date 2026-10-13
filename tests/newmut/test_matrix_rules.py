"""
Tests for the validity rules read on the bond-order matrix (newmut).

The tables are those of the specification (§2.2 to §2.5).
"""

import numpy as np
import pytest
from rdkit import Chem

from evomol import default_parameters as dp
from evomol.newmut import matrix_rules as mr
from evomol.representation import MolecularGraph


@pytest.fixture(autouse=True, scope="module")
def setup_parameters() -> None:
    """set the parameters for the tests."""
    dp.setup_default_parameters(
        accepted_atoms=["C", "O", "N", "F", "S"],
        max_heavy_atoms=38,
    )


def rule_ids(smiles: str) -> set[str]:
    """Ids of the rules broken by the molecule."""
    types, matrix = mr.to_matrix(MolecularGraph(smiles))
    return {violation.rule_id for violation in mr.rule_violations(types, matrix)}


def kekule(smiles: str) -> str:
    """Kekulé SMILES written by RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


PHTHALOCYANINE = (
    "C1=CC=C2C(=C1)C3=NC4=C5C=CC=CC5=C(N4)N=C6C7=CC=CC=C7C(=N6)"
    "N=C8C9=CC=CC=C9C(=N8)N=C2N3"
)


########################################
#                 Matrix
########################################
def test_to_matrix() -> None:
    """Diagonal = valence electrons, off diagonal = bond orders."""
    types, matrix = mr.to_matrix(MolecularGraph("C=CO"))
    assert types == ["C", "C", "O"]
    assert matrix.tolist() == [[4, 2, 0], [2, 4, 1], [0, 1, 6]]


def test_to_matrix_empty() -> None:
    """The empty molecule gives an empty matrix and no violation."""
    types, matrix = mr.to_matrix(MolecularGraph(""))
    assert types == []
    assert matrix.shape == (0, 0)
    assert mr.is_valid(types, matrix)


@pytest.mark.parametrize("smiles", ["C=CC=C", "c1ccccc1", "C1=CSC=C1C#N", "O=C1C=CC(=O)C=C1"])
def test_from_matrix_round_trip(smiles: str) -> None:
    """from_matrix keeps the matrix (and the Kekulé structure) as is."""
    types, matrix = mr.to_matrix(MolecularGraph(smiles))
    rebuilt = mr.from_matrix(types, matrix)
    types2, matrix2 = mr.to_matrix(rebuilt)
    assert types2 == types
    assert np.array_equal(matrix2, matrix)


@pytest.mark.parametrize(
    ("smiles", "tags"),
    (
        pytest.param("CC", ["s", "s"], id="ethane"),
        pytest.param("C=CC", ["d", "d", "s"], id="propene"),
        pytest.param("C#CC", ["t", "t", "s"], id="propyne"),
        pytest.param("C1=CC=CC=C1", ["a"] * 6, id="benzene"),
        pytest.param("CO", ["s", ""], id="methanol"),
    ),
)
def test_carbon_tag(smiles: str, tags: list[str]) -> None:
    """s / a / d / t tags read on the matrix."""
    types, matrix = mr.to_matrix(MolecularGraph(smiles))
    assert [mr.carbon_tag(matrix, types, i) for i in range(len(types))] == tags


def test_bond_in_ring() -> None:
    """A bond is in a ring when its atoms stay connected without it."""
    types, matrix = mr.to_matrix(MolecularGraph("CC1CC1"))
    assert not mr.bond_in_ring(matrix, 0, 1)
    assert mr.bond_in_ring(matrix, 1, 2)


def test_smallest_ring_per_bond() -> None:
    """Naphthalene: two 6-membered rings (deduplicated)."""
    _, matrix = mr.to_matrix(MolecularGraph("c1ccc2ccccc2c1"))
    rings = mr.smallest_ring_per_bond(matrix)
    assert sorted(len(ring) for ring in rings) == [6, 6]


########################################
#            R1, valence
########################################
def test_r1_valence() -> None:
    """A row whose sum exceeds the valence breaks R1."""
    types = ["C", "O"]
    matrix = np.array([[4, 3], [3, 6]])
    assert {v.rule_id for v in mr.rule_violations(types, matrix)} == {"R1"}


def test_r1_sulfur_valence() -> None:
    """The maximum valence of S follows the sulfur valence."""
    types = ["S", "O", "O"]
    matrix = np.array([[6, 2, 2], [2, 6, 0], [2, 0, 6]])
    assert mr.is_valid(types, matrix, sulfur_valence=6)
    assert not mr.is_valid(types, matrix, sulfur_valence=2)


########################################
#     R2, two bonded s carbons
########################################
@pytest.mark.parametrize(
    ("smiles", "valid"),
    (
        pytest.param("CC", False, id="ethane"),
        pytest.param("C=CC", True, id="propene"),
        pytest.param("CC(C)=C", True, id="isobutene"),
        pytest.param("Cc1ccccc1", True, id="toluene"),
        pytest.param("COc1ccccc1", True, id="anisole"),
        pytest.param("CCc1ccccc1", False, id="ethylbenzene"),
        pytest.param("CC(C)(C)c1ccccc1", False, id="tert-butylbenzene"),
        pytest.param("C1Cc2ccccc2-c2ccccc12", True, id="9,10-dihydrophenanthrene"),
        pytest.param("C1Cc2ccccc2C=C1", True, id="1,2-dihydronaphthalene"),
        pytest.param("C1Cc2cccc3cccc1c23", True, id="acenaphthene"),
        pytest.param("C1Cc2ccccc2C1", False, id="indane"),
        pytest.param("C1CCCCC1", False, id="cyclohexane"),
        pytest.param("C1COc2cscc2O1", True, id="EDOT"),
        pytest.param("C1COc2cscc2OC1", False, id="ProDOT"),
    ),
)
def test_r2(smiles: str, valid: bool) -> None:
    """R2 table of the specification."""
    assert ("R2" not in rule_ids(smiles)) == valid


########################################
#     R3, one double bond per carbon
########################################
@pytest.mark.parametrize(
    ("smiles", "valid"),
    (
        pytest.param("C=CC=C", True, id="butadiene"),
        pytest.param("C=CC#C", True, id="vinylacetylene"),
        pytest.param("C=C=C", False, id="allene"),
        pytest.param("C=C=O", False, id="ketene"),
        pytest.param("CN=C=O", False, id="methyl isocyanate"),
        pytest.param("O=C=O", False, id="CO2"),
    ),
)
def test_r3(smiles: str, valid: bool) -> None:
    """R3 table of the specification."""
    assert ("R3" not in rule_ids(smiles)) == valid


########################################
#            R4, flat rings
########################################
@pytest.mark.parametrize(
    ("smiles", "expected"),
    (
        pytest.param("c1ccc2ccccc2c1", set(), id="naphthalene"),
        pytest.param("c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", set(), id="coronene"),
        pytest.param("c1cc2cccc3c4cccc5cccc(c(c1)c23)c54", set(), id="perylene"),
        pytest.param("c1ccc2cccc2cc1", set(), id="azulene"),
        pytest.param("c1cc2ccc3cccc4ccc(c1)c2c34", set(), id="pyrene"),
        pytest.param("c1cc2cc3ccc2cc13", {"R4a"}, id="bridged naphthalene"),
        pytest.param("C1=CC2C=CC1C2", {"R4a"}, id="norbornadiene"),
        pytest.param("c1ccc2c(c1)C1c3ccccc3C2c2ccccc21", {"R4a"}, id="triptycene"),
        pytest.param(
            "C1=CC2=NC1=CC1=CC=C(C=C3C=CC(=N3)C=C3C=CC(=C2)N3)N1",
            set(),
            id="porphyrin",
        ),
        pytest.param(kekule(PHTHALOCYANINE), set(), id="phthalocyanine"),
        pytest.param("C1=CC=CC=CC=CC=CC=CC=CC=C1", set(), id="[18]annulene"),
        pytest.param("C1=CC=CC=CC=CC=C1", {"R4b"}, id="[10]annulene"),
        pytest.param(
            "c1cc2ccc1-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc2cc1",
            {"R4b"},
            id="[6]cycloparaphenylene",
        ),
        pytest.param("c1cc2ccc1CCc1ccc(cc1)CC2", {"R4b"}, id="[2.2]paracyclophane"),
    ),
)
def test_r4(smiles: str, expected: set[str]) -> None:
    """R4 table of the specification."""
    assert rule_ids(smiles) & {"R4a", "R4b"} == expected


def test_r4b_turning_angles() -> None:
    """[10]annulene: the turning angles sum to 600°, [6]CPP to 0°."""
    messages = [
        v.message
        for v in mr.rule_violations(*mr.to_matrix(MolecularGraph("C1=CC=CC=CC=CC=C1")))
    ]
    assert any("600°" in message for message in messages)
    smiles = "c1cc2ccc1-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc2cc1"
    messages = [
        v.message for v in mr.rule_violations(*mr.to_matrix(MolecularGraph(smiles)))
    ]
    assert any(" 0°" in message for message in messages)


########################################
#            Hybridization
########################################
@pytest.mark.parametrize(
    ("smiles", "expected"),
    (
        pytest.param("c1cc[nH]c1", ["sp2"] * 5, id="pyrrole"),
        pytest.param("c1ccoc1", ["sp2"] * 5, id="furan"),
        pytest.param("Nc1ccccc1", ["sp2"] * 7, id="aniline"),
        pytest.param("CC(N)=O", ["sp3", "sp2", "sp2", "sp"], id="acetamide"),
        pytest.param("COC", ["sp3", "sp2", "sp3"], id="dimethyl ether"),
        pytest.param("CC#N", ["sp3", "sp", "sp"], id="acetonitrile"),
    ),
)
def test_hybridization(smiles: str, expected: list[str]) -> None:
    """Hybridization read on the matrix (§2.5)."""
    types, matrix = mr.to_matrix(MolecularGraph(smiles))
    assert [mr.hybridization(matrix, types, i) for i in range(len(types))] == expected


########################################
#      Comparison before / after
########################################
def test_result_is_allowed_tolerance() -> None:
    """A molecule already outside the rules stays mutable."""
    cyclohexane = MolecularGraph("C1CCCCC1")
    methylcyclohexane = MolecularGraph("CC1CCCCC1")
    benzene = MolecularGraph("c1ccccc1")
    cyclohexene = MolecularGraph("C1=CCCCC1")
    # fewer violations: allowed
    assert mr.result_is_allowed(cyclohexane, cyclohexene)
    # new violation: refused
    assert not mr.result_is_allowed(benzene, cyclohexene)
    # cyclohexane (6 violations) -> methylcyclohexane (7): refused
    assert not mr.result_is_allowed(cyclohexane, methylcyclohexane)


def test_rule_violations_cache() -> None:
    """Two equal matrices give the same violations (memoized)."""
    types, matrix = mr.to_matrix(MolecularGraph("CCc1ccccc1"))
    first = mr.rule_violations(types, matrix)
    second = mr.rule_violations(types, matrix.copy())
    assert first == second


########################################
#   R5, no s carbon with 4 heavy atoms
########################################
@pytest.mark.parametrize(
    ("smiles", "valid"),
    (
        pytest.param("CC(C)C", True, id="isobutane"),
        pytest.param("CC(C)(C)C", False, id="neopentane"),
        pytest.param("CC(C)(C)O", False, id="tert-butanol"),
        pytest.param("CC(C)(C)c1ccccc1", False, id="tert-butylbenzene"),
        pytest.param("C1CCC2(CC1)CCCC2", False, id="spiro centre"),
        pytest.param(
            "c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12", False, id="spirobifluorene"
        ),
        pytest.param("C=C(C)C", True, id="isobutene (not an s carbon)"),
        pytest.param("c1ccc2c(c1)C1c3ccccc3C2c2ccccc21", True, id="triptycene (3 heavy atoms)"),
        pytest.param("CC1(C)C=CC=C1", False, id="5,5-dimethylcyclopentadiene"),
    ),
)
def test_r5(smiles: str, valid: bool) -> None:
    """R5: an s carbon never carries 4 heavy atoms."""
    assert ("R5" not in rule_ids(smiles)) == valid
