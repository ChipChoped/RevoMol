"""
Functions to set default parameters for evomol.
"""

import os
from typing import Optional

from evomol import evaluation as evaluator
from evomol.action import molecular_graph as mg
from evomol.representation import SMILES, MolecularGraph, Molecule


def setup_default_parameters(
    accepted_atoms: Optional[list[str]] = None,
    max_heavy_atoms: int = 38,
    unique_kekule: bool = False,
) -> None:
    """Set the default parameters for evomol.

    Args:
        accepted_atoms (Optional[list[str]], optional): List of accepted atoms
            for the molecules. Defaults to None (["C", "O", "N", "F", "S"]).
        max_heavy_atoms (int, optional): Maximum number of heavy atoms in a
            molecule. Defaults to 38.
        unique_kekule (bool, optional): one canonical SMILES per molecule
            whatever its Kekulé form (no duplicate aromatic molecules, but
            canonical_smiles ~6x slower on aromatic molecules). Defaults to
            False (original behaviour).
    """

    if accepted_atoms is None:
        accepted_atoms = ["C", "O", "N", "F", "S"]

    Molecule.id_representation_class = SMILES
    Molecule.representations_class = [MolecularGraph]
    Molecule.max_heavy_atoms = max_heavy_atoms

    Molecule.accepted_atoms = accepted_atoms
    MolecularGraph.unique_kekule = unique_kekule


# pylint: disable=too-many-arguments
def setup_default_action_space(
    with_add_group: bool = False,
    with_remove_group: bool = False,
    insert_carbon: bool = True,
    avoid_bond_breaking: bool = False,
    avoid_bond_forming: bool = False,
    avoid_sp3_on_sp3: bool = False,
    add_atom_multiple_bonds: bool = False,
    bond_change_by_one: bool = False,
) -> None:
    """Set the default action space for evomol.

    Args:
        with_add_group: add AddGroupMG to the action space.
        with_remove_group: add RemoveGroupMG to the action space.
        insert_carbon: keep InsertCarbonMG. Its inserted carbon is a
            non-terminal sp3 carbon, always rejected by HuckelCompatible.
        avoid_bond_breaking: ChangeBondMG never removes a bond (rings can no
            longer be opened by ChangeBondMG).
        avoid_bond_forming: ChangeBondMG never creates a bond between two
            unbonded atoms (rings can no longer be closed).
        avoid_sp3_on_sp3: AddAtomMG never attaches an sp3 carbon to an sp3
            carbon (would create a non-terminal sp3 carbon, rejected by
            HuckelCompatible).
        add_atom_multiple_bonds: AddAtomMG can also attach the new atom by a
            double or triple bond (=CH2, =O, #N…) when the valences allow it.
            False: original behaviour, single bond only.
        bond_change_by_one: ChangeBondMG only changes a bond order by one
            (0 -> 1, 1 -> 2…). False: original behaviour, any order in one
            action (0 -> 2, 1 -> 3, 3 -> 0…).
    """

    mg.AddGroupMG.groups = [
        mg.Group("C1=CC=CS1", 5, [0]),
        mg.Group("C1=CC=CC=C1", 6, [0]),
        mg.Group("[N+](=O)[O-]", 3, [0]),
        mg.Group("N=[N+]=[N-]", 3, [0]),
        mg.Group("S(=O)(=O)O", 4, [0]),
    ]

    mg.ChangeBondMG.avoid_bond_breaking = avoid_bond_breaking
    mg.ChangeBondMG.avoid_bond_forming = avoid_bond_forming
    mg.ChangeBondMG.by_one_step = bond_change_by_one
    mg.AddAtomMG.avoid_sp3_on_sp3 = avoid_sp3_on_sp3
    mg.AddAtomMG.allow_multiple_bonds = add_atom_multiple_bonds

    mg.RemoveGroupMG.remove_only_smallest = True
    mg.RemoveGroupMG.remove_only_single_bond = False
    mg.RemoveGroupMG.remove_charged_or_radical = True

    MolecularGraph.action_space = [
        mg.AddAtomMG,
        mg.ChangeBondMG,
        mg.CutAtomMG,
        mg.InsertCarbonMG,
        mg.MoveGroupMG,
        mg.RemoveAtomMG,
        mg.SubstituteAtomMG,
    ]
    if not insert_carbon:
        MolecularGraph.action_space.remove(mg.InsertCarbonMG)
    if with_add_group:
        MolecularGraph.action_space.append(mg.AddGroupMG)
    if with_remove_group:
        MolecularGraph.action_space.append(mg.RemoveGroupMG)


def setup_filters(filter_name: str) -> list[evaluator.Evaluation]:
    """Return the evaluations for the generic cyclic features and ECFP

    Args:
        filter_name (str): chembl or chembl_zinc as filter

    Returns:
        list[evaluator.Evaluation]: list of evaluations
    """
    if filter_name == "chembl":
        return [
            evaluator.UnknownGCF(
                path_db=os.path.join("external_data", "gcf1.txt"),
                name="chembl",
            ),
            evaluator.FilterUnknownGCF(
                threshold=0,
                name="chembl",
            ),
            evaluator.UnknownECFP(
                path_db=os.path.join("external_data", "ecfp4_ChEMBL.txt"),
                radius=2,
                name="chembl",
            ),
            evaluator.FilterUnknownECFP(
                threshold=0,
                name="chembl",
            ),
        ]
    if filter_name == "chembl_zinc":
        return [
            evaluator.UnknownGCF(
                path_db=os.path.join("external_data", "gcf2.txt"),
                name="chembl_zinc",
            ),
            evaluator.FilterUnknownGCF(
                threshold=0,
                name="chembl_zinc",
            ),
            evaluator.UnknownECFP(
                path_db=os.path.join("external_data", "ecfp4_ChEMBL_ZINC.txt"),
                radius=2,
                name="chembl_zinc",
            ),
            evaluator.FilterUnknownECFP(
                threshold=0,
                name="chembl_zinc",
            ),
        ]
    raise ValueError("filter_name must be 'chembl' or 'chembl_zinc'")


def chembl_and_chembl_zinc_evaluations() -> list[evaluator.Evaluation]:
    """Return the evaluations for the generic cyclic features and ECFP

    Returns:
        list[evaluator.Evaluation]: list of evaluations
    """
    return [
        evaluator.UnknownGCF(
            path_db=os.path.join("external_data", "gcf1.txt"),
            name="chembl",
        ),
        evaluator.UnknownECFP(
            path_db=os.path.join("external_data", "ecfp4_ChEMBL.txt"),
            radius=2,
            name="chembl",
        ),
        evaluator.UnknownGCF(
            path_db=os.path.join("external_data", "gcf2.txt"),
            name="chembl_zinc",
        ),
        evaluator.UnknownECFP(
            path_db=os.path.join("external_data", "ecfp4_ChEMBL_ZINC.txt"),
            radius=2,
            name="chembl_zinc",
        ),
    ]

