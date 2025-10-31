from evomol.distance.distance import Distance
from molfeat.Classes import MoleculeFeatures


def ged(molecule_1: str, molecule_2: str) -> float:
    """
    Calculate the graph edition distance between two molecules.

    Parameters:
        molecule_1 (str): The first molecule's SMILES
        molecule_2 (str): The second molecule's SMILES

    Return:
        float: The graphe edition distance.
    """

    return MoleculeFeatures(molecule_2).evaluate_distance(MoleculeFeatures(molecule_1), "min")[0]


def normalized_ged(molecule_1: str, molecule_2: str) -> float:
    """
    Calculate the graph edition distance between two molecules.

    Parameters:
        molecule_1 (str): The first molecule's SMILES
        molecule_2 (str): The second molecule's SMILES

    Return:
        float: The graphe edition distance.
    """

    return MoleculeFeatures(molecule_2).evaluate_distance(MoleculeFeatures(molecule_1), "min")[1]


GED = Distance("GED", ged)
NormalizedGED = Distance("NormalizedGED", normalized_ged)
