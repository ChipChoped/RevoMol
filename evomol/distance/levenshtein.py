import Levenshtein as lev

from evomol.distance.distance import Distance


def levenshtein(molecule_1: str, molecule_2: str) -> float:
    """
    Calculate the Levenshtein distance between two molecules.

    Parameters:
        molecule_1 (str): The first molecule's SMILES
        molecule_2 (str): The second molecule's SMILES

    Return:
        float: The Levenshtein distance.
    """

    return lev.distance(molecule_1, molecule_2)


Levenshtein = Distance("Levenshtein", levenshtein)
