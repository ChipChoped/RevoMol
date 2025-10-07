from rdkit.Chem import rdmolfiles, AllChem
from rdkit.DataStructs import TanimotoSimilarity

from evomol.distance.distance import Distance


def tanimoto(molecule_1: str, molecule_2: str) -> float:
    """
    Calculate the Tanimoto distance between two molecules.

    Parameters:
        molecule_1 (str): The first molecule's SMILES
        molecule_2 (str): The second molecule's SMILES

    Return:
        float: The Tanimoto distance
    """

    molecule_1 = rdmolfiles.MolFromSmiles(molecule_1)
    molecule_2 = rdmolfiles.MolFromSmiles(molecule_2)

    molecule_1_morgan_fp = AllChem.GetMorganFingerprint(molecule_1, 2)
    molecule_2_morgan_fp = AllChem.GetMorganFingerprint(molecule_2, 2)

    return TanimotoSimilarity(molecule_1_morgan_fp, molecule_2_morgan_fp)


Tanimoto = Distance("Tanimoto", tanimoto)
