from evomol.evaluation.evaluation import Function
from evomol.evaluation.generic_cyclic_features import UnknownGCF, list_gcf
from evomol.representation import Molecule, MolecularGraph


def gcf(molecule: Molecule) -> float:
    """
    Calculate the generic cyclic features (GCF) of a molecule
    (Score of 0 is a molecule with only known GCF)

    Args:
        molecule: Molecule to evaluate

    Returns:
        float: GCF score
    """

    if molecule:
        total_gcf = len(list_gcf(molecule.get_representation(MolecularGraph).canonical_smiles))

        if total_gcf > 0:
            return UnknownGCF().evaluate(molecule) / total_gcf
        else:
            return 0.0
    else:
        return 0.0


GCF = Function("GCF", gcf)
