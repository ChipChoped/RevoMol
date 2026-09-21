from evomol.evaluation import Silly_Walks, GCF
from evomol.evaluation.evaluation import Function
from evomol.representation import Molecule


def realism(molecule: Molecule) -> float:
    """
    Calculate the realism of a molecule by aggregating the Silly Walks score and
    the GCF score

    Args:
        molecule: Molecule to evaluate

    Returns:
        float: Realism score
    """

    if molecule:
        return (1 - Silly_Walks.evaluate(molecule)) * (1 - GCF.evaluate(molecule))
    else:
        return 0.0


Realism = Function("Realism", realism)
