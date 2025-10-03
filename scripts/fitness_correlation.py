import os
import sys
import time
import typer
import random

from typing import cast

from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator

from Levenshtein import distance as levenshtein


# Add the parent directory to the path to import the module evomol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pylint: disable=wrong-import-position, import-error

from evomol import default_parameters as dp
from evomol import evaluation as evaluator
from evomol.representation import MolecularGraph, Molecule
from evomol.search import enumeration as en
from evomol.evaluation import Function
from evomol.evaluation.qed import QED
from evomol.evaluation.sa_score import SAScore
from evomol.evaluation.logp import LogP
from evomol.evaluation.plogp import PLogP
from evomol.evaluation.silly_walks import Silly_walk


def get_random_neighbor(start_smiles: str) -> str:
    """Get a random neighbor for a molecule without looking if it is realistic."""
    # convert the starting SMILES to its canonical form
    can_smi_start = (
        Molecule(start_smiles).get_representation(MolecularGraph).canonical_smiles
    )

    smiles_set: set[str] = en.find_neighbors(Molecule(can_smi_start), 1)

    if len(smiles_set) == 0:
        return ""
    else:
        return random.choice(list(smiles_set))


def random_walk(start_smiles: str, n_steps: int, fitness_functions: list[Function]
                )-> tuple[list[str], list[bool], list[float]]:
    """
    Perform a random walk with a starting molecule and a set of allowed action.

    Args:
        start_smiles (str): The smiles of the starting molecule
        n_steps (int): The number of steps to perform
        fitness_functions (list[Function]): A list of fitness functions

    Returns:
        list[str]: The path took during the random walk (list of smiles)
        list[bool]: Whether each molecule encountered is valid
        list[float]: Fitnesses scores for each molecule encountered
    """
    evaluations = dp.setup_filters("chembl_zinc")
    print(start_smiles)
    start_mol = Molecule(start_smiles)

    path: list[str] = [start_smiles]
    are_valid: list[bool] = [evaluator.is_valid_molecule(start_mol, evaluations)]
    fitnesses: list[float] = []

    for fitness_function in fitness_functions:
        try:
            fitnesses.append([fitness_function._evaluate(start_mol)])
            print(fitnesses)
        except ModuleNotFoundError:
            print("No fitness function named " + fitness_function, file=sys.stderr)
            exit(1)

    for step in range(n_steps):
        rand_neighbor = get_random_neighbor(start_smiles)
        rand_neighbor_mol = Molecule(rand_neighbor)

        path.append(rand_neighbor)
        are_valid.append(evaluator.is_valid_molecule(rand_neighbor_mol, evaluations))

        for fitness_function, i in zip(fitness_functions, range(len(fitness_functions))):
            fitnesses[i].append(fitness_functions(rand_neighbor_mol))

    return path, are_valid, fitnesses


def fitness_correlation(start_smiles: str, n_steps: int,
                        fitness_functions: list[Function], distance_functions: list[str]) -> float:
    """
    Compute the fitness correlation between a starting molecule and molecules encountered during a random walk.

    Args:
        start_smiles (str): The smiles of the starting molecule
        n_steps (int): The number of steps to perform
        fitness_functions (list[Function]): A list of fitness functions
        distance_functions (list[str]): A list of distance functions

    Return:
        float: The fitnesses correlation scores
    """
    dp.setup_default_parameters()

    path, are_valid, fitnesses = cast(tuple[list[str], list[bool], list[float]],
                                      random_walk(start_smiles, n_steps, fitness_functions))

    print(fitnesses)

    return 0


def main() -> None:
    """Compute the fitness correlation between molecules found during a random walk"""
    smiles = [
        "C",
        "C(O)(=O)C1=C(OC(C)=O)C=CC=C1",  # Aspirin
        "CN1C(=NC2=C1C(=O)N(C(=O)N2C)C)CO"  # Caffeine
    ]

    max_heavy_atoms = 38

    for smi in smiles:
        fitness_correlation(smi, 1000, [SAScore, Silly_walk],
                            [TanimotoSimilarity, levenshtein])

if __name__ == "__main__":
    main()
