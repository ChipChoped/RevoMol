import os
import random
import sys
from typing import cast

import numpy as np

from evomol.action.molecular_graph.action_molecular_graph import ActionMolGraph
from evomol.distance.distance import Distance
from evomol.distance.levenshtein import Levenshtein
from evomol.distance.tanimoto import Tanimoto

# Add the parent directory to the path to import the module evomol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pylint: disable=wrong-import-position, import-error

from evomol import default_parameters as dp
from evomol import evaluation as evaluator
from evomol.representation import MolecularGraph, Molecule
from evomol.search import enumeration as en
from evomol.evaluation import Function
from evomol.evaluation.qed import QED
from evomol.evaluation.sa_score import SAScore, NormalizedSAScore
from evomol.evaluation.logp import LogP, ZincNormalizedLogP
from evomol.evaluation.plogp import PLogP
from evomol.evaluation.silly_walks import Silly_Walks
from evomol.evaluation.cycle_score import NormalizedCycleScore, CycleScore
from evomol.action import molecular_graph as mg


def get_random_neighbor(start_smiles: str) -> str:
    """
    Get a random neighbor for a molecule without looking if it is realistic.

    Arg:
        start_smiles (str): The smiles of the starting molecule

    Return:
        str: A random neighbor of the molecule
    """
    # Convert the starting SMILES to its canonical form
    can_smi_start = (
        Molecule(start_smiles).get_representation(MolecularGraph).canonical_smiles
    )

    smiles_set: set[str] = en.find_neighbors(Molecule(can_smi_start), 1)

    if len(smiles_set) == 0:
        return ""
    else:
        return random.choice(list(smiles_set))


def random_walk(start_smiles: str, n_steps: int, action_space: list[ActionMolGraph],
                fitness_functions: list[Function])-> tuple[list[str], list[bool], list[float]]:
    """
    Perform a random walk with a starting molecule and a set of allowed action.

    Args:
        start_smiles (str): The smiles of the starting molecule
        n_steps (int): The number of steps to perform
        action_space(ActionMolGraph): Actions allowed to perform
        fitness_functions (list[Function]): A list of fitness functions

    Returns:
        list[str]: The path took during the random walk (list of smiles)
        list[bool]: Whether each molecule encountered is valid
        dict[list[float]]: Dictionary of fitnesses score for each molecule encountered
    """
    # Initialize and set the action space
    dp.setup_default_action_space()
    MolecularGraph.action_space = action_space

    evaluations = dp.setup_filters("chembl_zinc")
    start_mol = Molecule(start_smiles)

    print("----------Step 0----------")
    print("Molecule:", start_smiles)

    path: list[str] = [start_smiles]  # All molecules encountered
    are_valid: list[bool] = [evaluator.is_valid_molecule(start_mol, evaluations)]  # Validity of molecules encountered
    print("Is valid:", are_valid[0])

    fitnesses: dict[list[float]] = dict()  # All fitnesses of molecules encountered

    for fitness_function, i in zip(fitness_functions, range(len(fitness_functions))):
        # Evaluation needed for the PlogP calculation
        function_name = fitness_function.name

        if function_name == "PLogP":
            start_mol.set_value("zinc_normalized_logP",
                                ZincNormalizedLogP.evaluate(start_mol))
            start_mol.set_value("zinc_normalized_sa_score",
                                NormalizedSAScore.evaluate(start_mol))
            start_mol.set_value("CycleScore",
                                CycleScore.evaluate(start_mol))
            start_mol.set_value("zinc_normalized_cycle_score",
                                NormalizedCycleScore.evaluate(start_mol))

        fitness = fitness_function.evaluate(start_mol)
        fitnesses[function_name] = [fitness]
        start_mol.set_value(function_name, fitness)

        print(function_name, ": ", fitnesses[function_name][0])

    print()

    # At each step a random candidate of the molecule neighbor is chosen
    # Its validity and all its fitnesses are computed and saved
    for step in range(n_steps):
        print("----------Step " + str(step + 1) + "----------")

        rand_neighbor = get_random_neighbor(start_smiles)
        rand_neighbor_mol = Molecule(rand_neighbor)
        print("Molecule:", rand_neighbor)

        path.append(rand_neighbor)
        are_valid.append(evaluator.is_valid_molecule(rand_neighbor_mol, evaluations))
        print("Is valid:", (are_valid[-1]))

        for fitness_function, i in zip(fitness_functions, range(len(fitness_functions))):
            function_name = fitness_function.name

            if fitness_function.name == "PLogP":
                rand_neighbor_mol.set_value("zinc_normalized_logP",
                                            ZincNormalizedLogP.evaluate(rand_neighbor_mol))
                rand_neighbor_mol.set_value("zinc_normalized_sa_score",
                                            NormalizedSAScore.evaluate(rand_neighbor_mol))
                rand_neighbor_mol.set_value("CycleScore",
                                            CycleScore.evaluate(rand_neighbor_mol))
                rand_neighbor_mol.set_value("zinc_normalized_cycle_score",
                                            NormalizedCycleScore.evaluate(rand_neighbor_mol))

            fitness = fitness_function.evaluate(rand_neighbor_mol)
            fitnesses[function_name].append(fitness)
            rand_neighbor_mol.set_value(function_name, fitness)

            print(function_name + ":", fitnesses[fitness_function.name][-1])

        path.append(rand_neighbor)
        start_smiles = rand_neighbor

        print()

    return path, are_valid, fitnesses


def fitness_correlation(fitnesses: list[float], k:int=1) -> float:
    """
    Compute the correlation coefficient of a list of fitnesses with gap of size k.

    Args:
        fitnesses (list[float]): A list of fitness scores
    Return:
        list[float]: The correlation coefficient
    """
    return np.corrcoef(fitnesses[:-1][::k], fitnesses[1:][::k])[0, 1]


def correlations(start_smiles: str, n_steps: int, action_space: list[ActionMolGraph],
                 fitness_functions: list[Function], distance_functions: list[Distance],
                 distance_size: int=1) -> float:
    """
    Compute the fitnesses correlations and the distances-fitnesses correlations between a starting molecule
    and molecules encountered during a random walk.

    Args:
        start_smiles (str): The smiles of the starting molecule
        n_steps (int): The number of steps to perform
        action_space(ActionMolGraph): Actions allowed to perform
        fitness_functions (list[Function]): A list of fitness functions
        distance_functions (list[Distance]): A list of distance functions

    Return:
        list[float]: A list of fitnesses correlation
        list[float]: A list of distances-fitnesses correlation
    """
    dp.setup_default_parameters()

    path, are_valid, all_fitnesses = cast(tuple[list[str], list[bool], dict[list[float]]],
                                      random_walk(start_smiles, n_steps, action_space, fitness_functions))

    print("---Correlation coefficient(s)---")
    print()

    for function_name, fitnesses in zip(all_fitnesses.keys(), all_fitnesses.values()):
        print(function_name + ": ", fitness_correlation(fitnesses))

    return 0


def main() -> None:
    """Compute the fitness correlation between molecules found during a random walk"""
    smiles = [
        "C",
        # "C(O)(=O)C1=C(OC(C)=O)C=CC=C1",  # Aspirin
        # "CN1C(=NC2=C1C(=O)N(C(=O)N2C)C)CO"  # Caffeine
    ]

    for smi in smiles:
        correlations(smi, 5, [mg.AddAtomMG, mg.RemoveAtomMG],
                     [QED, SAScore, LogP, PLogP, Silly_Walks],
                     [Tanimoto, Levenshtein], 3)

        print()
        print()

if __name__ == "__main__":
    main()
