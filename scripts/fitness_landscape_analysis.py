import csv
import os
import random
import sys
from datetime import datetime
from typing import cast

import numpy as np

from scripts.search_space_walks import random_walk

# Add the parent directory to the path to import the module evomol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evomol.action.molecular_graph.action_molecular_graph import ActionMolGraph
from evomol.distance.distance import Distance
from evomol.distance.levenshtein import Levenshtein
from evomol.distance.tanimoto import Tanimoto

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
from evomol.action import molecular_graph as mg, Action


TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def fitness_auto_correlation(fitnesses: list[float]) -> list[float]:
    """
    Compute the auto-correlation coefficients of a list of fitnesses with gap of variable size l.

    Args:
        fitnesses (list[float]): A list of fitness scores
    Return:
        list[float]: The correlation coefficient
    """
    auto_correlations: list[float] = []

    for l in range(1, ((len(fitnesses) - 1) // 10 + 1)):
        auto_correlations.append(float(np.corrcoef(fitnesses[:-1][::l], fitnesses[1:][::l])[0, 1]))

    return auto_correlations


def distance_fitness_correlation(all_fitnesses: dict[str, list[float]], distance_functions: list[Distance],
                                 molecules: list[str], gap:int=1, sample_size:int=1, only_valid: bool=True)\
    -> dict[str, dict[str, float]]:
    """
    Compute the correlation coefficient of a list of fitnesses with gap of size k.

    Args:
        all_fitnesses (str, dict[list[float]]): A list of fitness scores
        distance_functions (list[Distance]): A distance function
        molecules (list[str]): A list of molecules
        gap (int): The gap size between two sampled molecules
        sample_size (int): The number of samples to use
        only_valid (bool): If True, only valid molecules will be kept during the random walk (for log purpose)

    Returns:
        float: The correlation coefficient
        list[float]: The sampled distances
        list[float]: The sampled delta fitness
        list[tuple[str, str]]: The sampled molecule pairs
    """
    only_valid_str: str = "only_valid" if only_valid else "not_all_valid"

    path = "./results/correlations/" + only_valid_str + "/" + str(len(molecules) - 1) + "/" + molecules[0] + "/"\
           + "_".join([action.__name__ for action in MolecularGraph.action_space]) + "/"

    os.makedirs(path + "/samples", exist_ok=True)

    with (open(path + "/samples/gap_" + str(gap)
                + ".csv", "a", newline='') as file):
        writer = csv.writer(file)

        csv_row: list[str] = ["smiles_1", "smiles_2"]
        csv_row.extend(all_fitnesses.keys())
        csv_row.extend([distance.name for distance in distance_functions])

        writer.writerow(csv_row)

        sampled_steps: list[int] = random.sample(range(len(molecules) - gap), sample_size)

        for step in sampled_steps:
            delta_fitnesses: list[str] = []
            distances: list[str] = []

            mol_1 = molecules[step]
            mol_2 = molecules[step + gap]

            for fitness_function_name, fitnesses in zip(all_fitnesses.keys(), all_fitnesses.values()):
                delta_fitnesses.append(str(fitnesses[step + gap] - fitnesses[step]))

            for distance_function in distance_functions:
                distances.append(str(distance_function.distance(mol_1, mol_2)))

            csv_row = [mol_1, mol_2]
            csv_row.extend(delta_fitnesses)
            csv_row.extend(distances)

            writer.writerow(csv_row)

        distance_fitness_correlations: dict[str, dict[str, float]] = dict()

        for fitness_function_name, fitnesses in zip(all_fitnesses.keys(), all_fitnesses.values()):
            distance_fitness_correlations[fitness_function_name] = dict()

            for distance_function in distance_functions:
                distance_function_name = distance_function.name
                distance_fitness_correlations[fitness_function_name][distance_function_name] = float(np.corrcoef(
                    [fitnesses[step + gap] - fitnesses[step]
                     for step in range(len(molecules) - gap)],
                    [distance_function.distance(molecules[step], molecules[step + gap])
                     for step in range(len(molecules) - gap)]
                )[0, 1])

        return distance_fitness_correlations


def correlations(start_smiles: str, n_steps: int, action_space: list[type[Action]],
                 fitness_functions: list[Function], distance_functions: list[Distance],
                 distance_size: int=1, only_valid: bool=True) -> float:
    """
    Compute the fitnesses correlations and the distances-fitnesses correlations between a starting molecule
    and molecules encountered during a random walk.

    Args:
        start_smiles (str): The smiles of the starting molecule
        n_steps (int): The number of steps to perform
        action_space (list[type[Action]]): Actions allowed to perform
        fitness_functions (list[Function]): A list of fitness functions
        distance_functions (list[Distance]): A list of distance functions
        distance_size (int): The size of the distance between two molecules
        only_valid (bool): If True, only valid molecules will be kept during the random walk

    Return:
        list[float]: A list of fitnesses correlation
        list[float]: A list of distances-fitnesses correlation
    """
    dp.setup_default_parameters()

    only_valid_str: str = "only_valid" if only_valid else "not_all_valid"

    molecules, are_valid, all_fitnesses = cast(tuple[list[str], list[bool], dict[str, list[float]]],
                                      random_walk(start_smiles, n_steps, action_space, fitness_functions, only_valid))

    print("\n---Correlation coefficient(s)---\n")

    with open("./results/correlations/" + only_valid_str + "/" + str(n_steps) + "/" + start_smiles + "/"
              + "_".join([action.__name__ for action in action_space]) + "/"
              + "/fitness_auto_correlations.csv", "a", newline='') as file:
        writer = csv.writer(file)

        row = ["lag"]
        row.extend(all_fitnesses.keys())

        writer.writerow(row)

        fitness_auto_correlation_coefficients: dict[str, list[float]] = dict()

        for fitness_function_name, fitnesses in zip(all_fitnesses.keys(), all_fitnesses.values()):
            fitness_auto_correlation_coefficients[fitness_function_name] = fitness_auto_correlation(fitnesses)
            print(fitness_function_name + ":", fitness_auto_correlation_coefficients[fitness_function_name][0])

        for l in range(n_steps // 10):
            row = [str(l + 1)]
            row.extend(iter(map(str, [fitness_auto_correlation_coefficients[fitness_function_name][l]
                        for fitness_function_name in all_fitnesses.keys()])))

            writer.writerow(row)

    print()

    path = "./results/correlations/" + only_valid_str + "/" + str(n_steps) + "/" + start_smiles + "/"\
           + "_".join([action.__name__ for action in action_space])

    os.makedirs(path + "/distance_fitness_correlations", exist_ok=True)

    for gap in range(1, distance_size + 1):
        with (open(path + "/distance_fitness_correlations/gap_" + str(gap) + ".csv", "a", newline='') as file):
            writer = csv.writer(file)
            writer.writerow(["distance_function", "fitness_function", "correlation_coefficient"])

            distance_fitness_correlation_coefficient: dict[str, dict[str, float]]\
                = distance_fitness_correlation(all_fitnesses, distance_functions, molecules,
                                               gap, n_steps // 10, only_valid)

            print("Gap size:", gap)
            print("-------------------------\n")

            for fitness_function_name in distance_fitness_correlation_coefficient.keys():
                for distance_function_name in distance_fitness_correlation_coefficient[fitness_function_name].keys():
                    print(distance_function_name + "-" + fitness_function_name + ":",
                          distance_fitness_correlation_coefficient[fitness_function_name][distance_function_name])

                    writer.writerow([distance_function_name, fitness_function_name,
                                     distance_fitness_correlation_coefficient
                                     [fitness_function_name][distance_function_name]])

                print()
            print()

    return 0


def main() -> None:
    """Compute the fitness correlation between molecules found during a random walk"""
    args = sys.argv[1:]

    if len(args) < 4:
        raise Exception("Unexpected number of arguments"
                        "Arg 1: SMILES of a molecule"
                        "Arg 2: Number of steps to perform"
                        "Arg 3: 0 for all molecules and 1 for only valid ones"
                        "Arg 4: Actions to perform")

    smiles: str = args[0]
    n_steps: int = int(args[1])
    action_space: list[type[Action]] = []

    actions: list[str] = [
        "AddAtomMG",
        "AddGroupMG",
        "ChangeBondMG",
        "CutAtomMG",
        "InsertCarbonMG",
        "MoveGroupMG",
        "RemoveAtomMG",
        "RemoveGroupMG",
        "SubstituteAtomMG"
    ]

    only_valid: bool = bool(int(args[2]))

    for action in args[3].split(" "):
        if action in actions:
            action_space.append(eval("mg." + action))
        else:
            raise ("Actions must be in the following list:\n\n"
                   "AddAtomMG\n"
                   "AddGroupMG\n"
                   "ChangeBondMG\n"
                   "CutAtomMG\n"
                   "InsertCarbonMG\n"
                   "MoveGroupMG\n"
                   "RemoveAtomMG\n"
                   "RemoveGroupMG\n"<
                   "SubstituteAtomMG\n"
                   )

    only_valid_str: str = "only_valid" if only_valid else "not_all_valid"

    path = "./results/correlations/" + only_valid_str + "/" + str(n_steps) + "/" + smiles + "/" \
           + args[3].replace(" ", "_")

    print()
    print(path)
    print()

    os.makedirs(path, exist_ok=True)

    correlations(smiles, n_steps, action_space,
                 [QED, SAScore, LogP, PLogP, Silly_Walks],
                 [Tanimoto, Levenshtein], 3, only_valid)

    print()
    print()

if __name__ == "__main__":
    main()
