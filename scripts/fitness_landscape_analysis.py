import csv
import os
import random
import sys
from datetime import datetime
from typing import cast

import numpy as np

from scripts.random_walk import random_walk

# Add the parent directory to the path to import the module evomol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evomol.distance.distance import Distance
from evomol.distance.levenshtein import Levenshtein
from evomol.distance.tanimoto import Tanimoto

# pylint: disable=wrong-import-position, import-error

from evomol import default_parameters as dp
from evomol.representation import MolecularGraph
from evomol.evaluation import Function
from evomol.evaluation.qed import QED
from evomol.evaluation.sa_score import SAScore
from evomol.evaluation.logp import LogP
from evomol.evaluation.plogp import PLogP
from evomol.evaluation.silly_walks import Silly_Walks
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


def delta_fitness_distance_correlation(all_fitnesses: dict[str, list[float]], distance_functions: list[Distance],
                                       molecules: list[str], sample_size:int=1, only_valid: bool=True)\
    -> dict[str, dict[str, float]]:
    """
    Compute the correlation coefficient between the delta fitness and the distance for a set of distance functions.

    Args:
        all_fitnesses (str, dict[list[float]]): A list of fitness scores
        distance_functions (list[Distance]): A distance function
        molecules (list[str]): A list of molecules
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
           + "_".join([action.__name__ for action in MolecularGraph.action_space]) + "/samples.csv"

    with (open(path, "a", newline='') as file):
        writer = csv.writer(file)

        csv_row: list[str] = ["smiles_1", "smiles_2"]
        csv_row.extend(all_fitnesses.keys())
        csv_row.extend([distance.name for distance in distance_functions])

        writer.writerow(csv_row)

        sampled_steps: list[int] = random.sample(range(len(molecules)), sample_size * 2)
        steps_1 = sampled_steps[0:sample_size]
        steps_2 = sampled_steps[sample_size:sample_size * 2]

        for step_1, step_2 in zip(steps_1, steps_2):
            delta_fitnesses: list[str] = []
            distances: list[str] = []

            mol_1 = molecules[step_1]
            mol_2 = molecules[step_2]

            for fitness_function_name, fitnesses in zip(all_fitnesses.keys(), all_fitnesses.values()):
                delta_fitnesses.append(str(fitnesses[step_2] - fitnesses[step_1]))

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
                    [np.abs(fitnesses[step_2] - fitnesses[step_1])
                     for step_1, step_2 in zip(steps_1, steps_2)],
                    [distance_function.distance(molecules[step_1], molecules[step_2])
                     for step_1, step_2 in zip(steps_1, steps_2)],
                )[0, 1])

        return distance_fitness_correlations


def correlations(start_smiles: str, n_steps: int, action_space: list[type[Action]],
                 fitness_functions: list[Function], distance_functions: list[Distance],
                 only_valid: bool=True) -> float:
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
           + "_".join([action.__name__ for action in action_space]) + "/distance_fitness_correlations.csv"

    with (open(path, "a", newline='') as file):
        writer = csv.writer(file)
        writer.writerow(["fitness_function", "distance_function", "correlation_coefficient"])

        delta_fitness_distance_correlation_coefficient: dict[str, dict[str, float]]\
            = delta_fitness_distance_correlation(all_fitnesses, distance_functions, molecules,
                                                 n_steps // 10, only_valid)

        for fitness_function_name in delta_fitness_distance_correlation_coefficient.keys():
            for distance_function_name in delta_fitness_distance_correlation_coefficient[fitness_function_name].keys():
                print(fitness_function_name + "-" + distance_function_name + ":",
                      delta_fitness_distance_correlation_coefficient[fitness_function_name][distance_function_name])

                writer.writerow([fitness_function_name, distance_function_name,
                                 delta_fitness_distance_correlation_coefficient
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
                 [Tanimoto, Levenshtein], only_valid)

    print()
    print()

if __name__ == "__main__":
    main()
