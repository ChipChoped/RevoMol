import argparse
import csv
import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import cast

import numpy as np

from evomol.distance.ged import GED, NormalizedGED
from scripts.walk import walk

# Add the parent directory to the path to import the module evomol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evomol.distance.distance import Distance
from evomol.distance.levenshtein import Levenshtein
from evomol.distance.tanimoto import Tanimoto

# pylint: disable=wrong-import-position, import-error

from evomol import default_parameters as dp
from evomol.evaluation import Function
from evomol.evaluation.qed import QED
from evomol.evaluation.sa_score import SAScore
from evomol.evaluation.logp import LogP
from evomol.evaluation.plogp import PLogP
from evomol.evaluation.silly_walks import Silly_Walks
from evomol.action import molecular_graph as mg, Action


def fitness_auto_correlation(fitnesses: list[float], max_lag: int) -> list[float]:
    """
    Compute the auto-correlation coefficients of a list of fitnesses with gap of variable size l.

    Args:
        fitnesses (list[float]): A list of fitness scores
        max_lag (int): The maximum lag to compute
    Return:
        list[float]: The correlation coefficient
    """
    auto_correlations: list[float] = []

    for l in range(1, max_lag):
        auto_correlations.append(float(np.corrcoef(fitnesses[:-l], fitnesses[l:])[0, 1]))

    return auto_correlations


def delta_fitness_distance_correlation(all_fitnesses: dict[str, list[float]], distance_functions: list[Distance],
                                       molecules: list[str], path: str, sample_size: int = 1) \
    -> dict[str, dict[str, float]]:
    """
    Compute the correlation coefficient between the delta fitness and the distance for a set of distance functions.

    Args:
        all_fitnesses (str, dict[list[float]]): A list of fitness scores
        distance_functions (list[Distance]): A distance function
        molecules (list[str]): A list of molecules
        path (str): The path of the directory
        sample_size (int): The number of samples to use

    Returns:
        float: The correlation coefficient
        list[float]: The sampled distances
        list[float]: The sampled delta fitness
        list[tuple[str, str]]: The sampled molecule pairs
    """
    with open(path + "/samples.csv", "a", newline='') as file:
        writer = csv.writer(file)

        csv_row: list[str] = ["smiles_1", "smiles_2"]
        csv_row.extend(all_fitnesses.keys())
        csv_row.extend([distance.name for distance in distance_functions])

        writer.writerow(csv_row)

        steps_1: list[int] = random.sample(range(len(molecules)), sample_size)
        steps_2: list[int] = random.sample(range(len(molecules)), sample_size)

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


def correlations(start_smiles: str, n_steps: int, action_space: list[Action],
                 fitness_functions: list[Function], distance_functions: list[Distance],
                 strategy: str = "random", evaluation_function: Function = None,
                 only_valid: bool = True, path: str = "results") -> float:
    """
    Compute the fitnesses correlations and the distances-fitnesses correlations between a starting molecule
    and molecules encountered during a random walk.

    Args:
        start_smiles (str): The smiles of the starting molecule
        n_steps (int): The number of steps to perform
        action_space (list[Action]): Actions allowed to perform
        fitness_functions (list[Function]): A list of fitness functions
        distance_functions (list[Distance]): A list of distance functions
        strategy (str): The type of walk to perform ("random" or "adaptive")
        evaluation_function (Function): The fitness function to evaluate neighbors in adaptive walks
        only_valid (bool): If True, only valid molecules will be kept during the random walk
        path (str): The path to the directory where results are stored

    Return:
        list[float]: A list of fitnesses correlation
        list[float]: A list of distances-fitnesses correlation
    """
    dp.setup_default_parameters()

    only_valid_str: str = "only_valid" if only_valid else "not_all_valid"

    if only_valid and Silly_Walks in fitness_functions:
        fitness_functions.remove(Silly_Walks)

    evaluation_function_str: str = ""

    if strategy == "adaptive":
        evaluation_function_str = evaluation_function.name + "/"

    molecules, are_valid, all_fitnesses = cast(tuple[list[str], list[bool], dict[str, list[float]]],
                                               walk(start_smiles, n_steps, action_space, fitness_functions,
                                                    strategy, evaluation_function, only_valid, path))

    print("\n---Correlation coefficient(s)---\n")

    with open(path + "fitness_auto_correlations.csv", "a", newline='') as file:
        writer = csv.writer(file)

        row = ["lag"]
        row.extend(all_fitnesses.keys())

        writer.writerow(row)

        max_lag = min(100, len(molecules) - 1)
        fitness_auto_correlation_coefficients: dict[str, list[float]] = dict()

        for fitness_function_name, fitnesses in zip(all_fitnesses.keys(), all_fitnesses.values()):
            fitness_auto_correlation_coefficients[fitness_function_name] = fitness_auto_correlation(fitnesses, max_lag)
            print(fitness_function_name + ":", fitness_auto_correlation_coefficients[fitness_function_name][0])

        for l in range(max_lag - 1):
            row = [str(l + 1)]
            row.extend(iter(map(str, [fitness_auto_correlation_coefficients[fitness_function_name][l]
                                      for fitness_function_name in all_fitnesses.keys()])))

            writer.writerow(row)

    print()
    with open(path + "distance_fitness_correlations.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["fitness_function", "distance_function", "correlation_coefficient"])

        sample_size = min(100, len(molecules))

        delta_fitness_distance_correlation_coefficient: dict[str, dict[str, float]] \
            = delta_fitness_distance_correlation(all_fitnesses, distance_functions, molecules, path, sample_size)

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
    parser: ArgumentParser = argparse.ArgumentParser()

    parser.add_argument("strategy", type=str, choices=("random", "adaptive"),
                        help="The type of walk to perform (random or adaptive)")
    parser.add_argument("smiles", type=str, help="SMILES of a molecule")
    parser.add_argument("n_steps", type=int, help="Number of steps to perform")
    parser.add_argument("-a", required=True, type=str, help="Actions to perform (space separated)",
                        choices=("AddAtomMG", "AddGroupMG", "ChangeBondMG", "CutAtomMG", "InsertCarbonMG",
                                 "MoveGroupMG", "RemoveAtomMG", "RemoveGroupMG", "SubstituteAtomMG"),
                        dest="actions", nargs="+")
    parser.add_argument("-e", type=str, choices=("QED", "SAScore", "LogP", "PLogP", "Silly_Walks"),
                        help="The evaluation function to use in adaptive walks", dest="evaluation_function",
                        default=None)
    parser.add_argument("--only-valid", action="store_true",
                        help="If set, only valid molecules will be kept during the walk", dest="only_valid")
    parser.add_argument("--seed", type=int, help="Random seed to use", dest="seed", default=0)

    arguments: argparse.Namespace = parser.parse_args()

    random.seed(arguments.seed)

    action_space: list[Action] = [eval("mg." + action) for action in arguments.actions]

    evaluation_function: Function | None = None

    if arguments.evaluation_function is not None:
        try:
            evaluation_function = eval(arguments.evaluation_function)
        except NameError:
            print("Error: Unknown evaluation function", arguments.evaluation_function)
            exit(1)

    evaluation_function_str: str = ""

    if arguments.strategy == "adaptive" and evaluation_function is None:
        print("Error: An evaluation function must be provided for adaptive walks")
        exit(1)
    elif arguments.strategy == "adaptive" and evaluation_function is not None:
        evaluation_function_str: str = arguments.evaluation_function + "/"

    only_valid_str: str = "only_valid" if parser.parse_args().only_valid else "not_all_valid"

    path = ("./results/" + arguments.strategy + "_walk/" + only_valid_str + "/" + str(arguments.n_steps) + "/"
            + arguments.smiles + "/" + str(arguments.actions).replace("', '", "_")
            .replace("['", "").replace("']", "") + "/" + evaluation_function_str
            + str(arguments.seed) + "/")

    print()
    print(path)
    print()

    os.makedirs(path, exist_ok=True)

    correlations(arguments.smiles, arguments.n_steps, action_space,
                 [QED, SAScore, LogP, PLogP, Silly_Walks],
                 [Tanimoto, Levenshtein, GED, NormalizedGED],
                 strategy=arguments.strategy, evaluation_function=evaluation_function,
                 only_valid=arguments.only_valid, path = path)

    print()


if __name__ == "__main__":
    main()
