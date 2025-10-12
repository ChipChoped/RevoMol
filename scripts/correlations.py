import csv
import os
import random
import sys
from datetime import datetime
from typing import cast

import numpy as np

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


def random_walk(start_smiles: str, n_steps: int, action_space: list[type[Action]],
                fitness_functions: list[Function])-> tuple[list[str], list[bool], dict[str, list[float]]]:
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
        dict[list[float]]: Dictionary of fitnesses scores for each molecule encountered
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

    fitnesses: dict[str, list[float]] = dict()  # All fitnesses of molecules encountered

    with open("./results/correlations/" + str(n_steps) + "/" + start_smiles + "/"
              + "_".join([action.__name__ for action in MolecularGraph.action_space]) + "/"
              + TIMESTAMP + "/random_walk.csv", "a", newline='') as file:
        writer = csv.writer(file)

        csv_row = ["smiles", "is_valid"]
        csv_row.extend([function.name for function in fitness_functions])

        writer.writerow(csv_row)

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

        csv_row = [start_smiles, are_valid[-1]]
        csv_row.extend(iter([str(fitness[-1]) for fitness in fitnesses.values()]))

        writer.writerow(csv_row)

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

            start_smiles = rand_neighbor

            csv_row = [start_smiles, are_valid[-1]]
            csv_row.extend(iter([str(fitness[-1]) for fitness in fitnesses.values()]))

            writer.writerow(csv_row)

            print()

    return path, are_valid, fitnesses


def fitness_correlation(fitnesses: list[float], k:int=1) -> float:
    """
    Compute the correlation coefficient of a list of fitnesses with gap of size k.

    Args:
        fitnesses (list[float]): A list of fitness scores
        k (int): The size of the gap between each fitness
    Return:
        list[float]: The correlation coefficient
    """
    return float(np.corrcoef(fitnesses[:-1][::k], fitnesses[1:][::k])[0, 1])


def distance_fitness_correlation(all_fitnesses: dict[str, list[float]], distance_functions: list[Distance],
                                 molecules: list[str], gap:int=1, sample_size:int=1)\
    -> dict[str, dict[str, float]]:
    """
    Compute the correlation coefficient of a list of fitnesses with gap of size k.

    Args:
        all_fitnesses (str, dict[list[float]]): A list of fitness scores
        distance_functions (list[Distance]): A distance function
        molecules (list[str]): A list of molecules
        gap (int): The gap size between two sampled molecules
        sample_size (int): The number of samples to use

    Returns:
        float: The correlation coefficient
        list[float]: The sampled distances
        list[float]: The sampled delta fitness
        list[tuple[str, str]]: The sampled molecule pairs
    """
    path = "./results/correlations/" + str(len(molecules) - 1) + "/" + molecules[0] + "/"\
           + "_".join([action.__name__ for action in MolecularGraph.action_space]) + "/" + TIMESTAMP

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
                 distance_size: int=1) -> float:
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

    Return:
        list[float]: A list of fitnesses correlation
        list[float]: A list of distances-fitnesses correlation
    """
    dp.setup_default_parameters()

    molecules, are_valid, all_fitnesses = cast(tuple[list[str], list[bool], dict[str, list[float]]],
                                      random_walk(start_smiles, n_steps, action_space, fitness_functions))

    print("\n---Correlation coefficient(s)---\n")

    with open("./results/correlations/" + str(n_steps) + "/" + start_smiles + "/"
              + "_".join([action.__name__ for action in action_space]) + "/"
              + TIMESTAMP + "/fitness_correlations.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["fitness_function", "correlation_coefficient"])

        for fitness_function_name, fitnesses in zip(all_fitnesses.keys(), all_fitnesses.values()):
            fitness_correlation_coefficient: float = fitness_correlation(fitnesses)

            print(fitness_function_name + ":", fitness_correlation_coefficient)
            writer.writerow([fitness_function_name, fitness_correlation_coefficient])

    print()

    path = "./results/correlations/" + str(n_steps) + "/" + start_smiles + "/"\
           + "_".join([action.__name__ for action in action_space]) + "/" + TIMESTAMP

    os.makedirs(path + "/distance_fitness_correlations", exist_ok=True)

    for gap in range(1, distance_size + 1):
        with (open(path + "/distance_fitness_correlations/gap_" + str(gap) + ".csv", "a", newline='') as file):
            writer = csv.writer(file)
            writer.writerow(["distance_function", "fitness_function", "correlation_coefficient"])

            distance_fitness_correlation_coefficient: dict[str, dict[str, float]]\
                = distance_fitness_correlation(all_fitnesses, distance_functions, molecules, gap, n_steps // 10)

            print("Gap size:", gap)
            print("-------------------------\n")

            for fitness_function_name in distance_fitness_correlation_coefficient.keys():
                for distance_function_name in distance_fitness_correlation_coefficient[fitness_function_name].keys():
                    print(fitness_function_name + "-" + distance_function_name + ":",
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

    if len(args) < 3:
        raise Exception("Unexpected number of arguments"
                        "Arg 1: SMILES of a molecule"
                        "Arg 2: Number of steps to perform"
                        "Arg 3: Actions to perform")

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

    for action in args[2].split(" "):
        if action in actions:
            action_space.append(eval("mg." + action))
        else:
            raise ("Actions must be in the following list:\n\n"
                   "ddAtomMG\n"
                   "AddGroupMG\n"
                   "ChangeBondMG\n"
                   "CutAtomMG\n"
                   "InsertCarbonMG\n"
                   "MoveGroupMG\n"
                   "RemoveAtomMG\n"
                   "RemoveGroupMG\n"<
                   "SubstituteAtomMG\n"
                   )

    os.makedirs("./results/correlations/" + str(n_steps) + "/" + smiles + "/" + args[2].replace(" ", "_")
                + "/" + TIMESTAMP, exist_ok=True)

    correlations(smiles, n_steps, action_space,
                 [QED, SAScore, LogP, PLogP, Silly_Walks],
                 [Tanimoto, Levenshtein], 3)

    print()
    print()

if __name__ == "__main__":
    main()
