import csv
import random
from typing import cast

from evomol.action import Action
from evomol.evaluation import Function, ZincNormalizedLogP, NormalizedSAScore, CycleScore, NormalizedCycleScore, \
    Evaluation
from evomol.representation import Molecule, MolecularGraph
from evomol.search import enumeration as en
from evomol import default_parameters as dp
from evomol import evaluation as evaluator


def get_random_neighbor(start_smiles: str, only_valid: bool = True) -> tuple[str, int]:
    """
    Get a random neighbor for a molecule without looking if it is realistic.

    Arg:
        start_smiles (str): The smiles of the starting molecule
        only_valid (bool): If true, only valid smiles will be returned.

    Return:
        str: A random neighbor of the molecule
        int: Size of the neighborhood
    """
    # Convert the starting SMILES to its canonical form
    can_smi_start = (
        Molecule(start_smiles).get_representation(MolecularGraph).canonical_smiles
    )

    smiles_set: set[str] = en.find_neighbors(Molecule(can_smi_start), 1)

    if only_valid:
        evaluations = dp.setup_filters("chembl_zinc")

        valid_smiles: set[str] = {smiles for smiles in smiles_set
                                  if evaluator.is_valid_molecule(Molecule(smiles), evaluations)}

        smiles_set = valid_smiles

    if len(smiles_set) == 0:
        return "", 0
    else:
        return random.choice(list(smiles_set)), len(smiles_set)


def get_best_neighbor(start_smiles: str, fitness_function: Function, only_valid: bool = True) -> tuple[str, float, int]:
    """
    Get the neighbor with the highest fitness equal or higher than the starting molecule.

    Arg:
        start_smiles (str): The smiles of the starting molecule
        fitness_function (Function): The fitness function to evaluate neighbors
        only_valid (bool): If true, only valid smiles will be considered.

    Return:
        str: The best neighbor of the molecule
        float: The fitness of the best neighbor
        int: Size of the neighborhood
    """
    # Convert the starting SMILES to its canonical form
    can_smi_start = (
        Molecule(start_smiles).get_representation(MolecularGraph).canonical_smiles
    )

    smiles_set: set[str] = en.find_neighbors(Molecule(can_smi_start), 1)

    if only_valid:
        evaluations = dp.setup_filters("chembl_zinc")

        valid_smiles: set[str] = {smiles for smiles in smiles_set
                                  if evaluator.is_valid_molecule(Molecule(smiles), evaluations)}

        smiles_set = valid_smiles

    if len(smiles_set) == 0:
        return "", 0, 0
    else:
        start_mol = Molecule(start_smiles)
        start_fitness = fitness_function.evaluate(start_mol)

        best_smiles: str = ""
        best_fitness: float = start_fitness

        for smiles in smiles_set:
            neighbor_mol = Molecule(smiles)
            neighbor_fitness = fitness_function.evaluate(neighbor_mol)

            if neighbor_fitness >= best_fitness:
                best_fitness = neighbor_fitness
                best_smiles = smiles

        return best_smiles, best_fitness, len(smiles_set)


def walk(start_smiles: str, n_steps: int, action_space: list[Action],
         fitness_functions: list[Function], strategy: str= "random", evaluation_function: Function = None,
         only_valid: bool = True)\
    -> tuple[list[str], list[bool], dict[str, list[float]]]:
    """
    Perform an adaptive walk with a starting molecule and a set of allowed action.

    Args:
        start_smiles (str): The smiles of the starting molecule
        n_steps (int): The number of steps to perform
        action_space(list[Action]): Actions allowed to perform
        fitness_functions (list[Function]): A list of fitness functions
        strategy (str): The type of walk to perform ("random" or "adaptive")
        evaluation_function (Function): The fitness function to evaluate neighbors in adaptive walks
        only_valid (bool): If True, only valid molecules will be kept during the random walk

    Returns:
        list[str]: The path took during the random walk (list of smiles)
        list[bool]: Whether each molecule encountered is valid
        dict[list[float]]: Dictionary of fitnesses scores for each molecule encountered
    """
    # Initialize and set the action space
    dp.setup_default_action_space()
    MolecularGraph.action_space = cast(list[type[Action]], cast(object, action_space))

    evaluations: list[Evaluation] = dp.setup_filters("chembl_zinc")
    start_mol: Molecule = Molecule(start_smiles)

    print("----------Step 0----------")
    print("Molecule:", start_smiles)

    path: list[str] = [start_smiles]  # All molecules encountered
    are_valid: list[bool] = [evaluator.is_valid_molecule(start_mol, evaluations)]  # Validity of molecules encountered
    print("Is valid:", are_valid[0])

    only_valid_str: str = "only_valid" if only_valid else "not_all_valid"

    fitnesses: dict[str, list[float]] = dict()  # All fitnesses of molecules encountered

    evaluation_function_str: str = ""

    if strategy == "adaptive":
        evaluation_function_str: str = evaluation_function.name + "/"
        start_fitness: float = evaluation_function.evaluate(start_mol)

        plateau: list[int] = []
        plateaus: list[tuple[float, list[int]]] = []

    with open("./results/" + strategy + "_walk/" + only_valid_str + "/" + str(n_steps) + "/" + start_smiles + "/"
              + "_".join([action.__name__ for action in MolecularGraph.action_space]) + "/" + evaluation_function_str
              + "walk.csv", "a", newline='') as file:
        writer = csv.writer(file)

        csv_row = ["smiles", "is_valid"]
        csv_row.extend([function.name for function in fitness_functions])
        csv_row.append("neighborhood_size")

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

        init_smiles: str = start_smiles

        # At each step a random candidate of the molecule neighbor is chosen
        # Its validity and all its fitnesses are computed and saved
        for step in range(n_steps):

            # Get a random neighbor
            if strategy == "random":
                neighbor, neighborhood_size = get_random_neighbor(start_smiles, only_valid)

                csv_row.append(str(neighborhood_size))
                writer.writerow(csv_row)

                print("Neighborhood size:", neighborhood_size)
                print()
                print("----------Step " + str(step + 1) + "----------")
            # Get the best neighbor according to the evaluation function
            # and stops the walk if no better neighbor is found
            elif strategy == "adaptive":
                neighbor, neighbor_fitness, neighborhood_size = get_best_neighbor(start_smiles, evaluation_function,
                                                                                  only_valid)

                csv_row.append(str(neighborhood_size))
                writer.writerow(csv_row)

                print("Neighborhood size:", neighborhood_size)
                print()
                print("----------Step " + str(step + 1) + "----------")

                start_fitness: float
                plateaus: list[tuple[float, list[int]]]

                if neighbor == "":
                    print("No better neighbor found, stopping the walk.")

                    if len(plateau) > 0:
                        plateaus.append((start_fitness, plateau))

                        with open("./results/" + strategy + "_walk/" + only_valid_str + "/" + str(n_steps) + "/"
                                  + path[0] + "/" + "_".join([action.__name__ for action in MolecularGraph.action_space]) + "/"
                                  + evaluation_function_str + "plateaus.csv", "w") as plateau_file:
                            plateau_writer = csv.writer(plateau_file)
                            plateau_writer.writerow(["step", "smiles", "start_smiles", "fitness"])

                            for p in plateaus:
                                for s in p[1]:
                                    plateau_writer.writerow([s, start_smiles, path[0], p[0]])

                    break
                elif neighbor_fitness == start_fitness:
                    if len(plateau) == 0:
                        plateau = [step]
                    elif plateau[-1] == step - 1:
                        plateau.append(step)
                    else:
                        plateaus.append((start_fitness, plateau))
                        plateau = [step]
            else:
                raise ValueError("Strategy must be 'random' or 'adaptive'!")

            if neighbor == "":
                neighbor = init_smiles

            neighbor_mol = Molecule(neighbor)
            print("Molecule:", neighbor)

            path.append(neighbor)
            are_valid.append(evaluator.is_valid_molecule(neighbor_mol, evaluations))
            print("Is valid:", (are_valid[-1]))

            for fitness_function, i in zip(fitness_functions, range(len(fitness_functions))):
                function_name = fitness_function.name

                if fitness_function.name == "PLogP":
                    neighbor_mol.set_value("zinc_normalized_logP",
                                                ZincNormalizedLogP.evaluate(neighbor_mol))
                    neighbor_mol.set_value("zinc_normalized_sa_score",
                                                NormalizedSAScore.evaluate(neighbor_mol))
                    neighbor_mol.set_value("CycleScore",
                                                CycleScore.evaluate(neighbor_mol))
                    neighbor_mol.set_value("zinc_normalized_cycle_score",
                                                NormalizedCycleScore.evaluate(neighbor_mol))

                fitness = fitness_function.evaluate(neighbor_mol)
                fitnesses[function_name].append(fitness)
                neighbor_mol.set_value(function_name, fitness)

                print(function_name + ":", fitnesses[fitness_function.name][-1])

            start_smiles = neighbor

            csv_row = [start_smiles, are_valid[-1]]
            csv_row.extend(iter([str(fitness[-1]) for fitness in fitnesses.values()]))

    if strategy == "adaptive":
        with open("./results/" + strategy + "_walk/" + only_valid_str + "/" + str(n_steps) + "/"
                  + path[0] + "/" + "_".join([action.__name__ for action in MolecularGraph.action_space]) + "/"
                  + evaluation_function_str + "local_optimum.csv", "w") as file:
            writer = csv.writer(file)

            if len(path) - 1 != n_steps:
                row = ["smiles", "start_smiles", "is_valid", "steps_taken", "evaluation_function"]
                row.extend([function.name for function in fitness_functions])

                writer.writerow(row)

                row = [start_smiles, path[0], are_valid[-1], len(path) - 1, evaluation_function.name]
                row.extend([str(fitness[-1]) for fitness in fitnesses.values()])

                writer.writerow(row)
            else:
                writer.writerow(["No local optimum found in " + str(n_steps) + " steps."])

    return path, are_valid, fitnesses
