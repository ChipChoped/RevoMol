import csv
import random
from typing import cast

from evomol import default_parameters as dp
from evomol import evaluation as evaluator
from evomol.action import Action
from evomol.evaluation import Function, ZincNormalizedLogP, NormalizedSAScore, CycleScore, NormalizedCycleScore, \
    Evaluation, LogP, SAScore
from evomol.representation import Molecule, MolecularGraph
from evomol.search import enumeration as en
from evomol.action import molecular_graph as mg


def set_plogp_values(mol: Molecule) -> None:
    """
    Set all the values needed for the PLogP evaluation in a molecule.

    Arg:
        mol (Molecule): The molecule to set the values for
    """
    mol.set_value("logP", LogP.evaluate(mol))
    mol.set_value("zinc_normalized_logP", ZincNormalizedLogP.evaluate(mol))
    mol.set_value("sa_score", SAScore.evaluate(mol))
    mol.set_value("zinc_normalized_sa_score", NormalizedSAScore.evaluate(mol))
    mol.set_value("CycleScore", CycleScore.evaluate(mol))
    mol.set_value("zinc_normalized_cycle_score", NormalizedCycleScore.evaluate(mol))


def get_random_neighbor(start_smiles: str, only_valid: bool = True) -> tuple[str, Action | None, int]:
    """
    Get a random neighbor for a molecule without looking if it is realistic.

    Arg:
        start_smiles (str): The smiles of the starting molecule
        only_valid (bool): If true, only valid smiles will be returned.

    Return:
        str: A random neighbor of the molecule
        Action: The action taken to get to the neighbor
        int: Size of the neighborhood
    """
    # Convert the starting SMILES to its canonical form
    can_smi_start = (
        Molecule(start_smiles).get_representation(MolecularGraph).canonical_smiles
    )

    possible_smiles, possible_actions = en.find_neighbors(Molecule(can_smi_start), max_depth=1, info=True)
    neighborhood: set[tuple[str, Action]] = {(Molecule(smiles).get_representation(MolecularGraph).canonical_smiles,
                                              action) for smiles, action in zip(possible_smiles, possible_actions)}

    if only_valid:
        evaluations = dp.setup_filters("chembl_zinc")

        valid_smiles: set[tuple[str, Action]] = {neighbor for neighbor in neighborhood
                                              if evaluator.is_valid_molecule(Molecule(neighbor[0]), evaluations)}

        neighborhood = valid_smiles

    if len(neighborhood) == 0:
        return "", None, 0
    else:
        chosen_smiles, chosen_action = random.choice(list(neighborhood))
        return chosen_smiles, chosen_action, len(neighborhood)


def get_best_neighbor(start_smiles: str, fitness_function: Function, only_valid: bool = True)\
    -> tuple[str, Action | None, float, int]:
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

    possible_smiles, possible_actions = en.find_neighbors(Molecule(can_smi_start), max_depth=1, info=True)
    neighborhood: set[tuple[str, Action]] = {(Molecule(smiles).get_representation(MolecularGraph).canonical_smiles,
                                              action) for smiles, action in zip(possible_smiles, possible_actions)}

    if only_valid:
        evaluations = dp.setup_filters("chembl_zinc")

        valid_smiles: set[tuple[str, Action]] = {neighbor for neighbor in neighborhood
                                              if evaluator.is_valid_molecule(Molecule(neighbor[0]), evaluations)}

        neighborhood = valid_smiles

    if len(neighborhood) == 0:
        return "", None, 0, 0
    else:
        start_mol = Molecule(start_smiles)

        if fitness_function.name == "PLogP":
            set_plogp_values(start_mol)

        start_fitness = fitness_function.evaluate(start_mol)

        best_neighbor: tuple[str, Action | None] = ("", None)
        best_fitness: float = start_fitness

        for neighbor in neighborhood:
            neighbor_mol = Molecule(neighbor[0])

            if fitness_function.name == "PLogP":
                set_plogp_values(neighbor_mol)

            neighbor_fitness = fitness_function.evaluate(neighbor_mol)

            if neighbor_fitness >= best_fitness:
                best_fitness = neighbor_fitness
                best_neighbor = neighbor

        return best_neighbor[0], best_neighbor[1], best_fitness, len(neighborhood)


def walk(start_smiles: str, n_steps: int, action_space: list[Action],
         fitness_functions: list[Function], strategy: str= "random", evaluation_function: Function = None,
         only_valid: bool = True, path: str = "results", soft_change_bond: bool = False)\
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
        path (str): The path to the directory where results are stored
        soft_change_bond (bool): If True, bond breaking and formation won't be allowed (False by default)

    Returns:
        list[str]: The path took during the random walk (list of smiles)
        list[bool]: Whether each molecule encountered is valid
        dict[list[float]]: Dictionary of fitnesses scores for each molecule encountered
    """
    # Initialize and set the action space
    dp.setup_default_action_space()

    if soft_change_bond:
        mg.ChangeBondMG.avoid_bond_breaking = True
        mg.ChangeBondMG.avoid_bond_forming = True

    MolecularGraph.action_space = cast(list[type[Action]], cast(object, action_space))

    evaluations: list[Evaluation] = dp.setup_filters("chembl_zinc")
    start_mol: Molecule = Molecule(start_smiles)

    print("----------Step 0----------")
    print("Molecule:", start_smiles)

    molecules: list[str] = [start_smiles]  # All molecules encountered
    are_valid: list[bool] = [evaluator.is_valid_molecule(start_mol, evaluations)]  # Validity of molecules encountered
    print("Is valid:", are_valid[0])

    fitnesses: dict[str, list[float]] = dict()  # All fitnesses of molecules encountered

    if strategy == "adaptive":
        # Evaluation needed for the PlogP calculation
        if evaluation_function.name == "PLogP":
            set_plogp_values(start_mol)

        start_fitness: float = evaluation_function.evaluate(start_mol)

        plateau: list[int] = []
        plateaus: list[tuple[float, list[int]]] = []

    with open(path + "walk.csv", "a", newline='') as file:
        writer = csv.writer(file)

        csv_row = ["smiles", "is_valid"]
        csv_row.extend([function.name for function in fitness_functions])
        csv_row.extend(["action", "action_context", "neighborhood_size"])

        writer.writerow(csv_row)

        for fitness_function, i in zip(fitness_functions, range(len(fitness_functions))):
            # Evaluation needed for the PlogP calculation
            function_name = fitness_function.name

            if function_name == "PLogP":
                set_plogp_values(start_mol)

            fitness = fitness_function.evaluate(start_mol)
            fitnesses[function_name] = [fitness]
            start_mol.set_value(function_name, fitness)

            print(function_name, ": ", fitnesses[function_name][0])

        csv_row = [start_smiles, are_valid[-1]]
        csv_row.extend(iter([str(fitness[-1]) for fitness in fitnesses.values()]))
        csv_row.extend(["None", "None"])

        init_smiles: str = start_smiles

        # At each step a random candidate of the molecule neighbor is chosen
        # Its validity and all its fitnesses are computed and saved
        for step in range(n_steps):
            # Get a random neighbor
            if strategy == "random":
                neighbor, action, neighborhood_size = get_random_neighbor(start_smiles, only_valid)

                csv_row.append(str(neighborhood_size))
                writer.writerow(csv_row)

                print("Neighborhood size:", neighborhood_size)
                print()
                print("----------Step " + str(step + 1) + "----------")
            # Get the best neighbor according to the evaluation function
            # and stops the walk if no better neighbor is found
            elif strategy == "adaptive":
                neighbor, action, neighbor_fitness, neighborhood_size = get_best_neighbor(start_smiles, evaluation_function,
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

                        with open(path + "plateaus.csv", "w") as plateau_file:
                            plateau_writer = csv.writer(plateau_file)
                            row = ["step", "smiles", "start_smiles", "evaluation_function"]
                            row.extend([function.name for function in fitness_functions])
                            plateau_writer.writerow(row)

                            for p in plateaus:
                                for s in p[1]:
                                    plateau_writer.writerow([s, start_smiles, molecules[0], evaluation_function.name,
                                                             *[str(fitnesses[function.name][s]) for function in
                                                               fitness_functions]])

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

            molecules.append(neighbor)
            are_valid.append(evaluator.is_valid_molecule(neighbor_mol, evaluations))
            print("Is valid:", (are_valid[-1]))

            for fitness_function, i in zip(fitness_functions, range(len(fitness_functions))):
                function_name = fitness_function.name

                if fitness_function.name == "PLogP":
                    set_plogp_values(neighbor_mol)

                fitness = fitness_function.evaluate(neighbor_mol)
                fitnesses[function_name].append(fitness)
                neighbor_mol.set_value(function_name, fitness)

                print(function_name + ":", fitnesses[fitness_function.name][-1])

            start_smiles = neighbor
            action_context: str = str(cast(dict, action.__getstate__())).replace('"', "'")

            print("Action taken:", action.class_name(), action_context)

            csv_row = [start_smiles, are_valid[-1]]
            csv_row.extend(iter([str(fitness[-1]) for fitness in fitnesses.values()]))
            csv_row.extend([action.class_name(), action_context])

    if strategy == "adaptive":
        with open(path + "local_optimum.csv", "w") as file:
            writer = csv.writer(file)

            if len(molecules) - 1 != n_steps:
                row = ["smiles", "start_smiles", "is_valid", "steps_taken", "evaluation_function"]
                row.extend([function.name for function in fitness_functions])

                writer.writerow(row)

                row = [start_smiles, molecules[0], are_valid[-1], len(molecules) - 1, evaluation_function.name]
                row.extend([str(fitness[-1]) for fitness in fitnesses.values()])

                writer.writerow(row)
            else:
                writer.writerow(["No local optimum found in " + str(n_steps) + " steps."])



    return molecules, are_valid, fitnesses
