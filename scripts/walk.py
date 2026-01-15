import csv
import random
import time
from multiprocessing import Pool, Value, Array, Lock, Process
from multiprocessing.sharedctypes import Synchronized
from typing import cast, Tuple, Any

from tqdm import tqdm

from evomol import default_parameters as dp
from evomol import evaluation as evaluator
from evomol.action import Action
from evomol.evaluation import Function, ZincNormalizedLogP, NormalizedSAScore, CycleScore, NormalizedCycleScore, \
    Evaluation, LogP, SAScore, QED, PLogP, Silly_Walks
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


def get_deep_neighborhood(neighborhood: list[tuple[str, list[Action]]]) -> list[tuple[str, list[Action]]]:
    depth_smiles: list[str] = []
    depth_actions: list[list[Action]] = []

    for possible_neighbor in neighborhood:
        depth_neighborhood: Tuple[list[str], list[Action]] = en.find_neighbors(
            Molecule(Molecule(possible_neighbor[0]).get_representation(MolecularGraph).canonical_smiles),
            max_depth=1, info=True)

        depth_smiles.extend(depth_neighborhood[0])
        depth_actions.extend([possible_neighbor[1] + [depth_neighbor] for depth_neighbor in depth_neighborhood[1]])

    return [(Molecule(smiles).get_representation(MolecularGraph).canonical_smiles,
                                          action) for smiles, action in zip(depth_smiles, depth_actions)]


def get_random_neighbor(start_smiles: str, only_valid: bool = True, depth: int = 1) -> tuple[str, Action | None, int]:
    """
    Get a random neighbor for a molecule without looking if it is realistic.

    Arg:
        start_smiles (str): The smiles of the starting molecule
        only_valid (bool): If true, only valid smiles will be returned.
        depth (int): The depth of the search

    Return:
        str: A random neighbor of the molecule
        Action: The action taken to get to the neighbor
        int: Size of the neighborhood
    """
    can_smi_start = (
        Molecule(start_smiles).get_representation(MolecularGraph).canonical_smiles
    )

    possible_smiles, possible_actions = en.find_neighbors(Molecule(can_smi_start), max_depth=1, info=True)
    neighborhood: list[tuple[str, list[Action]]] = [(Molecule(smiles).get_representation(MolecularGraph)
                                                     .canonical_smiles, [action])
                                                    for smiles, action in zip(possible_smiles, possible_actions)]

    for _ in range(1, depth):
        neighborhood = get_deep_neighborhood(neighborhood)

    if only_valid:
        valid_smiles: list[tuple[str, list[Action]]] = [neighbor for neighbor in neighborhood
                                                        if evaluator.is_valid_molecule(Molecule(neighbor[0]),
                                                                                       dp.setup_filters("chembl_zinc"))]
        neighborhood = valid_smiles

    if len(neighborhood) == 0:
        return "", None, 0
    else:
        chosen_smiles, chosen_action = random.choice(list(neighborhood))
        return chosen_smiles, chosen_action, len(neighborhood)


def find_highest_fitness(lock: Lock, neighbors_indexes: list[int], neighborhood: list[tuple[str, list[Action]]],  # type: ignore
                         fitness_function: Function, best_index: Synchronized, best_fitness: Synchronized) -> None:
    """
    Find the neighbor with the highest fitness higher than the starting molecule one.

    Arg:
        lock (Lock): The lock to acquire lock
        neighbors_indexes (list[int]): The indexes of neighbors
        neighborhood (list[tuple[str, Action]]): List of tuple of smiles and list of actions to reach them
        fitness_function (Function): The fitness function to evaluate neighbors
        best_index (Synchronized): The index of the best neighbor
        best_fitness (Synchronized): The fitness of the best neighbor
    """
    dp.setup_default_parameters()

    for index in neighbors_indexes:
        neighbor_fitness = fitness_function.evaluate(Molecule(neighborhood[index][0]))

        lock.acquire()

        if neighbor_fitness < best_fitness.value:
            best_index.value = index
            best_fitness.value = neighbor_fitness

        lock.release()


def find_first_improvement(neighborhood: list[tuple[str, list[Action]]], start_mol: Molecule,
                           fitness_function: Function, aggregate_realism: bool = False)\
    -> tuple[str, list[Action] | None, float, int]:
    if fitness_function.name == "PLogP":
        set_plogp_values(start_mol)

    start_fitness = fitness_function.evaluate(start_mol)
    best_neighbor: tuple[str, list[Action] | None] = ("", None)
    best_fitness: float = start_fitness

    for neighbor in neighborhood:
        if neighbor[0] != "":
            neighbor_mol = Molecule(neighbor[0])

            if fitness_function.name == "PLogP":
                set_plogp_values(neighbor_mol)

            if aggregate_realism:
                if fitness_function.name == "SAScore":
                    neighbor_fitness = (fitness_function.evaluate(neighbor_mol)
                                        * (1 + Silly_Walks.evaluate(neighbor_mol)))
                else:
                    neighbor_fitness = (fitness_function.evaluate(neighbor_mol)
                                        * (1 - Silly_Walks.evaluate(neighbor_mol)))
            else:
                neighbor_fitness = fitness_function.evaluate(neighbor_mol)

            if (fitness_function.name == "QED" and neighbor_fitness > best_fitness) or \
                (fitness_function.name in ["SAScore", "LogP", "PLogP", "Silly_Walks"]
                 and neighbor_fitness < best_fitness):
                best_fitness = neighbor_fitness
                best_neighbor = neighbor

                break

    return best_neighbor[0], best_neighbor[1], best_fitness, len(neighborhood)


def get_best_neighbor(start_smiles: str, fitness_function: Function, strategy: str, aggregate_realism: bool = False,
                      only_valid: bool = True, depth: int = 1, seed: int = 0) -> tuple[str, None, int, int] | tuple[
    str | list[Action] | tuple[str, list[Action]], str | list[Action] | tuple[str, list[Action]], Any, int] | tuple[
                                             str, list[Action] | None, float | Synchronized, int] | tuple[
                                             str | Action | None, str | Action | None, float | Synchronized, int]:
    """
    Get the neighbor with the highest fitness equal or higher than the starting molecule.

    Arg:
        start_smiles (str): The smiles of the starting molecule
        fitness_function (Function): The fitness function to evaluate neighbors
        strategy (str): The strategy used to evaluate neighbors (best-improv, first-improv)
        aggregate_realism (bool): If True, makes an aggregation between the evaluation function if used with
        the silly walks function
        only_valid (bool): If true, only valid smiles will be considered.
        depth (int): Depth of the search
        seed (int): Seed for first improvement shuffle

    Return:
        str: The best neighbor of the molecule
        float: The fitness of the best neighbor
        int: Size of the neighborhood
    """
    can_smi_start = (
        Molecule(start_smiles).get_representation(MolecularGraph).canonical_smiles
    )

    possible_smiles, possible_actions = en.find_neighbors(Molecule(can_smi_start), max_depth=1, info=True)
    neighborhood: list[tuple[str, list[Action]]] = [(Molecule(smiles).get_representation(MolecularGraph)
                                                     .canonical_smiles, [action])
                                                    for smiles, action in zip(possible_smiles, possible_actions)]

    if len(neighborhood) == 0:
        return "", None, 0, 0
    elif strategy == "best_improv":
        for _ in range(1, depth):
            neighborhood = get_deep_neighborhood(neighborhood)

        if only_valid:
            valid_smiles: list[tuple[str, list[Action]]] = [neighbor for neighbor in neighborhood
                                                            if evaluator.is_valid_molecule(Molecule(neighbor[0]),
                                                                                           dp.setup_filters("chembl_zinc"))]
            neighborhood = valid_smiles

        start_mol = Molecule(start_smiles)

        if fitness_function.name == "PLogP":
            set_plogp_values(start_mol)

        if aggregate_realism:
            start_fitness = fitness_function.evaluate(start_mol) * (1 - Silly_Walks.evaluate(start_mol))
        else:
            start_fitness = fitness_function.evaluate(start_mol)

        best_neighbor: tuple[str, list[Action] | None] = ("", None)

        if fitness_function.name == "Silly_Walks":
            best_index: Synchronized = Value("i", -1)
            best_fitness: Synchronized = Value("d", start_fitness)
            lock: Lock = Lock()  # type: ignore

            max_processes = 4
            processes: list[Process] = []

            for n_process in range(max_processes):
                if n_process == max_processes - 1:
                    neighbors_indexes = range(len(neighborhood) // max_processes * max_processes,
                                              len(neighborhood))
                else:
                    neighbors_indexes = range(len(neighborhood) // max_processes * n_process,
                                              len(neighborhood) // max_processes * (n_process + 1))

                process = Process(target=find_highest_fitness,
                                  args=(lock, neighbors_indexes, neighborhood, fitness_function,
                                        best_index, best_fitness))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            if best_index.value == -1:
                return "", None, 0, len(neighborhood)

            return (neighborhood[best_index.value][0], neighborhood[best_index.value][1],
                    best_fitness.value, len(neighborhood))
        else:
            best_fitness: float = start_fitness
            for neighbor in neighborhood:
                if neighbor[0] != "":
                    neighbor_mol = Molecule(neighbor[0])

                    if fitness_function.name == "PLogP":
                        set_plogp_values(neighbor_mol)

                    if aggregate_realism:
                        if fitness_function.name == "SAScore":
                            neighbor_fitness = (fitness_function.evaluate(neighbor_mol)
                                               * (1 + Silly_Walks.evaluate(neighbor_mol)))
                        else:
                            neighbor_fitness = (fitness_function.evaluate(neighbor_mol)
                                               * (1 - Silly_Walks.evaluate(neighbor_mol)))
                    else:
                        neighbor_fitness = fitness_function.evaluate(neighbor_mol)

                    if (fitness_function.name == "QED" and neighbor_fitness > best_fitness) or \
                        (fitness_function.name in ["SAScore", "LogP", "PLogP", "Silly_Walks"]
                         and neighbor_fitness < best_fitness):
                        best_fitness = neighbor_fitness
                        best_neighbor = neighbor

            return best_neighbor[0], best_neighbor[1], best_fitness, len(neighborhood)
    elif strategy == "first_improv":
        start_mol = Molecule(start_smiles)

        # Shuffle the neighborhood to get different results by changing the seed
        random.Random(seed).shuffle(neighborhood)

        if depth == 1:
            return find_first_improvement(neighborhood, start_mol, fitness_function, aggregate_realism)
        elif depth > 1:
            root_neighborhood: list[tuple[str, list[Action]]] = neighborhood
            best_improvement: tuple[str, list[Action] | None, float, int] = ("", None, 0, len(root_neighborhood))
            neighborhood_size: int = 0

            for i in range(len(root_neighborhood)):
                for _ in range(1, depth):
                    neighborhood = get_deep_neighborhood([root_neighborhood[i]])

                neighborhood_size += len(neighborhood)
                best_improvement = find_first_improvement(neighborhood, start_mol, fitness_function, aggregate_realism)

                if best_improvement[1] is not None:
                    return (best_improvement[0], best_improvement[1], best_improvement[2],
                            int(neighborhood_size / (i+1) * len(root_neighborhood)))

            return best_improvement
        else:
            raise ValueError("Depth cannot be less than 1")
    else:
        raise ValueError(f"Unknown adaptive walk method: {strategy}")


def walk(start_smiles: str, n_steps: int, action_space: list[Action],
         fitness_functions: list[Function], strategy: str= "random", evaluation_function: Function = None,
         aggregate_realism: bool = False, only_valid: bool = True, path: str = "results", seed: int = 0,
         soft_change_bond: bool = False, depth: int = 1) -> tuple[list[str], list[bool], dict[str, list[float]]]:
    """
    Perform a random or adaptive walk with based on a starting molecule and a set of allowed actions.

    Args:
        start_smiles (str): The smiles of the starting molecule
        n_steps (int): The number of steps to perform
        action_space(list[Action]): Actions allowed to perform
        fitness_functions (list[Function]): A list of fitness functions
        strategy (str): The type of walk to perform ("random" or "adaptive")
        evaluation_function (Function): The fitness function to evaluate neighbors in adaptive walks
        aggregate_realism (bool): If True, makes an aggregation between the evaluation function if used with
        the silly walks function
        only_valid (bool): If True, only valid molecules will be kept during the random walk
        path (str): The path to the directory where results are stored
        seed (int): Seed for random operations
        soft_change_bond (bool): If True, bond breaking and formation won't be allowed (False by default)
        depth (int): Depth of the search

    Returns:
        list[str]: The path took during the random walk (list of smiles)
        list[bool]: Whether each molecule encountered is valid
        dict[list[float]]: Dictionary of fitnesses scores for each molecule encountered
    """
    # Initialize and set the actions space
    dp.setup_default_action_space()

    if soft_change_bond:
        mg.ChangeBondMG.avoid_bond_breaking = True
        mg.ChangeBondMG.avoid_bond_forming = True

    MolecularGraph.action_space = cast(list[type[Action]], cast(object, action_space))

    evaluations: list[Evaluation] = dp.setup_filters("chembl_zinc")
    start_mol: Molecule = Molecule(start_smiles)

    runtime: float = 0

    print("----------Step 0----------")
    print("Molecule:", start_smiles)

    molecules: list[str] = [start_smiles]  # All molecules encountered
    are_valid: list[bool] = [evaluator.is_valid_molecule(start_mol, evaluations)]  # Validity of molecules encountered
    print("Is valid:", are_valid[0])

    fitnesses: dict[str, list[float]] = dict()  # All fitnesses of molecules encountered

    if strategy in ["best_improv", "first_improv"]:
        # Evaluation needed for the PlogP calculation
        if evaluation_function.name == "PLogP":
            set_plogp_values(start_mol)

        start_fitness: float = evaluation_function.evaluate(start_mol)

        plateau: list[int] = []
        plateaus: list[tuple[float, list[int]]] = []

    with open(path + "walk.csv", "a", newline='') as file:
        writer = csv.writer(file, lineterminator='\n')

        csv_row = ["smiles", "is_valid"]
        csv_row.extend([function.name for function in fitness_functions])

        if aggregate_realism:
            csv_row.append(evaluation_function.name + "-Silly_Walks")

        if depth == 1:
            csv_row.extend(["action", "action_context", "neighborhood_size"])
        else:
            csv_row.extend(["actions", "actions_context", "neighborhood_size"])

        writer.writerow(csv_row)

        for fitness_function, i in zip(fitness_functions, range(len(fitness_functions))):
            # Evaluation needed for the PlogP calculation
            function_name = fitness_function.name

            if function_name == "PLogP":
                set_plogp_values(start_mol)

            fitness = fitness_function.evaluate(start_mol)
            fitnesses[function_name] = [fitness]
            start_mol.set_value(function_name, fitness)

            print(function_name, ":", fitnesses[function_name][0])

        csv_row = [start_smiles, are_valid[-1]]
        csv_row.extend(iter([str(fitness[-1]) for fitness in fitnesses.values()]))

        if aggregate_realism:
            aggregation: float = fitnesses[evaluation_function.name][0] * (1 - fitnesses["Silly_Walks"][0])
            print(evaluation_function.name + "-Silly_Walks:", aggregation)

            csv_row.append(str(aggregation))

        csv_row.extend(["None", "None"])

        init_smiles: str = start_smiles

        # At each step a random candidate of the molecule neighbor is chosen
        # Its validity and all its fitnesses are computed and saved
        for step in range(n_steps):
            # Get a random neighbor
            start_time: float = time.time()

            if strategy == "random":
                neighbor, actions, neighborhood_size = get_random_neighbor(start_smiles=start_smiles,
                                                                           only_valid=only_valid)

                csv_row.append(str(neighborhood_size))
                writer.writerow(csv_row)

                end_time = time.time() - start_time
                runtime += end_time

                print("Neighborhood size:", neighborhood_size)
                print("Time:", end_time)
                print()
                print("----------Step " + str(step + 1) + "----------")
            # Get the best neighbor according to the evaluation function
            # and stops the walk if no better neighbor is found
            elif strategy in ["best_improv", "first_improv"]:
                neighbor, actions, neighbor_fitness, neighborhood_size =\
                    get_best_neighbor(start_smiles=start_smiles, fitness_function=evaluation_function,
                                      strategy=strategy, aggregate_realism=aggregate_realism, only_valid=only_valid,
                                      depth=depth, seed=seed)

                csv_row.append(str(neighborhood_size))
                writer.writerow(csv_row)

                end_time = time.time() - start_time
                runtime += end_time

                print("Neighborhood size:", neighborhood_size)
                print("Time:", end_time)
                print()
                print("----------Step " + str(step + 1) + "----------")

                start_fitness: float
                plateaus: list[tuple[float, list[int]]]

                if neighbor == "":
                    print("No better neighbor found, stopping the walk.")

                    if len(plateau) > 0:
                        plateaus.append((start_fitness, plateau))

                        with open(path + "plateaus.csv", "w", newline='') as plateau_file:
                            plateau_writer = csv.writer(plateau_file, lineterminator='\n')

                            if aggregate_realism:
                                csv_row.append(evaluation_function.name + "-Silly_Walks")

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

            if aggregate_realism:
                print(evaluation_function.name + "-Silly_Walks:", neighbor_fitness)

            start_smiles = neighbor
            actions_context: list[str] = [str(cast(dict, action.__getstate__())).replace('"', "'")
                                          for action in actions]

            print("Action(s) taken:")

            for action, action_context in zip(actions, actions_context):
                print(action.class_name(), action_context)

            csv_row = [start_smiles, are_valid[-1]]

            if aggregate_realism:
                csv_row.append(neighbor_fitness)

            csv_row.extend(iter([str(fitness[-1]) for fitness in fitnesses.values()]))

            if depth == 1:
                csv_row.extend([actions[0].class_name(), actions_context[0]])
            else:
                csv_row.extend([[action.class_name() for action in actions], actions_context])

    if strategy in ["best_improv", "first_improv"]:
        with open(path + "local_optimum.csv", "a", newline='') as file:
            writer = csv.writer(file, lineterminator='\n')

            if len(molecules) - 1 != n_steps:
                row = ["smiles", "start_smiles", "is_valid", "steps_taken", "evaluation_function"]
                row.extend([function.name for function in fitness_functions])

                writer.writerow(row)

                row = [start_smiles, molecules[0], are_valid[-1], len(molecules) - 1, evaluation_function.name]
                row.extend([str(fitness[-1]) for fitness in fitnesses.values()])

                writer.writerow(row)
            else:
                writer.writerow(["No local optimum found in " + str(n_steps) + " steps."])

    with open(path + "running_time.csv", "a", newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(["time"])
        writer.writerow([runtime])

        print("\n Running time (s):", runtime)


    return molecules, are_valid, fitnesses
