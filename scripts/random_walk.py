import csv
import random

from evomol.action import Action
from evomol.evaluation import Function, ZincNormalizedLogP, NormalizedSAScore, CycleScore, NormalizedCycleScore
from evomol.representation import Molecule, MolecularGraph
from evomol.search import enumeration as en
from evomol import default_parameters as dp
from evomol import evaluation as evaluator


def get_random_neighbor(start_smiles: str, only_valid: bool=True) -> str:
    """
    Get a random neighbor for a molecule without looking if it is realistic.

    Arg:
        start_smiles (str): The smiles of the starting molecule
        only_valid (bool): If true, only valid smiles will be returned.

    Return:
        str: A random neighbor of the molecule
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
        return ""
    else:
        return random.choice(list(smiles_set))


def random_walk(start_smiles: str, n_steps: int, action_space: list[type[Action]],
                fitness_functions: list[Function], only_valid: bool=True)\
    -> tuple[list[str], list[bool], dict[str, list[float]]]:
    """
    Perform a random walk with a starting molecule and a set of allowed action.

    Args:
        start_smiles (str): The smiles of the starting molecule
        n_steps (int): The number of steps to perform
        action_space(ActionMolGraph): Actions allowed to perform
        fitness_functions (list[Function]): A list of fitness functions
        only_valid (bool): If True, only valid molecules will be kept during the random walk

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

    only_valid_str: str = "only_valid" if only_valid else "not_all_valid"

    fitnesses: dict[str, list[float]] = dict()  # All fitnesses of molecules encountered

    with open("./results/correlations/" + only_valid_str + "/" + str(n_steps) + "/" + start_smiles + "/"
              + "_".join([action.__name__ for action in MolecularGraph.action_space])
              + "/random_walk.csv", "a", newline='') as file:
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

        init_smiles: str = start_smiles

        # At each step a random candidate of the molecule neighbor is chosen
        # Its validity and all its fitnesses are computed and saved
        for step in range(n_steps):
            print("----------Step " + str(step + 1) + "----------")

            rand_neighbor = get_random_neighbor(start_smiles, only_valid)

            if rand_neighbor == "":
                rand_neighbor = init_smiles

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
