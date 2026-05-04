import argparse
import os
import random
import sys
from multiprocessing import Process

import pandas as pd
from pandas import DataFrame

# Add the parent directory to the path to import the module evomol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evomol.action import Action, molecular_graph as mg
from evomol.distance.ged import GED, NormalizedGED
from evomol.distance.levenshtein import Levenshtein
from evomol.distance.tanimoto import Tanimoto
from evomol.evaluation import Function

from evomol.evaluation.qed import QED
from evomol.evaluation.sa_score import SAScore
from evomol.evaluation.logp import LogP, LogP_Max
from evomol.evaluation.plogp import PLogP, PLogP_Max
from evomol.evaluation.silly_walks import Silly_Walks
from scripts.correlations import correlations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", type=str, choices=("random", "best_improv", "first_improv"),
                        help="The type of walk to perform")
    parser.add_argument("input_file", type=str,
                        help="Path to the input file containing starting molecules",)
    parser.add_argument("-a", required=True, type=str, help="Actions to perform (space separated)",
                        choices=("AddAtomMG", "AddGroupMG", "ChangeBondMG", "CutAtomMG", "InsertCarbonMG",
                                 "MoveGroupMG", "RemoveAtomMG", "RemoveGroupMG", "SubstituteAtomMG"),
                        dest="actions", nargs="+")
    parser.add_argument("-e", "--evaluation", type=str, dest="evaluation_function", default=None,
                        choices=("QED", "SAScore", "LogP", "PLogP", "LogP_Max", "PLogP_Max", "Silly_Walks"),
                        help="The evaluation function to use in adaptive walks")
    parser.add_argument("-r", "--aggregate-realism", action="store_true",
                        help="If set, makes an aggregation between the evaluation function if used with the silly walks"
                             "function")
    parser.add_argument("-n", "--max_steps", type=int, help="Maximum number of optimization steps",
                        default=1000)
    parser.add_argument("-o", "--only-valid", action="store_true",
                        help="If set, only valid molecules will be kept during the walk", dest="only_valid")
    parser.add_argument("-s", "--seed", type=int, help="Random seed to use", dest="seed", default=0)
    parser.add_argument("-b", "--soft-change-bond", action="store_true",
                        help="If set, bond breaking and formation won't be allowed", dest="soft_change_bond")
    parser.add_argument("-d", "--depth", type=int, help="Depth of the search", dest="depth", default=1)

    arguments = parser.parse_args()

    random.seed(arguments.seed)

    action_space: list[Action] = [eval("mg." + action) for action in arguments.actions]

    only_valid_str: str = "only_valid" if parser.parse_args().only_valid else "not_all_valid"

    if arguments.soft_change_bond and "ChangeBondMG" in arguments.actions:
        path_actions = arguments.actions.copy()
        path_actions.remove("ChangeBondMG")
        path_actions.append("SoftChangeBondMG")
    else:
        path_actions = arguments.actions

    if arguments.strategy == "first_improv":
        seed: str = str(arguments.seed) + "/"
    else:
        seed: str = ""

    if arguments.strategy != "random":
        if arguments.evaluation_function is None:
            raise Exception("You must specify the evaluation function to use for adaptive walks")
        else:
            evaluation_function: Function = eval(arguments.evaluation_function)
            evaluation_function_str: str = arguments.evaluation_function

            if arguments.aggregate_realism:
                evaluation_function_str += "-Silly_Walks/"
            else:
                evaluation_function_str += "/"
    else:
        evaluation_function = None
        evaluation_function_str: str = ""

    molecules : DataFrame = pd.read_csv(arguments.input_file)

    processes: list[Process] = []

    for index, row in molecules.iterrows():
        smiles: str = row["final_molecule"]

        path = ("./results/" + arguments.strategy + "_walk/" + only_valid_str + "/" + str(arguments.depth) + "/" +
                smiles + "/" + str(path_actions).replace("', '", "_").replace("['", "")
                .replace("']", "") + "/" + evaluation_function_str + seed)

        print()
        print(path)
        print()

        os.makedirs(path, exist_ok=True)

        process = Process(target=correlations,
                          args=(smiles, arguments.max_steps, action_space,
                                [QED, SAScore, LogP, PLogP, Silly_Walks],
                                [Tanimoto, Levenshtein, GED, NormalizedGED],
                                arguments.strategy, evaluation_function, arguments.aggregate_realism,
                                arguments.only_valid, path, seed, arguments.soft_change_bond, arguments.depth))

        process.start()
        processes.append(process)

        print()

    for process in processes:
        process.join()

    print("done")
