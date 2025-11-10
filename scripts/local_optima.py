import argparse
import os
import sys
from tqdm import tqdm

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
from evomol.evaluation.logp import LogP
from evomol.evaluation.plogp import PLogP
from evomol.evaluation.silly_walks import Silly_Walks
from scripts.fitness_landscape_analysis import correlations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str,
                        help="Path to the input file containing starting molecules",)
    parser.add_argument("-a", required=True, type=str, help="Actions to perform (space separated)",
                        choices=("AddAtomMG", "AddGroupMG", "ChangeBondMG", "CutAtomMG", "InsertCarbonMG",
                                 "MoveGroupMG", "RemoveAtomMG", "RemoveGroupMG", "SubstituteAtomMG"),
                        dest="actions", nargs="+")
    parser.add_argument("-e", "--evaluation", required=True, type=str, dest="evaluation_function",
                        choices=("QED", "SAScore", "LogP", "PLogP", "Silly_Walks"),
                        help="The evaluation function to use in adaptive walks")
    parser.add_argument("-n", "--max_steps", type=int, help="Maximum number of optimization steps",
                        default=1000)
    parser.add_argument("-o", "--only-valid", action="store_true",
                        help="If set, only valid molecules will be kept during the walk", dest="only_valid")
    parser.add_argument("-b", "--soft-change-bond", action="store_true",
                        help="If set, bond breaking and formation won't be allowed", dest="soft_change_bond")

    arguments = parser.parse_args()

    action_space: list[Action] = [eval("mg." + action) for action in arguments.actions]
    evaluation_function: Function = eval(arguments.evaluation_function)

    only_valid_str: str = "only_valid/" if parser.parse_args().only_valid else "not_all_valid/"

    if arguments.soft_change_bond and "ChangeBondMG" in arguments.actions:
        path_actions = arguments.actions.copy()
        path_actions.remove("ChangeBondMG")
        path_actions.append("SoftChangeBondMG")
    else:
        path_actions = arguments.actions

    molecules : DataFrame = pd.read_csv(arguments.input_file)
    molecules = molecules[molecules["mode"] == "not_all_valid"]

    for (index, row), tqdm in zip(molecules.iterrows(),
                                tqdm(range(len(molecules)), desc="Processing molecules", total=len(molecules))):
        smiles: str = row["final_molecule"]

        path = ("./results/adaptive_walk/" + only_valid_str + smiles + "/"
                + str(path_actions).replace("', '", "_").replace("['", "")
                .replace("']", "") + "/"+ arguments.evaluation_function + "/")

        print()
        print(path)
        print()

        os.makedirs(path, exist_ok=True)

        correlations(smiles, arguments.max_steps, action_space,
                     [QED, SAScore, LogP, PLogP, Silly_Walks],
                     [Tanimoto, Levenshtein, GED, NormalizedGED],
                     strategy="adaptive", evaluation_function=evaluation_function,
                     only_valid=arguments.only_valid, path=path, soft_change_bond=arguments.soft_change_bond)

        print()
