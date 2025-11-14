import argparse
import csv
import sys
import os

import pandas as pd
from pandas import DataFrame

from tqdm import tqdm

# Add the parent directory to the path to import the module evomol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evomol import default_parameters as dp
from evomol.evaluation import QED, SAScore, LogP, PLogP, Function
from evomol.evaluation.silly_walks import Silly_Walks
from evomol.action import molecular_graph as mg

from scripts.walk import walk


FUNCTIONS = [
    QED,
    SAScore,
    LogP,
    PLogP,
    Silly_Walks
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input file containing local optima")
    parser.add_argument("steps", type=int,
                        help="Number of steps to perform to get further away from the local optima")

    arguments = parser.parse_args()

    local_optima_df = pd.read_csv(arguments.input_file)
    local_optima_df = local_optima_df[local_optima_df["evaluation_function"] != "Silly_Walks"]

    steps: int = arguments.steps

    dp.setup_default_parameters()

    path: str = "./results/local_optima_attractivity/" + str(steps)
    os.makedirs(path, exist_ok=True)

    with open(path + "/attractivity.csv", "w") as file:
        writer = csv.writer(file, lineterminator="\n")

        writer.writerow(["local_optimum", "action_space", "evaluation_function", "fitness", "random_steps",
                         "adaptive_steps", "random_walk_final", "adaptive_walk_final", "is_equal"])

        for (index, row), tqdm in zip(local_optima_df.iterrows(),
                                      tqdm(range(len(local_optima_df.values)), desc="Processing local optima",
                                           total=len(local_optima_df.values))):
            local_optimum: str = row["local_optimum"]
            action_space_str: str = row["action_space"]
            evaluation_function: Function = eval(row["evaluation_function"]) if row["evaluation_function"] != "logP" else LogP

            soft_change_bond: bool = False

            if "Soft" in action_space_str:
                action_space = action_space_str.replace("Soft", "")
                soft_change_bond = True
            else:
                action_space = action_space_str

            action_space = action_space.split("_")
            action_space = [eval("mg." + action) for action in action_space]

            path = ("./tmp/random_walk/not_only_valid/" + str(steps) + "/" + local_optimum + "/" + action_space_str + "/")

            os.makedirs(path, exist_ok=True)

            print("\n\nStarting random walk from local optimum:", local_optimum)

            random_walk = walk(local_optimum, steps, action_space, FUNCTIONS,
                               strategy="random", only_valid=False, path=path, soft_change_bond=soft_change_bond)

            print("\n\nStarting adaptive walk from the last molecule of the random walk\nwith evaluation function:",
                  evaluation_function.name, "and action space:", action_space_str)

            adaptive_walk = walk(random_walk[0][-1], steps * 25, action_space, FUNCTIONS,
                                 strategy="adaptive", only_valid=False, path=path, soft_change_bond=soft_change_bond,
                                 evaluation_function=evaluation_function)

            print("\n")

            print("Local optimum:", local_optimum)
            print("Random walk final molecule:", random_walk[0][-1])
            print("Adaptive walk final molecule:", adaptive_walk[0][-1])
            print(local_optimum == adaptive_walk[0][-1])

            print("-----")

            writer.writerow([local_optimum, action_space_str, row["evaluation_function"],
                             row[row["evaluation_function"]], str(steps), str(len(adaptive_walk) - 1),
                             random_walk[0][-1], adaptive_walk[0][-1], str(local_optimum == adaptive_walk[0][-1])])


