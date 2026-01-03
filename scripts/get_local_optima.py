import csv
import sys
import os

import pandas as pd

# Add the parent directory to the path to import the module evomol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evomol.distance.ged import GED, NormalizedGED
from evomol.distance.levenshtein import Levenshtein
from evomol.distance.tanimoto import Tanimoto


if __name__ == "__main__":
    MODES = ["not_all_valid"]
    SEEDS = [1, 2, 3, 4, 5]
    PATH = "./results/adaptive_walk"

    ACTION_SPACES = [
        "ChangeBondMG",
        "SoftChangeBondMG",
        "AddAtomMG_RemoveAtomMG",
        "AddAtomMG_RemoveAtomMG_ChangeBondMG",
        "AddAtomMG_RemoveAtomMG_SoftChangeBondMG"
    ]

    FUNCTIONS = [
        "QED",
        "SAScore",
        "LogP",
        "PLogP",
        "Silly_Walks"
    ]

    with open(os.path.join(PATH, "local_optima_sample_summary.csv"), "w") as f:
        writer = csv.writer(f, lineterminator='\n')

        row = ["mode", "steps_taken", "action_space", "random_molecule", "local_optimum", "is_valid",
               "evaluation_function", "tanimoto", "levenstein", "ged", "normalized_ged"]
        row.extend(FUNCTIONS)

        writer.writerow(row)

        MOLECULES = pd.read_csv("./results/random_walk/random_molecules_sample.csv")
        MOLECULES = MOLECULES["final_molecule"]

        for mode in MODES:
            mol_count = 0
            error_count = 0
            key_error_count = 0
            file_error_count = 0

            for start_molecule in MOLECULES:
                for action_space in ACTION_SPACES:
                    for evaluation_function in FUNCTIONS:
                        if not (start_molecule == "C" and action_space in ["ChangeBondMG", "SoftChangeBondMG"]):
                            try:
                                df = pd.read_csv(f"{PATH}/{mode}/{start_molecule}/"
                                                 f"{action_space}/{evaluation_function}/local_optimum.csv")

                                row = [mode, df["steps_taken"].values[0], action_space, start_molecule,
                                       df["smiles"].values[0], df["is_valid"].values[0], evaluation_function]

                                for distance in [Tanimoto, Levenshtein, GED, NormalizedGED]:
                                    row.append(distance.distance(start_molecule, df["smiles"].values[0]))

                                if mode == "not_all_valid":
                                    row.extend(df[FUNCTIONS].values[0].tolist())
                                elif mode == "only_valid":
                                    row.extend(df[FUNCTIONS[:-1]].values[0].tolist())
                                    row.append(None)  # Silly_Walks is not computed in only_valid mode

                                writer.writerow(row)

                                mol_count += 1
                            except FileNotFoundError as e:
                                print(f"Could not process: {mode}/{start_molecule}/{action_space}/{evaluation_function}")
                                # print(e, "\n")
                                try:
                                    print(pd.read_csv(f"{PATH}/{mode}/{start_molecule}/{action_space}"
                                                      f"/{evaluation_function}/walk.csv"), "\n")
                                except Exception as e:
                                    print(e, "\n")

                                error_count += 1
                                file_error_count += 1
                            except KeyError as e:
                                print(f"Could not process: {mode}/{start_molecule}/{action_space}/{evaluation_function}")
                                print(e, "\n")

                                print(pd.read_csv(f"{PATH}/{mode}/{start_molecule}/{action_space}"
                                                  f"/{evaluation_function}/local_optimum.csv").columns, "\n")

                                error_count += 1
                                key_error_count += 1

    print(mol_count, error_count)
