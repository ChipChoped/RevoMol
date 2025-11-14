import os
import csv

import pandas as pd

from evomol.distance.ged import GED, NormalizedGED
from evomol.distance.levenshtein import Levenshtein
from evomol.distance.tanimoto import Tanimoto

if __name__ == "__main__":
    MODES = ["not_all_valid", "only_valid"]
    STEPS = 1000
    SEEDS = [1, 2, 3, 4, 5]
    PATH = "./results/random_walk"

    MOLECULES = [
        "C",  # Methane
        "C(O)(=O)C1=C(OC(C)=O)C=CC=C1",  # Aspirin
        "CN1C(=NC2=C1C(=O)N(C(=O)N2C)C)CO",  # Caffeine
        "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",  # Citric acid
        "S1C=CSC1=C2SC=CS2",  # TTF
        "C1CN2C(=NN=C2C(F)(F)F)CN1C(=O)C[C@@H](CC3=CC(=C(C=C3F)F)F)N"  # Sitagliptin
    ]

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
        "logP",
        "PLogP",
        "Silly_Walks"
    ]


    with open(os.path.join(PATH, "random_molecules_summary.csv"), "w") as f:
        writer = csv.writer(f, lineterminator='\n')

        row = ["mode", "steps", "seed", "action_space", "start_molecule", "final_molecule", "is_valid",
               "tanimoto", "levenstein", "ged", "normalized_ged"]
        row.extend(FUNCTIONS)

        writer.writerow(row)

        for mode in MODES:
            for start_molecule in MOLECULES:
                for action_space in ACTION_SPACES:
                    if not (start_molecule == "C" and action_space in ["ChangeBondMG", "SoftChangeBondMG"]):
                        for seed in SEEDS:
                            try:
                                df = pd.read_csv(f"{PATH}/{mode}/{STEPS}/{start_molecule}/{action_space}/{seed}"
                                                 f"/walk.csv", skiprows=list(range(1, STEPS)))

                                row = [mode, STEPS, seed, action_space, start_molecule, df["smiles"].values[0],
                                       df["is_valid"].values[0]]

                                for distance in [Tanimoto, Levenshtein, GED, NormalizedGED]:
                                    row.append(distance.distance(start_molecule, df["smiles"].values[0]))

                                if mode == "not_all_valid":
                                    row.extend(df[FUNCTIONS].values[0].tolist())
                                elif mode == "only_valid":
                                    row.extend(df[FUNCTIONS[:-1]].values[0].tolist())
                                    row.append(None)  # Silly_Walks is not computed in only_valid mode

                                writer.writerow(row)
                            except Exception as e:
                                print(f"Could not process: {mode}/{STEPS}/{start_molecule}/{action_space}/{seed}")
                                print(e)
