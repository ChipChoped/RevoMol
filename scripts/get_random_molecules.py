import os
import csv

import pandas as pd

from evomol import default_parameters as dp

from evomol.distance.ged import GED, NormalizedGED
from evomol.distance.levenshtein import Levenshtein
from evomol.distance.tanimoto import Tanimoto

from evomol.evaluation import QED, SAScore, LogP, PLogP
from evomol.evaluation.silly_walks import Silly_Walks
from evomol.representation import Molecule
from scripts.walk import set_plogp_values

if __name__ == "__main__":
    MODES = ["not_all_valid"]
    STEPS = 1000
    SEEDS = [1]
    PATH = "./results/random_walk"

    MOLECULES = [
        'C1=C2C1NC13CN2C1=N3',
        'CC12OC1C2=C1NN1',
        'C=CC12C3C4=NC31C42',
        'O=C1C(O)=CC12C=NO2',
        'OC1=C2N1NN(O)N2F',
        'C#CC(N)C(=C)C(N)N',
        'C1=NC=NC1C1=C2CN21',
        'N=C(O)N(N)N1C2=C1N2',
        'C1=C2NON=C3N(N1)N23',
        'NC(O)C1=C2C=C(C1)N2',
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
        "LogP",
        "PLogP",
        "Silly_Walks"
    ]


    with open(os.path.join(PATH, "random_molecules_summary.csv"), "w") as f:
        writer = csv.writer(f, lineterminator='\n')

        row = ["mode", "steps", "seed", "action_space", "start_molecule", "final_molecule", "is_valid",
               "tanimoto", "levenstein", "ged", "normalized_ged"]
        row.extend(FUNCTIONS)

        writer.writerow(row)

        dp.setup_default_parameters()

        # for smiles in MOLECULES:
        #     molecule = Molecule(smiles)
        #
        #     row = ["OD9", None, None, None, None, smiles, Silly_Walks.evaluate(molecule) == 0]
        #     row.extend([None] * 4)
        #
        #     set_plogp_values(molecule)
        #
        #     for function in [QED, SAScore, LogP, PLogP, Silly_Walks]:
        #         row.append(function.evaluate(molecule))
        #     writer.writerow(row)

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
