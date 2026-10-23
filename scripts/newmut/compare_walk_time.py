"""
Temps des nouvelles mutations comparé aux anciennes (newmut).

Mêmes réglages des deux côtés (départ benzène, atomes C, O, N, 38 atomes max, aucun filtre, mêmes graines) :
- « anciennes » : espace d'action par défaut d'evomol, sans règles ;
- « nouvelles » : règles R1-R5 et toutes les nouvelles mutations (setup_newmut_action_space).

Trois mesures :
1. marches aléatoires de N_STEPS pas comme dans newmut_walk.ipynb (random_walk_core.random_walk, voisinage complet
   listé à chaque pas) : temps par pas, détaillé (t_neighbors : lister et appliquer tous les voisins) ;
2. tirage de l'algorithme évolutionnaire : N_STEPS mutations enchaînées par NewMutNeighborhoodStrategy.random_mutation
   (anciennes, nouvelles avec listes complètes, nouvelles avec lazy_rules) ;
3. par type de mutation : temps de list_actions sur les mêmes molécules (celles visitées par les marches « nouvelles »),
   sans cache.

Résultats : notebooks/random_walk/results/newmut_walk/compare_time/ (CSV et figure).

Usage :
    python scripts/newmut/compare_walk_time.py [--steps 200] [--walks 4] [--jobs 4]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT, "scripts", "newmut"))

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import pandas as pd

import newmut_walk as nw  # se place à la racine du projet
from evomol.newmut import rule_filtered as rf
from evomol.newmut.neighborhood_strategy import NewMutNeighborhoodStrategy
from evomol.representation import MolecularGraph, Molecule
from random_walk_core import random_walk

OUT = Path("notebooks/random_walk/results/newmut_walk/compare_time")

NEW_OPTIONS = ["MATRIX_RULES", "SUBSTITUTION_SHIFT", "GRAFT", "ANNELATE", "FLIP_PATH", "AROMATIZE",
               "INSERT_CARBON_KEEP_DOUBLE"]
BASE = {
    "START": "c1ccccc1", "BURN_IN_STEPS": 0, "MAX_ATOMS": 38, "ACCEPTED_ATOMS": ["C", "O", "N"],
    "ONLY_VALID": True, "UNIFORM_ACTION_TYPE": True, "EXCLUDE_CURRENT": True,
    "INSERT_CARBON": True, "ADD_ATOM_MULTIPLE_BONDS": False, "BOND_CHANGE_BY_ONE": False,
    "WITH_ADD_GROUP": False, "WITH_REMOVE_GROUP": False,
}
CONDITIONS = {
    # anciennes : AddAtomMG d'evomol ; nouvelles : AddAtomMG de newmut (ex-AppendAtomRuleMG)
    "anciennes": nw.make_config(**BASE, EVOMOL_ADD_ATOM=True, **{option: False for option in NEW_OPTIONS}),
    "nouvelles": nw.make_config(**BASE, EVOMOL_ADD_ATOM=False, **{option: True for option in NEW_OPTIONS}),
}
COLORS = {"anciennes": "#2a78d6", "nouvelles": "#eb6834"}  # palette de pi_walk (bleu, orange)


def _n_atoms(smiles: str) -> int:
    return Molecule(smiles).get_representation(MolecularGraph).nb_atoms


def walk_task(task: tuple[str, int, int]) -> pd.DataFrame:
    """Une marche (mesure 1)."""
    label, k, n_steps = task
    config = CONDITIONS[label]
    filters, kwargs = nw.configure(config)
    rows = random_walk(config["START"], n_steps, filters, seed=1000 + k, uniform_action_type=True, **kwargs)
    table = pd.DataFrame(rows)[["step", "smiles", "action", "neighborhood_size", "t_neighbors", "t_filters"]]
    table["n_atoms"] = [_n_atoms(s) for s in table.smiles]
    table["condition"], table["walk"] = label, k
    return table


def draw_task(task: tuple[str, bool, int, int]) -> pd.DataFrame:
    """Mutations enchaînées comme dans l'algorithme évolutionnaire (mesure 2)."""
    label, lazy, k, n_steps = task
    nw.configure(CONDITIONS[label])
    random.seed(2000 + k)
    strategy = NewMutNeighborhoodStrategy(depth=1, lazy_rules=lazy)
    molecule = Molecule(CONDITIONS[label]["START"])
    rows = []
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        molecule = strategy.random_mutation(molecule)
        rows.append({"step": step, "t": time.perf_counter() - t0,
                     "n_atoms": molecule.get_representation(MolecularGraph).nb_atoms})
    table = pd.DataFrame(rows)
    table["condition"] = label + (" (lazy_rules)" if lazy else "")
    table["walk"] = k
    return table


def per_type(smiles_list: list[str]) -> pd.DataFrame:
    """Temps de list_actions par type sur les mêmes molécules, sans cache (mesure 3)."""
    rows = []
    for label, config in CONDITIONS.items():
        nw.configure(config)
        rf.set_cache_size(0)
        for action_class in MolecularGraph.action_space:
            total, count = 0.0, 0
            for smiles in smiles_list:
                molecule = Molecule(smiles)
                t0 = time.perf_counter()
                count += len(action_class.list_actions(molecule))
                total += time.perf_counter() - t0
            rows.append({"condition": label, "mutation": action_class.class_name(),
                         "ms par molécule": 1000 * total / len(smiles_list),
                         "actions par molécule": count / len(smiles_list)})
        rf.set_cache_size(10_000)
    return pd.DataFrame(rows)


def summary_walks(walks: pd.DataFrame) -> pd.DataFrame:
    """Par condition : moyenne ± écart-type entre marches."""
    steps = walks[walks.step > 0]
    per_walk = steps.groupby(["condition", "walk"]).agg(
        temps_total_s=("t_neighbors", "sum"),  # aucun filtre : t_filters ~ 0
        ms_par_pas=("t_neighbors", lambda t: 1000 * t.mean()),
        voisins_par_pas=("neighborhood_size", "mean"),
        atomes_moyens=("n_atoms", "mean"),
    )
    per_walk["ms_par_voisin"] = per_walk.ms_par_pas / per_walk.voisins_par_pas
    mean, std = per_walk.groupby("condition").mean(), per_walk.groupby("condition").std()
    return mean.round(2).astype(str) + " ± " + std.round(2).astype(str)


def plot(walks: pd.DataFrame, save: Path) -> None:
    """Temps par pas selon la taille de la molécule, et temps cumulé le long des marches."""
    steps = walks[walks.step > 0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, group in steps.groupby("condition"):
        left.scatter(group.n_atoms, 1000 * group.t_neighbors, s=10, alpha=0.45, color=COLORS[label],
                     edgecolors="none", label=label)
        cumulative = group.pivot(index="step", columns="walk", values="t_neighbors").cumsum()
        right.plot(cumulative.index, cumulative.mean(axis=1), lw=2, color=COLORS[label], label=label)
        right.fill_between(cumulative.index, cumulative.min(axis=1), cumulative.max(axis=1),
                           color=COLORS[label], alpha=0.15, lw=0)
        right.annotate(label, (cumulative.index[-1], cumulative.mean(axis=1).iloc[-1]), xytext=(4, 0),
                       textcoords="offset points", va="center", color="#52514e")
    left.set(xlabel="atomes lourds de la molécule", ylabel="ms par pas (lister + appliquer les voisins)",
             title="Temps d'un pas selon la taille")
    left.legend(frameon=False)
    right.set(xlabel="pas", ylabel="temps cumulé (s)", title="Temps cumulé (moyenne, min–max entre marches)")
    right.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=150)
    plt.close(fig)


def main() -> None:
    """Lance les trois mesures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--walks", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    # conditions alternées pour que les deux subissent la même charge de la machine
    walk_tasks = [(label, k, args.steps) for k in range(args.walks) for label in CONDITIONS]
    draw_tasks = [(label, lazy, k, args.steps) for k in range(args.walks)
                  for label, lazy in (("anciennes", False), ("nouvelles", False), ("nouvelles", True))]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs, mp_context=get_context("spawn")) as pool:
        walks = pd.concat(pool.map(walk_task, walk_tasks), ignore_index=True)
        draws = pd.concat(pool.map(draw_task, draw_tasks), ignore_index=True)
    walks.to_csv(OUT / "walks.csv.gz", index=False)
    draws.to_csv(OUT / "ea_draws.csv.gz", index=False)

    print(f"\n1. Marches de {args.steps} pas ({args.walks} par condition, {args.jobs} en parallèle) :")
    print(summary_walks(walks).to_string())

    print(f"\n2. Tirage de l'algorithme évolutionnaire, {args.steps} mutations enchaînées :")
    per_draw = draws.groupby(["condition", "walk"]).agg(temps_total_s=("t", "sum"), ms_par_mutation=("t", lambda t: 1000 * t.mean()),
                                                        atomes_moyens=("n_atoms", "mean"))
    print((per_draw.groupby("condition").mean().round(2).astype(str) + " ± "
           + per_draw.groupby("condition").std().round(2).astype(str)).to_string())

    sample = (walks[(walks.condition == "nouvelles") & (walks.step > 0)].drop_duplicates("smiles")
              .sample(n=min(40, walks.smiles.nunique()), random_state=0).smiles.tolist())
    types = per_type(sample)
    types.to_csv(OUT / "per_type.csv", index=False)
    print(f"\n3. list_actions par type, sans cache, sur {len(sample)} molécules des marches « nouvelles » :")
    print(types.pivot(index="mutation", columns="condition", values="ms par molécule").round(1).to_string())

    plot(walks, OUT / "temps_par_pas.png")
    print(f"\nterminé en {(time.perf_counter() - t0) / 60:.1f} min ; résultats dans {OUT}")


if __name__ == "__main__":
    main()
