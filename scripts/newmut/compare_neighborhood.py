"""
Temps du voisinage complet, anciennes et nouvelles mutations, sur les mêmes molécules (newmut).

1. 100 molécules (atomes C, O, N) : la moitié atteinte par les anciennes mutations, l'autre par les nouvelles
   (chaînes de 5 à 60 mutations au hasard depuis le benzène), pour ne favoriser aucun des deux côtés.
2. Pour chaque molécule, son voisinage complet comme dans une marche aléatoire (random_walk_core.neighbor_actions :
   toutes les mutations de tous les types listées, puis appliquées), avec :
   - « anciennes » : espace d'action par défaut d'evomol, sans règles ;
   - « nouvelles » : règles R1-R5 et toutes les nouvelles mutations.
   Mémoires vidées avant chaque molécule (règles, listes filtrées) : temps « à froid ».
3. Temps par molécule, nombre de voisins, rapport nouvelles / anciennes, par taille et par type de mutation.

Résultats : notebooks/random_walk/results/newmut_walk/compare_neighborhood/ (CSV et figure).

Usage :
    python scripts/newmut/compare_neighborhood.py [--molecules 100] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT, "scripts", "newmut"))

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import pandas as pd

import newmut_walk as nw  # se place à la racine du projet
from compare_walk_time import COLORS, CONDITIONS
from evomol.newmut import fast_rules
from evomol.newmut import rule_filtered as rf
from evomol.representation import MolecularGraph, Molecule
from random_walk_core import neighbor_actions

OUT = Path("notebooks/random_walk/results/newmut_walk/compare_neighborhood")


def pick_molecules(n: int, seed: int) -> pd.DataFrame:
    """n molécules distinctes, la moitié par chaque espace d'action."""
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()
    for label in CONDITIONS:
        nw.configure(CONDITIONS[label])
        while sum(row["origine"] == label for row in rows) < n // 2 + (n % 2 if label == "nouvelles" else 0):
            molecule = Molecule("c1ccccc1")
            for _ in range(rng.randint(5, 60)):
                action_class = rng.choice(MolecularGraph.action_space)
                actions = action_class.list_actions(molecule)
                if actions:
                    molecule = rng.choice(actions).apply()
            smiles = str(molecule)
            if smiles and smiles not in seen:
                seen.add(smiles)
                rows.append({"smiles": smiles, "origine": label,
                             "n_atoms": molecule.get_representation(MolecularGraph).nb_atoms})
    return pd.DataFrame(rows)


def neighborhood(smiles: str) -> tuple[float, int, int, dict[str, float]]:
    """(secondes, voisins distincts, actions, ms par type de mutation) du voisinage complet, à froid."""
    rf.clear_cache()
    fast_rules.clear_cache()
    molecule = Molecule(smiles)
    t0 = time.perf_counter()
    neighbors = neighbor_actions(molecule)
    elapsed = time.perf_counter() - t0
    # temps par type (lister + appliquer), sur une copie fraîche de la molécule
    per_type = {}
    for action_class in MolecularGraph.action_space:
        rf.clear_cache()
        fast_rules.clear_cache()
        copy = Molecule(smiles)
        t1 = time.perf_counter()
        for action in action_class.list_actions(copy):
            action.apply_smiles()  # type: ignore[attr-defined]
        per_type[action_class.class_name()] = 1000 * (time.perf_counter() - t1)
    return elapsed, len(neighbors), molecule.nb_actions, per_type


def plot(table: pd.DataFrame, save: Path) -> None:
    """Temps du voisinage selon la taille, et rapport nouvelles / anciennes par molécule."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for label in CONDITIONS:
        left.scatter(table.n_atoms, table[f"ms_{label}"], s=14, alpha=0.6, color=COLORS[label],
                     edgecolors="none", label=label)
    left.set(xlabel="atomes lourds de la molécule", ylabel="ms pour le voisinage complet", yscale="log",
             title="Voisinage complet, mêmes molécules")
    left.legend(frameon=False)
    right.scatter(table.n_atoms, table.rapport, s=14, alpha=0.7, color="#52514e", edgecolors="none")
    right.axhline(1, color="#8c8c8c", lw=1)
    right.set(xlabel="atomes lourds de la molécule", ylabel="temps nouvelles / temps anciennes",
              title="Rapport par molécule")
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=150)
    plt.close(fig)


def main() -> None:
    """Compare les voisinages."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--molecules", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    table = pick_molecules(args.molecules, args.seed)
    per_type_rows = []
    for label, config in CONDITIONS.items():
        nw.configure(config)
        results = [neighborhood(smiles) for smiles in table.smiles]
        table[f"ms_{label}"] = [1000 * r[0] for r in results]
        table[f"voisins_{label}"] = [r[1] for r in results]
        table[f"actions_{label}"] = [r[2] for r in results]
        for smiles, r in zip(table.smiles, results):
            per_type_rows += [{"smiles": smiles, "condition": label, "mutation": k, "ms": v} for k, v in r[3].items()]
    table["rapport"] = table.ms_nouvelles / table.ms_anciennes
    table.to_csv(OUT / "molecules.csv", index=False)
    per_type = pd.DataFrame(per_type_rows)
    per_type.to_csv(OUT / "par_type.csv", index=False)

    print(f"{len(table)} molécules, {table.n_atoms.min()} à {table.n_atoms.max()} atomes "
          f"(médiane {table.n_atoms.median():.0f}) ; origine : {table.origine.value_counts().to_dict()}\n")
    summary = pd.DataFrame({
        label: {
            "ms par molécule (moyenne)": table[f"ms_{label}"].mean(),
            "ms par molécule (médiane)": table[f"ms_{label}"].median(),
            "voisins distincts (moyenne)": table[f"voisins_{label}"].mean(),
            "actions listées (moyenne)": table[f"actions_{label}"].mean(),
            "ms par action": table[f"ms_{label}"].sum() / table[f"actions_{label}"].sum(),
        } for label in CONDITIONS
    }).round(2)
    print(summary.to_string())
    quartiles = table.rapport.quantile([0.25, 0.5, 0.75]).round(1).tolist()
    print(f"\nrapport nouvelles / anciennes par molécule : médiane {quartiles[1]}, quartiles {quartiles[0]} – "
          f"{quartiles[2]}, min {table.rapport.min():.1f}, max {table.rapport.max():.1f}")
    print(f"temps total : anciennes {table.ms_anciennes.sum() / 1000:.1f} s, "
          f"nouvelles {table.ms_nouvelles.sum() / 1000:.1f} s (x{table.ms_nouvelles.sum() / table.ms_anciennes.sum():.1f})")

    bins = pd.cut(table.n_atoms, [0, 10, 20, 30, 60], labels=["≤ 10", "11-20", "21-30", "> 30"])
    by_size = table.groupby(bins, observed=True).agg(
        molécules=("smiles", "size"), ms_anciennes=("ms_anciennes", "mean"), ms_nouvelles=("ms_nouvelles", "mean"),
        rapport_médian=("rapport", "median")).round(1)
    print("\npar taille (atomes lourds) :")
    print(by_size.to_string())

    print("\npar type de mutation (ms par molécule, moyenne, lister + appliquer) :")
    print(per_type.pivot_table(index="mutation", columns="condition", values="ms", aggfunc="mean")
          .round(1).fillna("—").to_string())

    plot(table, OUT / "voisinage_memes_molecules.png")
    print(f"\nrésultats dans {OUT}")


if __name__ == "__main__":
    main()
