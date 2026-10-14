"""
Temps de chaque étape de chaque mutation, selon la taille de la molécule (notebooks/random_walk/newmut_time.ipynb).

Pour chaque molécule et chaque type de mutation, trois étapes chronométrées, à froid (mémoires vidées avant chaque
molécule) :
  1. lister   : construire les mutations candidates (list_actions pour les anciennes ; list_unchecked_actions pour
                les nouvelles, c.-à-d. avant les règles) ;
  2. vérifier : tester les règles R1-R5 sur chaque candidate (nouvelles seulement ; 0 pour les anciennes). Le calcul
                des règles de la molécule de départ (cycles, ponts, violations), fait une fois par molécule, est
                compté à part (« règles de départ ») ;
  3. appliquer : appliquer chaque mutation permise et écrire le SMILES canonique du résultat (apply_smiles), comme
                pour construire le voisinage d'une marche.
SubstituteShiftMG applique déjà les règles pendant qu'elle liste (elle choisit le déplacement des doubles liaisons
qui les respecte) : son temps de vérification est inclus dans « lister ».

Molécules : états visités par des chaînes de mutations au hasard depuis le benzène (anciennes et nouvelles mutations,
atomes C, O, N, au plus MAX_ATOMS = 100 atomes), choisies par classe de taille.
"""

from __future__ import annotations

import os
import random
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT, "scripts", "newmut"))

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

import newmut_walk as nw  # se place à la racine du projet
from compare_walk_time import COLORS, CONDITIONS
from evomol.newmut import fast_rules
from evomol.newmut.neighborhood_strategy import draw_allowed_action
from evomol.newmut import rule_filtered as rf
from evomol.representation import MolecularGraph, Molecule

SIZE_BINS = [0, 10, 20, 30, 50, 70, 100,]
# taille maximale des molécules (Molecule.max_heavy_atoms) pour construire et muter les molécules
MAX_ATOMS = SIZE_BINS[-1]
STEPS = ["lister", "vérifier", "appliquer"]
STEP_COLORS = {"lister": "#2a78d6", "vérifier": "#eb6834", "appliquer": "#1baf7a"}


def size_class(n_atoms: int) -> str:
    """Classe de taille, ex. « 6-10 »."""
    for low, high in zip(SIZE_BINS, SIZE_BINS[1:]):
        if n_atoms <= high:
            return f"{low + 1}-{high}"
    return f"> {SIZE_BINS[-1]}"


def configs(max_atoms: int = MAX_ATOMS) -> dict[str, dict[str, Any]]:
    """Les deux conditions (anciennes, nouvelles), avec la taille maximale max_atoms."""
    return {label: {**config, "MAX_ATOMS": max_atoms} for label, config in CONDITIONS.items()}


def pick_molecules(per_class: int = 8, seed: int = 0, max_steps: int = 1000, chains: int = 12,
                   max_atoms: int = MAX_ATOMS) -> pd.DataFrame:
    """per_class molécules distinctes par classe de taille, prises dans des chaînes de mutations au hasard (moitié
    anciennes, moitié nouvelles mutations quand c'est possible : les anciennes font rarement de grosses molécules).

    Chaque chaîne s'arrête à max_steps mutations ou quand la molécule atteint max_atoms. Côté nouvelles mutations, la
    mutation est tirée comme avec lazy_rules (même tirage, plus rapide sur les grosses molécules).
    """
    rng = random.Random(seed)
    pool: dict[str, dict[str, Any]] = {}
    bar = tqdm(total=len(CONDITIONS) * chains * max_steps, desc="chaînes de mutations", unit="mutation")
    for label, config in configs(max_atoms).items():
        nw.configure(config)
        for _ in range(chains):
            molecule = Molecule("c1ccccc1")
            for step in range(max_steps):
                action = draw_allowed_action(rng.choice(MolecularGraph.action_space), molecule) \
                    if label == "nouvelles" else None
                if label != "nouvelles":
                    actions = rng.choice(MolecularGraph.action_space).list_actions(molecule)
                    action = rng.choice(actions) if actions else None
                if action is not None:
                    molecule = action.apply()
                smiles = str(molecule)
                n_atoms = molecule.get_representation(MolecularGraph).nb_atoms
                if smiles and smiles not in pool:
                    pool[smiles] = {"smiles": smiles, "origine": label, "n_atoms": n_atoms}
                if n_atoms >= max_atoms:
                    bar.update(max_steps - step)
                    break
                bar.update(1)
    bar.close()
    table = pd.DataFrame(pool.values())
    table["taille"] = table.n_atoms.map(size_class)
    chosen = []
    for _, group in table.groupby("taille", sort=False):
        halves = [g.sample(n=min(len(g), per_class // 2), random_state=seed) for _, g in group.groupby("origine")]
        picked = pd.concat(halves)
        rest = group.drop(picked.index)
        if len(picked) < per_class and len(rest):
            picked = pd.concat([picked, rest.sample(n=min(len(rest), per_class - len(picked)), random_state=seed)])
        chosen.append(picked)
    order = [size_class(high) for high in SIZE_BINS[1:]]
    result = pd.concat(chosen)
    result["taille"] = pd.Categorical(result.taille, categories=order, ordered=True)
    return result.sort_values(["taille", "n_atoms"]).reset_index(drop=True)


def _clear() -> None:
    rf.clear_cache()
    fast_rules.clear_cache()


def time_molecule(smiles: str, label: str) -> list[dict[str, Any]]:
    """Temps des trois étapes de chaque type de mutation sur une molécule (espace d'action déjà configuré)."""
    _clear()
    molecule = Molecule(smiles)
    mol_graph = molecule.get_representation(MolecularGraph)
    rows = []
    t_parent = 0.0
    if label == "nouvelles":
        t0 = time.perf_counter()
        fast_rules.parent_rules(mol_graph)
        t_parent = time.perf_counter() - t0
    for action_class in MolecularGraph.action_space:
        checks = getattr(action_class, "checks_rules", False)
        t0 = time.perf_counter()
        candidates = action_class.list_unchecked_actions(molecule) if checks else action_class.list_actions(molecule)
        t1 = time.perf_counter()
        allowed = [a for a in candidates if action_class.action_is_allowed(a, molecule)] if checks else candidates
        t2 = time.perf_counter()
        for action in allowed:
            action.apply_smiles()  # type: ignore[attr-defined]
        t3 = time.perf_counter()
        rows.append({"mutation": action_class.class_name(), "condition": label, "candidates": len(candidates),
                     "permises": len(allowed), "lister": 1000 * (t1 - t0), "vérifier": 1000 * (t2 - t1),
                     "appliquer": 1000 * (t3 - t2)})
    if label == "nouvelles":
        rows.append({"mutation": "(règles de départ)", "condition": label, "candidates": 0, "permises": 0,
                     "lister": 0.0, "vérifier": 1000 * t_parent, "appliquer": 0.0})
    return rows


def measure(molecules: pd.DataFrame, max_atoms: int = MAX_ATOMS) -> pd.DataFrame:
    """Temps de chaque étape, pour chaque molécule, chaque type de mutation et chaque condition (ms)."""
    rows = []
    bar = tqdm(total=len(CONDITIONS) * len(molecules), desc="mesure", unit="molécule")
    for label, config in configs(max_atoms).items():
        nw.configure(config)
        for _, molecule in molecules.iterrows():
            bar.set_postfix_str(f"{label}, {molecule.n_atoms} atomes")
            for row in time_molecule(molecule.smiles, label):
                rows.append({**row, "smiles": molecule.smiles, "n_atoms": molecule.n_atoms,
                             "taille": molecule.taille})
            bar.update(1)
    bar.close()
    table = pd.DataFrame(rows)
    table["total"] = table[STEPS].sum(axis=1)
    table["taille"] = pd.Categorical(table.taille, categories=molecules.taille.cat.categories, ordered=True)
    return table


def step_table(times: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Une ligne par type de mutation, colonnes (taille, étape) : ms par molécule, moyenne."""
    data = times[times.condition == condition]
    table = data.pivot_table(index="mutation", columns="taille", values=STEPS + ["total"], aggfunc="mean",
                             observed=True)
    table = table.reorder_levels([1, 0], axis=1)
    sizes = [s for s in data.taille.cat.categories if s in data.taille.unique()]
    table = table[[(size, step) for size in sizes for step in [*STEPS, "total"]]]
    total = table.sum().to_frame("TOTAL").T
    return pd.concat([table, total]).round(2)


def long_list(times: pd.DataFrame) -> pd.DataFrame:
    """Liste : une ligne par (condition, mutation, taille), avec le nombre de candidates et de mutations permises."""
    return (times.groupby(["condition", "mutation", "taille"], observed=True)
            [["candidates", "permises", *STEPS, "total"]].mean().round(2).reset_index())


def summary(times: pd.DataFrame) -> pd.DataFrame:
    """Par condition et taille : ms par molécule de chaque étape, tous types confondus."""
    per_molecule = times.groupby(["condition", "taille", "smiles"], observed=True)[[*STEPS, "total"]].sum()
    table = per_molecule.groupby(["condition", "taille"], observed=True).mean().round(1)
    ratio = (table.loc["nouvelles", "total"] / table.loc["anciennes", "total"]).round(1)
    table = table.unstack(0)
    table[("rapport", "nouvelles / anciennes")] = ratio
    return table


def plot_steps(times: pd.DataFrame, save: str | None = None) -> plt.Figure:
    """Pour chaque condition : ms par molécule de chaque étape selon la taille (barres empilées)."""
    per_molecule = times.groupby(["condition", "taille", "smiles"], observed=True)[STEPS].sum()
    means = per_molecule.groupby(["condition", "taille"], observed=True).mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, label in zip(axes, CONDITIONS):
        data = means.loc[label]
        bottom = None
        for step in STEPS:
            ax.bar(data.index.astype(str), data[step], bottom=bottom, color=STEP_COLORS[step], label=step,
                   edgecolor="white", linewidth=1)
            bottom = data[step] if bottom is None else bottom + data[step]
        ax.set(title=f"Mutations {label}", xlabel="atomes lourds")
    axes[0].set_ylabel("ms par molécule (tous types)")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def plot_mutation(times: pd.DataFrame, mutation: str, save: str | None = None) -> plt.Figure:
    """Une mutation : ms de chaque étape selon la taille (une courbe par étape et par condition)."""
    data = times[times.mutation == mutation].groupby(["condition", "taille"], observed=True)[STEPS].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label in data.index.get_level_values(0).unique():
        for step in STEPS:
            values = data.loc[label][step]
            if values.max() == 0:
                continue
            ax.plot(values.index.astype(str), values, marker="o", lw=2, color=STEP_COLORS[step],
                    linestyle="-" if label == "nouvelles" else "--", label=f"{step} ({label})")
    ax.set(title=mutation, xlabel="atomes lourds", ylabel="ms par molécule")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


__all__ = ["COLORS", "CONDITIONS", "pick_molecules", "measure", "step_table", "long_list", "summary",
           "plot_steps", "plot_mutation"]
