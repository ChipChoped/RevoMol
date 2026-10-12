"""
Évolvabilité détaillée : toutes les variables (observables + λ HOMO-3 … LUMO+3) sur tous les voisins
des molécules échantillonnées le long des marches déjà faites (notebooks/random_walk/ancien/evolvabilite.ipynb).

Aucune marche n'est relancée. Les marches écrites depuis le 2026-10-07 ont déjà leurs voisinages
complets (rep<k>_voisins.parquet) : ils sont copiés, sans recalcul. Chaque marche a gardé sa configuration (runs/<config>/config.json) et les
NEIGHBORHOOD_SAMPLE molécules dont random_walk.analyse_neighborhoods a évalué le voisinage
(rep<k>_voisinages.csv.gz). Leurs voisinages sont régénérés à l'identique (même espace d'action, mêmes
filtres, même règle EXCLUDE_CURRENT / ONLY_VALID), mais on garde cette fois, pour chaque voisin valide,
son SMILES, sa mutation et la valeur de toutes les variables.

Analyses, sur Δ = valeur du voisin − valeur de la molécule de départ (parent) :
  - parts de voisins qui augmentent / laissent inchangée (|Δ| <= ε) / baissent chaque variable,
    par type de mutation et selon le contexte (taille du parent, valeur actuelle de la variable) ;
  - optima locaux de chaque variable (aucun voisin ne l'augmente / ne la baisse) ;
  - corrélations des Δ entre variables, compromis entre objectifs.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/Lea

import random_walk as rw  # se place à la racine du projet, configure les chemins d'evomol

# pylint: disable=wrong-import-position,wrong-import-order
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import random_walk_analysis as rwa
from evomol.evaluation import is_valid_molecule
from evomol.representation import Molecule
from random_walk_core import neighbor_actions

NEUTRAL_EPS = rwa.NEUTRAL_EPS
ORBITALS = rw.ORBITALS  # λ des orbitales ajoutées aux observables (HOMO, LUMO et λ HOMO-1 y sont déjà)
KEYS = ["plan", "config_id", "rep", "step"]  # identifie une molécule de départ (parent)


# ==============================================================================
#  RÉGÉNÉRATION DES VOISINAGES
# ==============================================================================
def load_config(path: str | Path) -> dict[str, Any]:
    """config.json d'une marche, avec sa fitness. FITNESS_GOALS est relue dans fitness.json (écrit à
    côté depuis le 2026-10-07, valeurs exactes) ; sinon (anciennes marches, fitness enregistrée en
    texte) elle est remplacée par celle de DEFAULT_CONFIG, après avoir vérifié que c'est bien la même."""
    config = json.loads(Path(path).read_text())
    fitness_path = Path(path).with_name("fitness.json")
    if fitness_path.exists():
        fitness = json.loads(fitness_path.read_text())
        goals = rw.goals_from_json(fitness["FITNESS_GOALS"])
        if config["FITNESS_GOALS"] != rw._jsonable(goals):  # pylint: disable=protected-access
            raise ValueError(f"{fitness_path} ne correspond pas à {path}")
        return rw.make_config(**{**config, "FITNESS_GOALS": goals})
    default_goals = rw.DEFAULT_CONFIG["FITNESS_GOALS"]
    if config["FITNESS_GOALS"] != rw._jsonable(default_goals):  # pylint: disable=protected-access
        raise ValueError(f"{path} : fitness différente de DEFAULT_CONFIG et pas de fitness.json, non reconstructible")
    return rw.make_config(**{**config, "FITNESS_GOALS": default_goals})


orbital_columns = rw.orbital_columns  # colonne -> orbitale (orbitales de ORBITALS hors observables)


def variable_columns(config: dict[str, Any]) -> list[str]:
    """Toutes les variables enregistrées sur chaque molécule, dans l'ordre d'affichage."""
    return [*rw.observable_columns(config), *orbital_columns(config)]


def neighborhood_values(config: dict[str, Any], parents: pd.DataFrame) -> pd.DataFrame:
    """Toutes les variables de chaque parent (lignes is_parent) et de chacun de ses voisins valides.

    Args:
        parents: colonnes step, smiles (rep<k>_voisinages.csv.gz).
    """
    filters, walk_kwargs = rw.configure(config)
    evaluations, _ = rw.build_evaluations(config)
    orbitals = orbital_columns(config)
    rows = []
    for step, smiles in zip(parents.step, parents.smiles):
        parent = Molecule(smiles)
        rows.append({"step": int(step), "parent_smiles": smiles, "smiles": smiles, "mutation": "",
                     "is_parent": True, **rw.molecule_values(parent, evaluations, orbitals)})
        neighbors = neighbor_actions(parent, walk_kwargs["action_types"])
        if config["EXCLUDE_CURRENT"]:
            neighbors.pop(smiles, None)
        for neighbor_smiles, action in neighbors.items():
            neighbor = Molecule(neighbor_smiles)
            if config["ONLY_VALID"] and not is_valid_molecule(neighbor, filters):
                continue
            rows.append({"step": int(step), "parent_smiles": smiles, "smiles": neighbor_smiles,
                         "mutation": action.class_name(), "is_parent": False,
                         **rw.molecule_values(neighbor, evaluations, orbitals)})
    return pd.DataFrame(rows)


def _run_one(task: tuple[str, str, int, str, str]) -> tuple[str, str, int, float, int]:
    plan, config_id, rep, run_dir, out_path = task
    t0 = time.perf_counter()
    written = Path(run_dir) / f"rep{rep}_voisins.parquet"  # voisinages déjà écrits par la marche
    if written.exists():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(written, out_path)
        values = pd.read_parquet(out_path, columns=["is_parent"])
        return plan, config_id, rep, time.perf_counter() - t0, int((~values.is_parent).sum())
    config = load_config(Path(run_dir) / "config.json")
    parents = pd.read_csv(Path(run_dir) / f"rep{rep}_voisinages.csv.gz")
    values = neighborhood_values(config, parents)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    values.to_parquet(out_path, index=False)  # écrit en dernier : sa présence marque la tâche comme faite
    return plan, config_id, rep, time.perf_counter() - t0, int((~values.is_parent).sum())


def tasks(plans: dict[str, str | Path], out_dir: str | Path) -> tuple[list[tuple], int]:
    """(tâches à faire, nb déjà faites) : une par marche terminée (plan, configuration, répétition)."""
    todo, n_done = [], 0
    for plan, plan_dir in plans.items():
        for meta in sorted(Path(plan_dir).resolve().glob("runs/*/rep*_meta.json")):
            run_dir, rep = meta.parent, int(meta.name[3:].split("_")[0])
            if not (run_dir / f"rep{rep}_voisinages.csv.gz").exists():
                continue
            out_path = Path(out_dir).resolve() / plan / run_dir.name / f"rep{rep}_voisins.parquet"
            if out_path.exists():
                n_done += 1
            else:
                todo.append((plan, run_dir.name, rep, str(run_dir), str(out_path)))
    return todo, n_done


def run_plans(plans: dict[str, str | Path], out_dir: str | Path, n_jobs: int | None = None,
              max_tasks: int | None = None) -> None:
    """Régénère les voisinages de toutes les marches des plans, en parallèle et reprenable.

    Args:
        plans: nom du plan -> dossier de résultats (contient runs/).
        out_dir: out_dir/<plan>/<config>/rep<k>_voisins.parquet.
        max_tasks: au plus ce nombre de marches (essai, estimation du temps).
    """
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    todo, n_done = tasks(plans, out_dir)
    print(f"{len(todo)} marches à faire, {n_done} déjà faites")
    todo = todo[:max_tasks] if max_tasks else todo
    if not todo:
        return
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    errors, t0 = [], time.perf_counter()
    pool = ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("spawn"))
    try:
        futures = {pool.submit(_run_one, task): task[:3] for task in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc="voisinages"):
            try:
                future.result()
            except Exception as exc:  # pylint: disable=broad-except
                errors.append((*futures[future], f"{type(exc).__name__}: {exc}"))
    except KeyboardInterrupt:
        print("Interruption : fin des marches en cours, les autres sont annulées (relancer pour reprendre)…")
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown()
    print(f"terminé en {(time.perf_counter() - t0) / 60:.1f} min")
    for plan, config_id, rep, message in errors:
        print(f"ÉCHEC {plan} {config_id} rep{rep} : {message}")
    if errors:
        raise RuntimeError(f"{len(errors)} marches ont échoué (relancer pour réessayer)")


# ==============================================================================
#  CHARGEMENT, Δ
# ==============================================================================
def load(out_dir: str | Path, plans: list[str] | None = None) -> pd.DataFrame:
    """Toutes les lignes (parents et voisins) des marches déjà traitées."""
    frames = []
    for path in sorted(Path(out_dir).glob("*/*/rep*_voisins.parquet")):
        plan, config_id = path.parent.parent.name, path.parent.name
        if plans is None or plan in plans:
            frames.append(pd.read_parquet(path).assign(plan=plan, config_id=config_id,
                                                        rep=int(path.name[3:].split("_")[0])))
    return pd.concat(frames, ignore_index=True)


def deltas(data: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Une ligne par voisin : KEYS, mutation, Δ de chaque variable (colonnes = noms des variables) et
    valeur au parent (colonnes « parent <variable> »)."""
    parents = data[data.is_parent].set_index(KEYS)[variables]
    neighbors = data[~data.is_parent]
    parent_values = parents.reindex(pd.MultiIndex.from_frame(neighbors[KEYS])).to_numpy()
    d = neighbors[variables].to_numpy(dtype=float) - parent_values
    return pd.concat([
        neighbors[[*KEYS, "mutation"]].reset_index(drop=True),
        pd.DataFrame(d, columns=variables),
        pd.DataFrame(parent_values, columns=[f"parent {v}" for v in variables]),
    ], axis=1)


# ==============================================================================
#  ANALYSES
# ==============================================================================
def _shares(d: np.ndarray, eps: float) -> dict[str, float]:
    d = d[np.isfinite(d)]
    if not len(d):
        return {"up_pct": np.nan, "neutral_pct": np.nan, "down_pct": np.nan, "abs_delta_mean": np.nan,
                "delta_min": np.nan, "delta_max": np.nan, "n": 0}
    return {"up_pct": 100 * (d > eps).mean(), "neutral_pct": 100 * (np.abs(d) <= eps).mean(),
            "down_pct": 100 * (d < -eps).mean(), "abs_delta_mean": np.abs(d).mean(),
            "delta_min": d.min(), "delta_max": d.max(), "n": len(d)}


def shares(dl: pd.DataFrame, variables: list[str], by: str | list[str] | None = None,
           eps: float = NEUTRAL_EPS) -> pd.DataFrame:
    """Par variable (et par groupe `by`) : % de voisins qui l'augmentent / neutres / la baissent,
    |Δ| moyen, Δ min / max, nb de voisins où elle est définie."""
    groups = [((), dl)] if by is None else dl.groupby(by, observed=True)
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        labels = dict(zip([by] if isinstance(by, str) else (by or []), key))
        rows += [{"variable": v, **labels, **_shares(g[v].to_numpy(float), eps)} for v in variables]
    return pd.DataFrame(rows)


def local_optima(dl: pd.DataFrame, variables: list[str], by: str | list[str] | None = None,
                 eps: float = NEUTRAL_EPS) -> pd.DataFrame:
    """Par variable : % de parents sans voisin qui l'augmente (maximum local) / la baisse (minimum local)."""
    d, keys = dl[variables], [dl[k] for k in KEYS]
    up = (d > eps).groupby(keys).any().astype(float)
    down = (d < -eps).groupby(keys).any().astype(float)
    defined = d.notna().groupby(keys).any()
    up, down = up.where(defined), down.where(defined)
    if by is None:
        return pd.DataFrame({"max_local_pct": 100 * (1 - up.mean()), "min_local_pct": 100 * (1 - down.mean())})
    level = [by] if isinstance(by, str) else by
    return pd.concat({"max_local_pct": 100 * (1 - up.groupby(level=level).mean()),
                      "min_local_pct": 100 * (1 - down.groupby(level=level).mean())}, axis=1)


def context_shares(dl: pd.DataFrame, variables: list[str], context: str | None = None, n_bins: int = 5,
                   eps: float = NEUTRAL_EPS) -> pd.DataFrame:
    """Parts de voisins qui augmentent / baissent chaque variable selon le contexte du parent :
    context = colonne du parent (ex. « parent n_atoms »), en n_bins classes de même effectif ;
    None : la valeur de la variable elle-même au parent (« plus dur de progresser quand on est haut ? »)."""
    rows = []
    for v in variables:
        column = context or f"parent {v}"
        x = dl[column]
        ok = x.notna() & dl[v].notna()
        if ok.sum() == 0 or x[ok].nunique() < 2:
            continue
        bins = pd.qcut(x[ok], n_bins, duplicates="drop")
        for interval, g in dl[ok].groupby(bins, observed=True):
            rows.append({"variable": v, "context": column, "bin": interval, "x": g[column].median(),
                         **_shares(g[v].to_numpy(float), eps)})
    return pd.DataFrame(rows)


def delta_correlations(dl: pd.DataFrame, variables: list[str], method: str = "spearman") -> pd.DataFrame:
    """Corrélation des Δ entre variables sur les voisins (paires de variables définies)."""
    return dl[variables].corr(method=method)


def pair_correlations(dl: pd.DataFrame, variables: list[str], by: str, method: str = "spearman") -> pd.DataFrame:
    """Corrélation des Δ de chaque paire de variables, par groupe `by` (lignes : paires, colonnes : groupes)."""
    return pd.DataFrame({
        key: {f"{a} / {b}": g[a].corr(g[b], method=method) for a, b in itertools.combinations(variables, 2)}
        for key, g in dl.groupby(by, observed=True)
    })


def tradeoffs(dl: pd.DataFrame, objectives: dict[str, int], by: str | None = None,
              eps: float = NEUTRAL_EPS) -> pd.DataFrame:
    """Pour chaque paire d'objectifs (variable -> +1 à augmenter, −1 à baisser) : % de voisins qui
    améliorent les deux, un seul (et dégradent l'autre), dégradent les deux, ou laissent l'un inchangé."""
    rows = []
    groups = [(None, dl)] if by is None else dl.groupby(by, observed=True)
    for key, g in groups:
        for a, b in itertools.combinations(objectives, 2):
            da, db = objectives[a] * g[a].to_numpy(float), objectives[b] * g[b].to_numpy(float)
            ok = np.isfinite(da) & np.isfinite(db)
            da, db = da[ok], db[ok]
            if not len(da):
                continue
            row = {"pair": f"{a} / {b}", "a": a, "b": b, "n": len(da),
                   "both_better_pct": 100 * ((da > eps) & (db > eps)).mean(),
                   "a_better_b_worse_pct": 100 * ((da > eps) & (db < -eps)).mean(),
                   "a_worse_b_better_pct": 100 * ((da < -eps) & (db > eps)).mean(),
                   "both_worse_pct": 100 * ((da < -eps) & (db < -eps)).mean()}
            row["one_neutral_pct"] = 100 - sum(v for k, v in row.items() if k.endswith("_pct"))
            rows.append(row if by is None else {by: key, **row})
    return pd.DataFrame(rows)


# ==============================================================================
#  FIGURES
# ==============================================================================
def plot_matrix(table: pd.DataFrame, title: str = "", center: float | None = 0, fmt: str = "{:.0f}",
                vmax: float | None = None, label: str = "", save=None) -> plt.Figure:
    """Heatmap annotée (lignes × colonnes). center : échelle divergente autour de cette valeur
    (bleu au-dessus, orange en dessous) ; None : échelle séquentielle."""
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.75 * table.shape[1] + 3.5, 0.34 * table.shape[0] + 1.8))
    if center is None:
        im = ax.imshow(values, aspect="auto", cmap="Blues", vmin=np.nanmin(values), vmax=vmax or np.nanmax(values))
    else:
        span = vmax or np.nanmax(np.abs(values - center))
        im = ax.imshow(values, aspect="auto", cmap=rwa.DIVERGING, vmin=center - span, vmax=center + span)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    for i, j in itertools.product(range(table.shape[0]), range(table.shape[1])):
        if np.isfinite(values[i, j]):
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=7, color="#111111")
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=label)
    fig.tight_layout()
    return rwa._save(fig, save)  # pylint: disable=protected-access


def cluster_order(corr: pd.DataFrame) -> list[str]:
    """Ordre des variables qui regroupe les plus corrélées (classification hiérarchique sur 1 − |ρ|)."""
    c = corr.dropna(how="all").dropna(axis=1, how="all")
    distance = (1 - c.abs()).clip(lower=0).fillna(1).to_numpy()
    np.fill_diagonal(distance, 0)
    return list(c.index[leaves_list(linkage(squareform(distance, checks=False), "average"))])


def plot_context(context: pd.DataFrame, variables: list[str], xlabel: str = "", save=None) -> plt.Figure:
    """Pour chaque variable : % de voisins qui l'augmentent (bleu) / la baissent (orange) selon le contexte."""
    variables = [v for v in variables if v in set(context.variable)]
    fig, axes = rwa._grid(len(variables), ncols=4, width=3.6, height=2.6)  # pylint: disable=protected-access
    for ax, v in zip(axes.flat, variables):
        c = context[context.variable == v]
        ax.plot(c.x, c.up_pct, "o-", color=rwa.BLUE, label="augmente")
        ax.plot(c.x, c.down_pct, "o-", color=rwa.ORANGE, label="baisse")
        ax.set_title(v, fontsize=9)
        ax.set_xlabel(xlabel or f"{v} du parent", fontsize=8)
        ax.set_ylabel("% voisins", fontsize=8)
        ax.set_ylim(0, 100)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    return rwa._save(fig, save)  # pylint: disable=protected-access


def plot_tradeoffs(table: pd.DataFrame, save=None) -> plt.Figure:
    """Barres empilées par paire d'objectifs : améliorent les deux / un seul / dégradent les deux / neutre."""
    parts = {"both_better_pct": ("les deux mieux", rwa.BLUE), "a_better_b_worse_pct": ("1er mieux, 2e moins bien", "#9ec5ee"),
             "a_worse_b_better_pct": ("1er moins bien, 2e mieux", "#f5b394"), "both_worse_pct": ("les deux moins bien", rwa.ORANGE),
             "one_neutral_pct": ("au moins un inchangé", "#d9d9d9")}
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(table) + 1.4))
    left = np.zeros(len(table))
    for column, (label, color) in parts.items():
        ax.barh(table.pair, table[column], left=left, color=color, label=label)
        left += table[column].to_numpy()
    ax.invert_yaxis()
    ax.set_xlabel("% des voisins")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / max(len(table), 1)))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return rwa._save(fig, save)  # pylint: disable=protected-access


def plot_delta_pairs(dl: pd.DataFrame, pairs: list[tuple[str, str]], max_points: int = 200_000,
                     save=None) -> plt.Figure:
    """Densité de (Δ x, Δ y) sur les voisins, une case par paire (hexbin, échelle log)."""
    fig, axes = rwa._grid(len(pairs), ncols=3, width=3.8, height=3.2)  # pylint: disable=protected-access
    sample = dl.sample(min(max_points, len(dl)), random_state=0)
    for ax, (x, y) in zip(axes.flat, pairs):
        s = sample[[x, y]].dropna()
        ax.hexbin(s[x], s[y], gridsize=50, bins="log", cmap="Blues", mincnt=1)
        ax.axhline(0, color=rwa.GRAY, lw=0.6)
        ax.axvline(0, color=rwa.GRAY, lw=0.6)
        ax.set_xlabel(f"Δ {x}", fontsize=8)
        ax.set_ylabel(f"Δ {y}", fontsize=8)
        ax.set_title(f"ρ = {s[x].corr(s[y], method='spearman'):.2f}", fontsize=9)
    fig.tight_layout()
    return rwa._save(fig, save)  # pylint: disable=protected-access
