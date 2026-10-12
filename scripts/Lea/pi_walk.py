"""
Marches aléatoires « ligne de base » (notebooks/random_walk/pi_walk.ipynb) : evomol peut tout faire
(toutes les mutations, aucun filtre), et on suit pas à pas le système π et Hückel de chaque molécule.

À chaque pas :
  - n_atoms : nb d'atomes lourds ;
  - PI_CHAIN_SIZE, PCT_PI_ATOMS : taille du plus grand système π connexe et % des atomes lourds qui
    sont π (evomol/evaluation/pi_systems.py : comme _pi_components de gap_pi_relation.ipynb, atome π =
    aromatique ou hybridé SP2 / SP, composantes connexes, plus la règle du soufre) ;
  - HOMO, LUMO, gap (λ) : calcul de Hückel d'evomol (NaN s'il échoue).

Plusieurs marches tournent en parallèle (run_walks) ; une barre de chargement suit les pas.
Fichiers d'une expérience (out_dir, relatif à la racine du projet) : config.json, puis par marche k
walk<k>.csv.gz (une ligne par pas) et walk<k>_meta.json (écrit en dernier : marche terminée).
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
import os
import queue as queue_module
import random
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts", "Lea"))  # huckel_filters, random_walk_core
os.chdir(ROOT)  # les filtres ChEMBL/ZINC lisent external_data/ en relatif

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

import huckel_filters as hf
from evomol import default_parameters as dp
from evomol.action import molecular_graph as mg
from evomol.evaluation import Evaluation, EvaluationError, Huckel
from evomol.evaluation.pi_systems import pi_systems  # PI_CHAIN_SIZE, PCT_PI_ATOMS
from evomol.representation import MolecularGraph, Molecule
from random_walk_core import random_walk

RDLogger.DisableLog("rdApp.*")

MAX_LAG = 100  # plus grand décalage k de ρ(k)
BLUE, GRAY = "#2a78d6", "#8c8c8c"
STYLE = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e5", "axes.axisbelow": True, "font.size": 10,
}
plt.rcParams.update(STYLE)

DEFAULT_CONFIG: dict[str, Any] = {
    # ── départ et marche
    "START": "c1ccccc1",          # SMILES d'où part le burn-in
    "SEED": 0,                    # marche k : burn-in SEED + 2k, marche SEED + 2k + 1 ;
                                  # None : graines tirées au hasard (enregistrées dans walk<k>_meta.json)
    "BURN_IN_STEPS": 100,         # pas pour obtenir un départ aléatoire (0 : la marche part de START)
    "N_STEPS": 500,               # pas de la marche observée
    "MAX_ATOMS": 38,              # nb max d'atomes lourds (défaut d'evomol)
    "ACCEPTED_ATOMS": ["C", "O", "N", "F", "S"],  # défaut d'evomol
    "UNIQUE_KEKULE": False,       # un seul SMILES par molécule aromatique (plus lent)
    "ONLY_VALID": True,           # n'avancer que vers des voisins qui passent les filtres
    "COUNT_VALID_NEIGHBORS": False,  # valider tout le voisinage à chaque pas
    "EXCLUDE_CURRENT": True,      # la molécule courante n'est jamais son propre voisin
    "ACTION_TYPES": None,         # None : toutes les mutations de l'espace d'action
    "UNIFORM_ACTION_TYPE": True,  # tirer d'abord le type de mutation (uniforme parmi les types autorisés
                                  # qui ont un voisin), puis un voisin de ce type ; False : voisin uniforme
    # ── espace d'action : tout ce que sait faire evomol
    "INSERT_CARBON": True,        # InsertCarbonMG
    "AVOID_SP3_ON_SP3": False,    # AddAtomMG ne greffe pas de C sp3 sur un C sp3
    "ADD_ATOM_MULTIPLE_BONDS": True,  # AddAtomMG peut greffer par double / triple liaison
    "AVOID_BOND_BREAKING": False,  # ChangeBondMG ne supprime jamais de liaison
    "AVOID_BOND_FORMING": False,  # ChangeBondMG ne crée jamais de liaison
    "BOND_CHANGE_BY_ONE": False,  # ChangeBondMG change l'ordre d'une liaison d'un cran à la fois (0→1, 1→2…) ;
                                  # False : comme l'EvoMol d'origine, n'importe quel ordre (0→2, 1→3, 3→0…)
    "WITH_ADD_GROUP": True,       # AddGroupMG
    "WITH_REMOVE_GROUP": True,    # RemoveGroupMG
    # ── filtres (ordre : Hückel → sp3 → ChEMBL/ZINC → Hetero) : tous désactivés
    "FILTER_HUCKEL": False,       # tout C sp3 doit être en bout de chaîne
    "MAX_SP3": None,              # None : inactif ; entier : nb max d'atomes sans liaison π
    "FILTER_REALISM": None,       # "chembl_zinc", "chembl" ou None
    "FILTER_HETERO": False,       # chaînes d'hétéroatomes
    "MAX_RING_N": 3,              # nb max d'azotes en cycle (si FILTER_HETERO)
    # ── comptage des atomes π
    "SULFUR_PI_RULE": True,       # True : règle du soufre (pi_systems.apply_sulfur_rule) ; False : comme gap_pi_relation
}

# variables suivies à chaque pas, dans l'ordre d'affichage
VARIABLES = ["n_atoms", "PI_CHAIN_SIZE", "PCT_PI_ATOMS", "HOMO", "LUMO", "gap"]
UNITS = {"n_atoms": "atomes lourds", "PI_CHAIN_SIZE": "atomes", "PCT_PI_ATOMS": "% des atomes",
         "HOMO": "λ", "LUMO": "λ", "gap": "λ"}
FAILED = -2.0  # valeur tracée quand le calcul a échoué (NaN dans les fichiers)


def make_config(**overrides: Any) -> dict[str, Any]:
    """DEFAULT_CONFIG modifiée par overrides (une clé inconnue est une erreur)."""
    unknown = set(overrides) - set(DEFAULT_CONFIG)
    if unknown:
        raise KeyError(f"Réglages inconnus : {sorted(unknown)}")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(copy.deepcopy(overrides))
    return config


_FILTERS: dict[tuple, list[Evaluation]] = {}


def configure(config: dict[str, Any]) -> tuple[list[Evaluation], dict[str, Any]]:
    """Configure evomol pour cette configuration ; renvoie (filtres, options de random_walk).
    L'espace d'action est global au processus : à rappeler avant chaque marche."""
    dp.setup_default_parameters(accepted_atoms=list(config["ACCEPTED_ATOMS"]), max_heavy_atoms=config["MAX_ATOMS"],
                                unique_kekule=config["UNIQUE_KEKULE"])
    dp.setup_default_action_space(
        insert_carbon=config["INSERT_CARBON"], avoid_sp3_on_sp3=config["AVOID_SP3_ON_SP3"],
        avoid_bond_breaking=config["AVOID_BOND_BREAKING"], avoid_bond_forming=config["AVOID_BOND_FORMING"],
        with_add_group=config["WITH_ADD_GROUP"], with_remove_group=config["WITH_REMOVE_GROUP"],
        add_atom_multiple_bonds=config["ADD_ATOM_MULTIPLE_BONDS"],
        bond_change_by_one=config.get("BOND_CHANGE_BY_ONE", False),
    )
    key = (config["FILTER_HUCKEL"], config["MAX_SP3"], config["FILTER_REALISM"],
           config["FILTER_HETERO"], config["MAX_RING_N"])
    if key not in _FILTERS:  # les filtres ChEMBL/ZINC chargent des fichiers : une fois par processus
        _FILTERS[key] = hf.setup_huckel_filters(
            huckel=config["FILTER_HUCKEL"], max_sp3=config["MAX_SP3"], realism=config["FILTER_REALISM"],
            hetero=config["FILTER_HETERO"], max_ring_n=config["MAX_RING_N"],
        )
    walk_kwargs = {
        "only_valid": config["ONLY_VALID"],
        "count_valid_neighbors": config["COUNT_VALID_NEIGHBORS"],
        "exclude_current": config["EXCLUDE_CURRENT"],
        "action_types": [getattr(mg, a) for a in config["ACTION_TYPES"]] if config["ACTION_TYPES"] else None,
    }
    return _FILTERS[key], walk_kwargs


# ==============================================================================
#  MESURES SUR UNE MOLÉCULE
# ==============================================================================
def huckel_values(molecule: Molecule) -> dict[str, float]:
    """HOMO, LUMO, gap (λ) ; NaN si Hückel échoue ou si la valeur n'existe pas."""
    try:
        result = Huckel().evaluate(molecule)
    except (EvaluationError, KeyError, ValueError, RuntimeError, np.linalg.LinAlgError):
        return {"HOMO": math.nan, "LUMO": math.nan, "gap": math.nan}

    def number(value: float | None) -> float:
        return math.nan if value is None else value

    return {"HOMO": number(result.homo_lambda), "LUMO": number(result.lumo_lambda),
            "gap": number(result.gap_lambda)}


def molecule_values(smiles: str, sulfur_rule: bool = True) -> dict[str, float]:
    molecule = Molecule(smiles)
    return {"n_atoms": molecule.get_representation(MolecularGraph).nb_atoms,
            **pi_systems(smiles, sulfur_rule), **huckel_values(molecule)}


# ==============================================================================
#  FIGURES (chaque fonction renvoie la figure ; save=chemin l'enregistre en PNG)
# ==============================================================================
def _save(fig: plt.Figure, save: str | Path | None) -> plt.Figure:
    """Enregistre la figure ; dans <expérience>/figures/, le nom de l'expérience est mis en en-tête."""
    if save is not None and Path(save).parent.name == "figures":
        current = fig._suptitle.get_text() if fig._suptitle is not None else ""  # pylint: disable=protected-access
        fig.suptitle(f"{Path(save).parent.parent.name}\n{current}".strip(), x=0.01, y=1.0, ha="left",
                     va="bottom", fontsize=11)
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=120, bbox_inches="tight")
    return fig


def _grid(n: int, ncols: int = 3, width: float = 4.0, height: float = 2.8, **kwargs):
    ncols = min(n, ncols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False, **kwargs)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    return fig, axes


# ==============================================================================
#  MARCHES
# ==============================================================================
def run_walk(config: dict[str, Any], k: int, progress: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Marche k : burn-in (graine SEED + 2k), marche observée (graine SEED + 2k + 1), valeurs de chaque
    pas ; SEED = None : base tirée au hasard pour chaque marche.
    progress : file où envoyer 1 à chaque pas (barre de chargement), ou None."""
    t0 = time.perf_counter()
    filters, walk_kwargs = configure(config)
    walk_kwargs["uniform_action_type"] = config["UNIFORM_ACTION_TYPE"]

    def on_step(row: dict[str, Any]) -> None:
        if progress is not None and row["step"] > 0:
            progress.put(1)

    base = config["SEED"] if config["SEED"] is not None else random.SystemRandom().randrange(10**9)
    seed = base + 2 * k
    if config["BURN_IN_STEPS"]:
        start = random_walk(config["START"], config["BURN_IN_STEPS"], filters, seed=seed,
                            on_step=on_step, **walk_kwargs)[-1]["smiles"]
    else:
        start = Molecule(config["START"]).get_representation(MolecularGraph).canonical_smiles
    t1 = time.perf_counter()
    rows = random_walk(start, config["N_STEPS"], filters, seed=seed + 1, on_step=on_step, **walk_kwargs)
    t2 = time.perf_counter()
    walk = pd.DataFrame(rows)[["step", "smiles", "action", "valid", "neighborhood_size"]]
    by_smiles = {s: molecule_values(s, config.get("SULFUR_PI_RULE", True)) for s in walk.smiles.unique()}  # une fois par molécule distincte
    for column in VARIABLES:
        walk[column] = [by_smiles[s][column] for s in walk.smiles]
    t3 = time.perf_counter()
    meta = {"walk": k, "seed_burn_in": seed, "seed_walk": seed + 1, "walk_start": start,
            "filters": [f.name for f in filters],
            "actions": [a.__name__ for a in walk_kwargs["action_types"] or MolecularGraph.action_space],
            "t_burn_in": t1 - t0, "t_walk": t2 - t1, "t_values": t3 - t2, "t_total": t3 - t0}
    return walk, meta


def _run_one(task: tuple[dict[str, Any], int, str, Any]) -> int:
    config, k, out_dir, progress = task
    walk, meta = run_walk(config, k, progress)
    out = Path(out_dir)
    walk.to_csv(out / f"walk{k}.csv.gz", index=False)
    # écrit en dernier : sa présence marque la marche comme terminée
    (out / f"walk{k}_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return k


def run_walks(config: dict[str, Any], n_walks: int, out_dir: str | Path, n_jobs: int | None = None) -> None:
    """n_walks marches (graines 0 … n_walks − 1) en parallèle, avec une barre de chargement sur
    l'ensemble des pas (burn-in compris). Les marches et figures déjà présentes dans out_dir sont
    supprimées : chaque lancement les écrase. out_dir est relatif à la racine du projet.
    Interruption (Ctrl-C, bouton stop) : les marches en cours se terminent et sont écrites."""
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    old = [*out_dir.glob("walk*.csv.gz"), *out_dir.glob("walk*_meta.json"), *out_dir.glob("figures/*.png"),
           *out_dir.glob("decompte.csv")]
    for path in old:
        path.unlink()
    if old:
        print(f"{out_dir} : marches précédentes écrasées")
    (out_dir / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False))

    todo = list(range(n_walks))
    n_jobs = min(len(todo), n_jobs or max(1, (os.cpu_count() or 2) - 1))
    errors = []
    t0 = time.perf_counter()
    with multiprocessing.get_context("spawn").Manager() as manager:
        progress = manager.Queue()
        # spawn : chaque processus importe ce module et configure evomol lui-même
        pool = ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("spawn"))
        bar = tqdm(total=len(todo) * (config["BURN_IN_STEPS"] + config["N_STEPS"]), desc="pas")
        try:
            futures = {pool.submit(_run_one, (config, k, str(out_dir), progress)): k for k in todo}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.3, return_when=FIRST_COMPLETED)
                n = 0
                while True:
                    try:
                        n += progress.get_nowait()
                    except queue_module.Empty:
                        break
                bar.update(n)
                for future in done:
                    try:
                        future.result()
                    except Exception as exc:  # pylint: disable=broad-except
                        errors.append((futures[future], f"{type(exc).__name__}: {exc}"))
        except KeyboardInterrupt:
            print("Interruption : fin des marches en cours, les autres sont annulées…")
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            bar.close()
        pool.shutdown()
    print(f"terminé en {(time.perf_counter() - t0) / 60:.1f} min")
    write_counts(out_dir)  # décompte des atomes et des mutations des marches terminées
    for k, message in errors:
        print(f"ÉCHEC marche {k} : {message}")
    if errors:
        raise RuntimeError(f"{len(errors)} marches ont échoué")


def load_walks(out_dir: str | Path) -> dict[int, pd.DataFrame]:
    """Marches terminées d'une expérience : numéro -> DataFrame (une ligne par pas)."""
    out_dir = Path(out_dir)
    ks = sorted(int(p.name[4:-10]) for p in out_dir.glob("walk*_meta.json"))
    walks = {k: pd.read_csv(out_dir / f"walk{k}.csv.gz", converters={"smiles": str}) for k in ks}
    for walk in walks.values():  # marches écrites avant PCT_PI_ATOMS (nb d'atomes π dans TOTAL_PI_ATOMS)
        if "PCT_PI_ATOMS" not in walk and "TOTAL_PI_ATOMS" in walk:
            walk["PCT_PI_ATOMS"] = 100 * walk.TOTAL_PI_ATOMS / walk.n_atoms.where(walk.n_atoms > 0)
    return walks


def load_meta(out_dir: str | Path) -> dict[int, dict[str, Any]]:
    out_dir = Path(out_dir)
    return {int(p.name[4:-10]): json.loads(p.read_text()) for p in sorted(out_dir.glob("walk*_meta.json"))}


def mutation_counts(walks: dict[int, pd.DataFrame], actions: list[str] | None = None) -> pd.DataFrame:
    """Nombre de fois où chaque mutation a été faite (pas de la marche observée, burn-in exclu) :
    une colonne par marche, puis le total et sa part en %. actions : mutations disponibles, affichées
    même si elles n'ont jamais été faites ; « restart » : retour au départ faute de voisin."""
    counts = pd.DataFrame({f"marche {k}": w.action.iloc[1:].value_counts() for k, w in walks.items()})
    order = [*(actions or []), *sorted(set(counts.index) - set(actions or []))]
    counts = counts.reindex(order).fillna(0).astype(int)
    counts["total"] = counts.sum(axis=1)
    counts["%"] = (100 * counts["total"] / counts["total"].sum()).round(1)
    return counts.sort_values("total", ascending=False)


ELEMENTS = ["C", "O", "N", "S", "F"]  # toujours dans le décompte (H en dernier, autres éléments s'il y en a)


def atom_counts(smiles: str) -> dict[str, int]:
    """Nombre d'atomes de chaque élément d'une molécule, H compris (implicites et explicites)."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if smiles and mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
    counts: dict[str, int] = {}
    for atom in (mol.GetAtoms() if mol is not None else []):
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        counts["H"] = counts.get("H", 0) + atom.GetTotalNumHs()
    return counts


def count_table(out_dir: str | Path) -> pd.DataFrame:
    """Décompte d'une expérience : pour chaque marche, total de chaque élément (C, O, N, S, F, H…) sur
    toutes les molécules visitées (pas 0 compris, burn-in exclu) et part de chaque mutation en % des
    pas de la marche ; puis la moyenne entre marches. Colonnes : decompte (atome / mutation), nom,
    marche 0 … marche k, moyenne."""
    walks, meta = load_walks(out_dir), load_meta(out_dir)
    if not walks:
        raise ValueError(f"{out_dir} : aucune marche terminée")
    atoms = {}
    for k, walk in walks.items():
        by_smiles = {s: atom_counts(s) for s in walk.smiles.unique()}
        total: dict[str, int] = {}
        for s in walk.smiles:
            for element, n in by_smiles[s].items():
                total[element] = total.get(element, 0) + n
        atoms[f"marche {k}"] = total
    atoms = pd.DataFrame(atoms)
    others = sorted(set(atoms.index) - set(ELEMENTS) - {"H"})
    atoms = atoms.reindex([*ELEMENTS, *others, "H"]).fillna(0).astype(int)
    actions = next(iter(meta.values()))["actions"] if meta else None
    mutations = mutation_counts(walks, actions).drop(columns=["total", "%"])
    mutations = 100 * mutations / mutations.sum()  # % des pas de chaque marche
    table = pd.concat([atoms.assign(decompte="atome"), mutations.assign(decompte="mutation")])
    walk_columns = [c for c in table.columns if c.startswith("marche ")]
    table["moyenne"] = table[walk_columns].mean(axis=1)
    table[[*walk_columns, "moyenne"]] = table[[*walk_columns, "moyenne"]].round(2)
    table = table.rename_axis("nom").reset_index()
    table["decompte"] = table.decompte.replace({"mutation": "mutation (%)"})
    is_atom = table.decompte == "atome"
    table[walk_columns] = table[walk_columns].astype(object)
    table.loc[is_atom, walk_columns] = table.loc[is_atom, walk_columns].astype(int)  # totaux d'atomes : entiers
    return table[["decompte", "nom", *atoms.columns, "moyenne"]]


def plot_counts(table: pd.DataFrame, save=None) -> plt.Figure:
    """Histogramme des moyennes entre marches de count_table, en % : part de chaque élément parmi tous
    les atomes de la marche, H compris (à gauche ; le CSV garde les nombres) et part de chaque mutation
    parmi les pas (à droite)."""
    walk_columns = [c for c in table.columns if c.startswith("marche ")]
    atoms = table[table.decompte == "atome"].set_index("nom")[walk_columns].astype(float)
    atom_pct = (100 * atoms / atoms.sum()).mean(axis=1)  # % de chaque marche, puis moyenne entre marches
    mutations = table[table.decompte == "mutation (%)"].set_index("nom")["moyenne"]
    parts = [(atom_pct, "Atomes : part moyenne de chaque élément (H compris)", "% des atomes"),
             (mutations, "Mutations : part moyenne des pas", "% des pas")]
    widths = [max(1, len(values)) for values, *_ in parts]
    fig, axes = plt.subplots(1, 2, figsize=(4 + 0.9 * sum(widths), 4), gridspec_kw={"width_ratios": widths})
    for ax, (values, title, unit) in zip(axes, parts):
        labels = [n.removesuffix("MG") for n in values.index]
        bars = ax.bar(labels, values, width=0.6, color=BLUE)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=2, fontsize=8, color="#333333")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    axes[1].tick_params(axis="x", labelrotation=30)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    return _save(fig, save)


def write_counts(out_dir: str | Path) -> pd.DataFrame:
    """Écrit le décompte (count_table) dans out_dir/decompte.csv et le renvoie."""
    table = count_table(out_dir)
    table.to_csv(Path(out_dir) / "decompte.csv", index=False)
    return table


# ==============================================================================
#  DÉTAIL DES MUTATIONS (atome ajouté / retiré, saut d'ordre de liaison)
# ==============================================================================
DETAILED_ACTIONS = ["AddAtomMG", "RemoveAtomMG", "ChangeBondMG"]
BOND_SYMBOLS = {1: "", 2: "=", 3: "#"}
ELEMENT_COLORS = {"C": "#2a78d6", "O": "#eb6834", "N": "#1baf7a", "S": "#eda100", "F": "#e87ba4"}
BOND_UP, BOND_DOWN = "#4a3aa7", "#e34948"  # ChangeBond : ordre qui monte / qui descend


def _details_one(task: tuple[dict[str, Any], int, str]) -> pd.DataFrame:
    """Détail des mutations d'une marche (voir mutation_details)."""
    config, k, out_dir = task
    configure(config)  # même espace d'action que pendant la marche
    walk = pd.read_csv(Path(out_dir) / f"walk{k}.csv.gz", converters={"smiles": str})
    rows = []
    for t in range(1, len(walk)):
        name = walk.action.iloc[t]
        if name not in DETAILED_ACTIONS:
            continue
        parent, child = Molecule(walk.smiles.iloc[t - 1]), walk.smiles.iloc[t]
        graph = parent.get_representation(MolecularGraph)
        # la marche suit la 1re action du type qui donne ce SMILES (random_walk_core.neighbor_actions_by_type)
        action = None
        for candidate in getattr(mg, name).list_actions(parent):
            try:
                if candidate.apply_smiles() == child:
                    action = candidate
                    break
            except Exception:  # pylint: disable=broad-except
                continue
        row = {"walk": k, "step": t, "action": name, "detail": None, "element": None,
               "order_from": None, "order_to": None}
        if action is None:
            row["detail"] = "non retrouvée"
        elif name == "AddAtomMG":
            row.update(element=action.atom_type, order_from=0, order_to=action.bond_order,
                       detail=f"+{BOND_SYMBOLS[action.bond_order]}{action.atom_type}")
        elif name == "RemoveAtomMG":
            element = graph.atom_type(action.atom_idx)
            row.update(element=element, detail=f"−{element}")
        else:
            old = graph.bond_order(action.atom1, action.atom2)
            row.update(element="".join(sorted([graph.atom_type(action.atom1), graph.atom_type(action.atom2)])),
                       order_from=old, order_to=action.bond_type, detail=f"{old}→{action.bond_type}")
        rows.append(row)
    return pd.DataFrame(rows, columns=["walk", "step", "action", "detail", "element", "order_from", "order_to"])


def mutation_details(out_dir: str | Path, n_jobs: int | None = None) -> pd.DataFrame:
    """Retrouve, sans relancer les marches, le détail de chaque AddAtomMG (élément et ordre de la
    liaison créée), RemoveAtomMG (élément retiré) et ChangeBondMG (ordre avant → après, éléments de la
    liaison) : à chaque pas, l'action du type enregistré qui mène de la molécule précédente à la
    suivante. Écrit out_dir/mutations_detail.csv et le renvoie (une ligne par mutation)."""
    out_dir = Path(out_dir)
    config = json.loads((out_dir / "config.json").read_text())
    ks = sorted(load_meta(out_dir))
    n_jobs = min(len(ks), n_jobs or max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        parts = list(pool.map(_details_one, [(config, k, str(out_dir)) for k in ks]))
    details = pd.concat(parts, ignore_index=True)
    details.to_csv(out_dir / "mutations_detail.csv", index=False)
    return details


def _shade(color: str, factor: float) -> str:
    """Éclaircit (factor < 1 : mélange avec du blanc) ou fonce (factor > 1 : avec du noir) une couleur."""
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    rgb = rgb + (255 - rgb) * (1 - factor) if factor < 1 else rgb * (2 - factor)
    return "#" + "".join(f"{int(round(v)):02x}" for v in rgb)


def _detail_color(action: str, row: pd.Series) -> str:
    if action == "ChangeBondMG":
        base = BOND_UP if row.order_to > row.order_from else BOND_DOWN
        return _shade(base, {0: 0.45, 1: 0.7, 2: 1.0, 3: 1.3}[int(max(row.order_from, row.order_to))])
    base = ELEMENT_COLORS.get(row.element, GRAY)
    return _shade(base, {0: 1.0, 1: 0.6, 2: 1.0, 3: 1.35}[0 if pd.isna(row.order_to) else int(row.order_to)])  # simple clair, triple foncé


def plot_mutation_details(experiments: dict[str, pd.DataFrame], save=None, min_label: float = 3.0) -> plt.Figure:
    """Une barre empilée par mutation (AddAtomMG, RemoveAtomMG, ChangeBondMG), un graphe par expérience :
    hauteur = nombre de mutations (toutes marches), segments = atome ajouté (+=O : O greffé par une double
    liaison), atome retiré, ou saut d'ordre de la liaison (1→2…). Couleur des atomes : élément (clair :
    liaison simple, foncé : triple) ; ChangeBond : violet si l'ordre monte, rouge s'il descend, d'autant
    plus foncé que l'ordre est élevé. Étiquette dans le segment s'il fait au moins min_label % de la barre."""
    n = len(experiments)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    for ax, (label, details) in zip(axes.flat, experiments.items()):
        present = [a for a in DETAILED_ACTIONS if (details.action == a).any()]
        for x, action in enumerate(present):
            sub = details[(details.action == action) & (details.detail != "non retrouvée")]
            groups = (sub.groupby(["detail", "element", "order_from", "order_to"], dropna=False).size()
                      .reset_index(name="n").sort_values("n", ascending=False))
            if action == "ChangeBondMG":  # même saut, éléments différents : un seul segment
                groups = (groups.groupby(["detail", "order_from", "order_to"], dropna=False).n.sum()
                          .reset_index().sort_values("n", ascending=False))
            total, bottom = groups.n.sum(), 0
            for _, row in groups.iterrows():
                ax.bar(x, row.n, bottom=bottom, width=0.62, color=_detail_color(action, row),
                       edgecolor="white", linewidth=1.5)
                if 100 * row.n / total >= min_label:
                    rgb = [int(_detail_color(action, row)[i:i + 2], 16) for i in (1, 3, 5)]
                    dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 150  # texte blanc sur fond foncé
                    ax.text(x, bottom + row.n / 2, f"{row.detail}  {100 * row.n / total:.0f} %", ha="center",
                            va="center", fontsize=7.5, color="white" if dark else "#0b0b0b")
                bottom += row.n
            missing = (details.action == action).sum() - total
            ax.text(x, bottom, f"{bottom}" + (f"\n({missing} non retrouvées)" if missing else ""),
                    ha="center", va="bottom", fontsize=8, color="#52514e")
        ax.set_xticks(range(len(present)), [a.removesuffix("MG") for a in present])
        if n > 1:  # une seule expérience : son nom est l'en-tête (_save)
            ax.set_title(label, loc="left", fontsize=9)
        ax.set_ylabel("nombre de mutations")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.1)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return _save(fig, save)


def _axes(n: int) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.3 * n), sharex=True, squeeze=False)
    return fig, axes[:, 0]


def plot_walks(walks: dict[int, pd.DataFrame], variables: list[str] = VARIABLES, save=None) -> plt.Figure:
    """Un graphe par variable, une courbe par marche ; un calcul échoué est tracé à FAILED (−2)."""
    fig, axes = _axes(len(variables))
    for ax, c in zip(axes, variables):
        for k, walk in walks.items():
            ax.plot(walk.step, walk[c].fillna(FAILED), linewidth=1, label=f"marche {k}")
        ax.set_title(c, loc="left", fontsize=10)
        ax.set_ylabel(UNITS.get(c, ""))
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("pas")
    if len(walks) <= 12:
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.suptitle(f"Marches individuelles (calcul échoué : {FAILED:g})", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, save)


def plot_mean(walks: dict[int, pd.DataFrame], variables: list[str] = VARIABLES, save=None) -> plt.Figure:
    """Moyenne ± écart-type entre marches à chaque pas (calculs échoués exclus)."""
    table = pd.concat(walks.values())
    stats = table.groupby("step")[variables].agg(["mean", "std"])
    fig, axes = _axes(len(variables))
    for ax, c in zip(axes, variables):
        mean, std = stats[(c, "mean")], stats[(c, "std")].fillna(0)
        ax.fill_between(stats.index, mean - std, mean + std, color=BLUE, alpha=0.2, linewidth=0)
        ax.plot(stats.index, mean, color=BLUE, linewidth=1.4)
        ax.set_title(c, loc="left", fontsize=10)
        ax.set_ylabel(UNITS.get(c, ""))
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("pas")
    fig.suptitle(f"Moyenne ± écart-type sur {len(walks)} marches", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, save)


def _autocorrelation(values: pd.Series, max_lag: int) -> pd.Series:
    """ρ(k) = corrélation de Pearson entre values[t] et values[t + k], k = 1 … max_lag."""
    values = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    # série constante ou trop courte : ρ non défini (NaN), sans avertissement numpy
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pd.Series({k: values.autocorr(k) for k in range(1, max_lag + 1)}, dtype=float)


def correlation_length(rho1: float) -> float:
    if not np.isfinite(rho1) or rho1 <= 0:
        return math.nan
    return math.inf if rho1 >= 1 else -1 / math.log(rho1)


def plot_autocorrelation(rho: pd.DataFrame, columns: list[str], save=None) -> plt.Figure:
    """ρ(k) moyen ± écart-type entre marches (rho en format long : observable, k, rho)."""
    fig, axes = _grid(len(columns), sharey=True)
    for ax, c in zip(axes.flat, columns):
        r = rho[rho.observable == c].groupby("k").rho.agg(["mean", "std"])
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.fill_between(r.index, r["mean"] - r["std"], r["mean"] + r["std"], color=BLUE, alpha=0.2, linewidth=0)
        ax.plot(r.index, r["mean"], color=BLUE, linewidth=1.8)
        rho1 = r["mean"].iloc[0] if len(r) else math.nan
        ax.set_title(f"{c}   ρ(1) = {rho1:.2f}   ℓ = {correlation_length(rho1):.1f}", loc="left", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("décalage k (pas)")
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("ρ(k)")
    fig.tight_layout()
    return _save(fig, save)


def autocorrelation(walks: dict[int, pd.DataFrame], variables: list[str] = VARIABLES,
                    max_lag: int = MAX_LAG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ρ(1) et ℓ = −1/ln ρ(1) par marche et par variable (calculs échoués exclus), et ρ(k) en
    format long (observable, k, rho, walk) pour plot_autocorrelation."""
    table, rhos = [], []
    for k, walk in walks.items():
        lag = min(max_lag, len(walk) - 2)
        for c in variables:
            rho = _autocorrelation(walk[c], lag)
            rho1 = rho.iloc[0] if len(rho) else math.nan
            table.append({"walk": k, "variable": c, "rho1": rho1, "corr_length": correlation_length(rho1)})
            rhos.append(pd.DataFrame({"observable": c, "k": rho.index, "rho": rho.values, "walk": k}))
    return pd.DataFrame(table), pd.concat(rhos, ignore_index=True)


def autocorrelation_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par variable : longueur de corrélation ℓ de chaque marche, puis moyenne et écart-type."""
    wide = table.pivot(index="variable", columns="walk", values="corr_length")
    wide = wide.reindex([v for v in VARIABLES if v in wide.index] + [v for v in wide.index if v not in VARIABLES])
    wide.columns = [f"ℓ marche {k}" for k in wide.columns]
    values = wide.replace([np.inf, -np.inf], np.nan)
    wide["ℓ moyenne"] = values.mean(axis=1)
    wide["ℓ écart-type"] = values.std(axis=1)
    return wide


def recount_pi(out_dir: str | Path, sulfur_rule: bool = True) -> None:
    """Recalcule PI_CHAIN_SIZE et PCT_PI_ATOMS des marches d'une expérience à partir des SMILES
    enregistrés (sans relancer les marches), réécrit les fichiers et config.json (SULFUR_PI_RULE),
    puis refait les figures."""
    out_dir = Path(out_dir)
    for k, walk in load_walks(out_dir).items():
        values = {s: pi_systems(s, sulfur_rule) for s in walk.smiles.unique()}
        for column in ["PI_CHAIN_SIZE", "PCT_PI_ATOMS"]:
            walk[column] = [values[s][column] for s in walk.smiles]
        walk.drop(columns=["TOTAL_PI_ATOMS"], errors="ignore").to_csv(out_dir / f"walk{k}.csv.gz", index=False)
    config_path = out_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["SULFUR_PI_RULE"] = sulfur_rule
    config_path.write_text(json.dumps(config, indent=1, ensure_ascii=False))
    make_figures(out_dir)


def plot_pairs(walks: dict[int, pd.DataFrame], variables: list[str] = VARIABLES, save=None) -> plt.Figure:
    """Chaque variable contre chaque autre (sns.pairplot, kind="reg") sur tous les pas de toutes les
    marches : nuage de points, droite de régression (bande : intervalle de confiance à 95 %) et son R²,
    histogramme sur la diagonale. Calculs échoués (NaN) exclus paire par paire."""
    import seaborn as sns  # pylint: disable=import-outside-toplevel

    data = pd.concat(walks.values(), ignore_index=True)[variables]
    grid = sns.pairplot(data, kind="reg", height=2.0, diag_kws={"color": BLUE},
                        plot_kws={"color": BLUE, "scatter_kws": {"s": 6, "alpha": 0.25, "linewidths": 0},
                                  "line_kws": {"color": "#0b0b0b", "linewidth": 1.5}})
    for i, y in enumerate(variables):  # R² de la droite de régression (= r de Pearson au carré)
        for j, x in enumerate(variables):
            if i == j:
                continue
            pair = data[[x, y]].dropna()
            r2 = pair[x].corr(pair[y]) ** 2 if len(pair) > 2 else float("nan")
            grid.axes[i, j].text(0.03, 0.97, f"R² = {r2:.2f}", transform=grid.axes[i, j].transAxes,
                                 ha="left", va="top", fontsize=8, color="#0b0b0b",
                                 bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8,
                                       "linewidth": 0})
    grid.figure.suptitle(f"Variables deux à deux ({len(data)} pas, {len(walks)} marches)", x=0.01, ha="left")
    return _save(grid.figure, save)


def make_figures(out_dir: str | Path) -> None:
    """Refait les figures d'une expérience à partir de ses fichiers, sans relancer les marches :
    figures/01_marches.png, 02_autocorrelation.png, 03_moyenne.png, 04_decompte.png, 06_paires.png
    (comme le notebook)."""
    walks = load_walks(out_dir)
    figures = Path(out_dir) / "figures"
    for fig in [plot_walks(walks, save=figures / "01_marches.png"),
                plot_autocorrelation(autocorrelation(walks)[1], VARIABLES, save=figures / "02_autocorrelation.png"),
                plot_mean(walks, save=figures / "03_moyenne.png"),
                plot_counts(write_counts(out_dir), save=figures / "04_decompte.png"),
                plot_pairs(walks, save=figures / "06_paires.png")]:
        plt.close(fig)


def show_steps(out_dir: str | Path, walk: int, start: int, stop: int, per_row: int = 3,
               size: tuple[int, int] = (360, 300)) -> Any:
    """Molécules des pas start … stop (inclus) de la marche walk d'une expérience, per_row par ligne,
    avec la mutation et les valeurs de chaque pas."""
    table = load_walks(out_dir)[walk]
    rows = table[(table.step >= start) & (table.step <= stop)]
    mols, legends = [], []
    for _, r in rows.iterrows():
        mol = Chem.MolFromSmiles(r.smiles) or Chem.MolFromSmiles(r.smiles, sanitize=False)
        mols.append(mol)
        legends.append(f"pas {r.step} : {r.action if isinstance(r.action, str) else 'départ'}\n"
                       f"{r.n_atoms:g} atomes | pi max {r.PI_CHAIN_SIZE:g} | {r.PCT_PI_ATOMS:.0f} % pi\n"
                       f"HOMO {r.HOMO:.3f} | LUMO {r.LUMO:.3f} | gap {r.gap:.3f}")
    return Draw.MolsToGridImage(mols, legends=legends, molsPerRow=per_row, subImgSize=size)
