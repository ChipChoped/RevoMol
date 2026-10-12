"""
Marche aléatoire dans l'espace chimique (scripts/Lea/random_walk_core.py), en ligne de commande.

Un burn-in de --burn-in pas depuis --start donne un point de départ aléatoire ; la
marche observée part de la molécule atteinte. Les pas du burn-in ne sont pas gardés.
Résultat : un CSV avec une ligne par molécule visitée (SMILES, mutation, temps,
observables).

Usage :
    python scripts/Lea/ancien/random_walk.py --steps 500 --out output/random_walk.csv
    python scripts/Lea/ancien/random_walk.py --actions AddAtomMG,SubstituteAtomMG --realism none
    python scripts/Lea/ancien/random_walk.py --no-insert-carbon --avoid-sp3-on-sp3 --observe HOMO,gap

Expériences (notebooks/random_walk/ancien/) : make_config, run_walk, run_many — une configuration
complète (DEFAULT_CONFIG), toutes les observables, la fitness et l'évolvabilité du voisinage,
plusieurs répétitions en parallèle, un fichier par marche.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

CWD = os.getcwd()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts", "Lea"))  # fitness_functions, huckel_filters, random_walk_core
os.chdir(ROOT)  # les filtres ChEMBL/ZINC lisent external_data/ en relatif

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import Descriptors

import fitness_functions as ff
from fitness_functions import Gaussian, Sigmoid
from evomol import default_parameters as dp
import huckel_filters as hf
from evomol.action import molecular_graph as mg
from evomol.evaluation import (
    CycleScore, Evaluation, EvaluationError, Function, Huckel, HuckelGap, HuckelHOMO, HuckelLUMO,
    HuckelOrbitalLambda, HuckelOrbitalLambdaOverlap, LogP, NormalizedCycleScore, PLogP, QED,
    SAScore, ZincNormalizedLogP, ZincNormalizedSAScore, is_valid_molecule,
)
from evomol.evaluation.huckel import resolve_mo_index
from evomol.representation import MolecularGraph, Molecule
from random_walk_core import neighbor_actions, random_walk

RDLogger.DisableLog("rdApp.*")

ACTIONS = ["AddAtomMG", "ChangeBondMG", "CutAtomMG", "InsertCarbonMG", "MoveGroupMG", "RemoveAtomMG", "SubstituteAtomMG"]
# nom de l'observable -> (nom de l'évaluation evomol, classe ou instance)
OBSERVABLES: dict[str, tuple[str, Any]] = {
    "HOMO": ("huckel_homo_lambda", HuckelHOMO),
    "LUMO": ("huckel_lumo_lambda", HuckelLUMO),
    "gap": ("huckel_gap_lambda", HuckelGap),
    "SAscore": ("SAScore", SAScore),
}


def _seed(value: str) -> int | None:
    return None if value.lower() == "none" else int(value)


def add_walk_arguments(parser: argparse.ArgumentParser) -> None:
    """Options de la marche, de l'espace d'action et des filtres."""
    walk = parser.add_argument_group("marche")
    walk.add_argument("--start", default="C=CC=Cc2cccc(C=Cc1ccccc1)c2", help="SMILES d'où part le burn-in")
    walk.add_argument("--burn-in", type=int, default=200, help="pas de burn-in (0 : la marche part de --start)")
    walk.add_argument("--steps", type=int, default=500, help="pas de la marche observée")
    walk.add_argument("--max-atoms", type=int, default=30, help="nb max d'atomes lourds")
    walk.add_argument("--unique-kekule", action="store_true",
                      help="un seul SMILES par molécule aromatique, quelle que soit sa forme kékulé (plus lent)")
    walk.add_argument("--only-valid", action=argparse.BooleanOptionalAction, default=True,
                      help="n'avancer que vers des voisins valides")
    walk.add_argument("--count-valid-neighbors", action="store_true",
                      help="valider tout le voisinage à chaque pas (~6x plus lent)")
    walk.add_argument("--seed", type=_seed, default=0,
                      help="entier : marche reproductible ; none : différente à chaque exécution")
    walk.add_argument("--exclude-current", action=argparse.BooleanOptionalAction, default=True,
                      help="la molécule courante n'est jamais tirée comme son propre voisin")
    walk.add_argument("--actions", default=None,
                      help=f"mutations autorisées, séparées par des virgules, parmi {','.join(ACTIONS)} "
                           "(défaut : toutes celles de l'espace d'action)")

    space = parser.add_argument_group("espace d'action")
    space.add_argument("--insert-carbon", action=argparse.BooleanOptionalAction, default=True,
                       help="garder InsertCarbonMG")
    space.add_argument("--avoid-sp3-on-sp3", action="store_true",
                       help="AddAtomMG ne greffe pas de C sp3 sur un C sp3")
    space.add_argument("--add-atom-multiple-bonds", action=argparse.BooleanOptionalAction, default=True,
                       help="AddAtomMG peut greffer par double / triple liaison (=CH2, =O, #N…)")
    space.add_argument("--avoid-bond-breaking", action="store_true",
                       help="ChangeBondMG ne supprime jamais de liaison")
    space.add_argument("--avoid-bond-forming", action="store_true",
                       help="ChangeBondMG ne crée jamais de liaison")

    filters = parser.add_argument_group("filtres (ordre : Hückel → sp3 → ChEMBL/ZINC → Hetero)")
    filters.add_argument("--huckel", action=argparse.BooleanOptionalAction, default=True,
                         help="HuckelCompatible")
    filters.add_argument("--max-sp3", type=int, default=None,
                         help="FilterSP3 : nb max d'atomes sans liaison π (défaut : filtre inactif)")
    filters.add_argument("--realism", choices=["chembl_zinc", "chembl", "none"], default="chembl_zinc",
                         help="filtres GCF + ECFP sur cette base ; none : aucun")
    filters.add_argument("--hetero", action=argparse.BooleanOptionalAction, default=True,
                         help="HeteroatomChainFilter")
    filters.add_argument("--max-ring-n", type=int, default=3,
                         help="nb max d'azotes en cycle (HeteroatomChainFilter)")


def setup_from_args(args: argparse.Namespace) -> tuple[list[Evaluation], dict[str, Any]]:
    """Configure evomol et renvoie (filtres, arguments de random_walk)."""
    dp.setup_default_parameters(accepted_atoms=hf.HUCKEL_ACCEPTED_ATOMS, max_heavy_atoms=args.max_atoms,
                                unique_kekule=args.unique_kekule)
    dp.setup_default_action_space(
        insert_carbon=args.insert_carbon,
        avoid_bond_breaking=args.avoid_bond_breaking,
        avoid_bond_forming=args.avoid_bond_forming,
        avoid_sp3_on_sp3=args.avoid_sp3_on_sp3,
        add_atom_multiple_bonds=args.add_atom_multiple_bonds,
    )
    filters = hf.setup_huckel_filters(
        huckel=args.huckel,
        max_sp3=args.max_sp3,
        realism=None if args.realism == "none" else args.realism,
        hetero=args.hetero,
        max_ring_n=args.max_ring_n,
    )
    walk_kwargs = {
        "only_valid": args.only_valid,
        "count_valid_neighbors": args.count_valid_neighbors,
        "exclude_current": args.exclude_current,
        "action_types": [getattr(mg, a) for a in args.actions.split(",")] if args.actions else None,
    }
    return filters, walk_kwargs


def burn_in(start: str, n_steps: int, filters: list[Evaluation], seed: int | None, walk_kwargs: dict[str, Any]) -> str:
    """SMILES canonique atteint après n_steps pas depuis start (start si n_steps = 0)."""
    if n_steps == 0:
        return Molecule(start).get_representation(MolecularGraph).canonical_smiles
    return random_walk(start, n_steps, filters, seed=seed, **walk_kwargs)[-1]["smiles"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_walk_arguments(parser)
    parser.add_argument("--observe", default="HOMO,LUMO,gap,SAscore",
                        help=f"observables, séparées par des virgules, parmi {','.join(OBSERVABLES)} ; none : aucune")
    parser.add_argument("--out", default="output/random_walk.csv", help="CSV de sortie")
    args = parser.parse_args()

    filters, walk_kwargs = setup_from_args(args)
    observed = [] if args.observe.lower() == "none" else args.observe.split(",")
    evaluations: list[Evaluation] = []
    if set(observed) & {"HOMO", "LUMO", "gap"}:
        evaluations.append(Huckel())  # calcul Hückel complet, partagé par HOMO/LUMO/gap
    for name in observed:
        evaluation = OBSERVABLES[name][1]
        evaluations.append(evaluation() if isinstance(evaluation, type) else evaluation)

    print(f"mutations : {args.actions or [a.__name__ for a in MolecularGraph.action_space]}")
    print(f"filtres   : {[f.name for f in filters] or 'aucun'}")

    t0 = time.perf_counter()
    walk_start = burn_in(args.start, args.burn_in, filters, args.seed, walk_kwargs)
    print(f"départ    : {walk_start}  (burn-in de {args.burn_in} pas en {time.perf_counter() - t0:.1f} s)")

    # burn-in et marche ont chacun leur seed : la marche ne rejoue pas les tirages du burn-in
    walk_seed = None if args.seed is None else args.seed + 1
    t0 = time.perf_counter()
    rows = random_walk(walk_start, args.steps, filters, evaluations=evaluations, seed=walk_seed, **walk_kwargs)
    wall = time.perf_counter() - t0

    walk = pd.DataFrame(rows).drop(columns=["huckel"], errors="ignore")
    walk = walk.rename(columns={OBSERVABLES[n][0]: n for n in observed})
    walk["n_atoms"] = [Molecule(s).get_representation(MolecularGraph).nb_atoms for s in walk.smiles]

    out = os.path.join(CWD, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    walk.to_csv(out, index=False)
    print(f"marche    : {args.steps} pas en {wall:.1f} s, {walk.smiles.nunique()} molécules distinctes, "
          f"{walk.valid.mean():.0%} valides")
    print(f"résultat  : {out}")
    return 0


# ==============================================================================
#  EXPÉRIENCES : configuration complète, toutes les observables, répétitions
# ==============================================================================
ALL_ACTIONS = ["AddAtomMG", "RemoveAtomMG", "ChangeBondMG", "CutAtomMG", "MoveGroupMG",
               "SubstituteAtomMG", "InsertCarbonMG", "AddGroupMG", "RemoveGroupMG"]

DEFAULT_CONFIG: dict[str, Any] = {
    # ── départ et marche
    "START": "c1ccccc1",          # SMILES d'où part le burn-in
    "BURN_IN_STEPS": 100,         # pas pour obtenir un départ aléatoire (0 : la marche part de START)
    "N_STEPS": 500,               # pas de la marche observée
    "MAX_ATOMS": 50,              # nb max d'atomes lourds
    "ACCEPTED_ATOMS": ["C", "N", "O", "S", "F"],
    "UNIQUE_KEKULE": False,       # un seul SMILES par molécule aromatique (plus lent)
    "ONLY_VALID": True,           # n'avancer que vers des voisins qui passent les filtres
    "COUNT_VALID_NEIGHBORS": False,  # valider tout le voisinage à chaque pas (~6x plus lent)
    "EXCLUDE_CURRENT": True,      # la molécule courante n'est jamais son propre voisin
    "ACTION_TYPES": ["AddAtomMG", "RemoveAtomMG", "ChangeBondMG"],  # None : toutes
    # ── espace d'action
    "INSERT_CARBON": False,       # InsertCarbonMG disponible (toujours rejeté par Hückel)
    "AVOID_SP3_ON_SP3": True,     # AddAtomMG ne greffe pas de C sp3 sur un C sp3
    "ADD_ATOM_MULTIPLE_BONDS": True,  # AddAtomMG peut greffer par double / triple liaison
    "AVOID_BOND_BREAKING": False,  # ChangeBondMG ne supprime jamais de liaison
    "AVOID_BOND_FORMING": False,  # ChangeBondMG ne crée jamais de liaison
    "WITH_ADD_GROUP": False,      # AddGroupMG disponible
    "WITH_REMOVE_GROUP": False,   # RemoveGroupMG disponible
    # ── filtres (ordre : Hückel → sp3 → ChEMBL/ZINC → Hetero)
    "FILTER_HUCKEL": True,        # tout C sp3 doit être en bout de chaîne
    "MAX_SP3": None,              # None : inactif ; entier : nb max d'atomes sans liaison π
    "FILTER_REALISM": "chembl_zinc",  # "chembl_zinc", "chembl" ou None
    "FILTER_HETERO": False,       # chaînes d'hétéroatomes
    "MAX_RING_N": 3,              # nb max d'azotes en cycle (si FILTER_HETERO)
    # ── orbitales observées
    "ORBITAL_LAMBDA": "HOMO-1",
    "LAMBDA_OVERLAP_ORBITALS": ("HOMO", "LUMO"),
    # ── orbitales en plus, calculées seulement pour FITNESS_GOALS (pas de colonne observable)
    "ORBITAL_LAMBDAS": [],        # ex. ["HOMO-1", "HOMO", "LUMO"] -> huckel_orbital_lambda_<o>
    "LAMBDA_OVERLAPS": [],        # ex. [("HOMO", "LUMO")] -> huckel_lambda_<o1>_<o2>
    # ── fitness mesurée (objectifs de notebooks/search_fitness_functions.ipynb) ; [] : pas de fitness
    "FITNESS_GOALS": [
        ("huckel_homo_lambda", Sigmoid(center=1.005, steepness=23.1707, direction="down"), 1),
        ("SAScore", Sigmoid(center=3.9925, steepness=1.60986, direction="down"), 1),
        ("huckel_lumo_lambda", Gaussian(center=-1, sigma=0.511646), 1.5),
    ],
    "AGGREGATION": "sum",         # "sum" ou "geometric"
    # ── évolvabilité : nb de molécules visitées dont on évalue tout le voisinage (0 : aucune)
    "NEIGHBORHOOD_SAMPLE": 20,
}


def make_config(**overrides: Any) -> dict[str, Any]:
    """DEFAULT_CONFIG modifiée par overrides (une clé inconnue est une erreur)."""
    unknown = set(overrides) - set(DEFAULT_CONFIG)
    if unknown:
        raise KeyError(f"Réglages inconnus : {sorted(unknown)}")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(copy.deepcopy(overrides))
    return config


def action_config(actions: list[str]) -> dict[str, Any]:
    """Réglages qui rendent exactement ces mutations disponibles (InsertCarbonMG, AddGroupMG et
    RemoveGroupMG n'existent dans l'espace d'action que si leur option est activée)."""
    return {
        "ACTION_TYPES": list(actions),
        "INSERT_CARBON": "InsertCarbonMG" in actions,
        "WITH_ADD_GROUP": "AddGroupMG" in actions,
        "WITH_REMOVE_GROUP": "RemoveGroupMG" in actions,
    }


def _mol_weight(molecule: Molecule) -> float:
    return Descriptors.MolWt(molecule.get_representation(MolecularGraph).mol)


def _n_heavy_atoms(molecule: Molecule) -> int:
    return molecule.get_representation(MolecularGraph).mol.GetNumHeavyAtoms()


def _property_chains(config: dict[str, Any]) -> dict[str, list[Evaluation]]:
    """Propriété evomol -> évaluations à lancer, dépendances d'abord (catalogue de
    scripts/Lea/evolutionary_algorithm.py : noms utilisables dans FITNESS_GOALS)."""
    huckel = Huckel()
    orbital = HuckelOrbitalLambda(config["ORBITAL_LAMBDA"])
    overlap = HuckelOrbitalLambdaOverlap(list(config["LAMBDA_OVERLAP_ORBITALS"]))
    extra = [HuckelOrbitalLambda(o) for o in config["ORBITAL_LAMBDAS"]]
    extra += [HuckelOrbitalLambdaOverlap(list(s)) for s in config["LAMBDA_OVERLAPS"]]
    return {
        **{evaluation.name: [huckel, evaluation] for evaluation in extra},
        "huckel_homo_lambda": [huckel, HuckelHOMO()],
        "huckel_lumo_lambda": [huckel, HuckelLUMO()],
        "huckel_gap_lambda": [huckel, HuckelGap()],
        orbital.name: [huckel, orbital],
        overlap.name: [huckel, overlap],
        "n_heavy_atoms": [Function("n_heavy_atoms", _n_heavy_atoms)],
        "mol_weight": [Function("mol_weight", _mol_weight)],
        "SAScore": [SAScore],
        "zinc_normalized_sa_score": [SAScore, ZincNormalizedSAScore],
        "QED": [QED],
        "logP": [LogP],
        "zinc_normalized_logP": [LogP, ZincNormalizedLogP],
        "CycleScore": [CycleScore],
        "zinc_normalized_cycle_score": [CycleScore, NormalizedCycleScore],
        "PLogP": [LogP, ZincNormalizedLogP, SAScore, ZincNormalizedSAScore, CycleScore,
                  NormalizedCycleScore, PLogP],
    }


def observable_properties(config: dict[str, Any]) -> dict[str, str]:
    """Colonne de la marche -> propriété evomol, pour toutes les observables."""
    orbital = HuckelOrbitalLambda(config["ORBITAL_LAMBDA"]).name
    overlap = HuckelOrbitalLambdaOverlap(list(config["LAMBDA_OVERLAP_ORBITALS"])).name
    return {
        "HOMO": "huckel_homo_lambda", "LUMO": "huckel_lumo_lambda", "gap": "huckel_gap_lambda",
        f"λ {config['ORBITAL_LAMBDA']}": orbital,
        f"|overlap| {'·'.join(map(str, config['LAMBDA_OVERLAP_ORBITALS']))}": overlap,
        "SAscore": "SAScore", "mol_weight": "mol_weight", "QED": "QED", "logP": "logP",
        "logP_zinc": "zinc_normalized_logP", "CycleScore": "CycleScore",
        "CycleScore_zinc": "zinc_normalized_cycle_score", "PLogP": "PLogP",
    }


def observable_columns(config: dict[str, Any]) -> list[str]:
    """Colonnes des observables d'une marche, dans l'ordre d'affichage."""
    return [*observable_properties(config), *(["fitness"] if config["FITNESS_GOALS"] else []), "n_atoms"]


# λ des orbitales enregistrées en plus des observables, sur la marche et les voisinages (pas analysées
# comme observables : elles servent à recalculer une autre fitness sans relancer de marche)
ORBITALS = ["HOMO-3", "HOMO-2", "HOMO-1", "HOMO", "LUMO", "LUMO+1", "LUMO+2", "LUMO+3"]


def orbital_columns(config: dict[str, Any]) -> dict[str, str]:
    """Colonne -> orbitale, pour les orbitales de ORBITALS qui ne sont pas déjà des observables."""
    existing = set(observable_columns(config))
    names = {"HOMO": "HOMO", "LUMO": "LUMO"}
    return {f"λ {o}": o for o in ORBITALS if names.get(o, f"λ {o}") not in existing}


def recorded_properties(config: dict[str, Any]) -> dict[str, str]:
    """Colonne -> propriété evomol, pour tout ce qui est enregistré sur chaque molécule (observables
    et orbitales de ORBITALS) : propriétés utilisables pour recalculer une fitness."""
    return {**observable_properties(config),
            **{column: HuckelOrbitalLambda(o).name for column, o in orbital_columns(config).items()}}


def orbital_values(molecule: Molecule, orbitals: dict[str, str]) -> dict[str, float]:
    """λ de chaque orbitale (colonne -> orbitale) ; NaN si Hückel échoue ou si l'orbitale n'existe pas."""
    row = {}
    try:
        result = Huckel().evaluate(molecule)
    except EvaluationError:
        result = None
    for column, orbital in orbitals.items():
        try:
            row[column] = result.eigenvalues_lambda[resolve_mo_index(result, orbital)]
        except (AttributeError, EvaluationError):
            row[column] = math.nan
    return row


def molecule_values(molecule: Molecule, evaluations: list[Evaluation], orbitals: dict[str, str]) -> dict[str, float]:
    """Toutes les variables d'une molécule : évaluations (observables, fitness), n_atoms, orbitales."""
    row = {e.name: _value(e, molecule) for e in evaluations}
    row["n_atoms"] = molecule.get_representation(MolecularGraph).nb_atoms
    row.update(orbital_values(molecule, orbitals))
    return row


# ── fitness enregistrée sous une forme relisible (runs/<config>/fitness.json, valeurs exactes)
def curve_to_json(curve: Any) -> dict[str, Any]:
    if isinstance(curve, ff.Product):
        return {"type": "Product", "curves": [curve_to_json(c) for c in curve.curves]}
    return {"type": type(curve).__name__, **vars(curve)}


def curve_from_json(data: dict[str, Any]) -> Any:
    data = dict(data)
    kind = data.pop("type")
    if kind == "Product":
        return ff.Product(*[curve_from_json(c) for c in data["curves"]])
    return getattr(ff, kind)(**data)


def goals_to_json(goals: list[tuple]) -> list[dict[str, Any]]:
    return [{"property": name, "curve": curve_to_json(curve), "weight": weight} for name, curve, weight in goals]


def goals_from_json(data: list[dict[str, Any]]) -> list[tuple]:
    return [(g["property"], curve_from_json(g["curve"]), g["weight"]) for g in data]


UNITS = {
    "HOMO": "λ", "LUMO": "λ", "gap": "λ", "SAscore": "1 (facile) – 10", "mol_weight": "g/mol",
    "QED": "0 – 1", "logP": "lipophilie", "logP_zinc": "normalisé ZINC", "CycleScore": "0 si cycles ≤ 6",
    "CycleScore_zinc": "normalisé ZINC", "PLogP": "logP pénalisé", "fitness": "0 – 1",
    "n_atoms": "atomes lourds",
}


def unit(column: str) -> str:
    if column.startswith("λ "):
        return "λ"
    if column.startswith("|overlap|"):
        return "Σ ∏|c|"
    return UNITS.get(column, "")


def _guarded(name: str, compute: Callable[[Molecule], Any]) -> Function:
    """Évaluation nommée name ; une erreur de calcul (ex. Hückel impossible, dont dépend la
    propriété) donne EvaluationError, donc la valeur None dans la marche, sans l'arrêter."""
    def evaluate(molecule: Molecule) -> Any:
        try:
            return compute(molecule)
        except (EvaluationError, KeyError, TypeError, ValueError, ZeroDivisionError, RuntimeError) as exc:
            raise EvaluationError(f"{name} : {exc}") from exc
    return Function(name, evaluate)


def _run_chain(chain: list[Evaluation]) -> Callable[[Molecule], Any]:
    def compute(molecule: Molecule) -> Any:
        value = None
        for evaluation in chain:
            value = evaluation.evaluate(molecule)
        return value
    return compute


def build_evaluations(config: dict[str, Any]) -> tuple[list[Evaluation], Evaluation | None]:
    """Une évaluation par colonne observable (+ fitness) et l'évaluation de la fitness seule."""
    chains = _property_chains(config)
    evaluations: list[Evaluation] = [
        _guarded(column, _run_chain(chains[prop])) for column, prop in observable_properties(config).items()
    ]
    goals = config["FITNESS_GOALS"]
    if not goals:
        return evaluations, None
    ff.check_goals(goals, config["AGGREGATION"])
    unknown = [name for name, *_ in goals if name not in chains]
    if unknown:
        raise ValueError(f"Propriétés inconnues dans FITNESS_GOALS : {unknown}. Disponibles : {sorted(chains)}")
    weights = [w for *_, w in goals]

    def fitness(molecule: Molecule) -> float:
        for name, *_ in goals:
            _run_chain(chains[name])(molecule)
        return ff.aggregate([s for *_, s in ff.goal_scores(molecule, goals)], weights, config["AGGREGATION"])

    fitness_evaluation = _guarded("fitness", fitness)
    return [*evaluations, fitness_evaluation], fitness_evaluation


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


def _value(evaluation: Evaluation, molecule: Molecule) -> float:
    try:
        return evaluation.evaluate(molecule)
    except EvaluationError:
        return math.nan


def analyse_neighborhoods(
    walk: pd.DataFrame, config: dict[str, Any], filters: list[Evaluation],
    walk_kwargs: dict[str, Any], evaluations: list[Evaluation],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tout le voisinage de NEIGHBORHOOD_SAMPLE molécules réparties le long de la marche (même règle
    que la marche : EXCLUDE_CURRENT, voisins valides seulement si ONLY_VALID), avec toutes les
    variables (observables, fitness, n_atoms, orbitales de ORBITALS) de chaque voisin.

    Returns:
        molécules : une ligne par molécule analysée ; voisins : une ligne par voisin évalué (fitness, Δ) ;
        valeurs : une ligne par molécule analysée (is_parent) et par voisin évalué, toutes les variables
        (même format que scripts/Lea/evolvability.py : sert à recalculer une autre fitness).
    """
    orbitals = orbital_columns(config)
    n = min(config["NEIGHBORHOOD_SAMPLE"], len(walk))
    positions = sorted(set(np.linspace(0, len(walk) - 1, n).round().astype(int)))
    molecule_rows, neighbor_rows, value_rows = [], [], []
    for pos in positions:
        smiles, step = walk.smiles.iloc[pos], int(walk.step.iloc[pos])
        parent = molecule_values(Molecule(smiles), evaluations, orbitals)
        value_rows.append({"step": step, "parent_smiles": smiles, "smiles": smiles, "mutation": "",
                           "is_parent": True, **parent})
        f0 = parent["fitness"]
        neighbors = neighbor_actions(Molecule(smiles), walk_kwargs["action_types"])
        if config["EXCLUDE_CURRENT"]:
            neighbors.pop(smiles, None)
        n_evaluated = 0
        for neighbor_smiles, action in neighbors.items():
            neighbor = Molecule(neighbor_smiles)
            if config["ONLY_VALID"] and not is_valid_molecule(neighbor, filters):
                continue
            n_evaluated += 1
            values = molecule_values(neighbor, evaluations, orbitals)
            value_rows.append({"step": step, "parent_smiles": smiles, "smiles": neighbor_smiles,
                               "mutation": action.class_name(), "is_parent": False, **values})
            f = values["fitness"]
            neighbor_rows.append({"step": step, "mutation": action.class_name(), "fitness": f, "delta": f - f0})
        molecule_rows.append({"step": step, "smiles": smiles, "fitness": f0,
                              "n_neighbors": len(neighbors), "n_evaluated": n_evaluated})
    return (pd.DataFrame(molecule_rows),
            pd.DataFrame(neighbor_rows, columns=["step", "mutation", "fitness", "delta"]),
            pd.DataFrame(value_rows))


def run_walk(config: dict[str, Any], rep: int) -> dict[str, Any]:
    """Une marche complète : burn-in, marche observée avec toutes les observables, voisinages.

    Graines : burn-in 2·rep, marche 2·rep + 1 (mêmes graines pour toutes les configurations).

    Returns:
        {"walk", "neighborhoods", "neighbors", "values" (DataFrames ou None), "meta" (dict)}.
    """
    t0 = time.perf_counter()
    filters, walk_kwargs = configure(config)
    evaluations, fitness = build_evaluations(config)
    seed = 2 * rep
    walk_start = burn_in(config["START"], config["BURN_IN_STEPS"], filters, seed, walk_kwargs)
    t1 = time.perf_counter()
    rows = random_walk(walk_start, config["N_STEPS"], filters, evaluations=evaluations, seed=seed + 1, **walk_kwargs)
    walk = pd.DataFrame(rows)
    walk["n_atoms"] = [Molecule(s).get_representation(MolecularGraph).nb_atoms for s in walk.smiles]
    # λ des orbitales de ORBITALS (hors observables), une fois par molécule distincte
    orbitals = orbital_columns(config)
    by_smiles = {s: orbital_values(Molecule(s), orbitals) for s in walk.smiles.unique()}
    for column in orbitals:
        walk[column] = [by_smiles[s][column] for s in walk.smiles]
    t2 = time.perf_counter()
    neighborhoods = neighbors = values = None
    if config["NEIGHBORHOOD_SAMPLE"] and fitness is not None:
        neighborhoods, neighbors, values = analyse_neighborhoods(walk, config, filters, walk_kwargs, evaluations)
    t3 = time.perf_counter()
    meta = {"rep": rep, "seed_burn_in": seed, "seed_walk": seed + 1, "walk_start": walk_start,
            "filters": [f.name for f in filters],
            "t_burn_in": t1 - t0, "t_walk": t2 - t1, "t_neighborhoods": t3 - t2, "t_total": t3 - t0}
    return {"walk": walk, "neighborhoods": neighborhoods, "neighbors": neighbors, "values": values, "meta": meta}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)  # courbes de fitness


def _run_one(task: tuple[str, dict[str, Any], int, str]) -> tuple[str, int, float]:
    config_id, config, rep, run_dir = task
    result = run_walk(config, rep)
    run_dir_ = Path(run_dir)
    run_dir_.mkdir(parents=True, exist_ok=True)
    result["walk"].to_csv(run_dir_ / f"rep{rep}_walk.csv.gz", index=False)
    if result["neighborhoods"] is not None:
        result["neighborhoods"].to_csv(run_dir_ / f"rep{rep}_voisinages.csv.gz", index=False)
        result["neighbors"].to_csv(run_dir_ / f"rep{rep}_voisins.csv.gz", index=False)
        result["values"].to_parquet(run_dir_ / f"rep{rep}_voisins.parquet", index=False)
    # écrit en dernier : sa présence marque la marche comme terminée (reprise)
    (run_dir_ / f"rep{rep}_meta.json").write_text(json.dumps(result["meta"], indent=1, ensure_ascii=False))
    return config_id, rep, result["meta"]["t_total"]


def run_many(configs: dict[str, dict[str, Any]], n_repeats: int, out_dir: str | Path,
             n_jobs: int | None = None, on_done: Callable[[str, int], None] | None = None) -> None:
    """Toutes les marches (configuration, répétition) en parallèle.

    Écrit out_dir/runs/<config>/config.json, fitness.json (FITNESS_GOALS relisible par goals_from_json)
    et, par répétition k, rep<k>_walk.csv.gz, rep<k>_voisinages.csv.gz, rep<k>_voisins.csv.gz,
    rep<k>_voisins.parquet (toutes les variables du voisinage), rep<k>_meta.json. Les marches dont le
    meta.json existe déjà sont sautées : relancer reprend là où on s'était arrêté.
    out_dir est relatif à la racine du projet.
    on_done(config_id, rep) est appelé après chaque marche terminée (affichage en direct).
    Interruption (Ctrl-C, bouton stop) : les marches en cours se terminent et sont écrites,
    les autres sont annulées.
    """
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    out_dir = Path(out_dir)
    tasks, n_done = [], 0
    for config_id, config in configs.items():
        run_dir = out_dir / "runs" / config_id
        run_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(_jsonable(config), indent=1, ensure_ascii=False)
        config_path = run_dir / "config.json"
        if config_path.exists():
            # réglage ajouté après coup : absent d'un ancien config.json, il y valait sa valeur par défaut
            saved = {**_jsonable(DEFAULT_CONFIG), **json.loads(config_path.read_text())}
            if saved != json.loads(text):
                raise ValueError(f"{run_dir} contient des marches d'une autre configuration : "
                                 "changer de dossier (ou de LABEL) ou supprimer ce dossier")
        config_path.write_text(text)
        (run_dir / "fitness.json").write_text(json.dumps(
            {"FITNESS_GOALS": goals_to_json(config["FITNESS_GOALS"]), "AGGREGATION": config["AGGREGATION"]},
            indent=1, ensure_ascii=False))
        for rep in range(n_repeats):
            if (run_dir / f"rep{rep}_meta.json").exists():
                n_done += 1
            else:
                tasks.append((config_id, config, rep, str(run_dir)))
    print(f"{len(tasks)} marches à faire, {n_done} déjà faites ({len(configs)} configurations × {n_repeats})")
    if not tasks:
        return
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    errors = []
    t0 = time.perf_counter()
    # spawn : chaque processus importe ce module et configure evomol lui-même
    pool = ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("spawn"))
    try:
        futures = {pool.submit(_run_one, task): task[:3:2] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="marches"):
            try:
                future.result()
            except Exception as exc:  # pylint: disable=broad-except
                errors.append((*futures[future], f"{type(exc).__name__}: {exc}"))
                continue
            if on_done is not None:
                on_done(*futures[future])
    except KeyboardInterrupt:
        print("Interruption : fin des marches en cours, les autres sont annulées "
              "(relancer pour reprendre)…")
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown()
    print(f"terminé en {(time.perf_counter() - t0) / 60:.1f} min")
    for config_id, rep, message in errors:
        print(f"ÉCHEC {config_id} rep{rep} : {message}")
    if errors:
        raise RuntimeError(f"{len(errors)} marches ont échoué (relancer pour réessayer)")


if __name__ == "__main__":
    sys.exit(main())
