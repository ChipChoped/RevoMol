"""
================================================================================
 Recherche de molécules par algorithme évolutionnaire + méthode de Hückel
================================================================================

OBJECTIF
--------
C'est un script EvoMol classique (que tu connais déjà) : on fait évoluer une
molécule pour optimiser un score « fitness ». LA SEULE NOUVEAUTÉ ici, c'est que
ce score est calculé à partir d'une propriété électronique estimée par la
méthode de Hückel.

Ce fichier n'explique donc QUE la partie Hückel ; le reste (population, sélection,
voisinage, espace d'actions…) est l'usage habituel d'EvoMol.


PETIT VOCABULAIRE HÜCKEL (juste ce qu'il faut savoir)
----------------------------------------------
- Méthode de Hückel : un calcul de chimie quantique très simplifié et rapide.
  Pour une molécule donnée, il estime l'énergie de chaque « orbitale » (un niveau
  d'énergie que peuvent occuper les électrons).

- λ (lambda) : le nombre que Hückel calcule pour chaque orbitale. C'est ce que le
  code manipule. ATTENTION au sens :

      énergie = α + λ × β   avec β < 0

  Comme β est négatif, un λ PLUS PETIT correspond à une énergie PLUS GRANDE.
  Donc « maximiser l'énergie » revient à « minimiser λ », et inversement.

- HOMO : l'orbitale occupée (par des électrons) la plus haute en énergie.
- LUMO : l'orbitale vide la plus basse en énergie.
- gap  : l'écart entre HOMO et LUMO. Un petit gap rend souvent la molécule plus
  réactive / colorée / conductrice ; un grand gap la rend plus stable.

  → Ces trois grandeurs sont les « propriétés » que l'on peut optimiser.


PROPRIÉTÉS DISPONIBLES (à utiliser dans `fitness_goals`)
--------------------------------------------------------
  "huckel_homo_lambda"  → λ de l'orbitale HOMO
  "huckel_lumo_lambda"  → λ de l'orbitale LUMO
  "huckel_gap_lambda"   → λ(HOMO) − λ(LUMO)  (l'écart, le « gap »)

  Autres propriétés (hors Hückel), calculées seulement si utilisées :
  "n_heavy_atoms", "mol_weight", "SAScore", "NormalizedSAScore",
  "zinc_normalized_sa_score", "QED", "logP", "zinc_normalized_logP",
  "CycleScore", "zinc_normalized_cycle_score", "PLogP"
  (voir `other_evaluations` dans main()).

  → OÙ EST CALCULÉ HÜCKEL / QUELLES AUTRES PROPRIÉTÉS SONT POSSIBLES ?
    Tout le calcul de Hückel et la liste complète des grandeurs calculables
    se trouvent dans le fichier :

        evomol/evaluation/huckel.py

    (depuis la racine du projet RevoMol). Y voir notamment les classes
    HuckelHOMO, HuckelLUMO, HuckelGap — chacune définit le "nom" de propriété
    utilisable ici — ainsi que des évaluateurs plus avancés (énergies d'autres
    orbitales, recouvrements d'orbitales, etc.) qu'on peut activer au besoin.


COMMENT DÉFINIR CE QU'ON CHERCHE (`fitness_goals`)
--------------------------------------------------
`fitness_goals` est une liste d'objectifs. Chaque objectif est un tuple :

      (nom_propriété, mode, poids, cible)

  - nom_propriété : une des propriétés ci-dessus, ex. "huckel_gap_lambda"
  - mode :
        "max"    → rendre la valeur la plus GRANDE possible   (cible ignorée)
        "min"    → rendre la valeur la plus PETITE possible    (cible ignorée)
        "target" → rapprocher la valeur d'une cible            (cible utilisée)
  - poids : importance relative de cet objectif (plus grand = plus important)
  - cible : le nombre visé en mode "target", sinon None

Exemples :
    # gap le plus petit possible
    [("huckel_gap_lambda", "min", 1.0, None)]

    # gap petit ET HOMO proche de 0.4, le HOMO comptant 3 fois plus
    [("huckel_gap_lambda", "min", 1.0, None),
     ("huckel_homo_lambda", "target", 3.0, 0.4)]

Quand il y a plusieurs objectifs, ils sont combinés en un seul score (« fitness »)
par une somme pondérée, et l'algorithme cherche TOUJOURS à maximiser ce score.
================================================================================
"""

import os
import random
import sys
from typing import Literal

from rdkit.Chem import Descriptors

# Permet d'importer le paquet `evomol` même si le script est lancé depuis ailleurs.
# Ce fichier est dans scripts/Lea/, donc la racine du projet (où se trouve le
# dossier `evomol`) est 2 dossiers au-dessus.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_ROOT)
# dossier de ce script : les processus de calcul (n_workers > 1) y importent
# ce module et fitness_functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position, import-error

from evomol import default_parameters as dp
import huckel_filters as hf
from evomol import evaluation as evaluator
from evomol.logging import init_logger
from evomol.representation import MolecularGraph, Molecule
from evomol.search import selection_strategy
from evomol.search.evolutionary_algorithm import (
    EvolutionaryAlgorithm,
    ParametersEvolutionaryAlgorithm,
)
from evomol.search.neighborhood_strategy import RandomNeighborhoodStrategy
from parallel_evolutionary_algorithm import ParallelEvolutionaryAlgorithm
from evomol.search.parameters import search_parameters


# ==============================================================================
#  OBJECTIF PAR DÉFAUT
# ==============================================================================
# Utilisé seulement si on appelle main() sans préciser `fitness_goals`.
# Le format est expliqué en haut du fichier ("COMMENT DÉFINIR CE QU'ON CHERCHE").
FITNESS_GOALS: list[tuple[str, Literal["max", "min", "target"], float, float | None]] = [
    ("huckel_homo_lambda", "min", 1.0, None),  # = maximiser l'énergie du HOMO
]


# ==============================================================================
#  FILTRES — même standard que la marche aléatoire (hf.setup_huckel_filters)
# ==============================================================================
# Appliqués dans cet ordre : Hückel → sp3 → ChEMBL/ZINC → Hetero
#   huckel     : True / False — compatibilité Hückel (HuckelCompatible) :
#                tout C sp3 doit être en bout de chaîne.
#   max_sp3    : None = filtre inactif ; entier >= 0 = nb max d'atomes sans
#                liaison π (FilterSP3, détaillé ci-dessous).
#   realism    : "chembl_zinc" = filtres GCF + ECFP sur ChEMBL + ZINC (le plus large)
#                "chembl"      = filtres GCF + ECFP sur ChEMBL seule
#                None          = aucun filtre de réalisme
#   hetero     : True / False — chaînes d'hétéroatomes (HeteroatomChainFilter).
#   max_ring_n : nb max d'azotes en cycle (ignoré si hetero = False).
#
# FILTRE sp3 — QU'EST-CE QU'UN ATOME "sp3" ICI ?
#   En chimie, un atome est dit "sp3" quand il n'a que des liaisons simples
#   (pas de double, triple ni aromatique). Concrètement, dans ce code, on
#   compte comme "sp3" tout atome qui ne participe à AUCUNE liaison π
#   (= double, triple ou aromatique).
#
# POURQUOI LIMITER LES sp3 ?
#   La méthode de Hückel ne décrit QUE les électrons π (ceux des doubles
#   liaisons, triples liaisons et cycles aromatiques). Les atomes sp3 n'ont pas
#   d'électron π : ils n'apportent rien au calcul et "diluent" la partie
#   intéressante de la molécule. On veut donc surtout des molécules conjuguées
#   (riches en liaisons π), d'où ce filtre.
#
# COMMENT LE RÉGLER (paramètre `max_sp3`) ?
#   - max_sp3 = 0   → AUCUN atome sp3 toléré : molécules entièrement conjuguées.
#   - max_sp3 = 3   → jusqu'à 3 atomes sp3 tolérés (un peu de souplesse).
#   - max_sp3 = None → filtre DÉSACTIVÉ (valeur par défaut) : tout passe.
#
# Le filtre est défini dans scripts/Lea/huckel_filters.py (FilterSP3).


# ==============================================================================
#  CONSTRUCTION DU SCORE (fitness) À PARTIR DES OBJECTIFS
# ==============================================================================
def _build_fitness(
    goals: list[tuple[str, str, float, float | None]],
) -> evaluator.Function:
    """Transforme la liste d'objectifs en UN seul score que l'algo maximise.

    Rappel : l'algorithme maximise toujours. Pour cela :
      - "max"    : on ajoute    poids × valeur        (grande valeur → bon score)
      - "min"    : on soustrait poids × valeur        (petite valeur → bon score)
      - "target" : on soustrait poids × |valeur − cible|  (proche de la cible → bon score)

    ⚠ POINT D'ATTENTION — PAS DE NORMALISATION
      Les valeurs sont utilisées brutes : les propriétés n'ont pas la même
      échelle (λ Hückel ~ 0.1–1, SAScore 1–10, n_heavy_atoms ~ 10–40,
      mol_weight ~ 100–500 g/mol…). À poids égaux, une propriété à grande
      échelle écrase les autres (ex. mol_weight noie λ HOMO). Pour l'instant,
      compenser à la main avec les poids, ou utiliser les variantes déjà
      normalisées (NormalizedSAScore, zinc_normalized_logP, …).
      À faire : normaliser chaque terme (échelle fixe par propriété, min-max,
      ou échelle donnée dans le tuple).
    """
    def _fitness(mol: Molecule) -> float:
        total = 0.0
        for name, mode, weight, target in goals:
            value = mol.value(name)  # lit la propriété (ex. le gap) de la molécule
            if mode == "max":
                total += weight * value
            elif mode == "min":
                total -= weight * value
            elif mode == "target":
                total -= weight * abs(value - target)
            else:
                raise ValueError(
                    f"Mode inconnu : {mode!r}  (utiliser 'max', 'min' ou 'target')"
                )
        return total

    return evaluator.Function(name="fitness", function=_fitness)


def _n_heavy_atoms(mol: Molecule) -> int:
    """Nombre d'atomes lourds (hors hydrogènes)."""
    return mol.get_representation(MolecularGraph).mol.GetNumHeavyAtoms()


def _mol_weight(mol: Molecule) -> float:
    """Masse molaire (g/mol), hydrogènes implicites compris."""
    return Descriptors.MolWt(mol.get_representation(MolecularGraph).mol)


_ORIGINAL_BUILD_FITNESS = _build_fitness


# ==============================================================================
#  ESPACE DE RECHERCHE ET CHAÎNE D'ÉVALUATION (aussi lancé dans chaque processus
#  quand n_workers > 1)
# ==============================================================================
def _fitness_builder(curve_aggregation: str | None):
    """Fonction qui construit la fitness à partir de fitness_goals.

    None : _build_fitness de ce script (objectifs (nom, mode, poids, cible)).
    "sum" / "geometric" : courbes de désirabilité de fitness_functions.py
    (objectifs (nom, courbe, poids)), avec cette agrégation.
    """
    if curve_aggregation is None:
        return _build_fitness
    import fitness_functions as ff  # pylint: disable=import-outside-toplevel
    return lambda goals: ff.build_fitness(goals, curve_aggregation)


# pylint: disable=too-many-arguments, too-many-locals
def _setup_search(
    accepted_atoms, max_heavy_atoms, unique_kekule, with_add_group, with_remove_group,
    insert_carbon, avoid_bond_breaking, avoid_bond_forming, avoid_sp3_on_sp3,
    add_atom_multiple_bonds, huckel, max_sp3, realism, hetero, max_ring_n,
    orbital_lambda, lambda_overlap_orbitals, orbital_lambdas, lambda_overlaps,
    fitness_goals, build_fitness,
) -> list:
    """Configure EvoMol (atomes, actions) et renvoie la chaîne d'évaluation."""
    # --- Espace de recherche EvoMol (usage habituel) ------------------------
    dp.setup_default_parameters(
        accepted_atoms=accepted_atoms,
        max_heavy_atoms=max_heavy_atoms,
        unique_kekule=unique_kekule,
    )
    dp.setup_default_action_space(
        with_add_group=with_add_group,
        with_remove_group=with_remove_group,
        insert_carbon=insert_carbon,
        avoid_bond_breaking=avoid_bond_breaking,
        avoid_bond_forming=avoid_bond_forming,
        avoid_sp3_on_sp3=avoid_sp3_on_sp3,
        add_atom_multiple_bonds=add_atom_multiple_bonds,
    )

    # --- Chaîne d'évaluation ------------------------------------------------
    # Partie spécifique à Hückel. Point important : Huckel() DOIT précéder
    # HuckelGap/LUMO/HOMO (ils relisent le résultat mis en cache par Huckel()),
    # et _build_fitness DOIT rester en dernier (il lit les propriétés ci-dessus).
    filters = hf.setup_huckel_filters(
        huckel=huckel, max_sp3=max_sp3, realism=realism,
        hetero=hetero, max_ring_n=max_ring_n,
    )
    orbital_evaluations = [
        # ──────────────────────────────────────────────────────────────────
        #  CATALOGUE COMPLET DES ÉVALUATEURS HÜCKEL (orbitales choisies via
        #  les paramètres de main())
        #  Seuls ceux dont la propriété apparaît dans fitness_goals sont
        #  calculés : une orbitale inexistante (ex. HOMO-4 sur le benzène)
        #  dans un évaluateur non utilisé ne rend pas la molécule invalide.
        # ──────────────────────────────────────────────────────────────────
        #  Notations : λₖ = valeur propre de l'orbitale k ;  cₖᵢ = coefficient
        #  de l'orbitale k sur l'atome i (composante du vecteur propre).
        #  Un "sélecteur" d'orbitale vaut : "HOMO", "LUMO", "HOMO-1", "LUMO+2",
        #  … ou un indice entier.  Chaque évaluateur crée la propriété indiquée,
        #  utilisable ensuite dans fitness_goals.
        #
        #  • λ d'une orbitale quelconque (orbital_lambda)        →  λₖ
        #    propriété : "huckel_orbital_lambda_<orb>"  (ex. "..._HOMO-1")
        evaluator.HuckelOrbitalLambda(orbital_lambda),
        #
        #  • Recouvrement spatial, ignore le signe (lambda_overlap_orbitals, 2 ou plus)
        #    formule  :  Σᵢ  ∏ⱼ |c_(selⱼ)ᵢ|   (ex. |c_HOMOᵢ| · |c_LUMOᵢ|)
        #    propriété : "huckel_lambda_<orb1>_<orb2>…"  (ex. "huckel_lambda_HOMO_LUMO")
        evaluator.HuckelOrbitalLambdaOverlap(list(lambda_overlap_orbitals)),
        #
        #  • Mêmes évaluateurs pour plusieurs orbitales / recouvrements à la fois
        #    (orbital_lambdas, lambda_overlaps)
        *(evaluator.HuckelOrbitalLambda(o) for o in orbital_lambdas or []),
        *(evaluator.HuckelOrbitalLambdaOverlap(list(sel)) for sel in lambda_overlaps or []),
        # ──────────────────────────────────────────────────────────────────
    ]
    # un même évaluateur ne doit être lancé qu'une fois (ex. orbital_lambda répété dans orbital_lambdas)
    orbital_evaluations = list({e.name: e for e in orbital_evaluations}.values())
    goal_names = {name for name, *_ in fitness_goals}

    # Autres propriétés (taille, SA score, …) : propriété → évaluateurs à lancer,
    # dépendances d'abord. Seules celles présentes dans fitness_goals sont calculées.
    other_evaluations = {
        # Taille : nb d'atomes lourds (hors H) / masse molaire (g/mol)
        "n_heavy_atoms": [evaluator.Function("n_heavy_atoms", _n_heavy_atoms)],
        "mol_weight": [evaluator.Function("mol_weight", _mol_weight)],
        # SA score : 1 (facile à synthétiser) → 10 (difficile)
        "SAScore": [evaluator.SAScore],
        # SA score normalisé : 1 (facile) → 0 (difficile)
        "NormalizedSAScore": [evaluator.SAScore, evaluator.NormalizedSAScore],
        # SA score normalisé sur ZINC (grand = facile)
        "zinc_normalized_sa_score": [evaluator.SAScore, evaluator.ZincNormalizedSAScore],
        # QED : « drug-likeness », 0 → 1
        "QED": [evaluator.QED],
        # logP (lipophilie) et sa version normalisée sur ZINC
        "logP": [evaluator.LogP],
        "zinc_normalized_logP": [evaluator.LogP, evaluator.ZincNormalizedLogP],
        # Score de cycle : −(taille du plus grand cycle − 6), 0 si cycles ≤ 6
        "CycleScore": [evaluator.CycleScore],
        "zinc_normalized_cycle_score": [evaluator.CycleScore, evaluator.NormalizedCycleScore],
        # logP pénalisé = logP + SA + cycles (normalisés ZINC)
        "PLogP": [
            evaluator.LogP, evaluator.ZincNormalizedLogP,
            evaluator.SAScore, evaluator.ZincNormalizedSAScore,
            evaluator.CycleScore, evaluator.NormalizedCycleScore,
            evaluator.PLogP,
        ],
    }
    used_other_evaluations = []
    for name, *_ in fitness_goals:
        for e in other_evaluations.get(name, []):
            if e not in used_other_evaluations:
                used_other_evaluations.append(e)

    evaluations = [
        # Filtres (voir FILTRES plus haut)
        *filters,

        # Calcul de Hückel et propriétes optimisables
        evaluator.Huckel(),
        evaluator.HuckelGap(),   # crée la propriété "huckel_gap_lambda"
        evaluator.HuckelLUMO(),  # crée la propriété "huckel_lumo_lambda"
        evaluator.HuckelHOMO(),  # crée la propriété "huckel_homo_lambda"

        # Évaluateurs orbitalaires utilisés dans fitness_goals
        *(e for e in orbital_evaluations if e.name in goal_names),

        # Autres propriétés utilisées dans fitness_goals (taille, SA score, …)
        *used_other_evaluations,

        # Score final (somme pondérée des objectifs) → propriété "fitness"
        build_fitness(fitness_goals),
    ]
    search_parameters.fitness_criteria = "fitness"
    return evaluations


def _worker_setup(setup_kwargs: dict, curve_aggregation: str | None) -> list:
    """Lancé une fois dans chaque processus de calcul (n_workers > 1)."""
    from rdkit import RDLogger  # pylint: disable=import-outside-toplevel

    RDLogger.DisableLog("rdApp.*")
    os.chdir(_ROOT)  # les filtres ChEMBL/ZINC lisent external_data/ en relatif
    return _setup_search(**setup_kwargs, build_fitness=_fitness_builder(curve_aggregation))


# ==============================================================================
#  FONCTION PRINCIPALE — lance une expérience
# ==============================================================================
def main(
    start_smiles: str = "c1ccccc1",            # molécule de départ (notation SMILES)
    accepted_atoms: list[str] | None = None,   # atomes autorisés, ex. ["C", "N", "O"]
    max_heavy_atoms: int = 100,                # taille max (atomes hors hydrogènes)
    unique_kekule: bool = False,               # True : un seul SMILES par molécule aromatique (plus lent)
    huckel: bool = False,                      # filtre HuckelCompatible (voir FILTRES)
    max_sp3: int | None = None,                # nb max d'atomes sp3 autorisés (None = inactif)
    realism: str | None = "chembl",            # "chembl_zinc", "chembl" ou None
    hetero: bool = False,                      # filtre HeteroatomChainFilter
    max_ring_n: int = 3,                       # nb max d'azotes en cycle (si hetero)
    orbital_lambda: str | int = "HOMO-1",      # orbitale dont on lit λ (voir CATALOGUE)
    lambda_overlap_orbitals: tuple = ("HOMO", "LUMO"),  # 2+ orbitales : recouvrement spatial
    orbital_lambdas: list | None = None,       # orbitales supplémentaires dont on lit λ (None : aucune)
    lambda_overlaps: list[tuple] | None = None,  # recouvrements supplémentaires (None : aucun)
    fitness_goals: list[tuple[str, str, float, float | None]] | None = None,
    max_generations: int = 1000,               # nb max de générations
    max_time: float = 3600,                    # temps max en secondes
    population_size: int = 100,                  # taille de la population (remplie depuis start_smiles)
    nb_replacements: int = 10,                  # nb de molécules remplacées par génération
    selection: str = "best",                   # "best", "random" ou "roulette"
    depth: int = 2,                            # nb (max) d'actions par mutation
    random_depth: bool = True,                 # True : 1 à depth actions ; False : exactement depth
    nb_max_tries: int = 50,                    # essais de mutation par parent
    remember_invalid: bool = True,             # ne pas réévaluer un SMILES déjà jugé invalide
    preselect_actions: bool = True,            # tirer le type d'action avant de lister les actions
    uniform_over_actions: bool = False,        # tirage uniforme sur toutes les actions
    insert_carbon: bool = True,                # garder InsertCarbonMG
    avoid_sp3_on_sp3: bool = False,            # AddAtomMG ne greffe pas de C sp3 sur un C sp3
    add_atom_multiple_bonds: bool = False,     # AddAtomMG peut aussi greffer par double/triple liaison
    avoid_bond_breaking: bool = False,         # ChangeBondMG ne supprime jamais de liaison
    avoid_bond_forming: bool = False,          # ChangeBondMG ne crée jamais de liaison
    with_add_group: bool = False,              # ajouter AddGroupMG
    with_remove_group: bool = False,           # ajouter RemoveGroupMG
    random_seed: int | None = 42,              # entier : reproductible ; None : différent à chaque run
    n_workers: int = 1,                        # nb de processus qui cherchent les remplacements (1 : algorithme d'origine)
    curve_aggregation: str | None = None,      # fitness par courbes (fitness_functions) : "sum" / "geometric" ; None : _build_fitness
    generation_callback=None,                  # fonction appelée pour l'affichage live
    callback_every_n: int = 1,                 # appeler le callback toutes les N gén.
    output_dir: str | None = None,             # dossier où sauver les résultats
    save_n_generations: int = 50,              # sauver la population toutes les N gén. (0 : jamais)
    save_final_population: bool = True,        # sauver la population finale
) -> EvolutionaryAlgorithm:
    """Lance l'algorithme évolutionnaire et renvoie l'objet résultat.

    L'objet renvoyé contient notamment :
      - .nb_generations : nombre de générations effectuées
      - .population      : les meilleures molécules trouvées à la fin
    """
    # --- Valeurs par défaut -------------------------------------------------
    if accepted_atoms is None:
        accepted_atoms = ["C", "O", "N"]  # atomes autorisés par défaut
    if fitness_goals is None:
        fitness_goals = FITNESS_GOALS

    # Les processus de calcul ne voient pas un _build_fitness remplacé à la volée :
    # passer la fitness par courbes avec curve_aggregation.
    if n_workers > 1 and curve_aggregation is None and _build_fitness is not _ORIGINAL_BUILD_FITNESS:
        raise ValueError(
            "_build_fitness a été remplacé : avec n_workers > 1, passer curve_aggregation "
            "(\"sum\" ou \"geometric\") au lieu de remplacer _build_fitness."
        )

    _root = _ROOT
    os.chdir(_root)  # les filtres ChEMBL/ZINC lisent external_data/ en relatif
    if output_dir is None:
        output_dir = os.path.join(_root, "output", "huckel_homo")

    logger = init_logger(
        config_file=os.path.join(_root, "logging_configs", "stdout-json-file.json"),
        output_file=os.path.join(_root, "logs", "evomol_huckel_homo.log"),
    )

    logger.info("Init the evolutionary algorithm.", extra={"random_seed": random_seed})
    random.seed(random_seed)

    # --- Chaîne d'évaluation (voir _setup_search) -------------------------
    setup_kwargs = dict(
        accepted_atoms=accepted_atoms, max_heavy_atoms=max_heavy_atoms,
        unique_kekule=unique_kekule, with_add_group=with_add_group,
        with_remove_group=with_remove_group, insert_carbon=insert_carbon,
        avoid_bond_breaking=avoid_bond_breaking, avoid_bond_forming=avoid_bond_forming,
        avoid_sp3_on_sp3=avoid_sp3_on_sp3, add_atom_multiple_bonds=add_atom_multiple_bonds,
        huckel=huckel, max_sp3=max_sp3, realism=realism, hetero=hetero, max_ring_n=max_ring_n,
        orbital_lambda=orbital_lambda, lambda_overlap_orbitals=lambda_overlap_orbitals,
        orbital_lambdas=orbital_lambdas, lambda_overlaps=lambda_overlaps,
        fitness_goals=fitness_goals,
    )
    evaluations = _setup_search(**setup_kwargs, build_fitness=_fitness_builder(curve_aggregation))

    # --- Lancement EvoMol (usage habituel) ----------------------------------
    start = Molecule(start_smiles)
    population = [Molecule(start.get_representation(MolecularGraph).canonical_smiles)]

    # n_workers > 1 : les remplacements d'une génération sont cherchés en même
    # temps sur plusieurs processus (une seule population, voir
    # scripts/Lea/parallel_evolutionary_algorithm.py)
    parallel = (
        {"n_workers": n_workers, "worker_setup": _worker_setup,
         "worker_setup_args": (setup_kwargs, curve_aggregation)}
        if n_workers > 1 else {}
    )
    evo_mol = (ParallelEvolutionaryAlgorithm if n_workers > 1 else EvolutionaryAlgorithm)(
        **parallel,
        population=population,
        parameters=ParametersEvolutionaryAlgorithm(
            size=population_size,
            max_generations=max_generations,
            max_time=max_time,
            nb_replacements=nb_replacements,
            output_dir=output_dir,
            save_n_generations=save_n_generations,
            save_final_population=save_final_population,
        ),
        selection_strategy={
            "best": selection_strategy.BestSelection,
            "random": selection_strategy.RandomSelection,
            "roulette": selection_strategy.RouletteSelection,
        }[selection](),
        evaluations=evaluations,
        neighborhood_strategy=RandomNeighborhoodStrategy(
            depth=depth,
            nb_max_tries=nb_max_tries,
            preselect_actions=preselect_actions,
            random_depth=random_depth,
            uniform_over_actions=uniform_over_actions,
            remember_invalid=remember_invalid,
        ),
        generation_callback=generation_callback,
        callback_every_n=callback_every_n,
        # valeurs des objectifs de la meilleure molécule dans la ligne "Generation"
        log_properties=[name for name, *_ in fitness_goals],
    )

    evo_mol.run()
    return evo_mol


# ==============================================================================
#  POINT D'ENTRÉE quand on lance le script directement (python ce_fichier.py)
# ==============================================================================
if __name__ == "__main__":
    # Exemple minimal : modifier les arguments ci-dessous pour tester rapidement.
    main(
        start_smiles="c1ccccc1",          # benzène
        accepted_atoms=["C", "N", "O"],
        max_heavy_atoms=30,
        max_sp3=None,                     # max d'atomes sp3 tolérés (None = filtre
                                          # désactivé ; 0 = molécules 100% conjuguées)
        fitness_goals=[("huckel_gap_lambda", "min", 1.0, None)],  # gap le plus petit
        max_generations=1000,
        max_time=60,
    )
