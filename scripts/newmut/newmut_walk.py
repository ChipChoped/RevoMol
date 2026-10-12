"""
Nouvelles mutations (newmut) : essai à la main et marches aléatoires
(notebooks/random_walk/newmut_walk.ipynb).

Même principe que scripts/Lea/pi_walk.py, dont ce module reprend les marches, les fichiers et les figures sans le
modifier : seule la configuration change. Elle accepte en plus les options de
evomol.newmut.setup_newmut_action_space (règles matricielles R1-R4 et nouvelles mutations).

Essai à la main, sur l'espace d'action configuré par configure(CONFIG) :
  - show_molecule(smiles) : dessin avec les numéros d'atomes ;
  - matrix_table(smiles), violations_table(smiles) : matrice (électrons de valence sur la diagonale, ordres de
    liaison ailleurs, H, étiquette s / a / d / t, hybridation) et règles non respectées ;
  - mutation_summary(smiles) : nombre de mutations candidates, permises et refusées par type ;
  - mutation_table(smiles, type) / show_mutations(smiles, type) : toutes les mutations d'un type, en tableau ou en
    dessins (vert : permise, rouge : refusée par les règles, avec la règle) ;
  - apply_mutation(smiles, type, n) : applique la mutation n et montre ce qui a changé ; renvoie le nouveau SMILES ;
  - random_mutations(smiles, ...) : tirage au hasard comme l'algorithme évolutionnaire.
"""

from __future__ import annotations

import copy
import os
import random
import re
import sys
from collections import Counter
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts", "Lea"))  # pi_walk, huckel_filters, random_walk_core

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import Draw

import huckel_filters as hf
import pi_walk as pw  # se place à la racine du projet (filtres ChEMBL/ZINC en relatif)
from evomol import default_parameters as dp
from evomol.action import Action
from evomol.evaluation import Evaluation
from evomol.newmut import matrix_rules as mr
from evomol.newmut import rule_filtered as rf
from evomol.newmut.default_parameters import setup_newmut_action_space
from evomol.newmut.neighborhood_strategy import draw_allowed_action
from evomol.representation import MolecularGraph, Molecule

# lecture et figures des marches : celles de pi_walk
from pi_walk import (  # noqa: F401  pylint: disable=unused-import
    VARIABLES,
    autocorrelation,
    autocorrelation_summary,
    load_meta,
    load_walks,
    mutation_counts,
    plot_autocorrelation,
    plot_counts,
    plot_mean,
    plot_pairs,
    plot_walks,
    show_steps,
    write_counts,
)

DEFAULT_CONFIG: dict[str, Any] = {
    **copy.deepcopy(pw.DEFAULT_CONFIG),
    "ACCEPTED_ATOMS": ["C", "O", "N"],
    # ── nouvelles mutations (evomol/newmut)
    "MATRIX_RULES": True,          # les mutations d'evomol respectent les règles R1-R4
    "APPEND_ATOM_RULE": True,      # AppendAtomRuleMG remplace AddAtomMG
    "SUBSTITUTION_SHIFT": True,    # SubstituteShiftMG remplace SubstituteAtomMG
    "GRAFT": True,                 # GraftFragmentMG
    "ANNELATE": True,              # AnnelateMG
    "FLIP_PATH": True,             # FlipAlternatingPathMG
    "AROMATIZE": True,             # AromatizeMG
    "GRAFT_FRAGMENTS": None,       # None : vocabulaire par défaut (evomol/newmut/fragments.py)
    "ANNELATION_FRAGMENTS": None,  # None : vocabulaire par défaut
}

# couleurs des atomes modifiés dans les dessins
GREEN, RED = (0.65, 0.9, 0.65), (1.0, 0.65, 0.65)


def make_config(**overrides: Any) -> dict[str, Any]:
    """DEFAULT_CONFIG modifiée par overrides (une clé inconnue est une erreur)."""
    unknown = set(overrides) - set(DEFAULT_CONFIG)
    if unknown:
        raise KeyError(f"Réglages inconnus : {sorted(unknown)}")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(copy.deepcopy(overrides))
    return config


_FILTERS: dict[tuple, list[Evaluation]] = {}


def action_classes(names: list[str] | None) -> list[type[Action]]:
    """Types de mutation de l'espace d'action configuré, par nom (None : tous)."""
    space = {action.class_name(): action for action in MolecularGraph.action_space}
    if names is None:
        return list(space.values())
    unknown = [name for name in names if name not in space]
    if unknown:
        raise KeyError(f"Mutations absentes de l'espace d'action : {unknown}. Disponibles : {sorted(space)}")
    return [space[name] for name in names]


def configure(config: dict[str, Any]) -> tuple[list[Evaluation], dict[str, Any]]:
    """Configure evomol pour cette configuration ; renvoie (filtres, options de random_walk).
    L'espace d'action est global au processus : à rappeler avant chaque marche."""
    dp.setup_default_parameters(accepted_atoms=list(config["ACCEPTED_ATOMS"]), max_heavy_atoms=config["MAX_ATOMS"],
                                unique_kekule=config["UNIQUE_KEKULE"])
    setup_newmut_action_space(
        insert_carbon=config["INSERT_CARBON"], avoid_sp3_on_sp3=config["AVOID_SP3_ON_SP3"],
        avoid_bond_breaking=config["AVOID_BOND_BREAKING"], avoid_bond_forming=config["AVOID_BOND_FORMING"],
        with_add_group=config["WITH_ADD_GROUP"], with_remove_group=config["WITH_REMOVE_GROUP"],
        add_atom_multiple_bonds=config["ADD_ATOM_MULTIPLE_BONDS"],
        bond_change_by_one=config["BOND_CHANGE_BY_ONE"],
        matrix_rules=config["MATRIX_RULES"], append_atom_rule=config["APPEND_ATOM_RULE"],
        substitution_shift=config["SUBSTITUTION_SHIFT"], graft=config["GRAFT"], annelate=config["ANNELATE"],
        flip_path=config["FLIP_PATH"], aromatize=config["AROMATIZE"],
        graft_fragments=config["GRAFT_FRAGMENTS"], annelation_fragments=config["ANNELATION_FRAGMENTS"],
    )
    key = (config["FILTER_HUCKEL"], config["MAX_SP3"], config["FILTER_REALISM"],
           config["FILTER_HETERO"], config["MAX_RING_N"])
    if key not in _FILTERS:  # les filtres ChEMBL/ZINC chargent des fichiers : une fois par processus
        _FILTERS[key] = hf.setup_huckel_filters(
            huckel=config["FILTER_HUCKEL"], max_sp3=config["MAX_SP3"], realism=config["FILTER_REALISM"],
            hetero=config["FILTER_HETERO"], max_ring_n=config["MAX_RING_N"],
        )
    selected = action_classes(config["ACTION_TYPES"]) if config["ACTION_TYPES"] else None
    walk_kwargs = {
        "only_valid": config["ONLY_VALID"],
        "count_valid_neighbors": config["COUNT_VALID_NEIGHBORS"],
        "exclude_current": config["EXCLUDE_CURRENT"],
        # random_walk compare le type des actions listées : une mutation enveloppée par les règles liste des
        # actions de la classe d'origine
        "action_types": [getattr(a, "inner_class", a) for a in selected] if selected else None,
    }
    return _FILTERS[key], walk_kwargs


# ==============================================================================
#  MARCHES ALÉATOIRES (celles de pi_walk, avec cette configuration)
# ==============================================================================
_PI_WALK_RUN_ONE = pw._run_one  # pylint: disable=protected-access


def _run_one(task: tuple[dict[str, Any], int, str, Any]) -> int:
    """Une marche dans un processus de calcul : celle de pi_walk, configurée par ce module."""
    pw.configure = configure  # processus dédié aux marches newmut
    return _PI_WALK_RUN_ONE(task)


def run_walks(config: dict[str, Any], n_walks: int, out_dir: str, n_jobs: int | None = None) -> None:
    """pi_walk.run_walks avec la configuration de ce module (règles et nouvelles mutations) : n_walks marches en
    parallèle, fichiers dans out_dir (relatif à la racine du projet), qui sont écrasés à chaque lancement."""
    pw._run_one = _run_one  # pylint: disable=protected-access
    try:
        pw.run_walks(config, n_walks, out_dir, n_jobs)
    finally:
        pw._run_one = _PI_WALK_RUN_ONE  # pylint: disable=protected-access


# ==============================================================================
#  ESSAI À LA MAIN
# ==============================================================================
def _graph(smiles: str) -> MolecularGraph:
    return Molecule(smiles).get_representation(MolecularGraph)


def _label(mol_graph: MolecularGraph, atom: int) -> str:
    return f"{mol_graph.atom_type(atom)}{atom}"


def _options(indices: bool = True) -> Draw.MolDrawOptions:
    options = Draw.MolDrawOptions()
    options.addAtomIndices = indices
    return options


def show_molecule(smiles: str, size: tuple[int, int] = (420, 320)) -> Any:
    """Dessin de la molécule avec les numéros d'atomes (ceux des mutations)."""
    mol_graph = _graph(smiles)
    violations = mr.molecular_graph_violations(mol_graph)
    legend = f"{mol_graph.canonical_smiles}   ({len(violations)} règle(s) non respectée(s))"
    return Draw.MolsToGridImage([Chem.Mol(mol_graph.mol)], molsPerRow=1, subImgSize=size, legends=[legend],
                                drawOptions=_options())


def matrix_table(smiles: str) -> pd.DataFrame:
    """Matrice de la molécule : électrons de valence sur la diagonale, ordres de liaison ailleurs ; puis H,
    étiquette du carbone (s / a / d / t) et hybridation (C, N, O)."""
    mol_graph = _graph(smiles)
    types, matrix = mr.to_matrix(mol_graph)
    labels = [f"{element}{i}" for i, element in enumerate(types)]
    table = pd.DataFrame(matrix, index=labels, columns=labels)
    hydrogens = [mr.implicit_h(element, int(matrix[i].sum() - matrix[i, i])) for i, element in enumerate(types)]
    table["H"] = hydrogens
    table["étiquette"] = [mr.carbon_tag(matrix, types, i) for i in range(len(types))]
    table["hybridation"] = [mr.hybridization(matrix, types, i, hydrogens[i]) for i in range(len(types))]
    return table


def violations_table(smiles: str) -> pd.DataFrame:
    """Règles non respectées par la molécule (vide : molécule dans les règles)."""
    mol_graph = _graph(smiles)
    return pd.DataFrame(
        [{"règle": v.rule_id, "atomes": ", ".join(_label(mol_graph, a) for a in v.atoms), "message": v.message}
         for v in mr.molecular_graph_violations(mol_graph)],
        columns=["règle", "atomes", "message"],
    )


def describe(action: Action) -> str:
    """Description courte d'une mutation, avec les numéros d'atomes de la molécule de départ."""
    mol_graph = action.molecule.get_representation(MolecularGraph)
    lab = lambda atom: _label(mol_graph, atom)  # noqa: E731
    bond = {1: "", 2: "=", 3: "#"}
    name = action.class_name()
    a: Any = action
    if name in ("AddAtomMG", "AppendAtomRuleMG"):
        if mol_graph.nb_atoms == 0:
            return f"atome {a.atom_type}"
        return f"+{bond[a.bond_order]}{a.atom_type} sur {lab(a.index_atom)}"
    if name == "ChangeBondMG":
        return f"liaison {lab(a.atom1)}–{lab(a.atom2)} : {mol_graph.bond_order(a.atom1, a.atom2)} → {a.bond_type}"
    if name == "SubstituteAtomMG":
        return f"{lab(a.atom_idx)} → {a.new_type}"
    if name == "SubstituteShiftMG":
        shift = f" (+ {len(a.bond_changes)} liaison(s) déplacée(s))" if a.bond_changes else ""
        return f"{lab(a.atom_idx)} → {a.new_type}{shift}"
    if name == "RemoveAtomMG":
        return f"retrait de {lab(a.atom_idx)}"
    if name == "CutAtomMG":
        return f"excision de {lab(a.atom_to_cut)}, {lab(a.atom1_to_bond)}–{lab(a.atom2_to_bond)} ordre {a.bond_type}"
    if name == "InsertCarbonMG":
        return f"C inséré dans {lab(a.atom1_idx_to_bond)}–{lab(a.atom2_idx_to_bond)}"
    if name == "MoveGroupMG":
        return (f"groupe de {lab(a.atom_moving)} détaché de {lab(a.atom_staying)}, "
                f"relié à {lab(a.atom_to_link)}")
    if name == "GraftFragmentMG":
        return f"greffe {a.name} sur {lab(a.anchor)}"
    if name == "AnnelateMG":
        return f"annélation {a.name} : [*:1] sur {lab(a.atom1)}, [*:2] sur {lab(a.atom2)}"
    if name == "FlipAlternatingPathMG":
        return f"inversion du chemin {'–'.join(lab(x) for x in a.path)}"
    if name == "AromatizeMG":
        return f"aromatisation du système de {lab(a.atom_idx)} ({len(a.bond_changes)} liaison(s) changée(s))"
    # autres mutations : leur repr sans la molécule
    match = re.match(r"^(\w+)\([^,]*,? ?(.*)\)$", repr(action))
    return f"{match.group(1)} {match.group(2)}" if match else repr(action)


def _candidates(action_class: type[Action], molecule: Molecule) -> list[Action]:
    """Mutations candidates : avant les règles pour une mutation qui les applique, sinon list_actions."""
    if getattr(action_class, "checks_rules", False):
        return action_class.list_unchecked_actions(molecule)  # type: ignore[attr-defined]
    return action_class.list_actions(molecule)


def _new_violations(before: MolecularGraph, after: MolecularGraph) -> str:
    """Règles dont le nombre de violations augmente, ex. « R2 +1 »."""
    count_before = Counter(v.rule_id for v in mr.molecular_graph_violations(before))
    count_after = Counter(v.rule_id for v in mr.molecular_graph_violations(after))
    return ", ".join(f"{rule} +{count_after[rule] - count_before[rule]}"
                     for rule in sorted(count_after) if count_after[rule] > count_before[rule])


def _changed_atoms(before: MolecularGraph, after: MolecularGraph) -> list[int]:
    """Atomes du résultat (numéros conservés) dont la ligne de la matrice a changé, et atomes ajoutés. Vide si
    des atomes ont été retirés (numéros décalés)."""
    types_b, matrix_b = mr.to_matrix(before)
    types_a, matrix_a = mr.to_matrix(after)
    n_b, n_a = len(types_b), len(types_a)
    if n_a < n_b:
        return []
    changed = [i for i in range(n_b) if types_a[i] != types_b[i] or not np.array_equal(matrix_a[i, :n_b], matrix_b[i])]
    return changed + list(range(n_b, n_a))


def _rows(smiles: str, action_type: str) -> tuple[Molecule, list[dict[str, Any]]]:
    """Toutes les mutations candidates d'un type, avec leur résultat et s'il est permis."""
    (action_class,) = action_classes([action_type])
    molecule = Molecule(smiles)
    before = molecule.get_representation(MolecularGraph)
    checks = getattr(action_class, "checks_rules", False)
    rows = []
    for k, action in enumerate(_candidates(action_class, molecule)):
        try:
            after = rf.result_graph(action, before)  # type: ignore[arg-type]
        except Exception as exc:  # pylint: disable=broad-except
            rows.append({"n°": k, "action": action, "mutation": describe(action), "résultat": None, "graph": None,
                         "permise": False, "raison": f"erreur : {exc!r}"})
            continue
        allowed = action_class.action_is_allowed(action, molecule) if checks else True  # type: ignore[attr-defined]
        rows.append({"n°": k, "action": action, "mutation": describe(action),
                     "résultat": after.aromatic_canonical_smiles, "graph": after, "permise": allowed,
                     "raison": "" if allowed else _new_violations(before, after) or "règles"})
    return molecule, rows


def mutation_table(smiles: str, action_type: str, only_allowed: bool = False) -> pd.DataFrame:
    """Toutes les mutations d'un type sur la molécule : description (numéros de la molécule de départ), résultat,
    permise ou non, et la règle qui la refuse. Le n° sert à apply_mutation."""
    _, rows = _rows(smiles, action_type)
    table = pd.DataFrame(rows, columns=["n°", "mutation", "résultat", "permise", "raison"])
    return table[table["permise"]] if only_allowed else table


def mutation_summary(smiles: str) -> pd.DataFrame:
    """Par type de mutation de l'espace d'action : nombre de mutations candidates, permises, refusées, et de
    molécules différentes produites par les mutations permises."""
    summary = []
    for action_class in MolecularGraph.action_space:
        _, rows = _rows(smiles, action_class.class_name())
        allowed = [row for row in rows if row["permise"]]
        summary.append({"mutation": action_class.class_name(), "candidates": len(rows), "permises": len(allowed),
                        "refusées": len(rows) - len(allowed),
                        "molécules différentes": len({row["résultat"] for row in allowed})})
    return pd.DataFrame(summary).set_index("mutation")


def show_mutations(smiles: str, action_type: str, only_allowed: bool = False, per_row: int = 4,
                   size: tuple[int, int] = (300, 240), max_images: int = 60) -> Any:
    """Dessins des molécules produites par les mutations d'un type. Atomes modifiés en vert si la mutation est
    permise, en rouge si les règles la refusent (règle dans la légende). Numéros d'atomes : ceux de la molécule de
    départ (les atomes ajoutés prennent les numéros suivants)."""
    molecule, rows = _rows(smiles, action_type)
    rows = [row for row in rows if row["graph"] is not None and (row["permise"] or not only_allowed)]
    if len(rows) > max_images:
        print(f"{len(rows)} mutations : seules les {max_images} premières sont dessinées (max_images)")
        rows = rows[:max_images]
    if not rows:
        print(f"aucune mutation {action_type} sur {molecule}")
        return None
    before = molecule.get_representation(MolecularGraph)
    mols, legends, highlights, colors = [], [], [], []
    for row in rows:
        changed = _changed_atoms(before, row["graph"])
        color = GREEN if row["permise"] else RED
        mols.append(Chem.Mol(row["graph"].mol))
        legends.append(f"n°{row['n°']} : {row['mutation']}\n"
                       + ("permise" if row["permise"] else f"refusée ({row['raison']})"))
        highlights.append(changed)
        colors.append({atom: color for atom in changed})
    return Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=size, legends=legends,
                                highlightAtomLists=highlights, highlightAtomColors=colors, drawOptions=_options())


def apply_mutation(smiles: str, action_type: str, number: int, size: tuple[int, int] = (420, 320)) -> str:
    """Applique la mutation n° number de mutation_table(smiles, action_type) et montre avant / après avec les
    atomes modifiés et les cases de la matrice qui changent. Renvoie le SMILES de la nouvelle molécule (ses atomes
    sont renumérotés : relancer show_molecule pour les nouveaux numéros)."""
    molecule, rows = _rows(smiles, action_type)
    row = next((r for r in rows if r["n°"] == number), None)
    if row is None or row["graph"] is None:
        raise IndexError(f"pas de mutation {action_type} n°{number} sur {smiles} ({len(rows)} candidates)")
    if not row["permise"]:
        print(f"Attention : mutation refusée par les règles ({row['raison']}), appliquée quand même pour la voir.")
    before = molecule.get_representation(MolecularGraph)
    after = row["graph"]
    changed = _changed_atoms(before, after)

    types_b, matrix_b = mr.to_matrix(before)
    types_a, matrix_a = mr.to_matrix(after)
    print(f"{describe(row['action'])}")
    if len(types_a) >= len(types_b):
        for i in range(len(types_b)):
            if types_a[i] != types_b[i]:
                print(f"  élément {types_b[i]}{i} → {types_a[i]}{i}")
        for i in range(len(types_a)):
            for j in range(i + 1, len(types_a)):
                old = int(matrix_b[i, j]) if j < len(types_b) else 0
                if int(matrix_a[i, j]) != old:
                    print(f"  case ({types_a[i]}{i}, {types_a[j]}{j}) : {old} → {int(matrix_a[i, j])}")
    else:
        print(f"  {len(types_b) - len(types_a)} atome(s) retiré(s) (numéros décalés)")
    new_violations = _new_violations(before, after)
    print(f"  règles : {new_violations or 'aucune nouvelle violation'}")

    new_molecule = row["action"].apply()
    display(Draw.MolsToGridImage(
        [Chem.Mol(before.mol), Chem.Mol(after.mol)], molsPerRow=2, subImgSize=size,
        legends=[f"avant : {before.canonical_smiles}", f"après : {new_molecule}"],
        highlightAtomLists=[[a for a in changed if a < len(types_b)], changed],
        highlightAtomColors=[{a: GREEN for a in changed}, {a: GREEN for a in changed}], drawOptions=_options(),
    ))
    return str(new_molecule)


def random_mutations(smiles: str, n_mutations: int = 1, lazy_rules: bool = False, seed: int | None = None,
                     per_row: int = 4, size: tuple[int, int] = (300, 240)) -> str:
    """Tirage au hasard comme l'algorithme évolutionnaire (NewMutNeighborhoodStrategy) : n_mutations fois, un
    type de mutation au hasard puis une mutation permise de ce type au hasard. lazy_rules : mutations essayées
    une par une (même tirage). Montre chaque étape ; renvoie le SMILES final."""
    rng_state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        molecule = Molecule(smiles)
        mols = [Chem.Mol(molecule.get_representation(MolecularGraph).mol)]
        legends = ["départ"]
        for _ in range(n_mutations):
            action_class = random.choice(MolecularGraph.action_space)
            if lazy_rules:
                action = draw_allowed_action(action_class, molecule)
            else:
                actions = action_class.list_actions(molecule)
                action = random.choice(actions) if actions else None
            if action is None:
                legends[-1] += f"\n(aucune {action_class.class_name()} permise)"
                continue
            text = describe(action)
            molecule = action.apply()
            mols.append(Chem.Mol(molecule.get_representation(MolecularGraph).mol))
            legends.append(f"{action_class.class_name()}\n{text}")
    finally:
        if seed is not None:
            random.setstate(rng_state)
    display(Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=size, legends=legends))
    return str(molecule)
