"""
Marches newmut avec les règles V2 (teste_newmut_V2/Walk_V2/newmut_walk_V2.ipynb).

Les règles ne sont pas recopiées ici : elles sont lues dans parametrage_regles_V2.ipynb (les cellules de code
jusqu'à celle qui crée REGLES, exécutées dans chaque processus de marche). Modifier une règle dans ce notebook et le
sauvegarder suffit : les marches suivantes l'utilisent.

Ce qui change par rapport à scripts/newmut/newmut_walk.py :
  - la vérification des règles d'evomol (matrix_rules.rule_violations, utilisée par toutes les mutations pour savoir
    si une mutation crée une nouvelle violation) est remplacée par REGLES.violations sur un GrapheV2 ; R1 et les
    autres règles sont comptées à part, comme en V1 (COMPTE_SEPARE = ["R1"]) ;
  - le contrôle rapide (fast_rules) est coupé : il est écrit pour les règles V1 (environ 2 fois plus lent, mesuré
    le 2026-10-22 sur 4 marches de 150 pas).
Le reste est celui de newmut_walk (configuration, mutations, fichiers écrits). Les options de règles V1 de la
configuration (RULES_VERSION, R2_HETERO, R2_HETERO_OUTSIDE_RINGS, R4B) sont sans effet. Les valences des mutations
restent celles d'evomol : les mêmes que VALENCES de V2 pour C, N, O, pas pour S (6 dans evomol, 2 en V2).

Option propre à ces marches (pas dans le notebook de paramétrage), désactivée par défaut : BORE_MAX_ORDRE, ordre
de liaison le plus haut permis sur un bore (règle RB, comptée avec les règles autres que R1) : 1 (B- seulement),
2 (B- et B=, pas de B#), None (pas de limite). Elle remplace BORE_SIMPLE (True = BORE_MAX_ORDRE 1), qui reste lu pour
les marches faites avec (test_Regle_V2, 2026-10-22).
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTEBOOK = HERE / "parametrage_regles_V2.ipynb"
for path in (ROOT / "scripts" / "newmut", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# pylint: disable=wrong-import-position
import numpy as np

import newmut_walk as nw  # scripts/newmut/newmut_walk.py
import pi_walk as pw
from evomol.newmut import fast_rules
from evomol.newmut import matrix_rules as mr
from evomol.newmut import rule_filtered as rf

CACHE_SIZE = 20_000  # molécules dont les violations sont gardées en mémoire (par processus)

# réglages ajoutés à ceux de newmut_walk (make_config)
DEFAULT_CONFIG: dict[str, Any] = {
    "BORE_MAX_ORDRE": None,        # ordre de liaison max sur un B (règle RB) : 1 (B-), 2 (B-, B=) ; None : pas de limite
}


def bore_max_order(config: dict[str, Any]) -> int | None:
    """BORE_MAX_ORDRE d'une configuration ; celles d'avant l'option : BORE_SIMPLE = True → 1."""
    if config.get("BORE_MAX_ORDRE") is not None:
        return int(config["BORE_MAX_ORDRE"])
    return 1 if config.get("BORE_SIMPLE", False) else None


def _no_display(*_: Any, **__: Any) -> None:
    pass


def load_rules(notebook: Path = NOTEBOOK) -> Any:
    """Exécute les cellules de code du notebook jusqu'à celle qui crée REGLES (sans rien afficher) ; renvoie
    REGLES (regles_V2.Regles)."""
    cells = json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
    namespace: dict[str, Any] = {"display": _no_display}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), str(notebook), "exec"), namespace)  # pylint: disable=exec-used
        namespace["display"] = _no_display
        if "REGLES" in namespace:
            return namespace["REGLES"]
    raise RuntimeError(f"REGLES introuvable dans {notebook}")


def make_config(**overrides: Any) -> dict[str, Any]:
    """nw.make_config, avec en plus les réglages de DEFAULT_CONFIG (une clé inconnue est une erreur)."""
    own = {key: overrides.pop(key) for key in list(overrides) if key in DEFAULT_CONFIG}
    return {**nw.make_config(**overrides), **DEFAULT_CONFIG, **own}


def rule_rb(max_order: int) -> Any:
    """Règle RB : un bore ne porte pas de liaison d'ordre plus haut que max_order."""
    import regles_V2 as rv  # pylint: disable=import-outside-toplevel

    def RB(g: Any) -> list[Any]:  # pylint: disable=invalid-name
        return [rv.Violation("RB", (int(i), *(j for j in g.adj[i] if g.order[i][j] > max_order)),
                             f"B{i} porte une liaison d'ordre {g.max_order[i]} (au plus {max_order})")
                for i in np.flatnonzero((g.elements == "B") & (g.max_order > max_order))]

    return RB


def walk_rules(bore_max: int | None = None, notebook: Path = NOTEBOOK) -> Any:
    """Règles des marches : REGLES du notebook, plus RB (comptée avec les règles autres que R1) si bore_max n'est pas
    None (ordre de liaison max sur un B)."""
    import regles_V2 as rv  # pylint: disable=import-outside-toplevel

    regles = load_rules(notebook)
    if bore_max is None:
        return regles
    return rv.Regles({**regles.rules, "RB": rule_rb(bore_max)}, regles.compte_separe)


def drop_forbidden_bore(essai: Any) -> Any:
    """Tableau de pr.mutation_test sans les mutations qui donneraient un bore d'ordre de liaison interdit (B# avec
    BORE_MAX_ORDRE = 2) : sur des molécules conformes, ce sont exactement celles qui cassent RB. Elles ne font pas
    partie des mutations possibles (ni candidates, ni refusées)."""
    broken = essai["règles cassées"].fillna("").str.split(", ")
    mask = broken.apply(lambda rules: "RB" in rules)
    kept = essai[~mask].reset_index(drop=True)
    kept.attrs.update(essai.attrs)
    print(f"{int(mask.sum())} mutations qui donneraient un bore interdit (RB) retirées des possibilités")
    return kept


_ORIGINAL_CANDIDATES: dict[str, Any] = {}


def hide_forbidden_bore(bore_max: int | None) -> None:
    """Les mutations qui donneraient un bore d'ordre de liaison plus haut que bore_max (B# avec 2) ne sont plus
    proposées par les outils de mutations_V2.ipynb (pr.candidates pour les tests et les sous-familles,
    nw._candidates pour les essais à la main) : ni candidates, ni refusées. None : retour aux listes d'origine.
    Les numéros de mutation (n°) sont ceux de la liste filtrée."""
    import regles_V2 as rv  # pylint: disable=import-outside-toplevel

    pr = rv.pr
    if not _ORIGINAL_CANDIDATES:
        _ORIGINAL_CANDIDATES.update({"pr": pr.candidates, "nw": nw._candidates})  # pylint: disable=protected-access
    pr.candidates = _ORIGINAL_CANDIDATES["pr"]
    nw._candidates = _ORIGINAL_CANDIDATES["nw"]  # pylint: disable=protected-access
    if bore_max is None:
        return

    def forbidden(matrix: np.ndarray, types: list[str]) -> int:
        orders = np.array(matrix, dtype=np.int64)
        np.fill_diagonal(orders, 0)
        return sum(1 for i, element in enumerate(types) if element == "B" and orders[i].max(initial=0) > bore_max)

    def creates_forbidden(action: Any, mol_graph: Any, types: list[str], matrix: np.ndarray, before: int) -> bool:
        try:
            effect = fast_rules.matrix_effect(action, types, matrix)
            if effect is None:
                new_types, new_matrix = mr.to_matrix(rf.result_graph(action, mol_graph))
            else:
                new_types, new_matrix = effect[0], effect[1]
        except Exception:  # pylint: disable=broad-except
            return False  # mutation impossible : jugée plus loin, comme avant
        return forbidden(new_matrix, list(new_types)) > before

    def keep(molecule: Any, actions: list[Any], get: Any) -> list[Any]:
        from evomol.representation import MolecularGraph  # pylint: disable=import-outside-toplevel

        mol_graph = molecule.get_representation(MolecularGraph)
        types, matrix = mr.to_matrix(mol_graph)
        before = forbidden(matrix, types)
        return [a for a in actions if not creates_forbidden(get(a), mol_graph, types, matrix, before)]

    def candidates(molecule: Any, *args: Any, **kwargs: Any) -> list[Any]:
        return keep(molecule, _ORIGINAL_CANDIDATES["pr"](molecule, *args, **kwargs), lambda pair: pair[1])

    def unchecked(action_class: Any, molecule: Any) -> list[Any]:
        return keep(molecule, _ORIGINAL_CANDIDATES["nw"](action_class, molecule), lambda action: action)

    pr.candidates = candidates
    nw._candidates = unchecked  # pylint: disable=protected-access


def install(notebook: Path = NOTEBOOK, bore_max: int | None = None) -> Any:
    """Les mutations d'evomol vérifient désormais les règles des marches (walk_rules) dans ce processus ; les
    renvoie."""
    import regles_V2 as rv  # pylint: disable=import-outside-toplevel

    regles = walk_rules(bore_max, notebook)

    @lru_cache(maxsize=CACHE_SIZE)
    def violations_of_key(types: tuple[str, ...], data: bytes, n: int) -> tuple[Any, ...]:
        matrix = np.frombuffer(data, dtype=np.int64).reshape(n, n).copy()
        return tuple(regles.violations(rv.GrapheV2(list(types), matrix)))

    def rule_violations(types, matrix, sulfur_valence=None):  # pylint: disable=unused-argument
        matrix = np.ascontiguousarray(matrix, dtype=np.int64)
        return list(violations_of_key(tuple(types), matrix.tobytes(), matrix.shape[0]))

    mr.rule_violations = rule_violations
    fast_rules.FAST_RULES = False
    fast_rules.clear_cache()
    rf.clear_cache()
    return regles


def configure(config: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """nw.configure, puis les règles V2 (et RB selon BORE_MAX_ORDRE)."""
    result = nw.configure(config)
    install(bore_max=bore_max_order(config))
    return result


def _run_one(task: tuple[dict[str, Any], int, str, Any]) -> int:
    """Une marche dans un processus de calcul : celle de pi_walk, configurée par ce module."""
    pw.configure = configure
    return nw._PI_WALK_RUN_ONE(task)  # pylint: disable=protected-access


def run_walks(config: dict[str, Any], n_walks: int, out_dir: str, n_jobs: int | None = None) -> None:
    """nw.run_walks avec les règles V2 : n_walks marches en parallèle, fichiers dans out_dir (relatif à la racine
    du projet), écrasés à chaque lancement."""
    pw._run_one = _run_one  # pylint: disable=protected-access
    try:
        pw.run_walks(config, n_walks, out_dir, n_jobs)
    finally:
        pw._run_one = nw._PI_WALK_RUN_ONE  # pylint: disable=protected-access
