"""
Paramétrage des règles de la matrice (teste/parametrage_regles.ipynb).

Les règles sont écrites dans le notebook : ce module n'en contient aucune. Il donne les outils pour les essayer :
  - Graphe : la molécule lue sur la matrice, telle que les règles la reçoivent ;
  - Regles(RULES, compte_separe) : les règles du notebook, avec le compte des violations (une mutation est permise
    si elle ne crée aucune nouvelle violation, comme dans evomol) ;
  - compare_evomol(smiles, regles) : molécules où les règles du notebook et celles d'evomol diffèrent ;
  - evaluate / summary / rule_table / style / show_list : listes de molécules bonnes et mauvaises passées dans les
    règles (faux positifs, faux négatifs), en tableaux et en dessins ;
  - show_rules / violations_table / matrix_table : détail d'une molécule, atomes et cases de la matrice en cause ;
  - configure / mutation_summary / mutation_table / show_mutations / apply_mutation / random_mutations : mutations
    d'evomol filtrées par les règles du notebook, avec le verdict des règles d'evomol à côté ;
  - candidates / subfamily_summary / subfamily_table / show_subfamily / mutation_matrices : mutations rangées par
    sous-famille (découpage écrit dans le notebook : SOUS_FAMILLE), permises et refusées séparées, matrices avant,
    après et de mutation ;
  - sample_valid / mutation_test / test_summary / test_rules / test_combinations / test_examples : test d'une
    sous-famille sur beaucoup de molécules, avec les règles cassées.

Rien n'est modifié dans evomol : les mutations sont listées sans les règles d'evomol (matrix_rules=False, et
list_unchecked_actions pour les mutations qui appliquent déjà les règles), puis filtrées ici.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import multiprocessing
import os
import random
import re
import sys
import time
import warnings
from collections import Counter
from fnmatch import fnmatchcase
from functools import cached_property
from itertools import combinations
from typing import Any, Callable, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "newmut"))

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

import newmut_walk as nw  # describe, _candidates, _changed_atoms, action_classes
from evomol import default_parameters as dp
from evomol.action import Action
from evomol.action import molecular_graph as mg
from evomol.newmut import fast_rules as fr
from evomol.newmut import matrix_rules as mr
from evomol.newmut import rule_filtered as rf
from evomol.newmut.default_parameters import setup_newmut_action_space
from evomol.newmut.matrix_diff import classify_action, matrix_mol, mcs_both_ways, operation_names
from evomol.representation import MolecularGraph, Molecule

Violation = mr.Violation

# une couleur par règle, dans l'ordre de RULES
PALETTE = ["#e5484d", "#f08c00", "#8e4ec6", "#0090ff", "#12a594", "#e93d82",
           "#a18072", "#3e63dd", "#7c8a00", "#6e56cf", "#d6409f", "#29a383"]
CHANGED = "#46a758"   # atomes modifiés par une mutation permise
REFUSED = "#8b8d98"   # atomes modifiés par une mutation refusée


def _rgb(color: str, tint: float = 0.55) -> tuple[float, float, float]:
    """Couleur hexadécimale → RGB (0-1), éclaircie vers le blanc (tint = part de la couleur gardée)."""
    rgb = [int(color[k:k + 2], 16) / 255 for k in (1, 3, 5)]
    return tuple(1 - tint * (1 - c) for c in rgb)  # type: ignore[return-value]


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(255 * c):02x}" for c in rgb)


# ==============================================================================
#  MOLÉCULE DONNÉE AUX RÈGLES
# ==============================================================================
class Graphe:
    """Molécule lue sur la matrice, telle que les règles du notebook la reçoivent.

    types[i] : élément de l'atome i ; matrix : la matrice (électrons de valence sur la diagonale, ordres de liaison
    ailleurs) ; n : nombre d'atomes lourds ; order[i][j] : ordre de la liaison (0 : pas de liaison, diagonale à 0) ;
    adj[i] : voisins de i ; bonds : liaisons (i, j) avec i < j ; bridges : liaisons (i < j) hors de tout cycle ;
    rings : plus petit cycle passant par chaque liaison (liste ordonnée d'atomes, sans doublon).
    Méthodes : bond_sum(i), has_multiple(i), in_ring(a, b), connected_without(x, y).
    """

    def __init__(self, types: Iterable[str], matrix: np.ndarray):
        self._base = mr._Graph(list(types), matrix)  # pylint: disable=protected-access
        self.types: list[str] = self._base.types
        self.matrix: np.ndarray = np.array(matrix, dtype=np.int64)
        self.n: int = self._base.n
        self.order: list[list[int]] = self._base.order
        self.adj: list[list[int]] = self._base.adj
        self.bonds: list[tuple[int, int]] = [(a, b) for a in range(self.n) for b in self.adj[a] if a < b]

    # liaisons hors cycle et cycles : calculés à la première demande (le nom d'une sous-famille n'en a en général
    # pas besoin, les règles si)
    @cached_property
    def bridges(self) -> set[tuple[int, int]]:
        return self._base.bridges()

    @cached_property
    def rings(self) -> list[list[int]]:
        return self._base.rings(self.bridges)

    @classmethod
    def from_graph(cls, mol_graph: MolecularGraph) -> Graphe:
        return cls(*mr.to_matrix(mol_graph))

    def bond_sum(self, i: int) -> int:
        """Somme de la ligne de l'atome i, hors diagonale."""
        return self._base.bond_sum(i)

    def has_multiple(self, i: int) -> bool:
        """True si l'atome i porte une liaison double ou triple."""
        return self._base.has_multiple(i)

    def in_ring(self, a: int, b: int) -> bool:
        """True si la liaison a-b est dans un cycle."""
        return (min(a, b), max(a, b)) not in self.bridges

    def connected_without(self, x: int, y: int) -> bool:
        """True si x et y restent reliés une fois la liaison x-y retirée."""
        return self._base.connected_without(x, y)


def _mol_graph(smiles: str) -> MolecularGraph:
    """Graphe moléculaire (kekulisé) d'un SMILES ; ValueError si le SMILES est illisible."""
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"SMILES illisible : {smiles}")
    return MolecularGraph(smiles)


# ==============================================================================
#  RÈGLES DU NOTEBOOK
# ==============================================================================
class Regles:
    """Les règles du notebook.

    rules : nom → fonction(g: Graphe) qui renvoie une liste de Violation (ou une Violation, ou None). Le nom de la
    violation est toujours celui de la clé. compte_separe : règles comptées chacune à part ; les autres sont comptées
    ensemble. Une mutation est permise si aucun de ces comptes n'augmente (evomol : R1 à part, R2 + R3 + R4 + R5
    ensemble).
    """

    def __init__(self, rules: dict[str, Callable[[Graphe], Any]], compte_separe: Iterable[str] = ("R1",)):
        self.rules = dict(rules)
        self.names = list(self.rules)
        self.compte_separe = [name for name in compte_separe if name in self.rules]
        self.colors = {name: PALETTE[k % len(PALETTE)] for k, name in enumerate(self.names)}

    def violations(self, g: Graphe) -> list[Violation]:
        """Violations de toutes les règles, dans l'ordre de RULES."""
        found: list[Violation] = []
        for name, rule in self.rules.items():
            try:
                result = rule(g)
            except Exception as exc:
                raise RuntimeError(f"règle {name} : {exc!r}") from exc
            if result is None:
                continue
            if isinstance(result, Violation):
                result = [result]
            found.extend(v._replace(rule_id=name) for v in result if v is not None)
        return found

    def of_graph(self, mol_graph: MolecularGraph) -> list[Violation]:
        return self.violations(Graphe.from_graph(mol_graph))

    def of_smiles(self, smiles: str) -> list[Violation]:
        return self.of_graph(_mol_graph(smiles))

    def counts(self, violations: list[Violation]) -> tuple[int, ...]:
        """Comptes comparés pour décider si une mutation est permise."""
        per_rule = Counter(v.rule_id for v in violations)
        separate = [per_rule[name] for name in self.compte_separe]
        return (*separate, sum(per_rule.values()) - sum(separate))

    def allowed(self, before: list[Violation], after: list[Violation]) -> bool:
        """True si after ne crée aucune nouvelle violation par rapport à before."""
        return all(a <= b for a, b in zip(self.counts(after), self.counts(before)))

    def new_violations(self, before: list[Violation], after: list[Violation]) -> str:
        """Règles dont le nombre de violations augmente, ex. « R2 +1 »."""
        count_before = Counter(v.rule_id for v in before)
        count_after = Counter(v.rule_id for v in after)
        return ", ".join(f"{name} +{count_after[name] - count_before[name]}"
                         for name in self.names if count_after[name] > count_before[name])

    def chip(self, name: str, text: str | None = None) -> str:
        """Étiquette HTML de la couleur d'une règle."""
        color = self.colors.get(name, REFUSED)
        return (f'<span style="background:{_hex(_rgb(color))};border:1px solid {color};border-radius:3px;'
                f'padding:0 4px;margin:0 3px 0 0;color:#000;white-space:nowrap">{html.escape(text or name)}</span>')

    def legend(self) -> HTML:
        """Couleur de chaque règle."""
        return HTML("<div>" + "".join(self.chip(name) for name in self.names) + "</div>")


def _summary_text(violations: list[Violation], regles: Regles) -> str:
    """« R2 ×2, R5 » (vide : aucune violation)."""
    counts = Counter(v.rule_id for v in violations)
    return ", ".join(name if counts[name] == 1 else f"{name} ×{counts[name]}" for name in regles.names if counts[name])


def compare_evomol(smiles_list: Iterable[Any], regles: Regles, sulfur_valence: int = 6, r2_hetero: bool = False,
                   r2_hetero_outside_rings: bool = False) -> pd.DataFrame:
    """Molécules où les règles du notebook et celles d'evomol (matrix_rules.py, avec ses propres paramètres et les
    options R2 données ici) ne trouvent pas les mêmes violations (même nombre par règle). Vide : aucune différence.
    Chaque SMILES n'est compté qu'une fois."""
    options = mr.R2_HETERO, mr.R2_HETERO_OUTSIDE_RINGS
    mr.R2_HETERO, mr.R2_HETERO_OUTSIDE_RINGS = r2_hetero, r2_hetero_outside_rings
    rows, n, seen = [], 0, set()
    try:
        for smiles, _ in _entries(smiles_list):
            if smiles in seen:  # une molécule présente dans plusieurs listes : une seule ligne
                continue
            seen.add(smiles)
            try:
                mol_graph = _mol_graph(smiles)
            except ValueError:
                continue
            n += 1
            ours = regles.of_graph(mol_graph)
            types, matrix = mr.to_matrix(mol_graph)
            theirs = mr.rule_violations(types, matrix, sulfur_valence)
            if Counter(v.rule_id for v in ours) != Counter(v.rule_id for v in theirs):
                names = regles.names + [v.rule_id for v in theirs if v.rule_id not in regles.names]
                rows.append({"SMILES": smiles, "notebook": _summary_text(ours, regles) or "aucune",
                             "evomol": ", ".join(f"{k} ×{c}" if c > 1 else k for k, c in
                                                 sorted(Counter(v.rule_id for v in theirs).items(),
                                                        key=lambda kc: names.index(kc[0]))) or "aucune"})
    finally:
        mr.R2_HETERO, mr.R2_HETERO_OUTSIDE_RINGS = options
    print(f"{len(rows)} molécule(s) sur {n} où les règles du notebook et celles d'evomol diffèrent")
    return pd.DataFrame(rows, columns=["SMILES", "notebook", "evomol"])


# ==============================================================================
#  DESSINS
# ==============================================================================
def _highlights(mol: Chem.Mol, g: Graphe, violations: list[Violation], regles: Regles
                ) -> tuple[dict[int, list[tuple]], dict[int, list[tuple]]]:
    """Atomes de chaque violation et liaisons entre ces atomes, de la couleur de la règle."""
    atoms: dict[int, list[tuple]] = {}
    bonds: dict[int, list[tuple]] = {}
    for v in violations:
        color = _rgb(regles.colors.get(v.rule_id, REFUSED))
        for a in v.atoms:
            if color not in atoms.setdefault(a, []):
                atoms[a].append(color)
        for a, b in combinations(v.atoms, 2):
            if g.order[a][b]:
                idx = mol.GetBondBetweenAtoms(a, b).GetIdx()
                if color not in bonds.setdefault(idx, []):
                    bonds[idx].append(color)
    return atoms, bonds


def _png(mol: Chem.Mol, atoms: dict[int, list[tuple]] | None = None, bonds: dict[int, list[tuple]] | None = None,
         size: tuple[int, int] = (300, 240)) -> bytes:
    """Dessin PNG avec les numéros d'atomes ; un atome ou une liaison peut avoir plusieurs couleurs. Si RDKit ne
    sait pas dessiner la molécule (cages très symétriques : « Maximum BFS search size exceeded »), nouvel essai
    avec des coordonnées calculées sans gabarits de cycles et sans préparation, sinon une case « dessin
    impossible » : une molécule ne bloque pas toute une planche."""
    from rdkit.Chem import rdDepictor  # pylint: disable=import-outside-toplevel

    def draw(molecule: Chem.Mol, prepare: bool) -> bytes:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        options = drawer.drawOptions()
        options.addAtomIndices = True
        options.continuousHighlight = False  # sinon RDKit ne trace pas une liaison entre deux atomes coloriés
        options.prepareMolsBeforeDrawing = prepare
        if atoms or bonds:
            drawer.DrawMoleculeWithHighlights(molecule, "", atoms or {}, bonds or {}, {}, {})
        else:
            drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    try:
        return draw(Chem.Mol(mol), True)
    except (ValueError, RuntimeError):
        pass
    try:
        simple = Chem.Mol(mol)
        rdDepictor.SetPreferCoordGen(False)
        rdDepictor.Compute2DCoords(simple, useRingTemplates=False)
        return draw(simple, False)
    except (ValueError, RuntimeError):
        pass
    from PIL import Image, ImageDraw  # pylint: disable=import-outside-toplevel
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).text((10, size[1] // 2 - 6), "dessin impossible (RDKit)", fill="#999")
    buffer = __import__("io").BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _cell(png: bytes, caption: str, width: int) -> str:
    image = base64.b64encode(png).decode()
    return (f'<div style="width:{width}px;margin:4px;font:12px sans-serif;vertical-align:top">'
            f'<img src="data:image/png;base64,{image}" width="{width}" style="background:#fff"/>'
            f'<div style="padding:2px">{caption}</div></div>')


def _grid(cells: list[str]) -> HTML:
    return HTML('<div style="display:flex;flex-wrap:wrap;align-items:flex-start">' + "".join(cells) + "</div>")


def _chips(violations: list[Violation], regles: Regles) -> str:
    counts = Counter(v.rule_id for v in violations)
    return "".join(regles.chip(name, name if counts[name] == 1 else f"{name} ×{counts[name]}")
                   for name in regles.names if counts[name])


# ==============================================================================
#  LISTES DE MOLÉCULES BONNES ET MAUVAISES
# ==============================================================================
def _entries(items: Iterable[Any]) -> list[tuple[str, str]]:
    """SMILES ou (SMILES, commentaire) → (SMILES, commentaire)."""
    return [(item, "") if isinstance(item, str) else (item[0], item[1] if len(item) > 1 else "") for item in items]


def evaluate(bonnes: Iterable[Any], mauvaises: Iterable[Any], autres: Iterable[Any] | Regles,
             regles: Regles | None = None) -> pd.DataFrame:
    """Passe les listes dans les règles : evaluate(BONNES, MAUVAISES, AUTRES, REGLES) (ou sans AUTRES). Une molécule
    est acceptée si elle ne viole aucune règle. Verdict : OK, faux positif (bonne refusée), faux négatif (mauvaise
    acceptée), acceptée / refusée (autres : rien d'attendu) ou illisible. Une colonne par règle : nombre de
    violations."""
    if regles is None:  # evaluate(BONNES, MAUVAISES, REGLES)
        autres, regles = [], autres  # type: ignore[assignment]
    assert isinstance(regles, Regles)
    rows = []
    for expected, items in (("bonne", bonnes), ("mauvaise", mauvaises), ("autre", autres)):
        for smiles, comment in _entries(items):
            row: dict[str, Any] = {"SMILES": smiles, "commentaire": comment, "attendue": expected}
            try:
                violations = regles.of_smiles(smiles)
            except ValueError:
                row["verdict"] = "illisible"
                rows.append(row)
                continue
            accepted = not violations
            if expected == "bonne":
                row["verdict"] = "OK" if accepted else "faux positif (bonne refusée)"
            elif expected == "autre":
                row["verdict"] = "acceptée" if accepted else "refusée"
            else:
                row["verdict"] = "faux négatif (mauvaise acceptée)" if accepted else "OK"
            counts = Counter(v.rule_id for v in violations)
            row.update({name: counts[name] for name in regles.names})
            rows.append(row)
    table = pd.DataFrame(rows, columns=["SMILES", "commentaire", "attendue", "verdict", *regles.names])
    table[regles.names] = table[regles.names].fillna(0).astype(int)
    return table


def summary(bilan: pd.DataFrame) -> pd.DataFrame:
    """Nombre de molécules par cas."""
    cases = [
        ("bonnes acceptées (OK)", (bilan.attendue == "bonne") & (bilan.verdict == "OK")),
        ("bonnes refusées (faux positifs)", bilan.verdict.str.startswith("faux positif")),
        ("mauvaises refusées (OK)", (bilan.attendue == "mauvaise") & (bilan.verdict == "OK")),
        ("mauvaises acceptées (faux négatifs)", bilan.verdict.str.startswith("faux négatif")),
        ("autres acceptées", (bilan.attendue == "autre") & (bilan.verdict == "acceptée")),
        ("autres refusées", (bilan.attendue == "autre") & (bilan.verdict == "refusée")),
        ("illisibles", bilan.verdict == "illisible"),
    ]
    return pd.DataFrame({"molécules": [int(mask.sum()) for _, mask in cases]}, index=[name for name, _ in cases])


def rule_table(bilan: pd.DataFrame, regles: Regles) -> pd.DataFrame:
    """Par règle : bonnes qu'elle refuse (à éviter), mauvaises qu'elle refuse, et mauvaises qu'elle est seule à
    refuser (sans elle, elles seraient acceptées)."""
    counts = bilan[regles.names]
    alone = counts.gt(0).sum(axis=1) == 1
    bad = bilan.attendue == "mauvaise"
    return pd.DataFrame({
        "bonnes refusées": [int((counts[name].gt(0) & (bilan.attendue == "bonne")).sum()) for name in regles.names],
        "mauvaises refusées": [int((counts[name].gt(0) & bad).sum()) for name in regles.names],
        "mauvaises refusées par elle seule": [int((counts[name].gt(0) & bad & alone).sum()) for name in regles.names],
        "autres refusées": [int((counts[name].gt(0) & (bilan.attendue == "autre")).sum()) for name in regles.names],
    }, index=regles.names)


def style(bilan: pd.DataFrame, regles: Regles) -> Any:
    """Tableau du bilan avec le verdict en couleur et les violations de la couleur de leur règle."""

    def verdict_css(value: str) -> str:
        if value == "OK":
            return f"background:{_hex(_rgb(CHANGED, 0.35))};color:#000"
        if value in ("acceptée", "refusée"):  # autres : rien d'attendu
            return ""
        return f"background:{_hex(_rgb(PALETTE[0], 0.45))};color:#000"

    styler = bilan.style
    mapper = getattr(styler, "map", None) or styler.applymap
    styler = mapper(verdict_css, subset=["verdict"])
    for name in regles.names:
        css = f"background:{_hex(_rgb(regles.colors[name]))};color:#000"
        styler = (getattr(styler, "map", None) or styler.applymap)(
            lambda value, css=css: css if value else "color:#999", subset=[name])
    return styler


def show_list(bilan: pd.DataFrame, regles: Regles, seulement_erreurs: bool = False,
              size: tuple[int, int] = (260, 200)) -> HTML:
    """Dessins des molécules du bilan : atomes et liaisons en cause de la couleur de leur règle, verdict et
    règles en légende. seulement_erreurs : seulement les faux positifs, faux négatifs et illisibles."""
    errors = bilan.verdict.str.startswith("faux") | (bilan.verdict == "illisible")
    rows = bilan[errors] if seulement_erreurs else bilan
    cells = []
    for _, row in rows.iterrows():
        if row.verdict == "illisible":
            print(f"illisible : {row.SMILES}")
            continue
        verdict_color = {"OK": CHANGED, "acceptée": REFUSED, "refusée": REFUSED}.get(row.verdict, PALETTE[0])
        caption = (f'<b style="color:{verdict_color}">{html.escape(row.verdict)}</b> · {row.attendue}<br>'
                   f'<code style="font-size:10px">{html.escape(row.SMILES)}</code>')
        if row.commentaire:
            caption += f"<br><i>{html.escape(row.commentaire)}</i>"
        mol_graph = _mol_graph(row.SMILES)
        g = Graphe.from_graph(mol_graph)
        violations = regles.violations(g)
        caption += "<br>" + (_chips(violations, regles) or "aucune règle non respectée")
        atoms, bonds = _highlights(mol_graph.mol, g, violations, regles)
        cells.append(_cell(_png(mol_graph.mol, atoms, bonds, size), caption, size[0]))
    if not cells:
        print("aucune molécule à dessiner")
    return _grid(cells)


# ==============================================================================
#  MOLÉCULES TIRÉES DANS LES EXPÉRIENCES (marches aléatoires)
# ==============================================================================
WALKS_DIR = os.path.join(ROOT, "notebooks", "random_walk", "results")


def experiments(results_dir: str = WALKS_DIR) -> list[str]:
    """Dossiers d'expériences (relatifs à results_dir) qui contiennent des marches walk<k>.csv.gz."""
    found = []
    for folder, _, files in os.walk(results_dir):
        if any(re.fullmatch(r"walk\d+\.csv\.gz", name) for name in files):
            found.append(os.path.relpath(folder, results_dir))
    return sorted(found)


def sample_walks(n: int = 30, seed: int | None = 0, names: Iterable[str] | None = None,
                 max_atoms: int | None = None, results_dir: str = WALKS_DIR) -> pd.DataFrame:
    """n molécules tirées au hasard dans les marches des expériences : une expérience au hasard, une marche au
    hasard, puis un pas au hasard (pas 0 exclu). names : expériences (None : celles des marches newmut,
    newmut_walk/… dans experiments()) ;
    max_atoms : seulement les pas d'au plus max_atoms atomes lourds (None : tous).
    Colonnes : expérience, marche, pas, mutation (celle qui mène à ce pas), atomes, SMILES."""
    rng = random.Random(seed)
    if names is None:
        names = [name for name in experiments(results_dir) if name.startswith("newmut_walk")]
    names = list(names)
    if not names:
        raise FileNotFoundError(f"aucune marche newmut dans {results_dir}")
    walks: dict[tuple[str, int], pd.DataFrame] = {}
    files = {name: sorted(int(m.group(1)) for f in os.listdir(os.path.join(results_dir, name))
                          if (m := re.fullmatch(r"walk(\d+)\.csv\.gz", f)))
             for name in names}
    rows = []
    for _ in range(n):
        name = rng.choice(names)
        k = rng.choice(files[name])
        if (name, k) not in walks:
            table = pd.read_csv(os.path.join(results_dir, name, f"walk{k}.csv.gz"), converters={"smiles": str})
            keep = (table.step > 0) & (table.smiles != "")
            if max_atoms is not None:
                keep &= table.n_atoms <= max_atoms
            walks[name, k] = table[keep].reset_index(drop=True)
        table = walks[name, k]
        if table.empty:
            continue
        t = rng.randrange(len(table))
        rows.append({"expérience": name, "marche": k, "pas": int(table.step[t]), "mutation": table.action[t],
                     "atomes": int(table.n_atoms[t]), "SMILES": table.smiles[t]})
    return pd.DataFrame(rows)


def as_list(tirage: pd.DataFrame) -> list[tuple[str, str]]:
    """Tirage → liste (SMILES, place dans l'expérience), même format que BONNES et MAUVAISES."""
    return [(row.SMILES, f"{row.expérience} · marche {row.marche} · pas {row.pas}") for row in tirage.itertuples()]


def print_list(tirage: pd.DataFrame, name: str = "GENERER") -> None:
    """Écrit le tirage comme une liste Python (SMILES, place dans l'expérience), à copier dans une cellule."""
    print(f"{name} = [")
    for row in tirage.itertuples():
        print(f'    ("{row.SMILES}", "{row.expérience} · marche {row.marche} · pas {row.pas}"),')
    print("]")


def inspect(items: Iterable[Any], regles: Regles) -> pd.DataFrame:
    """Règles violées par chaque molécule d'une liste (SMILES ou (SMILES, commentaire)) : une colonne par règle,
    nombre de violations ; « acceptée » si aucune."""
    rows = []
    for smiles, comment in _entries(items):
        row: dict[str, Any] = {"SMILES": smiles, "commentaire": comment}
        try:
            violations = regles.of_smiles(smiles)
        except ValueError:
            row["verdict"] = "illisible"
            rows.append(row)
            continue
        row["verdict"] = "refusée" if violations else "acceptée"
        counts = Counter(v.rule_id for v in violations)
        row.update({name: counts[name] for name in regles.names})
        rows.append(row)
    table = pd.DataFrame(rows, columns=["SMILES", "commentaire", "verdict", *regles.names])
    table[regles.names] = table[regles.names].fillna(0).astype(int)
    return table


def show_items(items: Iterable[Any], regles: Regles, size: tuple[int, int] = (260, 200)) -> HTML:
    """Dessins d'une liste de molécules : atomes et liaisons en cause de la couleur de leur règle, commentaire et
    règles en légende."""
    cells = []
    for smiles, comment in _entries(items):
        try:
            mol_graph = _mol_graph(smiles)
        except ValueError:
            print(f"illisible : {smiles}")
            continue
        g = Graphe.from_graph(mol_graph)
        violations = regles.violations(g)
        atoms, bonds = _highlights(mol_graph.mol, g, violations, regles)
        caption = (f"<i>{html.escape(comment)}</i><br>" if comment else "") + \
            f'<code style="font-size:10px">{html.escape(smiles)}</code><br>' + \
            (_chips(violations, regles) or "aucune règle non respectée")
        cells.append(_cell(_png(mol_graph.mol, atoms, bonds, size), caption, size[0]))
    return _grid(cells)


# ==============================================================================
#  DÉTAIL D'UNE MOLÉCULE
# ==============================================================================
def show_rules(smiles: str, regles: Regles, size: tuple[int, int] = (420, 320)) -> None:
    """Un dessin avec toutes les violations (une couleur par règle), puis un dessin par violation : ses atomes et
    les liaisons entre eux, avec le message de la règle."""
    mol_graph = _mol_graph(smiles)
    g = Graphe.from_graph(mol_graph)
    violations = regles.violations(g)
    atoms, bonds = _highlights(mol_graph.mol, g, violations, regles)
    caption = (f"<code>{html.escape(mol_graph.canonical_smiles)}</code><br>"
               + (_chips(violations, regles) or "aucune règle non respectée"))
    display(_grid([_cell(_png(mol_graph.mol, atoms, bonds, size), caption, size[0])]))
    display(regles.legend())
    if not violations:
        return
    small = (int(size[0] * 0.75), int(size[1] * 0.75))
    cells = []
    for v in violations:
        atoms, bonds = _highlights(mol_graph.mol, g, [v], regles)
        cells.append(_cell(_png(mol_graph.mol, atoms, bonds, small),
                           f"{regles.chip(v.rule_id)} {html.escape(v.message)}", small[0]))
    display(_grid(cells))


def violations_table(smiles: str, regles: Regles) -> pd.DataFrame:
    """Règles non respectées par la molécule (vide : aucune)."""
    mol_graph = _mol_graph(smiles)
    return pd.DataFrame(
        [{"règle": v.rule_id, "atomes": ", ".join(f"{mol_graph.atom_type(a)}{a}" for a in v.atoms),
          "message": v.message} for v in regles.of_graph(mol_graph)],
        columns=["règle", "atomes", "message"],
    )


def matrix_table(smiles: str, regles: Regles) -> Any:
    """Matrice de la molécule (électrons de valence sur la diagonale, ordres de liaison ailleurs), puis H, étiquette
    du carbone (s / a / d / t), hybridation et règles où l'atome est en cause. Diagonale des atomes en cause et cases
    des liaisons en cause de la couleur de la règle."""
    mol_graph = _mol_graph(smiles)
    g = Graphe.from_graph(mol_graph)
    types, matrix = g.types, g.matrix
    violations = regles.violations(g)
    labels = [f"{element}{i}" for i, element in enumerate(types)]
    table = pd.DataFrame(matrix, index=labels, columns=labels)
    hydrogens = [mr.implicit_h(element, g.bond_sum(i)) for i, element in enumerate(types)]
    table["H"] = hydrogens
    table["étiquette"] = [mr.carbon_tag(matrix, types, i) for i in range(len(types))]
    table["hybridation"] = [mr.hybridization(matrix, types, i, hydrogens[i]) for i in range(len(types))]
    rules_of_atom: dict[int, list[str]] = {}
    for v in violations:
        for a in v.atoms:
            if v.rule_id not in rules_of_atom.setdefault(a, []):
                rules_of_atom[a].append(v.rule_id)
    table["règles"] = [", ".join(rules_of_atom.get(i, [])) for i in range(len(types))]

    css = pd.DataFrame("", index=table.index, columns=table.columns)
    for v in violations:
        color = f"background:{_hex(_rgb(regles.colors.get(v.rule_id, REFUSED)))};color:#000;font-weight:bold"
        for a in v.atoms:
            if not css.iat[a, a]:
                css.iat[a, a] = color
            if not css.iat[a, len(types) + 3]:
                css.iat[a, len(types) + 3] = color
        for a, b in combinations(v.atoms, 2):
            if g.order[a][b]:
                for i, j in ((a, b), (b, a)):
                    if not css.iat[i, j]:
                        css.iat[i, j] = color
    zero = (table.iloc[:, : len(types)] == 0) & (css.iloc[:, : len(types)] == "")
    css.iloc[:, : len(types)] = css.iloc[:, : len(types)].mask(zero, "color:#bbb")
    return table.style.apply(lambda _: css, axis=None)


# ==============================================================================
#  MUTATIONS
# ==============================================================================
def configure(accepted_atoms: Iterable[str] = ("C", "O", "N"), max_atoms: int = 38, **options: Any) -> None:
    """Espace d'action d'evomol sans ses règles (matrix_rules=False) : les mutations sont filtrées ensuite par les
    règles du notebook. options : arguments de evomol.newmut.setup_newmut_action_space (sauf matrix_rules) ; les
    options R2 (r2_hetero…) servent au verdict des règles d'evomol."""
    if options.pop("matrix_rules", False):
        raise ValueError("matrix_rules : les mutations sont filtrées par les règles du notebook")
    dp.setup_default_parameters(accepted_atoms=list(accepted_atoms), max_heavy_atoms=max_atoms)
    setup_newmut_action_space(matrix_rules=False, **options)
    print("mutations de l'espace d'action :", ", ".join(a.class_name() for a in MolecularGraph.action_space))


def _rows(smiles: str, action_type: str, regles: Regles, evomol: bool = True) -> tuple[Molecule, list[dict]]:
    """Toutes les mutations candidates d'un type, avec leur résultat, le verdict du notebook et celui d'evomol."""
    (action_class,) = nw.action_classes([action_type])
    molecule = Molecule(smiles)
    before = molecule.get_representation(MolecularGraph)
    v_before = regles.of_graph(before)
    rows = []
    for k, action in enumerate(nw._candidates(action_class, molecule)):  # pylint: disable=protected-access
        row: dict[str, Any] = {"n°": k, "action": action, "mutation": nw.describe(action)}
        try:
            after = rf.result_graph(action, before)  # type: ignore[arg-type]
            v_after = regles.of_graph(after)
        except RuntimeError:
            raise  # erreur dans une règle du notebook
        except Exception as exc:  # pylint: disable=broad-except
            row.update({"résultat": None, "graph": None, "violations": [], "permise": False,
                        "permise evomol": False, "raison": f"erreur : {exc!r}"})
            rows.append(row)
            continue
        allowed = regles.allowed(v_before, v_after)
        row.update({"résultat": after.aromatic_canonical_smiles, "graph": after, "violations": v_after,
                    "permise": allowed,
                    "permise evomol": rf.action_is_allowed(action, before) if evomol else None,  # type: ignore
                    "raison": "" if allowed else regles.new_violations(v_before, v_after) or "règles"})
        rows.append(row)
    return molecule, rows


def mutation_summary(smiles: str, regles: Regles) -> pd.DataFrame:
    """Par type de mutation : candidates, permises par les règles du notebook, permises par celles d'evomol,
    mutations où les deux diffèrent, et molécules différentes produites par les mutations permises (notebook)."""
    summary_rows = []
    for action_class in MolecularGraph.action_space:
        _, rows = _rows(smiles, action_class.class_name(), regles)
        allowed = [row for row in rows if row["permise"]]
        summary_rows.append({
            "mutation": action_class.class_name(), "candidates": len(rows), "permises (notebook)": len(allowed),
            "permises (evomol)": sum(1 for row in rows if row["permise evomol"]),
            "différences": sum(1 for row in rows if row["permise"] != row["permise evomol"]),
            "molécules différentes": len({row["résultat"] for row in allowed}),
        })
    return pd.DataFrame(summary_rows).set_index("mutation")


def mutation_table(smiles: str, action_type: str, regles: Regles, only_allowed: bool = False) -> pd.DataFrame:
    """Toutes les mutations d'un type : description (numéros de la molécule de départ), résultat, permise par les
    règles du notebook, par celles d'evomol, et la règle qui la refuse. Le n° sert à apply_mutation."""
    _, rows = _rows(smiles, action_type, regles)
    table = pd.DataFrame(rows, columns=["n°", "mutation", "résultat", "permise", "permise evomol", "raison"])
    return table[table["permise"]] if only_allowed else table


def _new_rule_atoms(row: dict, v_before: list[Violation], regles: Regles) -> list[Violation]:
    """Violations du résultat pour les règles dont le nombre augmente."""
    increased = {name for name in regles.names
                 if sum(v.rule_id == name for v in row["violations"]) > sum(v.rule_id == name for v in v_before)}
    return [v for v in row["violations"] if v.rule_id in increased]


def _mutation_cell(before: MolecularGraph, row: dict, v_before: list[Violation], regles: Regles,
                   size: tuple[int, int], caption: str) -> str:
    """Dessin du résultat : atomes modifiés en vert (permise) ou en gris (refusée), atomes des nouvelles
    violations de la couleur de leur règle."""
    after = row["graph"]
    g = Graphe.from_graph(after)
    atoms, bonds = _highlights(after.mol, g, _new_rule_atoms(row, v_before, regles), regles)
    changed_color = _rgb(CHANGED if row["permise"] else REFUSED)
    for a in nw._changed_atoms(before, after):  # pylint: disable=protected-access
        atoms.setdefault(a, []).insert(0, changed_color)
    return _cell(_png(after.mol, atoms, bonds, size), caption, size[0])


def show_mutations(smiles: str, action_type: str, regles: Regles, only_allowed: bool = False,
                   size: tuple[int, int] = (260, 210), max_images: int = 60) -> HTML | None:
    """Dessins des molécules produites par les mutations d'un type. Atomes modifiés en vert si la mutation est
    permise par les règles du notebook, en gris sinon ; atomes des nouvelles violations de la couleur de leur règle.
    Numéros d'atomes : ceux de la molécule de départ (les atomes ajoutés prennent les numéros suivants)."""
    molecule, rows = _rows(smiles, action_type, regles)
    rows = [row for row in rows if row["graph"] is not None and (row["permise"] or not only_allowed)]
    if len(rows) > max_images:
        print(f"{len(rows)} mutations : seules les {max_images} premières sont dessinées (max_images)")
        rows = rows[:max_images]
    if not rows:
        print(f"aucune mutation {action_type} sur {molecule}")
        return None
    before = molecule.get_representation(MolecularGraph)
    v_before = regles.of_graph(before)
    cells = []
    for row in rows:
        if row["permise"]:
            verdict = f'<b style="color:{CHANGED}">permise</b>'
        else:
            verdict = (f'<b style="color:{PALETTE[0]}">refusée</b> '
                       + "".join(regles.chip(part.split()[0], part) for part in row["raison"].split(", ")))
        if row["permise"] != row["permise evomol"]:
            verdict += f" · evomol : {'permise' if row['permise evomol'] else 'refusée'}"
        caption = f"n°{row['n°']} : {html.escape(row['mutation'])}<br>{verdict}"
        cells.append(_mutation_cell(before, row, v_before, regles, size, caption))
    return _grid(cells)


def apply_mutation(smiles: str, action_type: str, number: int, regles: Regles,
                   size: tuple[int, int] = (380, 300)) -> str:
    """Applique la mutation n° number de mutation_table(smiles, action_type, regles) et montre avant / après : nom
    de l'opération, cases de la matrice qui changent, règles. Renvoie le SMILES de la nouvelle molécule (ses atomes
    sont renumérotés : relancer le détail pour les nouveaux numéros)."""
    molecule, rows = _rows(smiles, action_type, regles)
    row = next((r for r in rows if r["n°"] == number), None)
    if row is None or row["graph"] is None:
        raise IndexError(f"pas de mutation {action_type} n°{number} sur {smiles} ({len(rows)} candidates)")
    if not row["permise"]:
        print(f"Attention : mutation refusée par les règles du notebook ({row['raison']}), appliquée quand même "
              "pour la voir.")
    before, after = molecule.get_representation(MolecularGraph), row["graph"]
    v_before = regles.of_graph(before)
    types_b, matrix_b = mr.to_matrix(before)
    types_a, matrix_a = mr.to_matrix(after)
    print(nw.describe(row["action"]))
    print(f"  opération : {' + '.join(operation_names(classify_action(row['action']))) or 'aucune'}")
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
    print(f"  règles du notebook : {regles.new_violations(v_before, row['violations']) or 'aucune nouvelle violation'}"
          f" ; evomol : {'permise' if row['permise evomol'] else 'refusée'}")

    new_molecule = row["action"].apply()
    g_before = Graphe.from_graph(before)
    atoms, bonds = _highlights(before.mol, g_before, v_before, regles)
    changed = [a for a in nw._changed_atoms(before, after) if a < len(types_b)]  # pylint: disable=protected-access
    for a in changed:
        atoms.setdefault(a, []).insert(0, _rgb(CHANGED))
    cells = [
        _cell(_png(before.mol, atoms, bonds, size),
              f"avant : <code>{html.escape(before.canonical_smiles)}</code><br>" + _chips(v_before, regles), size[0]),
        _mutation_cell(before, row, v_before, regles, size,
                       f"après : <code>{html.escape(str(new_molecule))}</code><br>"
                       + _chips(row["violations"], regles)),
    ]
    display(_grid(cells))
    display(regles.legend())
    return str(new_molecule)


def random_mutations(smiles: str, n_mutations: int, regles: Regles, seed: int | None = None,
                     size: tuple[int, int] = (240, 200)) -> str:
    """Tirage au hasard comme l'algorithme : n_mutations fois, un type de mutation au hasard, puis une mutation
    permise par les règles du notebook au hasard. Montre chaque étape ; renvoie le SMILES final."""
    rng = random.Random(seed)
    molecule = Molecule(smiles)
    graph = molecule.get_representation(MolecularGraph)
    cells = [_cell(_png(graph.mol), f"départ<br><code>{html.escape(str(molecule))}</code><br>"
                   + _chips(regles.of_graph(graph), regles), size[0])]
    for _ in range(n_mutations):
        action_class = rng.choice(MolecularGraph.action_space)
        _, rows = _rows(str(molecule), action_class.class_name(), regles, evomol=False)
        allowed = [row for row in rows if row["permise"]]
        if not allowed:
            cells.append(_cell(_png(graph.mol), f"aucune {action_class.class_name()} permise", size[0]))
            continue
        row = rng.choice(allowed)
        action: Action = row["action"]
        text = nw.describe(action)
        molecule = action.apply()
        graph = molecule.get_representation(MolecularGraph)
        cells.append(_cell(_png(graph.mol), f"{action_class.class_name()}<br>{html.escape(text)}<br>"
                                            f"<code>{html.escape(str(molecule))}</code>", size[0]))
    display(_grid(cells))
    return str(molecule)


# ==============================================================================
#  SOUS-FAMILLES DE MUTATIONS
# ==============================================================================
# sous_famille(action, avant, après) → nom de la sous-famille (écrite dans le notebook : SOUS_FAMILLE)
SousFamille = Callable[[Action, Graphe, Graphe], str]

ADD_FAMILIES = ("AddAtomMG", "AppendAtomRuleMG")  # AppendAtomRuleMG : ancien nom (avant le 2026-10-19)

# une seule étiquette par famille : anciens noms et variantes → nom de la famille
FAMILY_LABELS = {
    "AppendAtomRuleMG": "AddAtomMG",          # ajout de newmut, renommé AddAtomMG le 2026-10-19
    "SubstituteAtomMG": "SubstituteShiftMG",  # substitution d'origine d'evomol (substitution_shift=False)
    "InsertCarbonDoubleMG": "InsertCarbonMG",  # variante qui garde la double (insert_carbon_keep_double)
}


def family_label(name: str) -> str:
    """Étiquette unique d'une famille (anciens noms et variantes ramenés au nom de la famille)."""
    return FAMILY_LABELS.get(name, name)


def _family_patterns(familles: str | Iterable[str] | None) -> list[str] | None:
    """FAMILLES avec une seule étiquette par famille, sans doublon."""
    names = _patterns(familles)
    return None if names is None else list(dict.fromkeys(family_label(n) for n in names))


def add_candidates(molecule: Molecule) -> list[Action]:
    """Ajout d'un atome, toutes liaisons : pour chaque atome non chargé et chaque élément accepté, liaison d'ordre
    k = 1, 2 ou 3 si l'atome porte au moins k H et que la valence max de l'élément est au moins k. Sans la
    règle sp3-sp3 de l'AddAtomMG de newmut (pas de C par une simple sur un C s) : ce sont les règles qui décident."""
    mol_graph = molecule.get_representation(MolecularGraph)
    if mol_graph.nb_atoms >= Molecule.max_heavy_atoms:
        return []
    if mol_graph.nb_atoms == 0:
        return [mg.AddAtomMG(molecule, 0, element) for element in Molecule.accepted_atoms]
    free = mol_graph.implicit_valences
    actions: list[Action] = []
    for i in range(mol_graph.nb_atoms):
        if mol_graph.atom_charged_or_radical(i):
            continue
        for element in Molecule.accepted_atoms:
            for order in (1, 2, 3):
                if free[i] >= order and mr.max_valence(element) >= order:
                    actions.append(mg.AddAtomMG(molecule, i, element, order))
    return actions


def families(ajout_complet: bool = True) -> list[str]:
    """Noms des familles de l'espace d'action (ceux à donner à FAMILLES)."""
    return list(dict.fromkeys(family_label(a.class_name()) for a in MolecularGraph.action_space))


def substitution_candidates(molecule: Molecule) -> list[Action]:
    """Substitutions sans le tri interne de SubstituteShiftMG (qui ne garde que le premier état permis par les règles
    d'evomol) : pour chaque atome non chargé et chaque autre élément accepté, TOUS les états — sans déplacement, puis
    chaque chemin alterné pour la double perdue (shift_states) — et toutes les liaisons qui peuvent monter
    (raise_states). Les options de SubstituteShiftMG (shift_bonds, raise_bonds) sont respectées. Ce sont les règles
    du notebook qui jugent ensuite."""
    from evomol.newmut.substitute_shift import (  # pylint: disable=import-outside-toplevel
        SubstituteShiftMG, raise_states, shift_states)
    graph = molecule.get_representation(MolecularGraph)
    types, matrix = mr.to_matrix(graph)
    out: list[Action] = []
    for atom in range(len(types)):
        if graph.atom_charged_or_radical(atom):
            continue
        for element in Molecule.accepted_atoms:
            if element == types[atom]:
                continue
            seen: set[tuple] = set()
            for state in shift_states(types, matrix, atom, element, SubstituteShiftMG.shift_bonds):
                new = np.array(state, dtype=np.int64)
                changes = tuple((int(i), int(j), int(new[i, j]))
                                for i, j in zip(*np.nonzero(np.triu(new != matrix, 1))))
                if changes not in seen:
                    seen.add(changes)
                    out.append(SubstituteShiftMG(molecule, atom, element, changes))
            if SubstituteShiftMG.raise_bonds:
                for change in raise_states(types, matrix, atom, element):
                    if (change,) not in seen:
                        seen.add((change,))
                        out.append(SubstituteShiftMG(molecule, atom, element, (change,)))
    return out


def candidates(molecule: Molecule, ajout_complet: bool = True,
               familles: str | Iterable[str] | None = None) -> list[tuple[str, Action]]:
    """(famille, mutation) pour les familles de l'espace d'action, avant toute règle. ajout_complet (toutes les
    candidates, sans tri interne) : l'AddAtomMG de l'espace d'action remplacé par add_candidates (famille
    « AddAtomMG »), et SubstituteShiftMG par substitution_candidates (tous les états). familles : seulement ces
    familles (noms ou motifs, ex. ["AddAtomMG"], ["Change*"] ; None : toutes) ; les autres ne sont pas listées.
    L'ordre est fixe : le rang d'une mutation dans cette liste (pour les mêmes familles) est son n°."""
    wanted = _family_patterns(familles)
    found: list[tuple[str, Action]] = []
    for action_class in MolecularGraph.action_space:
        name = action_class.class_name()
        label = family_label(name)
        if wanted is not None and not any(fnmatchcase(label, p) or fnmatchcase(name, p) for p in wanted):
            continue
        if label == "AddAtomMG" and ajout_complet:
            found += [("AddAtomMG", action) for action in add_candidates(molecule)]
        elif name == "SubstituteShiftMG" and ajout_complet:
            found += [(name, action) for action in substitution_candidates(molecule)]
        else:
            found += [(label, action) for action in nw._candidates(action_class, molecule)]  # pylint: disable=W0212
    return found


def _effect(action: Action, types: list[str], matrix: np.ndarray, before: MolecularGraph
            ) -> tuple[Graphe, list[int] | None]:
    """(molécule après, new_of_old) : calculée sur la matrice (fast_rules.matrix_effect), sinon par RDKit
    (AddGroupMG, RemoveGroupMG). new_of_old[i] : numéro après de l'atome i (−1 : retiré) ; None : inconnu."""
    effect = fr.matrix_effect(action, types, matrix)
    if effect is not None:
        new_types, new_matrix, new_of_old = effect
        return Graphe(new_types, new_matrix), list(new_of_old)
    after = rf.result_graph(action, before)  # type: ignore[arg-type]
    mapping = list(range(len(types))) if isinstance(action, mg.AddGroupMG) else None
    return Graphe.from_graph(after), mapping


def _patterns(selection: str | Iterable[str] | None) -> list[str] | None:
    if selection is None:
        return None
    return [selection] if isinstance(selection, str) else list(selection)


def _selected(patterns: list[str] | None, family: str, subfamily: str) -> bool:
    """Sous-famille ou famille qui correspond à un des noms (motifs permis : « add* », « bondCC* »). Un nom sans
    « @ » prend aussi toutes ses variantes : « addCs » → addCs@Cs, addCs@Cd…"""
    return patterns is None or any(fnmatchcase(subfamily, p) or fnmatchcase(family, p)
                                   or ("@" not in p and fnmatchcase(subfamily.split("@")[0], p))
                                   for p in patterns)


def _increased(regles: Regles, before: list[Violation], after: list[Violation]) -> list[str]:
    """Règles dont le nombre de violations augmente."""
    count_before = Counter(v.rule_id for v in before)
    count_after = Counter(v.rule_id for v in after)
    return [name for name in regles.names if count_after[name] > count_before[name]]


def touched_atoms(types: list[str], matrix: np.ndarray, after: Graphe, mapping: list[int] | None) -> list[int]:
    """Numéros de départ des atomes que la mutation touche : élément ou ligne de la matrice changés (liaisons vers
    les atomes ajoutés comprises), ou atome retiré. Vide si la correspondance des numéros est inconnue."""
    if mapping is None:
        return []
    image = set(a for a in mapping if a >= 0)
    added = [a for a in range(after.n) if a not in image]
    touched = []
    for i, a in enumerate(mapping):
        if a < 0 or after.types[a] != types[i]:
            touched.append(i)
            continue
        if any(after.matrix[a, b] for b in added) or any(
                matrix[i, j] != (after.matrix[a, mapping[j]] if mapping[j] >= 0 else 0)
                for j in range(len(types)) if j != i):
            touched.append(i)
    return touched


# contexte(action, avant, après, touchés) → description du contexte de la mutation (écrite dans un notebook)
Contexte = Callable[[Action, Graphe, Graphe, list[int]], str]


def _evaluate(smiles: str, regles: Regles, sous_famille: SousFamille, selection: str | Iterable[str] | None = None,
              ajout_complet: bool = True, keep: bool = False, familles: str | Iterable[str] | None = None,
              contexte: Contexte | None = None) -> tuple[Molecule, list[dict]]:
    """Toutes les mutations candidates (des sous-familles choisies) : famille, sous-famille, n°, description,
    permise par les règles du notebook, règles cassées. keep : garde aussi l'action, la molécule après (Graphe),
    new_of_old et les violations après. contexte : ajoute la colonne « contexte » (contexte(action, avant, après,
    atomes touchés))."""
    patterns = _patterns(selection)
    molecule = Molecule(smiles)
    before = molecule.get_representation(MolecularGraph)
    types, matrix = mr.to_matrix(before)
    g_before = Graphe(types, matrix)
    v_before = regles.violations(g_before)
    rows = []
    for k, (family, action) in enumerate(candidates(molecule, ajout_complet, familles)):
        row: dict[str, Any] = {"n°": k, "famille": family}
        try:
            g_after, mapping = _effect(action, types, matrix, before)
        except Exception as exc:  # pylint: disable=broad-except
            if _selected(patterns, family, "(erreur)"):
                row.update({"sous-famille": "(erreur)", "mutation": nw.describe(action), "permise": False,
                            "règles cassées": f"erreur : {exc!r}"})
                if keep:
                    row.update({"action": action, "après": None, "new_of_old": None, "violations": []})
                rows.append(row)
            continue
        name = sous_famille(action, g_before, g_after)
        if not _selected(patterns, family, name):
            continue
        v_after = regles.violations(g_after)
        allowed = regles.allowed(v_before, v_after)
        row.update({"sous-famille": name, "mutation": nw.describe(action), "permise": allowed,
                    "règles cassées": ", ".join(_increased(regles, v_before, v_after))})
        if contexte is not None:
            row["contexte"] = contexte(action, g_before, g_after, touched_atoms(types, matrix, g_after, mapping))
        if keep:
            row.update({"action": action, "après": g_after, "new_of_old": mapping, "violations": v_after})
        rows.append(row)
    return molecule, rows


def _family_order() -> dict[str, int]:
    """Rang de chaque famille dans l'espace d'action (pour trier les tableaux)."""
    order = {}
    for k, action_class in enumerate(MolecularGraph.action_space):
        name = action_class.class_name()
        order["AddAtomMG" if name in ADD_FAMILIES else name] = k
        order[name] = k
    return order


def _count_text(names: Iterable[str]) -> str:
    """« R2a ×5, R3 » à partir d'une liste de noms de règles."""
    counts = Counter(names)
    return ", ".join(name if c == 1 else f"{name} ×{c}" for name, c in counts.most_common())


def _broken(text: str) -> list[str]:
    return [name for name in text.split(", ") if name] if text and not text.startswith("erreur") else []


def subfamily_summary(smiles: str, regles: Regles, sous_famille: SousFamille, ajout_complet: bool = True
                      ) -> pd.DataFrame:
    """Par famille et sous-famille : candidates, permises, refusées, % acceptées, % refusées, règles cassées par les refusées
    (« R2a ×5 » : 5 mutations refusées cassent R2a)."""
    _, rows = _evaluate(smiles, regles, sous_famille, ajout_complet=ajout_complet)
    table = pd.DataFrame(rows, columns=["n°", "famille", "sous-famille", "mutation", "permise", "règles cassées"])
    order = _family_order()
    summary_rows = []
    for (family, name), group in table.groupby(["famille", "sous-famille"], sort=False):
        refused = group[~group.permise]
        summary_rows.append({
            "famille": family, "sous-famille": name, "candidates": len(group), "permises": int(group.permise.sum()),
            "refusées": len(refused), "% acceptées": round(100 * group.permise.mean(), 1),
            "% refusées": round(100 * (1 - group.permise.mean()), 1),
            "règles cassées (refusées)": _count_text(r for text in refused["règles cassées"] for r in _broken(text)),
        })
    result = pd.DataFrame(summary_rows)
    if result.empty:
        return result
    result = result.sort_values(["famille", "sous-famille"], key=lambda col: col.map(order).fillna(99)
                                if col.name == "famille" else col)
    return result.set_index(["famille", "sous-famille"])


def _verdict_style(table: pd.DataFrame) -> Any:
    styler = table.style
    mapper = getattr(styler, "map", None) or styler.applymap
    return mapper(lambda value: f"background:{_hex(_rgb(CHANGED, 0.35))};color:#000" if value is True
                  else f"background:{_hex(_rgb(PALETTE[0], 0.45))};color:#000", subset=["permise"])


def _title(text: str) -> HTML:
    return HTML(f'<h4 style="margin:12px 0 4px 0">{html.escape(text)}</h4>')


def subfamily_table(smiles: str, nom: str | Iterable[str], regles: Regles, sous_famille: SousFamille,
                    ajout_complet: bool = True, separe: bool = True) -> pd.DataFrame:
    """Les mutations d'une sous-famille (ou d'une famille, ou d'un motif « add* ») : n°, sous-famille, description,
    molécule produite, règles cassées. separe : affiche les permises puis les refusées dans deux tableaux. Renvoie
    le tableau complet. Le n° sert à mutation_matrices."""
    molecule, rows = _evaluate(smiles, regles, sous_famille, nom, ajout_complet, keep=True)
    before = molecule.get_representation(MolecularGraph)
    for row in rows:
        try:
            row["résultat"] = rf.result_graph(row["action"], before).aromatic_canonical_smiles  # type: ignore
        except Exception as exc:  # pylint: disable=broad-except
            row["résultat"] = f"erreur : {exc!r}"
    columns = ["n°", "sous-famille", "mutation", "résultat", "permise", "règles cassées"]
    table = pd.DataFrame(rows, columns=columns)
    if separe:
        allowed, refused = table[table.permise], table[~table.permise]
        display(_title(f"Permises : {len(allowed)}"))
        display(_verdict_style(allowed) if len(allowed) else "aucune")
        display(_title(f"Refusées : {len(refused)}"))
        display(_verdict_style(refused) if len(refused) else "aucune")
    return table


def show_subfamily(smiles: str, nom: str | Iterable[str], regles: Regles, sous_famille: SousFamille,
                   ajout_complet: bool = True, size: tuple[int, int] = (240, 200), max_images: int = 40) -> None:
    """Dessins des molécules produites, en deux blocs : permises (atomes modifiés en vert), puis refusées groupées
    par règles cassées (atomes modifiés en gris, atomes des nouvelles violations de la couleur de leur règle).
    Numéros d'atomes : ceux de la molécule de départ."""
    molecule, rows = _evaluate(smiles, regles, sous_famille, nom, ajout_complet, keep=True)
    before = molecule.get_representation(MolecularGraph)
    v_before = regles.of_graph(before)

    def cells(selected: list[dict]) -> list[str]:
        if len(selected) > max_images:
            print(f"{len(selected)} mutations : seules les {max_images} premières sont dessinées (max_images)")
            selected = selected[:max_images]
        out = []
        for row in selected:
            try:
                after = rf.result_graph(row["action"], before)
            except Exception:  # pylint: disable=broad-except
                continue
            drawn = {"graph": after, "permise": row["permise"], "violations": regles.of_graph(after)}
            caption = f"n°{row['n°']} · {html.escape(row['sous-famille'])}<br>{html.escape(row['mutation'])}"
            out.append(_mutation_cell(before, drawn, v_before, regles, size, caption))
        return out

    allowed = [row for row in rows if row["permise"]]
    refused = [row for row in rows if not row["permise"]]
    display(regles.legend())
    display(_title(f"Permises : {len(allowed)}"))
    if allowed:
        display(_grid(cells(allowed)))
    display(_title(f"Refusées : {len(refused)}"))
    groups: dict[str, list[dict]] = {}
    for row in refused:
        groups.setdefault(row["règles cassées"], []).append(row)
    for text, group in sorted(groups.items(), key=lambda item: -len(item[1])):
        chips = "".join(regles.chip(name) for name in _broken(text)) or html.escape(text)
        display(HTML(f'<div style="margin:8px 0 2px 0;font:13px sans-serif">{chips} : {len(group)}</div>'))
        display(_grid(cells(group)))


# ------------------------------------------------------------------------------
#  Matrices avant / après / mutation
# ------------------------------------------------------------------------------
def _matrix_html(table: pd.DataFrame, css: pd.DataFrame) -> str:
    styler = table.style.apply(lambda _: css, axis=None)
    styler = styler.set_table_styles([{"selector": "td, th", "props": "padding:1px 5px;text-align:center;"
                                                                      "font:12px monospace"}])
    return styler.to_html()


def mutation_matrices(smiles: str, nom: str, numero: int, regles: Regles, sous_famille: SousFamille,
                      ajout_complet: bool = True, size: tuple[int, int] = (360, 280)) -> str:
    """Mutation n° numero (n° de subfamily_table) : dessins avant / après, puis M avant, M après et la matrice de
    mutation ΔM = après − avant, alignées sur les numéros de départ (atomes ajoutés à la fin, atome retiré gardé
    en ligne grisée). ΔM : positif en vert, négatif en rouge. Renvoie le SMILES de la molécule produite."""
    molecule, rows = _evaluate(smiles, regles, sous_famille, None, ajout_complet, keep=True)
    row = next((r for r in rows if r["n°"] == numero), None)
    if row is None:
        raise IndexError(f"pas de mutation n°{numero} sur {smiles} ({len(rows)} candidates)")
    if not _selected(_patterns(nom), row["famille"], row["sous-famille"]):
        raise ValueError(f"la mutation n°{numero} est de la sous-famille {row['sous-famille']}, pas {nom} : "
                         "relancer subfamily_table pour choisir le n°")
    action: Action = row["action"]
    before = molecule.get_representation(MolecularGraph)
    after = rf.result_graph(action, before)  # type: ignore[arg-type]
    types_b, m_b = mr.to_matrix(before)
    types_a, m_a = mr.to_matrix(after)
    mapping = row["new_of_old"]
    if mapping is None:  # RemoveGroupMG : alignement cherché comme matrix_diff (mcs)
        found = mcs_both_ways(matrix_mol(before), matrix_mol(after))
        mapping = [found.get(i, -1) for i in range(len(types_b))]
    n_b = len(types_b)
    image = {a for a in mapping if a >= 0}
    added = [a for a in range(len(types_a)) if a not in image]
    position = list(mapping) + added            # position alignée → numéro dans après (−1 : retiré)
    size_all = len(position)
    aligned_b = np.zeros((size_all, size_all), dtype=int)
    aligned_b[:n_b, :n_b] = m_b
    aligned_a = np.zeros((size_all, size_all), dtype=int)
    for p, a in enumerate(position):
        for q, b in enumerate(position):
            if a >= 0 and b >= 0:
                aligned_a[p, q] = m_a[a, b]
    delta = aligned_a - aligned_b

    removed = [p for p, a in enumerate(position) if a < 0]
    new_rows = list(range(n_b, size_all))
    labels_b = [f"{types_b[p]}{p}" if p < n_b else f"({types_a[position[p]]}{p})" for p in range(size_all)]
    labels_a = [f"{types_a[a]}{p}" if a >= 0 else f"({types_b[p]}{p})" for p, a in enumerate(position)]
    labels_d = [labels_b[p] if p < n_b and labels_b[p] == labels_a[p] else
                (f"{types_b[p]}{p}→{types_a[position[p]]}" if p < n_b and position[p] >= 0 else
                 (f"−{types_b[p]}{p}" if p < n_b else f"+{labels_a[p]}")) for p in range(size_all)]

    grey, changed_css = "color:#bbb", f"background:{_hex(_rgb('#f5d90a', 0.45))};color:#000;font-weight:bold"
    absent = "background:#eee;color:#bbb"

    def css_for(values: np.ndarray, missing: list[int]) -> pd.DataFrame:
        css = pd.DataFrame("", index=range(size_all), columns=range(size_all))
        for p in range(size_all):
            for q in range(size_all):
                if p in missing or q in missing:
                    css.iat[p, q] = absent
                elif delta[p, q]:
                    css.iat[p, q] = changed_css
                elif values[p, q] == 0:
                    css.iat[p, q] = grey
        return css

    def frame(values: np.ndarray, labels: list[str]) -> pd.DataFrame:
        return pd.DataFrame(values, index=labels, columns=labels)

    def renamed(css: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
        css.index, css.columns = labels, labels
        return css

    css_d = pd.DataFrame("", index=range(size_all), columns=range(size_all))
    for p in range(size_all):
        for q in range(size_all):
            value = delta[p, q]
            css_d.iat[p, q] = (f"background:{_hex(_rgb(CHANGED, 0.5))};color:#000;font-weight:bold" if value > 0 else
                               f"background:{_hex(_rgb(PALETTE[0], 0.5))};color:#000;font-weight:bold" if value < 0
                               else grey)

    # dessins
    v_before = regles.of_graph(before)
    g_before = Graphe.from_graph(before)
    atoms, bonds = _highlights(before.mol, g_before, v_before, regles)
    for a in [p for p in range(n_b) if delta[p].any()]:
        atoms.setdefault(a, []).insert(0, _rgb(CHANGED if row["permise"] else REFUSED))
    verdict = (f'<b style="color:{CHANGED}">permise</b>' if row["permise"] else
               f'<b style="color:{PALETTE[0]}">refusée</b> '
               + "".join(regles.chip(name) for name in _broken(row["règles cassées"])))
    drawn = {"graph": after, "permise": row["permise"], "violations": regles.of_graph(after)}
    display(_grid([
        _cell(_png(before.mol, atoms, bonds, size), f"avant : <code>{html.escape(before.canonical_smiles)}</code>"
              f"<br>{_chips(v_before, regles)}", size[0]),
        _mutation_cell(before, drawn, v_before, regles, size,
                       f"après : <code>{html.escape(after.aromatic_canonical_smiles)}</code><br>"
                       f"{html.escape(row['sous-famille'])} · {html.escape(row['mutation'])}<br>{verdict}"),
    ]))

    # matrices
    blocks = [
        ("M avant", _matrix_html(frame(aligned_b, labels_b), renamed(css_for(aligned_b, new_rows), labels_b))),
        ("M après", _matrix_html(frame(aligned_a, labels_a), renamed(css_for(aligned_a, removed), labels_a))),
        ("ΔM = après − avant", _matrix_html(frame(delta, labels_d), renamed(css_d, labels_d))),
    ]
    display(HTML('<div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start">' + "".join(
        f'<div><div style="font:bold 13px sans-serif;margin-bottom:4px">{title}</div>{table}</div>'
        for title, table in blocks) + "</div>"))

    # ce qui change
    plain = [f"{types_b[p]}{p}" if p < n_b else labels_a[p] for p in range(size_all)]
    lines = []
    for p in range(size_all):
        if p in removed:
            lines.append(f"atome retiré : {types_b[p]}{p}")
        elif p in new_rows:
            lines.append(f"atome ajouté : {labels_a[p]}")
        elif delta[p, p]:
            lines.append(f"élément {types_b[p]}{p} → {types_a[position[p]]}{p} (diagonale {aligned_b[p, p]} → "
                         f"{aligned_a[p, p]})")
    for p in range(size_all):
        for q in range(p + 1, size_all):
            if delta[p, q]:
                lines.append(f"liaison {plain[p]}–{plain[q]} : {aligned_b[p, q]} → {aligned_a[p, q]}")
    print(f"{row['sous-famille']} ({row['famille']}) : {row['mutation']}")
    print("  " + "\n  ".join(lines))
    print(f"  règles du notebook : {'permise' if row['permise'] else 'refusée, cassées : ' + row['règles cassées']}")
    return after.aromatic_canonical_smiles


# ------------------------------------------------------------------------------
#  Test sur beaucoup de molécules
# ------------------------------------------------------------------------------
def sample_valid(n: int = 1000, regles: Regles | None = None, seed: int = 0, max_atoms: int | None = None,
                 max_rounds: int = 20, results_dir: str = WALKS_DIR, names: Iterable[str] | None = None
                 ) -> pd.DataFrame:
    """n molécules différentes tirées dans les marches newmut (sample_walks) qui respectent toutes les règles du
    notebook (aucune violation). Tirages successifs jusqu'à n (au plus max_rounds). names : expériences
    (ex. ["newmut_walk/test_Regle_V1"] ; None : toutes les marches newmut)."""
    assert regles is not None, "sample_valid(n, REGLES, …)"
    kept: list[dict] = []
    seen: set[str] = set()
    drawn = duplicates = refused = unreadable = 0
    for round_ in range(max_rounds):
        batch = sample_walks(max(2 * (n - len(kept)), 100), seed=seed + 1000 * round_, names=names,
                             max_atoms=max_atoms, results_dir=results_dir)
        for row in batch.to_dict("records"):
            drawn += 1
            if row["SMILES"] in seen:
                duplicates += 1
                continue
            seen.add(row["SMILES"])
            try:
                violations = regles.of_smiles(row["SMILES"])
            except ValueError:
                unreadable += 1
                continue
            if violations:
                refused += 1
                continue
            kept.append(row)
            if len(kept) == n:
                break
        if len(kept) == n:
            break
    print(f"{drawn} tirées : {duplicates} doublons, {unreadable} illisibles, {refused} hors règles, "
          f"{len(kept)} gardées" + ("" if len(kept) == n else f" (moins que les {n} demandées)"))
    return pd.DataFrame(kept)


def _smiles_list(molecules: Any) -> list[str]:
    if isinstance(molecules, pd.DataFrame):
        return list(molecules.SMILES)
    return [smiles for smiles, _ in _entries(molecules)]


# ------------------------------------------------------------------------------
#  Calcul sur plusieurs cœurs
# ------------------------------------------------------------------------------
# état lu par les processus de calcul : ils sont créés par fork et en héritent (les fonctions écrites dans le
# notebook, règles, SOUS_FAMILLE, CONTEXTE, ne pourraient pas leur être envoyées autrement)
PARALLEL_STATE: dict[str, Any] = {}


def parallel_map(func: Callable[[Any], Any], chunks: list[Any], n_jobs: int | None, state: dict[str, Any]
                 ) -> Iterable[Any]:
    """func(chunk) pour chaque paquet, résultats dans l'ordre des paquets. n_jobs : nombre de cœurs (None ou 1 : un
    seul, sans processus ; -1 : tous). func doit être une fonction de module qui lit PARALLEL_STATE."""
    PARALLEL_STATE.clear()
    PARALLEL_STATE.update(state)
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if not n_jobs or n_jobs <= 1 or len(chunks) <= 1:
        yield from map(func, chunks)
        return
    with warnings.catch_warnings():
        # Python 3.12 avertit que fork dans un processus à plusieurs fils (noyau Jupyter) peut bloquer ; les calculs
        # ici n'utilisent pas ces fils
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")
        with multiprocessing.get_context("fork").Pool(min(n_jobs, len(chunks))) as pool:
            yield from pool.imap(func, chunks)


def chunked(items: list[Any], n_jobs: int | None, per_job: int = 8, max_size: int = 500) -> list[list[Any]]:
    """Paquets d'éléments consécutifs : assez de paquets pour occuper les cœurs, pas trop gros."""
    n = max(1, n_jobs if n_jobs and n_jobs > 0 else (os.cpu_count() or 1))
    size = max(1, min(max_size, -(-len(items) // (n * per_job))))
    return [items[k:k + size] for k in range(0, len(items), size)]


def _test_chunk(pairs: list[tuple[int, str]]) -> list[dict]:
    """Lignes de mutation_test pour un paquet de (numéro, SMILES)."""
    st = PARALLEL_STATE
    rows = []
    for m, smiles in pairs:
        _, found = _evaluate(smiles, st["regles"], st["sous_famille"], st["selection"], st["ajout_complet"],
                             familles=st["familles"], contexte=st["contexte"])
        n_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
        rows.extend({"molécule": m, "SMILES": smiles, "atomes": n_atoms, **row} for row in found)
    return rows


# ------------------------------------------------------------------------------
#  Cache du test : empreinte de tout ce qui détermine le résultat
# ------------------------------------------------------------------------------
_SIMPLE = (int, float, str, bool, bytes, tuple, list, dict, set, frozenset, type(None))
_CODE_DIRS = ("evomol/newmut", "evomol/action/molecular_graph", "evomol/representation", "teste")
_CODE_FILES = ("teste/parametrage_regles.py", "scripts/newmut/newmut_walk.py")


def _code_fingerprint(code: Any, h: Any) -> None:
    h.update(code.co_code)
    h.update(repr(code.co_names).encode())
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            _code_fingerprint(const, h)
        else:
            h.update(repr(const).encode())


def _object_fingerprint(obj: Any, h: Any, seen: set[int]) -> None:
    """Code d'une fonction écrite dans un notebook (règle, SOUS_FAMILLE, CONTEXTE) et valeurs des noms qu'elle lit
    (paramètres, autres fonctions), récursivement : un paramètre ou une fonction modifiés changent l'empreinte,
    même si le notebook n'est pas enregistré."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "rules") and hasattr(obj, "compte_separe"):  # Regles (même d'un module rechargé)
        h.update(repr((obj.names, obj.compte_separe)).encode())
        for rule in obj.rules.values():
            _object_fingerprint(rule, h, seen)
        return
    code = getattr(obj, "__code__", None)
    if code is None:
        h.update(repr(obj).encode() if isinstance(obj, _SIMPLE) else type(obj).__name__.encode())
        return
    _code_fingerprint(code, h)
    module = (getattr(obj, "__module__", "") or "").split(".")[0]
    if module in ("evomol", "parametrage_regles", "contextes_mutations", "numpy", "pandas", "rdkit", "builtins"):
        return  # code des modules : couvert par l'empreinte des fichiers
    names = set(code.co_names)
    stack = [c for c in code.co_consts if hasattr(c, "co_code")]
    while stack:  # noms lus par les fonctions imbriquées
        inner = stack.pop()
        names |= set(inner.co_names)
        stack += [c for c in inner.co_consts if hasattr(c, "co_code")]
    for name in sorted(names):
        if name in obj.__globals__:
            value = obj.__globals__[name]
            h.update(name.encode())
            if callable(value) or hasattr(value, "compte_separe") or isinstance(value, _SIMPLE):
                _object_fingerprint(value, h, seen)


def test_fingerprint(smiles_list: list[str], regles: Regles, sous_famille: SousFamille, selection: Any,
                     ajout_complet: bool, familles: Any, contexte: Any) -> str:
    """Empreinte de ce qui détermine le résultat de mutation_test : molécules, règles et SOUS_FAMILLE / CONTEXTE
    (code et paramètres lus), réglages, espace d'action (classes et options) et code d'evomol et des modules."""
    h = hashlib.sha1()
    h.update("\n".join(smiles_list).encode())
    h.update(repr((selection, ajout_complet, familles)).encode())
    seen: set[int] = set()
    for obj in (regles, sous_famille, contexte):
        _object_fingerprint(obj, h, seen)
    options = {
        "space": [a.class_name() + "/" + getattr(a, "__module__", "") for a in MolecularGraph.action_space],
        "atoms": list(Molecule.accepted_atoms), "max": Molecule.max_heavy_atoms,
        "add": (mg.AddAtomMG.allow_multiple_bonds, mg.AddAtomMG.avoid_sp3_on_sp3),
        "bond": (mg.ChangeBondMG.by_one_step, mg.ChangeBondMG.avoid_bond_breaking, mg.ChangeBondMG.avoid_bond_forming),
        "rules": (mr.RULES_VERSION, mr.R4B, mr.R2_HETERO, mr.R2_HETERO_OUTSIDE_RINGS),
    }
    for action in MolecularGraph.action_space:
        for attr in ("shift_bonds", "raise_bonds", "fragments", "groups"):
            if hasattr(action, attr):
                options[f"{action.class_name()}.{attr}"] = repr(getattr(action, attr))
    h.update(json.dumps(options, sort_keys=True, default=str).encode())
    for folder in _CODE_DIRS:
        path = os.path.join(ROOT, folder)
        for name in sorted(os.listdir(path)):
            if name.endswith(".py") and (folder != "teste" or name in ("parametrage_regles.py",)):
                with open(os.path.join(path, name), "rb") as f:
                    h.update(f.read())
    for name in _CODE_FILES:
        with open(os.path.join(ROOT, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def cache_path(cache: str, molecules: Any, regles: Regles, sous_famille: SousFamille, selection: Any,
               ajout_complet: bool, familles: Any, contexte: Any) -> str:
    """Fichier du cache pour ces réglages (voir test_fingerprint)."""
    key = test_fingerprint(_smiles_list(molecules), regles, sous_famille, selection, ajout_complet, familles, contexte)
    return os.path.join(cache, f"{families_name(familles)}_{key}.parquet")


def save_to_cache(essai: pd.DataFrame, cache: str, molecules: Any, regles: Regles, sous_famille: SousFamille,
                  selection: Any, ajout_complet: bool, familles: Any, contexte: Any = None) -> str:
    """Enregistre un résultat de mutation_test déjà calculé (ex. encore en mémoire dans un notebook) dans le cache,
    avec les réglages qui l'ont produit : mutation_test(…, cache=…) le relira au lieu de tout refaire."""
    path = cache_path(cache, molecules, regles, sous_famille, selection, ajout_complet, familles, contexte)
    os.makedirs(cache, exist_ok=True)
    essai.to_parquet(path, index=False)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(essai.attrs, f, ensure_ascii=False, default=list)
    print(f"{len(essai)} mutations gardées dans le cache : {path}")
    return path


def mutation_test(molecules: Any, regles: Regles, sous_famille: SousFamille,
                  selection: str | Iterable[str] | None = None, ajout_complet: bool = True,
                  progress_every: int = 100, familles: str | Iterable[str] | None = None,
                  contexte: Contexte | None = None, n_jobs: int | None = None,
                  cache: str | None = None) -> pd.DataFrame:
    """Toutes les mutations candidates des sous-familles choisies (selection : noms de sous-familles ou de familles,
    motifs « add* » ; None : toutes) sur chaque molécule (tableau de sample_valid, ou liste de SMILES). Une ligne par
    candidate : molécule, SMILES, atomes, famille, sous-famille, n°, mutation, permise, règles cassées. familles :
    seulement les mutations de ces familles (beaucoup plus rapide ; None : toutes). Seules les mutations
    sélectionnées sont jugées par les règles (cycles calculés seulement pour elles). n_jobs : nombre de cœurs
    (None ou 1 : un seul ; -1 : tous) ; même résultat, dans le même ordre. cache : dossier où garder le résultat
    (parquet) ; il est relu tant que rien de ce qui le détermine n'a changé (test_fingerprint), refait sinon."""
    smiles_list = _smiles_list(molecules)
    if cache:
        path = cache_path(cache, smiles_list, regles, sous_famille, selection, ajout_complet, familles, contexte)
        if os.path.exists(path):
            essai = pd.read_parquet(path)
            with open(path + ".json", encoding="utf-8") as f:
                essai.attrs.update(json.load(f))
            print(f"{len(essai)} mutations sur {essai.attrs.get('molécules')} molécules lues dans le cache : {path}")
            return essai
        essai = mutation_test(smiles_list, regles, sous_famille, selection, ajout_complet, progress_every, familles,
                              contexte, n_jobs)
        save_to_cache(essai, cache, smiles_list, regles, sous_famille, selection, ajout_complet, familles, contexte)
        return essai
    start = time.perf_counter()
    columns = ["molécule", "SMILES", "atomes", "famille", "sous-famille", "n°", "mutation", "permise",
               "règles cassées", *(["contexte"] if contexte else [])]
    repeated = [c for c in ("famille", "sous-famille", "règles cassées", "contexte") if c in columns]
    frames: list[pd.DataFrame] = []
    n_rows = 0
    state = {"regles": regles, "sous_famille": sous_famille, "selection": selection, "ajout_complet": ajout_complet,
             "familles": familles, "contexte": contexte}
    chunks = chunked(list(enumerate(smiles_list)), n_jobs, max_size=progress_every or 500)
    done = 0
    for chunk, found in zip(chunks, parallel_map(_test_chunk, chunks, n_jobs, state)):
        if found:
            # un tableau par paquet (moins de mémoire qu'une liste de dictionnaires) ; textes répétés gardés en un
            # seul exemplaire
            frame = pd.DataFrame(found, columns=columns)
            for column in repeated:
                frame[column] = [sys.intern(value) for value in frame[column]]
            frames.append(frame)
            n_rows += len(frame)
        before, done = done, done + len(chunk)
        if progress_every and done // progress_every > before // progress_every:
            print(f"  {done}/{len(smiles_list)} molécules, {n_rows} mutations, "
                  f"{time.perf_counter() - start:.0f} s")
    elapsed = time.perf_counter() - start
    print(f"{n_rows} mutations sur {len(smiles_list)} molécules en {elapsed:.1f} s")
    essai = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    del frames
    essai["permise"] = essai["permise"].astype(bool)
    essai.attrs.update({"molécules": len(smiles_list), "ajout_complet": ajout_complet, "durée": elapsed,
                        "familles": familles})
    if essai.empty:
        print(f"ATTENTION : aucune mutation. SELECTION = {selection!r} ne correspond à aucune sous-famille des "
              f"familles {familles!r} (familles : {', '.join(families(ajout_complet))}). Ex. bond* → "
              "ChangeBondMG, add* → AddAtomMG, sub* → SubstituteShiftMG.")
    return essai


def test_summary(essai: pd.DataFrame) -> pd.DataFrame:
    """Par sous-famille : molécules où elle a au moins une candidate, candidates, permises, refusées, % acceptées, % refusées,
    permises par molécule (où elle existe), molécules où au moins une est permise."""
    order = _family_order()
    rows = []
    for (family, name), group in essai.groupby(["famille", "sous-famille"], sort=False):
        molecules = group.molécule.nunique()
        allowed = int(group.permise.sum())
        rows.append({
            "famille": family, "sous-famille": name, "molécules": molecules, "candidates": len(group),
            "permises": allowed, "refusées": len(group) - allowed, "% acceptées": round(100 * allowed / len(group), 1),
            "% refusées": round(100 * (len(group) - allowed) / len(group), 1),
            "permises par molécule": round(allowed / molecules, 2),
            "molécules avec une permise": int(group[group.permise].molécule.nunique()),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(["famille", "sous-famille"], key=lambda col: col.map(order).fillna(99)
                              if col.name == "famille" else col)
    return table.set_index(["famille", "sous-famille"])


def test_rules(essai: pd.DataFrame, regles: Regles, seule: bool = False, pourcentage: bool = False) -> Any:
    """Sous-famille × règle : nombre de mutations refusées qui cassent la règle (une mutation peut en casser
    plusieurs). seule : seulement les refus où c'est la seule règle cassée. pourcentage : en % des candidates de la
    sous-famille. Tableau coloré (dégradé par colonne)."""
    data = []
    for name, group in essai.groupby("sous-famille", sort=False):
        broken = [_broken(text) for text in group.loc[~group.permise, "règles cassées"]]
        row: dict[str, Any] = {"sous-famille": name, "candidates": len(group), "refusées": len(broken)}
        for rule in regles.names:
            count = sum(1 for b in broken if (b == [rule] if seule else rule in b))
            row[rule] = round(100 * count / len(group), 1) if pourcentage else count
        data.append(row)
    table = pd.DataFrame(data)
    if table.empty:
        return table
    order = {name: k for k, name in enumerate(test_summary(essai).index.get_level_values(1))}
    table = table.sort_values("sous-famille", key=lambda col: col.map(order)).set_index("sous-famille")
    used = [rule for rule in regles.names if table[rule].any()]
    return table.style.background_gradient(cmap="Reds", subset=used or None, axis=0).format(
        "{:.1f}" if pourcentage else "{:d}", subset=regles.names)


def test_combinations(essai: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Pour chaque sous-famille, les combinaisons de règles cassées les plus fréquentes parmi les refusées (top
    par sous-famille), avec leur part des refusées."""
    refused = essai[~essai.permise.astype(bool)]
    rows = []
    for name, group in refused.groupby("sous-famille", sort=False) if len(refused) else []:
        for combination, count in group["règles cassées"].value_counts().head(top).items():
            rows.append({"sous-famille": name, "règles cassées": combination, "mutations": int(count),
                         "% des refusées": round(100 * count / len(group), 1)})
    return pd.DataFrame(rows, columns=["sous-famille", "règles cassées", "mutations", "% des refusées"])


def test_examples(essai: pd.DataFrame, regles: Regles,
                  nom: str | Iterable[str] | None = None, regle: str | None = None, permise: bool | None = False,
                  n: int = 12, seed: int = 0, size: tuple[int, int] = (220, 180)) -> HTML | None:
    """Exemples tirés au hasard dans le test : avant (atomes modifiés en vert / gris) → après (atomes des nouvelles
    violations de la couleur de leur règle). nom : sous-famille(s) ou motif ; regle : seulement les refus qui
    cassent cette règle ; permise : True / False / None (toutes)."""
    ajout_complet = essai.attrs.get("ajout_complet", True)
    attrs = dict(essai.attrs)
    patterns = _patterns(nom)
    mask = pd.Series(True, index=essai.index)
    if patterns is not None:
        mask &= pd.Series([_selected(patterns, f, s) for f, s in zip(essai.famille, essai["sous-famille"])],
                          index=essai.index)
    if permise is not None:
        mask &= essai.permise == permise
    if regle is not None:
        mask &= essai["règles cassées"].map(lambda text: regle in _broken(text))
    rows = essai[mask]
    if rows.empty:
        print("aucune mutation ne correspond")
        return None
    rows = rows.sample(min(n, len(rows)), random_state=seed)
    boxes = []
    for row in rows.to_dict("records"):
        molecule = Molecule(row["SMILES"])
        before = molecule.get_representation(MolecularGraph)
        action = candidates(molecule, ajout_complet, attrs.get("familles"))[row["n°"]][1]
        after = rf.result_graph(action, before)  # type: ignore[arg-type]
        v_before = regles.of_graph(before)
        changed = [a for a in nw._changed_atoms(before, after) if a < before.nb_atoms]  # pylint: disable=W0212
        color = _rgb(CHANGED if row["permise"] else REFUSED)
        verdict = (f'<b style="color:{CHANGED}">permise</b>' if row["permise"] else
                   "".join(regles.chip(name) for name in _broken(row["règles cassées"])))
        drawn = {"graph": after, "permise": row["permise"], "violations": regles.of_graph(after)}
        caption = f"{html.escape(row['sous-famille'])} · {html.escape(row['mutation'])}<br>{verdict}"
        boxes.append('<div style="display:flex;border:1px solid #ddd;border-radius:4px;margin:4px">'
                     + _cell(_png(before.mol, {a: [color] for a in changed}, {}, size),
                             f"<code style='font-size:10px'>{html.escape(row['SMILES'])}</code>", size[0])
                     + _mutation_cell(before, drawn, v_before, regles, size, caption) + "</div>")
    return _grid(boxes)


# ------------------------------------------------------------------------------
#  Règles de mutation écrites à la main (REGLES_MUTATION du notebook)
# ------------------------------------------------------------------------------
VERDICTS = ("autorisée", "interdite", "dépend")


def _rule_entry(rules_mutation: list[tuple], family: str, subfamily: str) -> tuple | None:
    """Première ligne de la liste dont le motif correspond (l'ordre compte : écrire les cas précis d'abord)."""
    for entry in rules_mutation:
        if _selected([entry[0]], family, subfamily):
            return entry
    return None


def _check_entries(rules_mutation: list[tuple], regles: Regles) -> None:
    for entry in rules_mutation:
        if len(entry) < 3 or entry[1] not in VERDICTS:
            raise ValueError(f"ligne mal écrite : {entry!r} (attendu : (motif, {' / '.join(VERDICTS)}, règles, "
                             "commentaire))")
        unknown = [r for r in _rule_list(entry[2]) if r not in regles.names]
        if unknown:
            raise ValueError(f"{entry[0]} : règle(s) inconnue(s) {unknown} (règles : {regles.names})")


def _rule_list(text: str | None) -> list[str]:
    return [r.strip() for r in (text or "").replace("+", ",").split(",") if r.strip()]


def check_mutation_rules(essai: pd.DataFrame, rules_mutation: list[tuple], regles: Regles) -> Any:
    """Compare la liste écrite à la main au test : pour chaque sous-famille du test, la ligne qui s'applique, le
    verdict écrit, ce qui est observé (% acceptées, % refusées, règles cassées) et l'écart :
    - interdite : aucune mutation permise, et chaque refus casse au moins une des règles écrites ;
    - autorisée : toutes permises ;
    - dépend : chaque refus casse au moins une des règles écrites ;
    - « non écrite » : aucune ligne ne correspond."""
    _check_entries(rules_mutation, regles)
    rows = []
    summary = test_summary(essai)
    for (family, name), stat in summary.iterrows():
        group = essai[(essai.famille == family) & (essai["sous-famille"] == name)]
        refused = [_broken(t) for t in group.loc[~group.permise, "règles cassées"]]
        entry = _rule_entry(rules_mutation, family, name)
        row = {"famille": family, "sous-famille": name, "candidates": int(stat.candidates),
               "% acceptées": stat["% acceptées"], "% refusées": stat["% refusées"], "règles cassées (observées)": _count_text(r for b in refused for r in b)}
        if entry is None:
            row.update({"ligne": "", "verdict écrit": "", "règles écrites": "", "écart": "non écrite"})
        else:
            written = _rule_list(entry[2])
            problems = []
            if entry[1] == "autorisée" and refused:
                problems.append(f"{len(refused)} refusée(s)")
            if entry[1] == "interdite" and stat.permises:
                problems.append(f"{int(stat.permises)} permise(s)")
            if entry[1] in ("interdite", "dépend"):
                other = [b for b in refused if not set(b) & set(written)]
                if other:
                    problems.append(f"{len(other)} refus par d'autres règles ({_count_text(r for b in other for r in b)})")
            row.update({"ligne": entry[0], "verdict écrit": entry[1], "règles écrites": ", ".join(written),
                        "écart": "; ".join(problems) or "OK"})
        rows.append(row)
    table = pd.DataFrame(rows).set_index(["famille", "sous-famille"])
    unused = [entry[0] for entry in rules_mutation
              if not any(_selected([entry[0]], f, s) for f, s in summary.index)]
    ok = (table["écart"] == "OK").sum()
    print(f"{ok} sous-familles conformes, {(table['écart'] == 'non écrite').sum()} non écrites, "
          f"{len(table) - ok - (table['écart'] == 'non écrite').sum()} en écart, sur {len(table)} présentes dans le test")
    if unused:
        print("lignes sans sous-famille dans ce test :", ", ".join(unused))
    styler = table.style
    mapper = getattr(styler, "map", None) or styler.applymap
    return mapper(lambda v: f"background:{_hex(_rgb(CHANGED, 0.35))};color:#000" if v == "OK" else
                  ("background:#eee;color:#000" if v == "non écrite" else
                   f"background:{_hex(_rgb(PALETTE[0], 0.45))};color:#000"), subset=["écart"])


def write_mutation_rules(rules_mutation: list[tuple], regles: Regles, path: str,
                         essai: pd.DataFrame | None = None) -> str:
    """Écrit la liste dans un fichier markdown (un tableau par verdict). Avec essai : ajoute ce qui est observé
    dans le test pour chaque ligne (candidates, % acceptées, % refusées, règles cassées)."""
    _check_entries(rules_mutation, regles)
    observed: dict[str, dict] = {}
    if essai is not None:
        for entry in rules_mutation:
            mask = [_rule_entry(rules_mutation, f, s) is entry for f, s in zip(essai.famille, essai["sous-famille"])]
            group = essai[pd.Series(mask, index=essai.index)]
            refused = [_broken(t) for t in group.loc[~group.permise, "règles cassées"]]
            observed[entry[0]] = {
                "candidates": len(group),
                "acceptées": f"{100 * group.permise.mean():.1f} %" if len(group) else "—",
                "refusées": f"{100 * (1 - group.permise.mean()):.1f} %" if len(group) else "—",
                "cassées": _count_text(r for b in refused for r in b) or "—",
            }
    lines = ["# Règles de mutation", "",
             "Écrites à la main dans `teste/parametrage_regles.ipynb` (`REGLES_MUTATION`) ; ce fichier est réécrit "
             "par la cellule qui l'exporte. Une sous-famille prend la première ligne qui lui correspond.", "",
             f"Règles de la matrice utilisées : {', '.join(regles.names)}.", ""]
    if essai is not None:
        lines += [f"Observé : test sur {essai.attrs.get('molécules', '?')} molécules des marches newmut qui respectent "
                  "les règles (`mutation_test`).", ""]
    for verdict in VERDICTS:
        entries = [e for e in rules_mutation if e[1] == verdict]
        if not entries:
            continue
        lines += [f"## {dict(autorisée='Autorisées', interdite='Interdites', dépend='Selon la molécule')[verdict]}", ""]
        header = "| sous-famille | règle(s) qui interdisent | commentaire |"
        sep = "|---|---|---|"
        if essai is not None:
            header += " candidates | % acceptées | % refusées | règles cassées (observé) |"
            sep += "---|---|---|---|"
        lines += [header, sep]
        for e in entries:
            line = f"| `{e[0]}` | {', '.join(_rule_list(e[2])) or '—'} | {e[3] if len(e) > 3 else ''} |"
            if essai is not None:
                o = observed[e[0]]
                line += f" {o['candidates']} | {o['acceptées']} | {o['refusées']} | {o['cassées']} |"
            lines.append(line)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"écrit : {path}")
    return path


def test_table(essai: pd.DataFrame, regles: Regles) -> Any:
    """Un seul tableau par sous-famille : candidates, acceptées, refusées (nombres et %), règles qui refusent, une
    colonne par règle avec le nombre de refus qui la cassent (une mutation refusée peut en casser plusieurs), puis
    les refus où une seule règle est cassée : leur nombre et la règle seule (« R2a ×575, R3 ×12 »)."""
    order = _family_order()
    rows = []
    for (family, name), group in essai.groupby(["famille", "sous-famille"], sort=False):
        broken = [_broken(t) for t in group.loc[~group.permise, "règles cassées"]]
        accepted = int(group.permise.sum())
        row: dict[str, Any] = {
            "famille": family, "sous-famille": name, "candidates": len(group),
            "acceptées": accepted, "refusées": len(group) - accepted,
            "% acceptées": round(100 * accepted / len(group), 1),
            "% refusées": round(100 * (len(group) - accepted) / len(group), 1),
            "règles qui refusent": ", ".join(r for r in regles.names if any(r in b for b in broken)) or "—",
        }
        for rule in regles.names:
            row[rule] = sum(1 for b in broken if rule in b)
        # refus où une seule règle est cassée : laquelle, combien de fois
        alone = [b[0] for b in broken if len(b) == 1]
        row["refus par une seule règle"] = len(alone)
        row["règle seule (refus)"] = _count_text(alone) or "—"
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(["famille", "sous-famille"], key=lambda col: col.map(order).fillna(99)
                              if col.name == "famille" else col).set_index(["famille", "sous-famille"])
    styler = table.style.format("{:.1f}", subset=["% acceptées", "% refusées"])
    styler = styler.set_table_styles([{"selector": "th.col_heading", "props": "text-align:center"}])
    mapper = getattr(styler, "map", None) or styler.applymap
    styler = mapper(lambda v: f"background:{_hex(_rgb(CHANGED, 0.35))};color:#000" if v else "color:#bbb",
                    subset=["acceptées", "% acceptées"])
    styler = (getattr(styler, "map", None) or styler.applymap)(
        lambda v: f"background:{_hex(_rgb(PALETTE[0], 0.4))};color:#000" if v else "color:#bbb",
        subset=["refusées", "% refusées"])
    styler = (getattr(styler, "map", None) or styler.applymap)(
        lambda v: "font-weight:bold" if v not in (0, "—") else "color:#bbb",
        subset=["refus par une seule règle", "règle seule (refus)"])
    for rule in regles.names:  # nombre de refus par règle, de la couleur de la règle
        css = f"background:{_hex(_rgb(regles.colors[rule]))};color:#000;font-weight:bold"
        styler = (getattr(styler, "map", None) or styler.applymap)(
            lambda v, css=css: css if v else "color:#bbb", subset=[rule])
    return styler


# ------------------------------------------------------------------------------
#  Exemples acceptées / refusées en une image, résultats en fichiers
# ------------------------------------------------------------------------------
def families_name(familles: str | Iterable[str] | None) -> str:
    """Nom de fichier pour FAMILLES : « AddAtomMG », « AddAtomMG_ChangeBondMG », « toutes »."""
    names = _family_patterns(familles)
    if names is None:
        return "toutes"
    return re.sub(r"[^\w.-]+", "-", "_".join(names))


def selection_mask(essai: pd.DataFrame, nom: str | Iterable[str] | None) -> pd.Series:
    """Lignes dont la sous-famille (ou la famille) correspond à nom (nom, motif ; None : toutes). Le test des noms
    est fait une fois par sous-famille différente, pas ligne par ligne."""
    patterns = _patterns(nom)
    if patterns is None:
        return pd.Series(True, index=essai.index)
    pairs = essai[["famille", "sous-famille"]].drop_duplicates()
    kept = [(f, s) for f, s in zip(pairs.famille, pairs["sous-famille"]) if _selected(patterns, f, s)]
    return (essai["sous-famille"].isin({s for _, s in kept}) & essai.famille.isin({f for f, _ in kept}))


def rule_mask(broken: pd.Series, regle: str) -> pd.Series:
    """Lignes dont « règles cassées » (« R2a, R5 ») contient la règle regle."""
    return broken.str.contains(rf"(?:^|, ){re.escape(regle)}(?:,|$)", regex=True)


def _example_rows(essai: pd.DataFrame, nom: str | Iterable[str] | None, regle: str | None, permise: bool,
                  n: int, seed: int) -> tuple[list[dict], int]:
    mask = (essai.permise == permise) & selection_mask(essai, nom)
    if regle is not None and not permise:
        mask &= rule_mask(essai["règles cassées"], regle)
    rows = essai[mask]
    if not len(rows):
        return [], 0
    return rows.sample(min(n, len(rows)), random_state=seed).to_dict("records"), len(rows)


def examples_page(columns: list[tuple[str, str, list[dict], str]], regles: Regles, title: str,
                  ajout_complet: bool = True, familles: str | Iterable[str] | None = None, path: str | None = None,
                  size: tuple[int, int] = (260, 210)) -> Any:
    """Image en colonnes : columns = [(titre, couleur, lignes du test, texte après le titre)]. Chaque ligne : avant
    (atomes modifiés) → après (atomes des nouvelles violations de la couleur de leur règle), sous-famille, mutation,
    règles cassées, SMILES (et contexte si la ligne en a un). path : enregistre l'image (PNG). Renvoie l'image PIL."""
    # pylint: disable=import-outside-toplevel
    import io
    from PIL import Image, ImageDraw, ImageFont

    def font(name: str, px: int) -> Any:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            return ImageFont.load_default()

    text_font = font("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    bold_font = font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
    mono_font = font("/System/Library/Fonts/Menlo.ttc", 10)
    w, h = size
    with_context = any(row.get("contexte") for _, _, rows, _ in columns for row in rows)
    caption_h = 62 + (16 if with_context else 0)
    box_w, box_h = 2 * w + 30, h + caption_h

    def example(row: dict) -> Any:
        molecule = Molecule(row["SMILES"])
        before = molecule.get_representation(MolecularGraph)
        action = candidates(molecule, ajout_complet, familles)[row["n°"]][1]
        after = rf.result_graph(action, before)  # type: ignore[arg-type]
        changed = [a for a in nw._changed_atoms(before, after) if a < before.nb_atoms]  # pylint: disable=W0212
        color = _rgb(CHANGED if row["permise"] else REFUSED)
        img_before = _png(before.mol, {a: [color] for a in changed}, {}, size)
        g_after = Graphe.from_graph(after)
        v_before = regles.of_graph(before)
        v_after = regles.of_graph(after)
        new = _new_rule_atoms({"violations": v_after}, v_before, regles)
        atoms, bonds = _highlights(after.mol, g_after, new, regles)
        for a in nw._changed_atoms(before, after):  # pylint: disable=W0212
            atoms.setdefault(a, []).insert(0, color)
        img_after = _png(after.mol, atoms, bonds, size)
        box = Image.new("RGB", (box_w, box_h), "white")
        box.paste(Image.open(io.BytesIO(img_before)).convert("RGB"), (0, 0))
        box.paste(Image.open(io.BytesIO(img_after)).convert("RGB"), (w + 30, 0))
        draw = ImageDraw.Draw(box)
        draw.text((w + 8, h // 2 - 10), "→", fill="#555", font=bold_font)
        draw.text((4, h + 4), f"{row['sous-famille']} · {row['mutation']}", fill="black", font=text_font)
        x = 4
        for name in _broken(row["règles cassées"]):
            label_w = draw.textlength(name, font=text_font) + 8
            draw.rounded_rectangle([x, h + 22, x + label_w, h + 40], 3, fill=_hex(_rgb(regles.colors[name])),
                                   outline=regles.colors[name])
            draw.text((x + 4, h + 24), name, fill="black", font=text_font)
            x += label_w + 4
        smiles = row["SMILES"] if len(row["SMILES"]) <= 70 else row["SMILES"][:68] + "…"
        draw.text((4, h + 44), smiles, fill="#666", font=mono_font)
        if with_context and row.get("contexte"):
            draw.text((4, h + 60), f"contexte : {row['contexte']}"[:90], fill="#335", font=mono_font)
        draw.rectangle([0, 0, box_w - 1, box_h - 1], outline="#ddd")
        return box

    margin, header = 16, 70
    rows_max = max(1, *(len(rows) for _, _, rows, _ in columns))
    page = Image.new("RGB", (len(columns) * (box_w + margin) + margin, header + rows_max * (box_h + 8) + margin),
                     "white")
    draw = ImageDraw.Draw(page)
    draw.text((margin, 10), title, fill="black", font=bold_font)
    for k, (label, color, rows, text) in enumerate(columns):
        x0 = margin + k * (box_w + margin)
        draw.text((x0, 40), f"{label}{text}", fill=color, font=bold_font)
        if not rows:
            draw.text((x0, header), "aucune", fill="#777", font=text_font)
        for r, row in enumerate(rows):
            page.paste(example(row), (x0, header + r * (box_h + 8)))
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        page.save(path)
        print(f"image : {path}")
    return page


def test_examples_image(essai: pd.DataFrame, regles: Regles, nom: str | Iterable[str] | None = None,
                        regle: str | None = None, n: int = 8, seed: int = 0, path: str | None = None,
                        size: tuple[int, int] = (260, 210)) -> Any:
    """Une image : à gauche n mutations acceptées, à droite n refusées (regle : seulement les refus qui cassent
    cette règle), tirées au hasard dans le test. Chaque exemple : avant (atomes modifiés) → après (atomes des
    nouvelles violations de la couleur de leur règle). path : enregistre l'image (PNG). Renvoie l'image PIL."""
    familles = essai.attrs.get("familles")
    accepted, n_accepted = _example_rows(essai, nom, None, True, n, seed)
    refused, n_refused = _example_rows(essai, nom, regle, False, n, seed)
    columns = [("Acceptées", CHANGED, accepted, f" : {len(accepted)} exemples (sur {n_accepted} dans le test)"),
               ("Refusées" + (f" (cassent {regle})" if regle else ""), PALETTE[0], refused,
                f" : {len(refused)} exemples (sur {n_refused} dans le test)")]
    title = (f"{families_name(familles)} · {', '.join(_patterns(nom) or ['toutes les sous-familles'])} · "
             f"{essai.attrs.get('molécules', '?')} molécules")
    return examples_page(columns, regles, title, essai.attrs.get("ajout_complet", True), familles, path, size)


def save_test_table(essai: pd.DataFrame, regles: Regles, path: str) -> str:
    """Enregistre le tableau de test_table (une ligne par sous-famille) en CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table = test_table(essai, regles)
    (table.data if hasattr(table, "data") else table).to_csv(path)
    print(f"tableau : {path}")
    return path


def subfamily_file_name(name: str) -> str:
    """Nom de fichier d'une sous-famille : « addCs@Cd », « bondCC1-2 », « subCd-Os+dépl »."""
    return re.sub(r"[^\w@.+-]+", "_", name.replace("→", "-"))


def test_examples_images(essai: pd.DataFrame, regles: Regles, folder: str, regle: str | None = None, n: int = 8,
                         seed: int = 0) -> dict[str, str]:
    """Une image par sous-famille du test (test_examples_image : acceptées à gauche, refusées à droite), dans
    folder/{sous-famille}.png. Renvoie sous-famille → chemin."""
    os.makedirs(folder, exist_ok=True)
    paths = {}
    groups = list(essai.groupby("sous-famille", sort=False))  # découpé une seule fois
    for k, (name, group) in enumerate(groups, start=1):
        group.attrs.update(essai.attrs)
        path = os.path.join(folder, f"{subfamily_file_name(name)}.png")
        test_examples_image(group, regles, name, regle, n=n, seed=seed, path=None).save(path)
        paths[name] = path
        print(f"  {k}/{len(groups)} {name}")
    print(f"{len(paths)} images dans {folder}")
    return paths
