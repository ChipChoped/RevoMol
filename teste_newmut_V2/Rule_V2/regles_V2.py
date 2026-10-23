"""
Paramétrage des règles V2 (teste_newmut_V2/Rule_V2/parametrage_regles_V2.ipynb).

Les règles sont écrites dans le notebook : ce module n'en contient aucune. Il reprend les outils du notebook de V1
(teste_newmut_V1/parametrage_regles.py : Regles, listes bonnes / mauvaises, dessins, tirage dans les marches) et
ajoute ce qui change en V2 :
  - GrapheV2 : la molécule lue sur la matrice, avec les tableaux numpy dont les règles V2 ont besoin (degré, somme
    et ordre max de chaque ligne, nombre de doubles, état s / d / t de chaque atome, matrice d'adjacence), calculés
    une fois par molécule, et la molécule RDKit quand on part d'un SMILES (charges et radicaux, R3c). Il remplace
    Graphe dans les outils de V1 (pr.Graphe = GrapheV2) : evaluate, show_rules… donnent ce graphe aux règles ;
  - matrix_table : la matrice avec les H des valences V2 et l'état s / d / t de tous les atomes ;
  - sample_chembl / acceptance : molécules ou systèmes π de ChEMBL 34 (teste_newmut_V2/ChEMBL_comparaison/donnees),
    et part refusée par chaque règle ;
  - timing : temps des règles par molécule.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import Counter
from functools import cached_property
from itertools import combinations
from typing import Any, Iterable, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "teste_newmut_V1"))

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import parametrage_regles as pr  # teste_newmut_V1/parametrage_regles.py
from evomol.newmut import matrix_rules as mr
from evomol.representation import MolecularGraph

RDLogger.DisableLog("rdApp.*")

Violation = pr.Violation
Regles = pr.Regles

CHEMBL_DIR = os.path.join(ROOT, "teste_newmut_V2", "ChEMBL_comparaison", "donnees")
CHEMBL_FILES = {
    "molecules": os.path.join(CHEMBL_DIR, "chembl34_CNOFSB.tsv"),     # molécules entières (C N O F B, sans charge)
    "pi": os.path.join(CHEMBL_DIR, "systemes_pi_chembl34.tsv"),      # systèmes π extraits de ces molécules
}

_STATES = np.array(["s", "s", "d", "t"])   # état selon l'ordre de liaison le plus haut (0 : atome seul, s)


# ==============================================================================
#  MOLÉCULE DONNÉE AUX RÈGLES
# ==============================================================================
class GrapheV2(pr.Graphe):
    """Graphe de V1 (types, matrix, n, order, adj, bonds, bridges, rings, in_ring…) avec, en plus, des tableaux numpy
    (un élément par atome) :
      orders : matrice des ordres de liaison (diagonale à 0) ; adjacency : 1 si deux atomes sont liés ;
      deg : nombre de voisins lourds ; bond_sums : somme de la ligne ; max_order : ordre le plus haut ;
      n_double : nombre de doubles liaisons ; tags : état « s », « d » ou « t » (tous les éléments) ;
      is_dt : atome d ou t ; elements : élément de chaque atome ;
    components : nombre de morceaux ; mol : molécule RDKit si on part d'un SMILES (sinon None).
    """

    def __init__(self, types: Iterable[str], matrix: np.ndarray, mol: Chem.Mol | None = None):
        super().__init__(types, matrix)
        self.mol = mol
        orders = self.matrix.copy()
        np.fill_diagonal(orders, 0)
        self.orders = orders
        self.adjacency = (orders > 0).astype(np.int64)
        self.deg = self.adjacency.sum(axis=1)
        self.bond_sums = orders.sum(axis=1)
        self.max_order = orders.max(axis=1) if self.n else np.zeros(0, dtype=np.int64)
        self.n_double = (orders == 2).sum(axis=1)
        self.tags = _STATES[np.minimum(self.max_order, 3)]
        self.is_dt = self.max_order >= 2
        self.elements = np.array(self.types, dtype="<U2")

    @classmethod
    def from_graph(cls, mol_graph: MolecularGraph) -> GrapheV2:
        return cls(*mr.to_matrix(mol_graph), mol=mol_graph.mol)

    @cached_property
    def components(self) -> int:
        """Nombre de morceaux de la molécule."""
        parent = list(range(self.n))

        def root(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in self.bonds:
            parent[root(a)] = root(b)
        return len({root(x) for x in range(self.n)})

    @cached_property
    def cyclomatic(self) -> int:
        """Nombre de cycles indépendants (liaisons − atomes + morceaux)."""
        return len(self.bonds) - self.n + self.components


# les outils de V1 construisent maintenant des GrapheV2
pr.Graphe = GrapheV2

# outils de V1 repris tels quels
evaluate = pr.evaluate
summary = pr.summary
rule_table = pr.rule_table
style = pr.style
show_list = pr.show_list
sample_walks = pr.sample_walks
as_list = pr.as_list
print_list = pr.print_list
inspect = pr.inspect
show_items = pr.show_items
show_rules = pr.show_rules
violations_table = pr.violations_table


def graph_of_smiles(smiles: str) -> GrapheV2:
    return GrapheV2.from_graph(pr._mol_graph(smiles))  # pylint: disable=protected-access


# ==============================================================================
#  DÉTAIL D'UNE MOLÉCULE
# ==============================================================================
def matrix_table(smiles: str, regles: Regles, valences: dict[str, int]) -> Any:
    """Matrice de la molécule (électrons de valence sur la diagonale, ordres de liaison ailleurs), puis H (valences
    V2 : valences[élément], 4 pour un élément absent), état s / d / t, degré et règles où l'atome est en cause.
    Diagonale des atomes en cause et cases des liaisons en cause de la couleur de la règle."""
    g = graph_of_smiles(smiles)
    types, matrix = g.types, g.matrix
    violations = regles.violations(g)
    labels = [f"{element}{i}" for i, element in enumerate(types)]
    table = pd.DataFrame(matrix, index=labels, columns=labels)
    table["H"] = [max(valences.get(element, 4) - int(g.bond_sums[i]), 0) for i, element in enumerate(types)]
    table["état"] = list(g.tags)
    table["degré"] = g.deg
    rules_of_atom: dict[int, list[str]] = {}
    for v in violations:
        for a in v.atoms:
            if v.rule_id not in rules_of_atom.setdefault(a, []):
                rules_of_atom[a].append(v.rule_id)
    table["règles"] = [", ".join(rules_of_atom.get(i, [])) for i in range(len(types))]

    css = pd.DataFrame("", index=table.index, columns=table.columns)
    rules_column = len(types) + 3
    for v in violations:
        color = f"background:{pr._hex(pr._rgb(regles.colors.get(v.rule_id, pr.REFUSED)))};color:#000;font-weight:bold"
        for a in v.atoms:
            if not css.iat[a, a]:
                css.iat[a, a] = color
            if not css.iat[a, rules_column]:
                css.iat[a, rules_column] = color
        for a, b in combinations(v.atoms, 2):
            if g.order[a][b]:
                for i, j in ((a, b), (b, a)):
                    if not css.iat[i, j]:
                        css.iat[i, j] = color
    zero = (table.iloc[:, : len(types)] == 0) & (css.iloc[:, : len(types)] == "")
    css.iloc[:, : len(types)] = css.iloc[:, : len(types)].mask(zero, "color:#bbb")
    return table.style.apply(lambda _: css, axis=None)


# ==============================================================================
#  CHEMBL
# ==============================================================================
_chembl: dict[str, pd.DataFrame] = {}


def load_chembl(source: str = "pi") -> pd.DataFrame:
    """Table ChEMBL (gardée en mémoire) : source « molecules » (chembl_id, canonical_smiles) ou « pi » (smiles_pi,
    taille, occurrences, molecules, exemple)."""
    if source not in CHEMBL_FILES:
        raise ValueError(f"source inconnue : {source} (choix : {', '.join(CHEMBL_FILES)})")
    if source not in _chembl:
        _chembl[source] = pd.read_csv(CHEMBL_FILES[source], sep="\t", dtype=str, keep_default_na=False)
    return _chembl[source]


def sample_chembl(n: int = 30, seed: int | None = 0, source: str = "pi", max_atoms: int | None = None
                  ) -> list[tuple[str, str]]:
    """n molécules (source « molecules ») ou systèmes π (source « pi », tirés sans tenir compte de leur fréquence)
    de ChEMBL 34, au hasard, d'au plus max_atoms atomes lourds (None : toutes les tailles). Même format que
    BONNES : (SMILES, commentaire)."""
    table = load_chembl(source)
    order = list(range(len(table)))
    random.Random(seed).shuffle(order)
    found: list[tuple[str, str]] = []
    for k in order:
        if len(found) >= n:
            break
        row = table.iloc[k]
        if source == "pi":
            smiles, comment, size = row.smiles_pi, f"système π · {row.occurrences} fois · ex. {row.exemple}", None
            if max_atoms is not None:
                size = int(row.taille)
        else:
            smiles, comment, size = row.canonical_smiles, row.chembl_id, None
            if max_atoms is not None:
                mol = Chem.MolFromSmiles(smiles)
                size = mol.GetNumAtoms() if mol is not None else None
        if not smiles or (max_atoms is not None and (size is None or size > max_atoms)):
            continue
        found.append((smiles, comment))
    return found


def acceptance(items: Iterable[Any], regles: Regles) -> pd.DataFrame:
    """Part des molécules (SMILES ou (SMILES, commentaire)) acceptées, puis pour chaque règle : molécules qu'elle
    refuse (nombre et %), et celles qu'elle est seule à refuser."""
    broken: list[set[str]] = []
    unreadable = 0
    for smiles, _ in pr._entries(items):  # pylint: disable=protected-access
        try:
            g = graph_of_smiles(smiles)
        except ValueError:
            unreadable += 1
            continue
        broken.append({v.rule_id for v in regles.violations(g)})
    total = len(broken)
    rows = [{"règle": "acceptées", "molécules": sum(1 for b in broken if not b)}]
    for name in regles.names:
        rows.append({"règle": name, "molécules": sum(1 for b in broken if name in b),
                     "seule": sum(1 for b in broken if b == {name})})
    table = pd.DataFrame(rows, columns=["règle", "molécules", "seule"])
    table["%"] = (100 * table["molécules"] / max(total, 1)).round(1)
    table["seule"] = table["seule"].fillna(0).astype(int)
    table = table[["règle", "molécules", "%", "seule"]].set_index("règle")
    table.attrs["molécules lues"] = total
    if unreadable:
        print(f"{unreadable} SMILES illisibles ignorés")
    print(f"{total} molécules")
    return table


# ==============================================================================
#  VITESSE
# ==============================================================================
def timing(items: Iterable[Any], regles: Regles, sulfur_valence: int = 6) -> pd.DataFrame:
    """Temps moyen par molécule (µs) : lecture du SMILES (MolecularGraph), matrice et GrapheV2, chaque règle (sur
    des graphes neufs : les cycles, gardés dans le graphe, sont comptés dans la première règle qui les demande,
    ici celle qui les utilise), toutes les règles ensemble, et pour comparaison les règles d'evomol
    (matrix_rules.all_violations, version actuelle, sur ses propres graphes)."""
    smiles_list = [s for s, _ in pr._entries(items)]  # pylint: disable=protected-access
    start = time.perf_counter()
    mol_graphs = []
    for smiles in smiles_list:
        try:
            mol_graphs.append(pr._mol_graph(smiles))  # pylint: disable=protected-access
        except ValueError:
            pass
    t_read = time.perf_counter() - start
    n = len(mol_graphs)
    data = [(*mr.to_matrix(m), m.mol) for m in mol_graphs]

    def fresh() -> list[GrapheV2]:
        return [GrapheV2(types, matrix, mol) for types, matrix, mol in data]

    start = time.perf_counter()
    fresh()
    rows = [("lecture du SMILES", t_read), ("GrapheV2", time.perf_counter() - start)]
    for name, rule in regles.rules.items():
        graphs = fresh()
        start = time.perf_counter()
        for g in graphs:
            rule(g)
        rows.append((name, time.perf_counter() - start))
    graphs = fresh()
    start = time.perf_counter()
    for g in graphs:
        regles.violations(g)
    rows.append(("toutes les règles", time.perf_counter() - start))
    start = time.perf_counter()
    for types, matrix, _ in data:
        mr.all_violations(mr._Graph(types, matrix), sulfur_valence)  # pylint: disable=protected-access
    rows.append(("evomol (GrapheV1 + règles)", time.perf_counter() - start))
    table = pd.DataFrame(rows, columns=["étape", "total (s)"]).set_index("étape")
    table["µs / molécule"] = (1e6 * table["total (s)"] / max(n, 1)).round(1)
    table["total (s)"] = table["total (s)"].round(3)
    print(f"{n} molécules")
    return table


def atoms_text(g: GrapheV2, atoms: Sequence[int]) -> str:
    """« C1 s, O4 s » : élément, numéro et état de chaque atome."""
    return ", ".join(f"{g.types[a]}{a} {g.tags[a]}" for a in atoms)
