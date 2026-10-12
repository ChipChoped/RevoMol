"""
Name the operation that turns one molecule into another (newmut, optional).

Port of reference_core.js: mcs / identity_map (atom alignment), diff (aligned
bond-order matrices) and classify (named operations: substitution, append,
graft, annelation, insertion, excision, removal, bond change, alternating path
flip, Kekulé rewriting). The aromaticity, given by OpenChemLib in the
JavaScript version, comes from RDKit here. The titles of the operations are in
French, as in the reference.

Not used by the search: it is an analysis tool (e.g. to name the mutations
along a random walk). Main entry points: classify_mutation and
classify_action.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from rdkit import Chem

from evomol.action import Action
from evomol.newmut.matrix_rules import VALENCE_ELECTRONS, implicit_h, max_valence
from evomol.representation import MolecularGraph, Molecule


@dataclass
class MatrixMol:
    """Molecule as in reference_core.js: atoms (id, element) and bonds
    (id_a, id_b, order), in this order."""

    ids: list[int]
    elements: list[str]
    bonds: list[tuple[int, int, int]]


def matrix_mol(mol_graph: MolecularGraph) -> MatrixMol:
    """MatrixMol of a kekulized molecular graph (ids = atom indices)."""
    mol = mol_graph.mol
    n = mol.GetNumAtoms()
    return MatrixMol(
        ids=list(range(n)),
        elements=[mol.GetAtomWithIdx(i).GetSymbol() for i in range(n)],
        bonds=[
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble()))
            for b in mol.GetBonds()
        ],
    )


def key(a: int, b: int) -> str:
    """Key of an unordered pair."""
    return f"{a}|{b}" if a < b else f"{b}|{a}"


@dataclass
class _Index:
    idx: dict[int, int]
    n: int
    adj: list[dict[int, int]]
    sums: list[int]


def _index(mol: MatrixMol) -> _Index:
    idx = {atom_id: i for i, atom_id in enumerate(mol.ids)}
    n = len(mol.ids)
    adj: list[dict[int, int]] = [{} for _ in range(n)]
    for a, b, order in mol.bonds:
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None:
            continue
        adj[i][j] = order
        adj[j][i] = order
    return _Index(idx, n, adj, [sum(m.values()) for m in adj])


def _distances(adj: list[dict[int, int]], start: int) -> list[float]:
    dist = [math.inf] * len(adj)
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == math.inf:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


# pylint: disable=too-many-locals,too-many-statements
def _mcs_once(left: MatrixMol, right: MatrixMol, cap: int, rot: int) -> dict[int, int]:
    """Maximum common substructure, heuristic branch and bound (mcsOnce)."""
    li, ri = _index(left), _index(right)
    n, m = li.n, ri.n
    if not n or not m:
        return {}

    # circular environment labels (Morgan-like) to rank candidate pairs
    depth = 4
    intern: dict[Any, int] = {}

    def intern_id(value: Any) -> int:
        if value not in intern:
            intern[value] = len(intern)
        return intern[value]

    def labels(mol: MatrixMol, ix: _Index) -> list[list[int]]:
        levels = [[intern_id(element) for element in mol.elements]]
        for _ in range(depth):
            prev = levels[-1]
            levels.append([
                intern_id(f"{prev[i]}({','.join(sorted(str(prev[j]) for j in ix.adj[i]))})")
                for i in range(len(mol.elements))
            ])
        return levels

    ll, rl = labels(left, li), labels(right, ri)

    def sim(u: int, r: int) -> int:
        k = -1
        while k < depth and ll[k + 1][u] == rl[k + 1][r]:
            k += 1
        return k + 1

    r_count = [Counter(level) for level in rl]

    def uniq(u: int) -> int:
        for k in range(depth, -1, -1):
            if r_count[k].get(ll[k][u]) == 1:
                return k + 1
        return 0

    # BFS order over left, starting from the atom with the most distinctive environment
    order: list[int] = []
    seen = [False] * n
    starts0 = sorted(range(n), key=lambda a: (-uniq(a), -len(li.adj[a])))
    r0 = min(rot, n - 1)
    starts = [starts0[r0], *[x for i, x in enumerate(starts0) if i != r0]]
    for s in starts:
        if seen[s]:
            continue
        queue = deque([s])
        seen[s] = True
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in sorted(li.adj[u], key=lambda a: -len(li.adj[a])):
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
    pos = [0] * n
    for k, u in enumerate(order):
        pos[u] = k

    # remaining potential after position k
    rem_pot = [0] * (n + 1)
    for k in range(n - 1, -1, -1):
        u = order[k]
        potential = 2 + len(li.adj[u])
        for v in li.adj[u]:
            if pos[v] < k:
                potential += 11  # bonds back to earlier atoms
        rem_pot[k] = rem_pot[k + 1] + potential

    # shortest-path distances (tie-breaker: keep broken bonds' ends close)
    d_right = [_distances(ri.adj, s) for s in range(m)]
    d_left = [_distances(li.adj, s) for s in range(n)]

    pre = [-1] * m
    assign = [-1] * n
    used = [False] * m
    state = {"best": -1, "best_assign": assign[:], "nodes": 0}

    def gain(u: int, r: int) -> int:
        g = 2 if left.elements[u] == right.elements[r] else 1
        for v, o in li.adj[u].items():
            if pos[v] >= pos[u]:
                continue
            rv = assign[v]
            if rv < 0:
                continue
            ro = ri.adj[r].get(rv)
            if ro is not None:
                g += 10 + (1 if ro == o else 0)
            elif d_right[r][rv] <= 3:
                g += 1
        for rv in ri.adj[r]:
            v = pre[rv]
            if v >= 0 and v not in li.adj[u] and d_left[u][v] <= 3:
                g += 1
        return g

    def candidates(u: int) -> list[tuple[int, int]]:
        images = [assign[v] for v in li.adj[u] if pos[v] < pos[u] and assign[v] >= 0]
        if images:
            cand: dict[int, None] = {}
            for r in images:
                for r1 in ri.adj[r]:
                    if not used[r1]:
                        cand.setdefault(r1)
                    for r2 in ri.adj[r1]:
                        if not used[r2]:
                            cand.setdefault(r2)

            def kept(r: int) -> int:
                return sum(
                    1 for v in li.adj[u]
                    if pos[v] < pos[u] and assign[v] >= 0 and assign[v] in ri.adj[r]
                )

            # conserved bonds first, then environment similarity, then the full gain
            scored = [(r, gain(u, r), sim(u, r), kept(r)) for r in cand]
            scored.sort(key=lambda x: (-x[3], -x[2], -x[1]))
            return [(r, g) for r, g, _, _ in scored]
        scored2 = [(r, gain(u, r), sim(u, r)) for r in range(m) if not used[r]]
        scored2.sort(key=lambda x: (-x[2], -x[1]))
        return [(r, g) for r, g, _ in scored2[:12]]

    def rec(k: int, score: int) -> None:
        state["nodes"] += 1
        if state["nodes"] > cap:
            return
        if score + rem_pot[k] <= state["best"]:
            return
        if k == n:
            if score > state["best"]:
                state["best"] = score
                state["best_assign"] = assign[:]
            return
        u = order[k]
        for r, g in candidates(u):
            assign[u] = r
            used[r] = True
            pre[r] = u
            rec(k + 1, score + g)
            used[r] = False
            assign[u] = -1
            pre[r] = -1
            if state["nodes"] > cap:
                return
        rec(k + 1, score)  # leave unmapped

    rec(0, 0)

    # post-pass: an atom left unmapped on both sides with the same element is one
    # atom kept, not "removed + added"
    image = state["best_assign"]
    used_right = {r for r in image if r >= 0}
    for u in order:
        if image[u] >= 0:
            continue
        best_r, best_score = -1, 0
        for r in range(m):
            if r in used_right:
                continue
            sc = 2 if left.elements[u] == right.elements[r] else 0
            for v, o in li.adj[u].items():
                rv = image[v]
                if rv >= 0 and rv in ri.adj[r]:
                    sc += 10 + (1 if ri.adj[r][rv] == o else 0)
            if sc > best_score:
                best_score, best_r = sc, r
        if best_r >= 0:
            image[u] = best_r
            used_right.add(best_r)

    return {left.ids[u]: right.ids[r] for u, r in enumerate(image) if r >= 0}


def mapping_score(left: MatrixMol, right: MatrixMol, mapping: dict[int, int]) -> int:
    """Score of an alignment (score of mcs in reference_core.js): 2 per atom of
    the same element (1 otherwise), 10 per conserved bond (+1 same order)."""
    li, ri = _index(left), _index(right)
    image = {li.idx[l]: ri.idx[r] for l, r in mapping.items()}
    total = 0
    for u, r in image.items():
        total += 2 if left.elements[u] == right.elements[r] else 1
        for v, o in li.adj[u].items():
            if v > u and v in image:
                ro = ri.adj[r].get(image[v])
                if ro is not None:
                    total += 10 + (1 if ro == o else 0)
    return total


def mcs(left: MatrixMol, right: MatrixMol, cap: int = 250_000) -> dict[int, int]:
    """Several searches from different starting atoms; keep the alignment that
    conserves the most (mcs of reference_core.js). Returns left id -> right id."""
    best: dict[int, int] | None = None
    best_score = -1
    tries = min(4, len(left.ids))
    for t in range(max(1, tries)):
        mapping = _mcs_once(left, right, cap // max(1, tries), t)
        s = mapping_score(left, right, mapping)
        if s > best_score:
            best_score, best = s, mapping
    return best or {}


def mcs_both_ways(left: MatrixMol, right: MatrixMol, cap: int = 250_000) -> dict[int, int]:
    """Best of mcs(left, right) and of the inverse of mcs(right, left), by
    mapping_score. The search is heuristic and not symmetric: the workshop
    aligns with the first one when analysing, with the second one when a
    molecule is imported next to another (adoptIds)."""
    forward = mcs(left, right, cap)
    backward = {l: r for r, l in mcs(right, left, cap).items()}
    if mapping_score(left, right, backward) > mapping_score(left, right, forward):
        return backward
    return forward


def identity_map(left: MatrixMol, right: MatrixMol) -> dict[int, int]:
    """Atoms with the same id on both sides."""
    right_ids = set(right.ids)
    return {a: a for a in left.ids if a in right_ids}


# pylint: disable=too-many-branches
def diff(
    left: MatrixMol, right: MatrixMol, mapping: dict[int, int], s_max: int | None = None
) -> dict[str, Any]:
    """Aligned diff of the two matrices (diff of reference_core.js)."""
    li, ri = _index(left), _index(right)
    n = li.n
    inverse = {r: l for l, r in mapping.items()}
    r_al = [0] * ri.n
    added: list[int] = []
    for j, atom_id in enumerate(right.ids):
        if atom_id in inverse:
            r_al[j] = li.idx[inverse[atom_id]]
        else:
            r_al[j] = n + len(added)
            added.append(j)
    big_n = n + len(added)
    removed = [i for i in range(n) if left.ids[i] not in mapping]
    el_r: list[str | None] = [None] * big_n
    el_l: list[str | None] = [None] * big_n
    for i, element in enumerate(left.elements):
        el_l[i] = element
    for j, element in enumerate(right.elements):
        el_r[r_al[j]] = element
    ml = [[0] * big_n for _ in range(big_n)]
    mr_ = [[0] * big_n for _ in range(big_n)]
    for i in range(n):
        for j, o in li.adj[i].items():
            ml[i][j] = o
    for j in range(ri.n):
        for k, o in ri.adj[j].items():
            mr_[r_al[j]][r_al[k]] = o
    labels = [f"{el_r[i] or el_l[i]}{i}" for i in range(big_n)]
    s_l = [sum(row) for row in ml]
    s_r = [sum(row) for row in mr_]
    h_l = [implicit_h(e, s_l[i], s_max) if e else None for i, e in enumerate(el_l)]
    h_r = [implicit_h(e, s_r[i], s_max) if e else None for i, e in enumerate(el_r)]
    bad_l = [i for i, e in enumerate(el_l) if e and s_l[i] > max_valence(e, s_max)]
    bad_r = [i for i, e in enumerate(el_r) if e and s_r[i] > max_valence(e, s_max)]
    elem_changes = [
        {"i": i, "from": el_l[i], "to": el_r[i]}
        for i in range(n)
        if el_r[i] and el_l[i] != el_r[i]
    ]
    bond_changes = [
        {"i": i, "j": j, "from": ml[i][j], "to": mr_[i][j]}
        for i in range(big_n)
        for j in range(i + 1, big_n)
        if ml[i][j] != mr_[i][j]
    ]
    return {
        "n": n, "N": big_n, "labels": labels, "elL": el_l, "elR": el_r, "ML": ml, "MR": mr_,
        "HL": h_l, "HR": h_r, "sL": s_l, "sR": s_r, "badL": bad_l, "badR": bad_r,
        "removed": removed, "added": [n + k for k in range(len(added))],
        "elemChanges": elem_changes, "bondChanges": bond_changes, "rAl": r_al,
    }


def _components(nodes: Iterable[int], adj_fn: Callable[[int], Iterable[int]]) -> list[list[int]]:
    nodes = list(nodes)
    node_set, seen, out = set(nodes), set(), []
    for s in nodes:
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj_fn(u):
                if v in node_set and v not in seen:
                    seen.add(v)
                    stack.append(v)
        out.append(sorted(comp))
    return out


def _plural(count: int, word: str = "") -> str:
    return f" × {count}" if count > 1 else word


# pylint: disable=too-many-statements
def classify(d: dict[str, Any], extra: dict[str, set] | None = None) -> list[dict[str, Any]]:
    """Named operations explaining the diff (classify of reference_core.js)."""
    extra = extra or {}
    ml, mr_, labels, big_n = d["ML"], d["MR"], d["labels"], d["N"]

    def lab(i: int) -> str:
        return labels[i]

    def nb(matrix: list[list[int]], u: int) -> list[int]:
        return [v for v in range(big_n) if matrix[u][v]]

    consumed: set[str] = set()
    ops: list[dict[str, Any]] = []

    def frag_info(matrix: list[list[int]], comp: list[int]) -> dict[str, Any]:
        comp_set = set(comp)
        internal, multi, conn = 0, 0, []
        for u in comp:
            for v in range(big_n):
                o = matrix[u][v]
                if not o:
                    continue
                if v in comp_set:
                    if u < v:
                        internal += 1
                        if o > 1:
                            multi += 1
                else:
                    conn.append({"frag": u, "anchor": v, "o": o})
                    if o > 1:
                        multi += 1
        return {"internal": internal, "multi": multi, "conn": conn, "ring": internal >= len(comp)}

    # 1. added fragments
    for comp in _components(d["added"], lambda u: nb(mr_, u)):
        f = frag_info(mr_, comp)
        anchors = [c["anchor"] for c in f["conn"]]
        bonds_total = f["internal"] + len(f["conn"])
        cost = len(comp) + max(0, bonds_total - len(comp)) + f["multi"]
        atoms_txt = ", ".join(lab(u) for u in comp)
        if (len(f["conn"]) == 2 and anchors[0] != anchors[1] and ml[anchors[0]][anchors[1]]
                and not mr_[anchors[0]][anchors[1]] and not f["ring"]):
            a, b = anchors
            consumed.add(key(a, b))
            non_c = sum(1 for u in comp if d["elR"][u] != "C")
            single = len(comp) == 1
            evomol = ("insert_carbon" + (f" × {len(comp)}" if len(comp) > 1 else "")
                      + (" + substitution" + _plural(non_c) if non_c else "")
                      + (" + change_bond" + _plural(f["multi"]) if f["multi"] else ""))
            frag_a = f["conn"][0]["frag"] if f["conn"][0]["anchor"] == a else f["conn"][1]["frag"]
            frag_b = f["conn"][0]["frag"] if f["conn"][0]["anchor"] == b else f["conn"][1]["frag"]
            ops.append({
                "kind": "insert", "family": "native" if single else "macro",
                "title": (f"Insertion de {lab(comp[0])} dans la liaison {lab(a)}–{lab(b)}" if single
                          else f"Insertion d'un fragment de {len(comp)} atomes ({atoms_txt}) dans la "
                               f"liaison {lab(a)}–{lab(b)}"),
                "evomol": evomol, "cost": len(comp) + non_c + f["multi"],
                "matrix": (f"La case ({lab(a)}, {lab(b)}) passe de {ml[a][b]} à 0 ; "
                           + ("une ligne ajoutée" if single else f"{len(comp)} lignes ajoutées")
                           + f", reliées par ({lab(a)}, {lab(frag_a)}) et ({lab(frag_b)}, {lab(b)})."),
                "atoms": [*comp, a, b],
            })
            continue
        if not f["conn"]:
            ops.append({
                "kind": "isolated", "family": "other", "title": f"Fragment déconnecté ajouté ({atoms_txt})",
                "evomol": "impossible en une mutation : EvoMol garde une seule molécule connexe", "cost": None,
                "matrix": "Bloc ajouté sans aucune case de connexion.", "atoms": comp,
            })
            continue
        if len(f["conn"]) == 1:
            c = f["conn"][0]
            if len(comp) == 1:
                ops.append({
                    "kind": "append", "family": "native", "title": f"Ajout de {lab(comp[0])} sur {lab(c['anchor'])}",
                    "evomol": "append_atom" + (" + change_bond" if c["o"] > 1 else ""),
                    "cost": 1 + (1 if c["o"] > 1 else 0),
                    "matrix": (f"Une ligne et une colonne ajoutées, connexion ({lab(c['anchor'])}, "
                               f"{lab(comp[0])}) = {c['o']}."),
                    "atoms": [comp[0], c["anchor"]],
                })
            else:
                ops.append({
                    "kind": "graft", "family": "macro",
                    "title": (f"Greffe d'un cycle sur {lab(c['anchor'])}" if f["ring"]
                              else f"Greffe d'un fragment sur {lab(c['anchor'])}"),
                    "evomol": "macro-mutation (greffe par blocs)", "cost": cost,
                    "matrix": (f"Bloc de {len(comp)} atomes ({atoms_txt}) concaténé, une connexion "
                               f"({lab(c['anchor'])}, {lab(c['frag'])}) = {c['o']}."),
                    "fragment": comp, "anchor": c["anchor"], "atoms": [*comp, c["anchor"]],
                })
            continue
        if len(f["conn"]) == 2 and anchors[0] != anchors[1] and mr_[anchors[0]][anchors[1]]:
            a, b = anchors
            ring_size = len(comp) + 2
            arom = ""
            if "aromAR" in extra:
                arom = (", perçu aromatique" if all(u in extra["aromAR"] for u in comp)
                        else ", non aromatique (forme quinoïde ou non conjuguée)")
            ops.append({
                "kind": "annel", "family": "macro", "title": f"Annélation sur la liaison {lab(a)}–{lab(b)}",
                "evomol": "macro-mutation (greffe à deux connexions)", "cost": cost,
                "matrix": (f"Bloc de {len(comp)} atomes ({atoms_txt}) concaténé, connexions ({lab(a)}, "
                           f"{lab(f['conn'][0]['frag'])}) et ({lab(b)}, {lab(f['conn'][1]['frag'])}). Nouveau "
                           f"cycle à {ring_size} atomes fusionné{arom}."),
                "fragment": comp, "anchors": [a, b], "ringSize": ring_size, "atoms": [*comp, a, b],
            })
            continue
        ops.append({
            "kind": "multigraft", "family": "macro",
            "title": f"Greffe à {len(f['conn'])} points ({', '.join(lab(a) for a in anchors)})",
            "evomol": "macro-mutation (greffe multiple)", "cost": cost,
            "matrix": f"Bloc de {len(comp)} atomes concaténé avec {len(f['conn'])} cases de connexion.",
            "fragment": comp, "atoms": [*comp, *anchors],
        })

    # 2. removed fragments
    for comp in _components(d["removed"], lambda u: nb(ml, u)):
        f = frag_info(ml, comp)
        anchors = [c["anchor"] for c in f["conn"]]
        if (len(comp) == 1 and len(f["conn"]) == 2 and not ml[anchors[0]][anchors[1]]
                and mr_[anchors[0]][anchors[1]]):
            a, b = anchors
            consumed.add(key(a, b))
            ops.append({
                "kind": "cut", "family": "native", "title": f"Excision de {lab(comp[0])} entre {lab(a)} et {lab(b)}",
                "evomol": "cut_atom", "cost": 1,
                "matrix": f"La ligne {lab(comp[0])} disparaît ; la case ({lab(a)}, {lab(b)}) passe de 0 à {mr_[a][b]}.",
                "atoms": [comp[0], a, b],
            })
            continue
        bonds_total = f["internal"] + len(f["conn"])
        native = ("remove_atom" if len(comp) == 1
                  else "remove_group" if len(f["conn"]) == 1 else f"remove_atom × {len(comp)}")
        is_native = len(comp) == 1 or len(f["conn"]) == 1
        ops.append({
            "kind": "remove", "family": "native" if is_native else "macro",
            "title": f"Retrait de {lab(comp[0])}" if len(comp) == 1 else f"Retrait d'un fragment de {len(comp)} atomes",
            "evomol": native, "cost": 1 if is_native else len(comp) + max(0, bonds_total - len(comp)),
            "matrix": f"Lignes et colonnes {', '.join(lab(u) for u in comp)} supprimées.",
            "atoms": [*comp, *anchors],
        })

    # 3. element substitutions
    for c in d["elemChanges"]:
        aromatic = "aromAL" in extra and c["i"] in extra["aromAL"]
        ve_from, ve_to = VALENCE_ELECTRONS.get(c["from"]), VALENCE_ELECTRONS.get(c["to"])
        matrix = (f"Seule la case diagonale ({c['i']}, {c['i']}) change : électrons de valence {ve_from} → {ve_to}."
                  if ve_from != ve_to else
                  f"Même nombre d'électrons de valence ({ve_from}) : la matrice ne change pas, seule l'étiquette "
                  f"de la ligne {c['i']} passe de {c['from']} à {c['to']}.")
        ops.append({
            "kind": "subst", "family": "native",
            "title": f"Substitution {c['from']} → {c['to']} sur l'atome {c['i']}"
                     + (" (cycle aromatique)" if aromatic else ""),
            "evomol": "substitution", "cost": 1,
            "matrix": matrix + f" Ordres de liaison inchangés ; H implicites : {d['HL'][c['i']]} → {d['HR'][c['i']]}.",
            "atoms": [c["i"]],
        })

    # 4. bond changes between atoms present on both sides
    def present_both(i: int) -> bool:
        return bool(d["elL"][i] and d["elR"][i])

    bc = [c for c in d["bondChanges"]
          if present_both(c["i"]) and present_both(c["j"]) and key(c["i"], c["j"]) not in consumed]
    if "aromBL" in extra and "aromBR" in extra:
        kek = [c for c in bc if c["from"] >= 1 and c["to"] >= 1
               and key(c["i"], c["j"]) in extra["aromBL"] and key(c["i"], c["j"]) in extra["aromBR"]]
        if kek:
            bc = [c for c in bc if c not in kek]
            ops.append({
                "kind": "kekule", "family": "neutral",
                "title": f"Réécriture de Kekulé dans un cycle aromatique ({len(kek)} liaison{'s' if len(kek) > 1 else ''})",
                "evomol": "aucune : même structure électronique", "cost": 0,
                "matrix": " · ".join(f"({lab(c['i'])}, {lab(c['j'])}) : {c['from']} → {c['to']}" for c in kek)
                          + ". Liaisons aromatiques des deux côtés : simple choix de structure de Kekulé.",
                "atoms": sorted({x for c in kek for x in (c["i"], c["j"])}),
            })
    bc_adj: dict[int, list[dict[str, int]]] = {}
    for c in bc:
        bc_adj.setdefault(c["i"], []).append(c)
        bc_adj.setdefault(c["j"], []).append(c)
    for comp in _components(list(bc_adj), lambda u: [c["j"] if c["i"] == u else c["i"] for c in bc_adj[u]]):
        comp_set = set(comp)
        edges = [c for c in bc if c["i"] in comp_set]
        all_flip = all(c["from"] >= 1 and c["to"] >= 1 and abs(c["to"] - c["from"]) == 1 for c in edges)
        degree_ok = all(len(bc_adj[u]) <= 2 for u in comp)
        if all_flip and degree_ok and len(edges) >= 2:
            ends = [u for u in comp if len(bc_adj[u]) == 1]
            start = ends[0] if ends else comp[0]
            walk = [start]
            used_edges: list[dict[str, int]] = []
            u, ok, prev_sign = start, True, 0
            while True:
                edge = next((c for c in bc_adj[u] if not any(c is e for e in used_edges)), None)
                if edge is None:
                    break
                used_edges.append(edge)
                sign = (edge["to"] > edge["from"]) - (edge["to"] < edge["from"])
                if prev_sign and sign == prev_sign:
                    ok = False
                    break
                prev_sign = sign
                u = edge["j"] if edge["i"] == u else edge["i"]
                walk.append(u)
            if ok and len(used_edges) == len(edges):
                text = "–".join(lab(x) for x in walk)
                if not ends:
                    ops.append({
                        "kind": "kekule", "family": "macro", "title": f"Permutation de Kekulé sur le cycle {text}",
                        "evomol": "aucune : même molécule écrite autrement", "cost": 0,
                        "matrix": f"Les {len(edges)} ordres du cycle s'inversent (1 ↔ 2). Toutes les sommes de "
                                  "lignes sont conservées.",
                        "atoms": comp,
                    })
                else:
                    e0, e1 = walk[0], walk[-1]

                    def d_end(i: int) -> int:
                        return d["sR"][i] - d["sL"][i]

                    oxidation = d_end(e0) > 0 and d_end(e1) > 0
                    ops.append({
                        "kind": "flip", "family": "macro", "title": f"Inversion le long du chemin alterné {text}",
                        "evomol": f"{len(edges)} × change_bond, avec des intermédiaires qui cassent la conjugaison",
                        "cost": len(edges),
                        "matrix": (f"{len(edges)} ordres inversés (1 ↔ 2) en alternance. Atomes internes : somme "
                                   f"conservée. Extrémités {lab(e0)} et {lab(e1)} : "
                                   f"{'+1' if d_end(e0) > 0 else '−1'} / {'+1' if d_end(e1) > 0 else '−1'}"
                                   + (" (chacune perd un H : bascule vers une forme quinoïde)" if oxidation else "")
                                   + "."),
                        "atoms": comp,
                    })
                continue
        for c in edges:
            if c["from"] == 0:
                title = f"Création de la liaison {lab(c['i'])}–{lab(c['j'])} (ordre {c['to']})"
                evomol = "change_bond (création, ferme un cycle)"
            elif c["to"] == 0:
                title = f"Rupture de la liaison {lab(c['i'])}–{lab(c['j'])}"
                evomol = "change_bond (rupture)"
            else:
                title = f"Ordre de {lab(c['i'])}–{lab(c['j'])} : {c['from']} → {c['to']}"
                evomol = "change_bond"
            ops.append({
                "kind": "bond", "family": "native", "title": title, "evomol": evomol, "cost": 1,
                "matrix": f"Case ({lab(c['i'])}, {lab(c['j'])}) : {c['from']} → {c['to']}.",
                "atoms": [c["i"], c["j"]],
            })
    return ops


def _aromatic(mol_graph: MolecularGraph) -> tuple[set[int], set[str]]:
    """Aromatic atoms and bonds (keys) of a molecular graph.

    RDKit perception, closer to OpenChemLib (used by the workshop): a ring with
    an atom bearing an exocyclic double bond (quinoid form, C=O) is not
    aromatic, whereas RDKit may perceive it as aromatic.
    """
    mol = Chem.Mol(mol_graph.mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # pylint: disable=broad-except
        return set(), set()
    atoms: set[int] = set()
    bonds: set[str] = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = set(ring)
        if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        exocyclic_double = any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtomIdx(a) not in ring_set
            for a in ring
            for bond in mol.GetAtomWithIdx(a).GetBonds()
        )
        if exocyclic_double:
            continue
        atoms |= ring_set
        bonds |= {key(ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))}
    return atoms, bonds


def _as_graph(molecule: MolecularGraph | Molecule | str) -> MolecularGraph:
    if isinstance(molecule, MolecularGraph):
        return molecule
    if isinstance(molecule, Molecule):
        return molecule.get_representation(MolecularGraph)
    return MolecularGraph(molecule)


def classify_mutation(
    before: MolecularGraph | Molecule | str,
    after: MolecularGraph | Molecule | str,
    align: str = "mcs",
    cap: int = 250_000,
) -> list[dict[str, Any]]:
    """Named operations from before to after (analyse of reference_app.js).

    Args:
        before, after: molecules (SMILES, Molecule or MolecularGraph).
        align: "mcs" (automatic alignment, see mcs_both_ways) or "id" (same
            atom indices on both sides, e.g. a parent and the raw result of an
            action that adds atoms; falls back to "mcs" if no atom is shared).
        cap: maximum number of nodes of the alignment search.

    Returns:
        list of operations (dict: kind, family, title, evomol, cost, matrix,
        atoms). Kinds: subst, append, graft, annel, insert, isolated, cut,
        remove, multigraft, bond, flip, kekule.
    """
    graph_l, graph_r = _as_graph(before), _as_graph(after)
    left, right = matrix_mol(graph_l), matrix_mol(graph_r)
    mapping = identity_map(left, right) if align == "id" else {}
    if not mapping and left.ids and right.ids:
        mapping = mcs_both_ways(left, right, cap)
    d = diff(left, right, mapping)
    arom_al, arom_bl = _aromatic(graph_l)
    arom_r_atoms, arom_r_bonds = _aromatic(graph_r)
    position = {atom_id: j for j, atom_id in enumerate(right.ids)}
    extra = {
        "aromAL": arom_al,
        "aromBL": arom_bl,
        "aromAR": {d["rAl"][j] for j in arom_r_atoms},
        "aromBR": {
            key(d["rAl"][position[int(a)]], d["rAl"][position[int(b)]])
            for a, b in (k.split("|") for k in arom_r_bonds)
        },
    }
    return classify(d, extra)


def classify_action(action: Action) -> list[dict[str, Any]]:
    """Named operations of an action, read on its raw result (atom indices
    kept when no atom is removed)."""
    from evomol.newmut.rule_filtered import result_graph  # pylint: disable=import-outside-toplevel

    before = action.molecule.get_representation(MolecularGraph)
    after = result_graph(action, before)  # type: ignore[arg-type]
    align = "id" if after.nb_atoms >= before.nb_atoms else "mcs"
    return classify_mutation(before, after, align=align)


def operation_names(ops: list[dict[str, Any]]) -> list[str]:
    """Titles of the operations that change the molecule (Kekulé rewritings left out)."""
    return [op["title"] for op in ops if op["cost"] != 0]
