"""
Validity rules read on the bond-order matrix of a molecule (newmut).

For a molecule with n heavy atoms, the matrix M (n x n, symmetric, integers)
holds:
- on the diagonal: the number of valence electrons of the element;
- off the diagonal: the bond order (0 = no bond, 1, 2, 3).
Hydrogens are implicit: H = valence - sum of the row off the diagonal.

Rules (translated from csPairs, doubleCarbons, tangledRings of
reference_app.js, the executable specification):
- R1: valence, the sum of a row never exceeds the maximum valence;
- R2: two "s" carbons (only single bonds) are not bonded, unless the bond lies
  on a cycle and each of them has another sp2 neighbour;
- R3: a carbon carries at most one double bond;
- R4a: two small rings (<= 8 atoms) share nothing, one atom or exactly one bond;
- R4b: a large ring (> 8 atoms) can close flat.

An action is allowed when its result creates no new violation (see
result_is_allowed): a molecule that already breaks a rule stays mutable.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import NamedTuple, Sequence

import numpy as np
from rdkit import Chem

from evomol.representation import MolecularGraph
from evomol.representation import molecule as _molecule

VALENCE_ELECTRONS: dict[str, int] = {
    "B": 3,
    "C": 4,
    "Si": 4,
    "N": 5,
    "P": 5,
    "O": 6,
    "S": 6,
    "F": 7,
    "Cl": 7,
    "Br": 7,
    "I": 7,
}

# allowed valences (Core.VAL of reference_core.js); S is capped by the sulfur
# valence, other elements default to 4
ALLOWED_VALENCES: dict[str, tuple[int, ...]] = {
    "B": (3,),
    "C": (4,),
    "N": (3,),
    "O": (2,),
    "F": (1,),
    "Si": (4,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "Cl": (1,),
    "Br": (1,),
    "I": (1,),
}

# rings of at most SMALL_RING atoms are "small" (R4a), the others "large" (R4b)
SMALL_RING = 8
# R4b: tolerance on the sum of the turning angles (degrees)
FLAT_TOLERANCE = 40
# R4b: a free atom can point its H inwards only in rings of at least 14 atoms
INWARD_H_MIN_RING = 14

# size of the memoization caches (one entry per molecular graph)
CACHE_SIZE = 100_000

_BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


class Violation(NamedTuple):
    """A rule broken by a molecule."""

    rule_id: str
    atoms: tuple[int, ...]
    message: str


############################################################
#                     VALENCE AND MATRIX                   #
############################################################


def current_sulfur_valence() -> int:
    """Maximum valence of sulfur currently set in evomol (SULFUR_MAX_VALENCE)."""
    sulfur_valence: int = _molecule.SULFUR_MAX_VALENCE
    return sulfur_valence


def allowed_valences(element: str, sulfur_valence: int | None = None) -> tuple[int, ...]:
    """Allowed valences of an element, in increasing order."""
    valences = ALLOWED_VALENCES.get(element, (4,))
    if element == "S":
        if sulfur_valence is None:
            sulfur_valence = current_sulfur_valence()
        valences = tuple(v for v in valences if v <= sulfur_valence) or valences[:1]
    return valences


def max_valence(element: str, sulfur_valence: int | None = None) -> int:
    """Maximum valence of an element."""
    return allowed_valences(element, sulfur_valence)[-1]


def implicit_h(element: str, bond_sum: int, sulfur_valence: int | None = None) -> int:
    """Implicit hydrogens: smallest allowed valence >= bond_sum, minus bond_sum."""
    for valence in allowed_valences(element, sulfur_valence):
        if valence >= bond_sum:
            return valence - bond_sum
    return 0


def to_matrix(mol_graph: MolecularGraph) -> tuple[list[str], np.ndarray]:
    """Return (types, M): M[i, i] = valence electrons, M[i, j] = bond order.

    The molecular graph is kekulized, so the bond orders are integers.
    """
    mol = mol_graph.mol
    n = mol.GetNumAtoms()
    types = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n)]
    if n == 0:
        return types, np.zeros((0, 0), dtype=np.int64)
    matrix = np.asarray(Chem.GetAdjacencyMatrix(mol, useBO=True), dtype=np.int64)
    diagonal = np.arange(n)
    matrix[diagonal, diagonal] = [VALENCE_ELECTRONS.get(element, 0) for element in types]
    return types, matrix


def from_matrix(types: Sequence[str], matrix: np.ndarray) -> MolecularGraph:
    """Build a kekulized MolecularGraph from (types, M).

    No SMILES round trip: the Kekulé structure of the matrix is kept as is.
    """
    rwmol = Chem.RWMol()
    for element in types:
        atom = Chem.Atom(element)
        atom.SetBoolProp("mutability", True)
        rwmol.AddAtom(atom)
    n = len(types)
    for i in range(n):
        for j in range(i + 1, n):
            order = int(matrix[i][j])
            if order:
                rwmol.AddBond(i, j, _BOND_TYPES[order])

    mol_graph = MolecularGraph.__new__(MolecularGraph)
    mol_graph.str_id = ""
    mol_graph.sanitize = False
    mol_graph.mol = rwmol
    mol_graph.update_representation()
    return mol_graph


############################################################
#                    GRAPH READ ON THE MATRIX              #
############################################################


class _Graph:
    """Adjacency lists and bond orders read once from a matrix."""

    __slots__ = ("types", "n", "order", "adj", "sums", "multiple")

    def __init__(self, types: Sequence[str], matrix: np.ndarray | list[list[int]]):
        self.types: list[str] = list(types)
        n = len(self.types)
        rows = np.array(matrix, dtype=np.int64).reshape(n, n)
        self.n: int = len(rows)
        np.fill_diagonal(rows, 0)
        self.order: list[list[int]] = rows.tolist()
        self.adj: list[list[int]] = [np.flatnonzero(row).tolist() for row in rows]
        self.sums: list[int] = rows.sum(axis=1).tolist()
        self.multiple: list[bool] = (rows >= 2).any(axis=1).tolist()

    def bond_sum(self, i: int) -> int:
        """Sum of the row of atom i, off the diagonal."""
        return self.sums[i]

    def has_multiple(self, i: int) -> bool:
        """True if atom i has a double or triple bond."""
        return self.multiple[i]

    def bridges(self) -> set[tuple[int, int]]:
        """Bonds (i < j) that lie on no cycle (iterative Tarjan)."""
        bridges: set[tuple[int, int]] = set()
        index = [-1] * self.n
        low = [0] * self.n
        counter = 0
        for root in range(self.n):
            if index[root] != -1:
                continue
            index[root] = low[root] = counter
            counter += 1
            stack = [(root, -1, iter(self.adj[root]))]
            while stack:
                u, parent, neighbours = stack[-1]
                advanced = False
                for v in neighbours:
                    if v == parent:
                        continue
                    if index[v] == -1:
                        index[v] = low[v] = counter
                        counter += 1
                        stack.append((v, u, iter(self.adj[v])))
                        advanced = True
                        break
                    low[u] = min(low[u], index[v])
                if advanced:
                    continue
                stack.pop()
                if parent != -1:
                    low[parent] = min(low[parent], low[u])
                    if low[u] > index[parent]:
                        bridges.add((min(u, parent), max(u, parent)))
        return bridges

    def connected_without(self, x: int, y: int) -> bool:
        """True if x and y stay connected once the bond x-y is removed."""
        seen = {x}
        stack = [x]
        while stack:
            u = stack.pop()
            for v in self.adj[u]:
                if (u == x and v == y) or (u == y and v == x):
                    continue
                if v == y:
                    return True
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    def is_s_carbon(self, i: int) -> bool:
        """True if atom i is a carbon with only single bonds ("s" tag)."""
        return self.types[i] == "C" and not self.has_multiple(i)

    def carbon_tag(self, i: int) -> str:
        """s / a / d / t tag of a carbon (matTag of reference_app.js)."""
        if self.types[i] != "C":
            return ""
        if any(self.order[i][j] == 3 for j in self.adj[i]):
            return "t"
        doubles = [j for j in self.adj[i] if self.order[i][j] == 2]
        if not doubles:
            return "s"
        return "a" if any(self.connected_without(i, j) for j in doubles) else "d"

    def rings(self) -> list[list[int]]:
        """Smallest ring through each bond, deduplicated by atom set.

        For each bond (u, v), shortest path u -> v without the bond (BFS),
        closed by the bond. The ring is the ordered list of its atoms.
        """
        rings: dict[frozenset[int], list[int]] = {}
        # A bridge bond lies on no cycle, and a shortest path between the two
        # atoms of a ring bond never crosses a bridge: the search only follows
        # ring bonds (same paths, fewer atoms visited).
        bridges = self.bridges()
        ring_adj = [
            [v for v in self.adj[u] if (min(u, v), max(u, v)) not in bridges]
            for u in range(self.n)
        ]
        for a in range(self.n):
            for b in ring_adj[a]:
                if b < a:
                    continue
                prev: dict[int, int | None] = {a: None}
                queue = deque([a])
                while queue and b not in prev:
                    u = queue.popleft()
                    for v in ring_adj[u]:
                        if u == a and v == b:
                            continue
                        if v not in prev:
                            prev[v] = u
                            queue.append(v)
                if b not in prev:
                    continue
                path: list[int] = []
                x: int | None = b
                while x is not None:
                    path.append(x)
                    x = prev[x]
                key = frozenset(path)
                if key not in rings:
                    rings[key] = path
        return list(rings.values())


@lru_cache(maxsize=CACHE_SIZE)
def _graph_from_key(types: tuple[str, ...], data: bytes, n: int) -> _Graph:
    matrix = np.frombuffer(data, dtype=np.int64).reshape(n, n)
    return _Graph(types, matrix)


def _graph(types: Sequence[str], matrix: np.ndarray) -> _Graph:
    matrix = np.ascontiguousarray(matrix, dtype=np.int64)
    return _graph_from_key(tuple(types), matrix.tobytes(), matrix.shape[0])


def bond_in_ring(matrix: np.ndarray, i: int, j: int) -> bool:
    """True if i and j stay connected when the bond (i, j) is removed."""
    types = ["C"] * len(matrix)
    return _graph(types, matrix).connected_without(i, j)


def carbon_tag(matrix: np.ndarray, types: Sequence[str], i: int) -> str:
    """'s' | 'a' | 'd' | 't' for a carbon, '' for another element."""
    return _graph(types, matrix).carbon_tag(i)


def smallest_ring_per_bond(matrix: np.ndarray) -> list[list[int]]:
    """Smallest ring through each bond, deduplicated by atom set (see R4)."""
    types = ["C"] * len(matrix)
    return _graph(types, matrix).rings()


def hybridization(
    matrix: np.ndarray,
    types: Sequence[str],
    i: int,
    h_count: int | None = None,
    sulfur_valence: int | None = None,
) -> str:
    """'sp3' | 'sp2' | 'sp' | '?' for C, N, O ('' for another element).

    rowHyb of reference_app.js: steric number = heavy neighbours + H + lone
    pairs, minus the lone pair sitting in a p orbital (two lone pairs, or one
    lone pair conjugated with a neighbour carrying a double or triple bond
    while the atom has none itself).
    """
    element = types[i]
    if element not in ("C", "N", "O"):
        return ""
    graph = _graph(types, matrix)
    degree = len(graph.adj[i])
    bond_sum = graph.bond_sum(i)
    if h_count is None:
        h_count = implicit_h(element, bond_sum, sulfur_valence)
    non_bonding = VALENCE_ELECTRONS[element] - bond_sum - h_count
    if non_bonding < 0 or non_bonding % 2:
        return "?"
    lone_pairs = non_bonding // 2
    conjugated = False
    if lone_pairs > 0 and bond_sum == degree:
        conjugated = any(
            graph.bond_sum(j) - len(graph.adj[j]) > 0 for j in graph.adj[i]
        )
    p_lone_pair = 1 if lone_pairs >= 2 or (lone_pairs == 1 and conjugated) else 0
    steric_number = degree + h_count + lone_pairs - p_lone_pair
    return {4: "sp3", 3: "sp2", 2: "sp"}.get(steric_number, "?")


############################################################
#                           RULES                          #
############################################################


def _r1_valence(graph: _Graph, sulfur_valence: int) -> list[Violation]:
    """R1: the sum of a row never exceeds the maximum valence."""
    violations = []
    for i in range(graph.n):
        bond_sum = graph.bond_sum(i)
        valence = max_valence(graph.types[i], sulfur_valence)
        if bond_sum > valence:
            violations.append(
                Violation(
                    "R1",
                    (i,),
                    f"{graph.types[i]}{i} porte {bond_sum} liaisons "
                    f"pour une valence max de {valence}",
                )
            )
    return violations


def _r2_s_carbons(graph: _Graph) -> list[Violation]:
    """R2: two bonded "s" carbons (csPairs of reference_app.js)."""

    def sp2_like(v: int) -> bool:
        # a 2 or 3 in its row, or O / S, or an N with a conjugated lone pair
        if graph.has_multiple(v):
            return True
        element = graph.types[v]
        if element in ("O", "S"):
            return True
        if element == "N":
            return any(graph.has_multiple(w) for w in graph.adj[v])
        return False

    def sp2_next(x: int, y: int) -> bool:
        return any(v != y and sp2_like(v) for v in graph.adj[x])

    violations = []
    for a in range(graph.n):
        if not graph.is_s_carbon(a):
            continue
        for b in graph.adj[a]:
            if b < a or not graph.is_s_carbon(b):
                continue
            allowed = (
                graph.connected_without(a, b) and sp2_next(a, b) and sp2_next(b, a)
            )
            if not allowed:
                violations.append(
                    Violation(
                        "R2",
                        (a, b),
                        f"C{a} et C{b} sont deux carbones s liés hors des cas "
                        "permis (liaison dans un cycle, chacun ayant un autre "
                        "voisin sp²)",
                    )
                )
    return violations


def _r3_double_carbons(graph: _Graph) -> list[Violation]:
    """R3: a carbon carries at most one double bond."""
    return [
        Violation("R3", (i,), f"C{i} porte deux doubles liaisons")
        for i in range(graph.n)
        if graph.types[i] == "C"
        and sum(1 for j in graph.adj[i] if graph.order[i][j] == 2) > 1
    ]


def _ring_angle(ring: list[int]) -> float:
    """Interior angle of a regular ring."""
    return (len(ring) - 2) * 180 / len(ring)


def _r4_rings(graph: _Graph) -> list[Violation]:
    """R4: tangled small rings (R4a), large rings that cannot close flat (R4b).

    tangledRings of reference_app.js.
    """
    rings = graph.rings()
    small = [ring for ring in rings if len(ring) <= SMALL_RING]
    large = [ring for ring in rings if len(ring) > SMALL_RING]
    small_sets = [set(ring) for ring in small]
    violations = []

    # R4a: small rings share nothing, one atom, or exactly one bond
    for i, ring_i in enumerate(small):
        for j in range(i + 1, len(small)):
            shared = [x for x in ring_i if x in small_sets[j]]
            if len(shared) > 2 or (
                len(shared) == 2 and not graph.order[shared[0]][shared[1]]
            ):
                violations.append(
                    Violation(
                        "R4a",
                        tuple(shared),
                        f"cycles emmêlés : deux cycles partagent {len(shared)} "
                        "atomes au lieu d'une seule liaison",
                    )
                )

    def rings_with(*atoms: int) -> list[list[int]]:
        return [
            ring
            for ring, ring_set in zip(small, small_sets)
            if all(x in ring_set for x in atoms)
        ]

    # R4b: large rings close flat (sum of the turning angles close to 360°)
    for ring in large:
        n = len(ring)
        total = 0.0
        free = 0
        sp3: int | None = None
        for k in range(n):
            x, a, b = ring[k], ring[(k + n - 1) % n], ring[(k + 1) % n]
            others = [y for y in graph.adj[x] if y not in (a, b)]
            if graph.is_s_carbon(x):
                sp3 = x
                break
            if any(graph.order[x][y] == 3 for y in graph.adj[x]):
                continue  # sp: straight, turning angle 0
            ring_ab = rings_with(x, a, b)
            if ring_ab:
                phi = 360 - _ring_angle(ring_ab[0])
            else:
                t = others[0] if others else None
                ring_a = rings_with(x, a, t) if t is not None else []
                ring_b = rings_with(x, b, t) if t is not None else []
                if ring_a and ring_b:
                    phi = 360 - _ring_angle(ring_a[0]) - _ring_angle(ring_b[0])
                elif ring_a:
                    phi = (360 - _ring_angle(ring_a[0])) / 2
                elif ring_b:
                    phi = (360 - _ring_angle(ring_b[0])) / 2
                else:
                    phi = 120
                    if not others:
                        free += 1  # its H (or lone pair) may point in or out
            total += 180 - phi
        if sp3 is not None:
            violations.append(
                Violation(
                    "R4b",
                    (sp3,),
                    f"grand cycle de {n} atomes contenant C{sp3}, un carbone s : "
                    "il ne peut pas rester plan",
                )
            )
            continue
        # free atoms point outwards by default; turning one inwards costs 120°
        max_inward = free if n >= INWARD_H_MIN_RING else 0
        if not any(
            abs(total - 120 * k - 360) <= FLAT_TOLERANCE
            for k in range(max_inward + 1)
        ):
            violations.append(
                Violation(
                    "R4b",
                    tuple(ring),
                    f"grand cycle de {n} atomes qui ne peut pas se refermer à "
                    f"plat : la somme des virages vaut {round(total)}° au lieu "
                    "d'environ 360°",
                )
            )
    return violations


@lru_cache(maxsize=CACHE_SIZE)
def _violations_from_key(
    types: tuple[str, ...], data: bytes, n: int, sulfur_valence: int
) -> tuple[Violation, ...]:
    graph = _graph_from_key(types, data, n)
    return tuple(
        _r1_valence(graph, sulfur_valence)
        + _r2_s_carbons(graph)
        + _r3_double_carbons(graph)
        + _r4_rings(graph)
    )


def rule_violations(
    types: Sequence[str], matrix: np.ndarray, sulfur_valence: int | None = None
) -> list[Violation]:
    """All the violations of R1 to R4 (memoized by matrix)."""
    if sulfur_valence is None:
        sulfur_valence = current_sulfur_valence()
    matrix = np.ascontiguousarray(matrix, dtype=np.int64)
    return list(
        _violations_from_key(
            tuple(types), matrix.tobytes(), matrix.shape[0], sulfur_valence
        )
    )


def is_valid(
    types: Sequence[str], matrix: np.ndarray, sulfur_valence: int | None = None
) -> bool:
    """True if the molecule breaks none of the rules R1 to R4."""
    return not rule_violations(types, matrix, sulfur_valence)


def rule_counts(
    types: Sequence[str], matrix: np.ndarray, sulfur_valence: int | None = None
) -> tuple[int, int]:
    """(number of R1 violations, number of R2 + R3 + R4 violations)."""
    violations = rule_violations(types, matrix, sulfur_valence)
    nb_r1 = sum(1 for v in violations if v.rule_id == "R1")
    return nb_r1, len(violations) - nb_r1


def matrices_allowed(
    types_before: Sequence[str],
    matrix_before: np.ndarray,
    types_after: Sequence[str],
    matrix_after: np.ndarray,
    sulfur_valence: int | None = None,
) -> bool:
    """True if the state after creates no new violation.

    As in reference_app.js, the valence (R1) and the carbon / ring rules
    (R2 + R3 + R4, counted together) are compared separately: neither count
    may increase. A molecule already outside the rules stays mutable.
    """
    r1_before, rules_before = rule_counts(types_before, matrix_before, sulfur_valence)
    r1_after, rules_after = rule_counts(types_after, matrix_after, sulfur_valence)
    return r1_after <= r1_before and rules_after <= rules_before


def result_is_allowed(before: MolecularGraph, after: MolecularGraph) -> bool:
    """True if the result of an action creates no new violation."""
    types_before, matrix_before = to_matrix(before)
    types_after, matrix_after = to_matrix(after)
    return matrices_allowed(types_before, matrix_before, types_after, matrix_after)


def molecular_graph_violations(mol_graph: MolecularGraph) -> list[Violation]:
    """Violations of a molecular graph."""
    types, matrix = to_matrix(mol_graph)
    return rule_violations(types, matrix)
