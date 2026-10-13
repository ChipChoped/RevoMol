"""
Fast check of the matrix rules for the result of an action (newmut).

Same answer as the original check (apply the action on an RDKit copy, rebuild
its matrix, recompute every rule on the whole molecule: see
rule_filtered.action_is_allowed), obtained in two ways:

1. the matrix of the result is computed directly from the action (matrix_effect),
   without RDKit, for the actions whose effect is a simple matrix edit;
2. only the rules around the change are recomputed (ParentRules.allowed_after):
   - R1, R3, R5 depend on the row of one atom: only the changed rows;
   - R2 (bond between two s carbons) depends on the rows within distance 2 of
     the bond and on whether the bond lies on a cycle: only the bonds near the
     changed atoms, and those whose ring membership changed;
   - R4 depends on the ring bonds: if they are the same, the rings are the same
     (R4a unchanged) and only the large rings that contain a changed atom are
     checked again (R4b); otherwise everything is recomputed.

The equivalence with the original check is tested on every action of many
molecules (tests/newmut/test_fast_rules.py). FAST_RULES = False goes back to
the original check.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from typing import Hashable, Sequence

import numpy as np

from evomol.action import Action
from evomol.action import molecular_graph as mg
from evomol.newmut import matrix_rules as mr
from evomol.newmut.matrix_rules import _Graph
from evomol.representation import MolecularGraph

# False: original check (RDKit copy and full recomputation) everywhere
FAST_RULES = True

# True: on a molecule without any violation, the actions that surely create one
# are refused without being checked (ParentRules.surely_refused). Same answer.
PREFILTERS = True

# parents whose rules are kept in memory
CACHE_SIZE = 2_000
_parents: OrderedDict[Hashable, ParentRules] = OrderedDict()

Edge = tuple[int, int]


def _edges(graph: _Graph) -> set[Edge]:
    return {(a, b) for a in range(graph.n) for b in graph.adj[a] if a < b}


def _ball(graph: _Graph, atoms: set[int], radius: int) -> set[int]:
    """Atoms within distance radius of atoms."""
    ball = set(atoms)
    frontier = set(atoms)
    for _ in range(radius):
        frontier = {v for u in frontier for v in graph.adj[u]} - ball
        ball |= frontier
    return ball


class ParentRules:
    """Rules of a molecule, item by item, to check its mutants quickly."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, types: Sequence[str], matrix: np.ndarray, sulfur_valence: int):
        self.types = list(types)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.int64)
        self.n = len(self.types)
        self.sulfur_valence = sulfur_valence
        self.graph = _Graph(self.types, self.matrix)
        graph = self.graph
        self.bridges = graph.bridges()
        self.edges = _edges(graph)
        self.ring_edges = self.edges - self.bridges
        self.rings = graph.rings(self.bridges)
        self.small, self.large, self.small_sets = mr.split_rings(self.rings)

        self.r1 = {i for i in range(self.n) if mr.r1_atom(graph, i, sulfur_valence)}
        self.r3 = {i for i in range(self.n) if mr.r3_atom(graph, i)}
        self.r5 = {i for i in range(self.n) if mr.r5_atom(graph, i)}
        self.r2 = {(a, b) for a, b in self.edges if mr.r2_bond(graph, a, b, (a, b) not in self.bridges)}
        self.r4a = len(mr.r4a_violations(graph, self.small, self.small_sets))
        self.r4b = [mr.r4b_ring(graph, ring, self.small, self.small_sets) is not None for ring in self.large]
        self.nb_r1 = len(self.r1)
        self.charged = False  # set by parent_rules: charged or radical atom
        self.nb_rules = len(self.r2) + len(self.r3) + self.r4a + sum(self.r4b) + len(self.r5)
        self._distances: dict[int, list[int]] = {}

    def distance(self, a: int, b: int) -> int:
        """Number of bonds on the shortest path between a and b (-1: not connected)."""
        if a not in self._distances:
            dist = [-1] * self.n
            dist[a] = 0
            frontier = [a]
            while frontier:
                following = []
                for u in frontier:
                    for v in self.graph.adj[u]:
                        if dist[v] < 0:
                            dist[v] = dist[u] + 1
                            following.append(v)
                frontier = following
            self._distances[a] = dist
        return self._distances[a][b]

    def surely_refused(self, action: Action) -> bool:
        """True if the action surely creates a violation, on a molecule that has
        none (the count can only go up, so the action is refused):

        A. a single bond created with an s carbon, between two atoms more than
           mr.SMALL_RING - 1 bonds apart: the s carbon ends up in a large ring
           (R4b, which refuses any s carbon in a large ring; if R4b changes, the
           equivalence test of tests/newmut/test_fast_rules.py catches it);
        C. a 4th heavy atom bonded by a single bond to an s carbon that already
           carries 3 (R5): atom added, fragment grafted or fused, group moved,
           bond created;
        D. an alternating path inverted while one of its ends is a carbon that
           already carries a double bond: that end gets a second one (R3);
        E. a C=C outside any cycle, between two carbons that carry no other
           double bond, made single: two bonded s carbons outside a cycle (R2).
        """
        # pylint: disable=import-outside-toplevel
        from evomol.newmut.annelate import AnnelateMG
        from evomol.newmut.flip_path import FlipAlternatingPathMG
        from evomol.newmut.graft import GraftFragmentMG

        if self.nb_r1 or self.nb_rules:
            return False
        graph = self.graph

        def four_on_s(atom: int) -> bool:
            return graph.is_s_carbon(atom) and len(graph.adj[atom]) == 3

        if isinstance(action, mg.ChangeBondMG):
            i, j = action.atom1, action.atom2
            order = graph.order[i][j]
            if order == 0 and action.bond_type == 1:
                if (graph.is_s_carbon(i) or graph.is_s_carbon(j)) and self.distance(i, j) + 1 > mr.SMALL_RING:
                    return True  # A
                return four_on_s(i) or four_on_s(j)  # C
            if order == 2 and action.bond_type == 1 and (min(i, j), max(i, j)) in self.bridges:
                return (
                    self.types[i] == "C" and self.types[j] == "C"
                    and sum(graph.order[i][k] == 2 for k in graph.adj[i]) == 1
                    and sum(graph.order[j][k] == 2 for k in graph.adj[j]) == 1
                    and not any(graph.order[x][k] == 3 for x in (i, j) for k in graph.adj[x])
                )  # E
            return False
        if isinstance(action, mg.AddAtomMG):
            return self.n > 0 and action.bond_order == 1 and four_on_s(action.index_atom)  # C
        if isinstance(action, GraftFragmentMG):
            return four_on_s(action.anchor)  # C
        if isinstance(action, AnnelateMG):
            return four_on_s(action.atom1) or four_on_s(action.atom2)  # C
        if isinstance(action, mg.MoveGroupMG):
            return action.bond_type == 1 and four_on_s(action.atom_to_link)  # C
        if isinstance(action, FlipAlternatingPathMG):
            return any(
                self.types[end] == "C" and any(graph.order[end][k] == 2 for k in graph.adj[end])
                for end in (action.path[0], action.path[-1])
            )  # D
        return False

    def allowed_after(self, types: Sequence[str], matrix: np.ndarray, new_of_old: Sequence[int]) -> bool:
        """True if the state (types, matrix) creates no new violation.

        new_of_old[i]: index in the new state of atom i of this molecule (-1 if
        removed), increasing for the kept atoms; the other atoms of the new
        state are added atoms.
        """
        matrix = np.asarray(matrix, dtype=np.int64)
        if all(i == j for i, j in enumerate(new_of_old)):
            state = self._state_without_removal(types, matrix)
        else:
            state = self._state_general(types, matrix, np.asarray(new_of_old, dtype=np.int64))
        if state is None:
            return True  # same molecule
        return self._counts_allowed(*state)

    def _state_without_removal(self, types: Sequence[str], matrix: np.ndarray) -> tuple | None:
        """Changed atoms, graph, bridges and ring changes of a state whose atoms
        keep their indices (atoms possibly added at the end)."""
        n, n_new = self.n, len(types)
        diff = matrix[:n, :n] != self.matrix
        np.fill_diagonal(diff, False)
        changed_rows = diff.any(axis=1)
        if n_new > n:
            changed_rows |= (matrix[:n, n:] != 0).any(axis=1)
        changed = set(np.flatnonzero(changed_rows).tolist()) | set(range(n, n_new))
        changed |= {i for i in range(n) if types[i] != self.types[i]}
        if not changed:
            return None
        rows = {i: matrix[i].tolist() for i in changed}
        graph = _Graph.patched(self.graph, types, rows)

        added: set[Edge] = set()
        removed: set[Edge] = set()
        for i in changed:
            new_neighbours = set(graph.adj[i])
            old_neighbours = set(self.graph.adj[i]) if i < n else set()
            added |= {(min(i, j), max(i, j)) for j in new_neighbours - old_neighbours}
            removed |= {(min(i, j), max(i, j)) for j in old_neighbours - new_neighbours}
        identity = np.arange(n, dtype=np.int64)
        old_of_new = np.concatenate([identity, np.full(n_new - n, -1, dtype=np.int64)])
        if not added and not removed:
            # same bonds: same bridges, same rings
            return changed, set(changed) & set(range(n)), graph, self.bridges, self.edges, set(), True, identity, old_of_new
        if not removed and all(
            max(a, b) >= n and len(graph.adj[max(a, b)]) == 1 for a, b in added
        ):
            # only terminal atoms added: their bonds are bridges, rings unchanged
            return (changed, set(changed) & set(range(n)), graph, self.bridges | added, self.edges | added,
                    set(), True, identity, old_of_new)
        return self._topology(changed, set(changed) & set(range(n)), graph, identity, old_of_new)

    def _state_general(self, types: Sequence[str], matrix: np.ndarray, new_of_old: np.ndarray) -> tuple | None:
        """Changed atoms, graph, bridges and ring changes of any state."""
        n_new = len(types)
        kept_old = np.flatnonzero(new_of_old >= 0)
        kept_new = new_of_old[kept_old]
        old_of_new = np.full(n_new, -1, dtype=np.int64)
        old_of_new[kept_new] = kept_old
        deleted = set(np.flatnonzero(new_of_old < 0).tolist())

        # changed rows (in the new state): added atoms, element or row changed,
        # neighbours of removed atoms
        mapped = np.zeros((n_new, n_new), dtype=np.int64)
        mapped[np.ix_(kept_new, kept_new)] = self.matrix[np.ix_(kept_old, kept_old)]
        off_diagonal = ~np.eye(n_new, dtype=bool)
        row_changed = ((matrix != mapped) & off_diagonal).any(axis=1)
        changed = set(np.flatnonzero(row_changed | (old_of_new < 0)).tolist())
        changed |= {j for j in range(n_new) if old_of_new[j] >= 0 and types[j] != self.types[old_of_new[j]]}
        changed |= {int(new_of_old[v]) for i in deleted for v in self.graph.adj[i] if new_of_old[v] >= 0}
        if not changed and not deleted:
            return None
        changed_old = deleted | {int(old_of_new[j]) for j in changed if old_of_new[j] >= 0}
        return self._topology(changed, changed_old, _Graph(types, matrix), new_of_old, old_of_new)

    def _topology(self, changed: set[int], changed_old: set[int], graph: _Graph, new_of_old: np.ndarray,
                  old_of_new: np.ndarray) -> tuple:
        """Bridges, bonds whose ring membership changed and whether the ring
        bonds are the same, computed on the whole new graph."""
        bridges = graph.bridges()
        edges = _edges(graph)
        renumber = new_of_old.tolist()

        def to_new(edge: Edge) -> Edge | None:
            a, b = renumber[edge[0]], renumber[edge[1]]
            if a < 0 or b < 0:
                return None
            return (a, b) if a < b else (b, a)

        flipped: set[Edge] = set()
        for edge in self.edges:
            new_edge = to_new(edge)
            if new_edge is not None and new_edge in edges and ((edge in self.bridges) != (new_edge in bridges)):
                flipped.add(edge)
        same_rings = {to_new(edge) for edge in self.ring_edges} == edges - bridges
        return changed, changed_old, graph, bridges, edges, flipped, same_rings, new_of_old, old_of_new

    # pylint: disable=too-many-arguments,too-many-locals
    def _counts_allowed(self, changed: set[int], changed_old: set[int], graph: _Graph, bridges: set[Edge],
                        edges: set[Edge], flipped_old: set[Edge], same_rings: bool, new_of_old: np.ndarray,
                        old_of_new: np.ndarray) -> bool:
        """Count the violations around the change and compare."""
        renumber = new_of_old.tolist()
        back = old_of_new.tolist()

        # R1, R3, R5: one row each
        nb_r1 = self.nb_r1 - len(self.r1 & changed_old) + sum(
            1 for j in changed if mr.r1_atom(graph, j, self.sulfur_valence)
        )
        if nb_r1 > self.nb_r1:
            return False
        nb_rules = self.nb_rules
        nb_rules += sum(1 for j in changed if mr.r3_atom(graph, j)) - len(self.r3 & changed_old)
        nb_rules += sum(1 for j in changed if mr.r5_atom(graph, j)) - len(self.r5 & changed_old)

        # R2: bonds within distance 2 of a changed atom, or whose ring
        # membership changed
        ball_old = _ball(self.graph, changed_old, 2)
        region_new = _ball(graph, changed, 2) | {renumber[i] for i in ball_old if renumber[i] >= 0}
        region_old = ball_old | {back[j] for j in region_new if back[j] >= 0}
        flipped_new = set()
        for a, b in flipped_old:
            x, y = renumber[a], renumber[b]
            flipped_new.add((x, y) if x < y else (y, x))
        nb_rules -= sum(1 for a, b in self.r2 if a in region_old or b in region_old or (a, b) in flipped_old)
        candidates = {
            (min(a, b), max(a, b)) for a in region_new for b in graph.adj[a]
        } | flipped_new
        nb_rules += sum(1 for a, b in candidates if mr.r2_bond(graph, a, b, (a, b) not in bridges))

        # R4: same ring bonds -> same rings (the search only follows ring bonds,
        # in index order, which the kept atoms preserve)
        if same_rings:
            small = [[renumber[x] for x in ring] for ring in self.small]
            small_sets = [set(ring) for ring in small]
            for ring, violated in zip(self.large, self.r4b):
                new_ring = [renumber[x] for x in ring]
                if any(x in changed for x in new_ring):
                    nb_rules += (mr.r4b_ring(graph, new_ring, small, small_sets) is not None) - violated
        else:
            small, large, small_sets = mr.split_rings(graph.rings(bridges))
            nb_rules += len(mr.r4a_violations(graph, small, small_sets)) - self.r4a
            nb_rules += sum(mr.r4b_ring(graph, ring, small, small_sets) is not None for ring in large)
            nb_rules -= sum(self.r4b)

        return nb_rules <= self.nb_rules


def parent_rules(mol_graph: MolecularGraph) -> ParentRules:
    """Rules of a molecular graph, kept in memory (on the graph object, and by
    matrix for the other graphs of the same molecule)."""
    sulfur_valence = mr.current_sulfur_valence()
    parent: ParentRules | None = getattr(mol_graph, "_newmut_rules", None)
    if parent is not None and parent.sulfur_valence == sulfur_valence:
        return parent
    types, matrix = mr.to_matrix(mol_graph)
    mol = mol_graph.mol
    charges = tuple(
        (mol.GetAtomWithIdx(i).GetFormalCharge(), mol.GetAtomWithIdx(i).GetNumRadicalElectrons())
        for i in range(len(types))
    )
    key = (tuple(types), matrix.tobytes(), charges, sulfur_valence)
    parent = _parents.get(key)
    if parent is None:
        parent = ParentRules(types, matrix, sulfur_valence)
        parent.charged = any(c != (0, 0) for c in charges)
        _parents[key] = parent
        if len(_parents) > CACHE_SIZE:
            _parents.popitem(last=False)
    else:
        _parents.move_to_end(key)
    mol_graph._newmut_rules = parent  # type: ignore[attr-defined]  # pylint: disable=protected-access
    return parent


def clear_cache() -> None:
    """Forget the parents kept in memory."""
    _parents.clear()


# ==============================================================================
#  EFFECT OF AN ACTION ON THE MATRIX
# ==============================================================================
def _bond(matrix: np.ndarray, a: int, b: int, order: int) -> None:
    matrix[a, b] = matrix[b, a] = order


def _add_atom(types: list[str], matrix: np.ndarray, element: str) -> np.ndarray:
    n = len(types)
    new = np.zeros((n + 1, n + 1), dtype=np.int64)
    new[:n, :n] = matrix
    new[n, n] = mr.VALENCE_ELECTRONS.get(element, 0)
    types.append(element)
    return new


@lru_cache(maxsize=None)
def _fragment_matrix(fragment_smiles: str) -> tuple[tuple[str, ...], np.ndarray, tuple[int, ...]]:
    """(types, matrix, attachment atoms) of a fragment, atoms in the order of
    MolecularGraph.add_group."""
    # pylint: disable=import-outside-toplevel
    from evomol.newmut.fragments import parse_fragment

    fragment = parse_fragment(fragment_smiles)
    types, matrix = mr.to_matrix(MolecularGraph(fragment.smiles))
    return tuple(types), matrix, fragment.attach


# pylint: disable=too-many-return-statements
def matrix_effect(
    action: Action, types: Sequence[str], matrix: np.ndarray
) -> tuple[list[str], np.ndarray, list[int]] | None:
    """(types, matrix, new_of_old) of the result of the action, computed on the
    matrix as MolecularGraph does it with RDKit (added atoms at the end, removed
    atoms renumber the following ones). None for the actions whose effect is
    not a simple matrix edit (AddGroupMG, RemoveGroupMG): they go through RDKit.
    For a fragment (graft, annelation), RDKit may write the Kekulé structure of
    an aromatic fragment differently: the rules do not depend on it (each atom
    of an aromatic system has one double bond in any Kekulé structure), and the
    equivalence test checks it.
    """
    # pylint: disable=import-outside-toplevel
    from evomol.newmut.annelate import AnnelateMG
    from evomol.newmut.aromatize import AromatizeMG
    from evomol.newmut.flip_path import FlipAlternatingPathMG
    from evomol.newmut.graft import GraftFragmentMG
    from evomol.newmut.substitute_shift import SubstituteShiftMG

    new_types = list(types)
    new = np.array(matrix, dtype=np.int64)
    n = len(types)
    identity = list(range(n))

    if isinstance(action, mg.AddAtomMG):
        new = _add_atom(new_types, new, action.atom_type)
        if n > 0:
            _bond(new, n, action.index_atom, action.bond_order)
        return new_types, new, identity
    if isinstance(action, mg.ChangeBondMG):
        _bond(new, action.atom1, action.atom2, action.bond_type)
        return new_types, new, identity
    if isinstance(action, mg.SubstituteAtomMG):
        new_types[action.atom_idx] = action.new_type
        new[action.atom_idx, action.atom_idx] = mr.VALENCE_ELECTRONS.get(action.new_type, 0)
        return new_types, new, identity
    if isinstance(action, mg.MoveGroupMG):
        _bond(new, action.atom_moving, action.atom_staying, 0)
        _bond(new, action.atom_moving, action.atom_to_link, action.bond_type)
        return new_types, new, identity
    if isinstance(action, mg.InsertCarbonMG):
        _bond(new, action.atom1_idx_to_bond, action.atom2_idx_to_bond, 0)
        new = _add_atom(new_types, new, "C")
        _bond(new, n, action.atom1_idx_to_bond, action.bond_to_1)
        _bond(new, n, action.atom2_idx_to_bond, action.bond_to_2)
        return new_types, new, identity
    if isinstance(action, (mg.RemoveAtomMG, mg.CutAtomMG)):
        if isinstance(action, mg.CutAtomMG):
            removed = action.atom_to_cut
            _bond(new, action.atom_to_cut, action.atom1_to_bond, 0)
            _bond(new, action.atom_to_cut, action.atom2_to_bond, 0)
            _bond(new, action.atom1_to_bond, action.atom2_to_bond, action.bond_type)
        else:
            removed = action.atom_idx
        new = np.delete(np.delete(new, removed, axis=0), removed, axis=1)
        del new_types[removed]
        return new_types, new, [i if i < removed else (i - 1 if i > removed else -1) for i in range(n)]
    if isinstance(action, FlipAlternatingPathMG):
        for k in range(len(action.path) - 1):
            _bond(new, action.path[k], action.path[k + 1], 2 if k % 2 == 0 else 1)
        return new_types, new, identity
    if isinstance(action, (GraftFragmentMG, AnnelateMG)):
        # block concatenation [[M, B], [B^T, F]] with the connection cases at 1
        fragment_types, fragment_matrix, attach = _fragment_matrix(action.fragment_smiles)
        size = len(fragment_types)
        block = np.zeros((n + size, n + size), dtype=np.int64)
        block[:n, :n] = new
        block[n:, n:] = fragment_matrix
        new_types += fragment_types
        if isinstance(action, GraftFragmentMG):
            _bond(block, action.anchor, n + attach[0], 1)
        else:
            _bond(block, action.atom1, n + attach[0], 1)
            _bond(block, action.atom2, n + attach[1], 1)
        return new_types, block, identity
    if isinstance(action, (AromatizeMG, SubstituteShiftMG)):
        if isinstance(action, SubstituteShiftMG):
            new_types[action.atom_idx] = action.new_type
            new[action.atom_idx, action.atom_idx] = mr.VALENCE_ELECTRONS.get(action.new_type, 0)
        for atom1, atom2, order in action.bond_changes:
            _bond(new, atom1, atom2, order)
        return new_types, new, identity
    return None
