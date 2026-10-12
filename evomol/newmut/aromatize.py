"""
Aromatize ("Arom" of the specification): give a Kekulé structure to a whole
fused ring system (aromatize of reference_app.js).

System: atoms reachable through ring bonds. Roles, read on the system with its
ring bonds counted as single:
- carbon without exocyclic double bond and with free valence: "need" (must
  receive one double bond inside the system);
- N without substituent: "opt" (may receive one);
- O, S, NR, atoms with an exocyclic double bond, carbons with four bonds:
  "none" (left with single ring bonds).
Algorithm (fewest bond changes): start from the existing double bonds of the
system, then flip the shortest alternating paths that reach the "need" atoms
still unmatched; if that fails, exhaustive search (backtracking, at most
200 000 steps) preferring the current partners.

One action per ring system (its smallest atom index, so that a system is not
drawn more often because it has more atoms). Valid if a Kekulé structure
exists, if the bonds change, and if the result respects the matrix rules.
"""

from __future__ import annotations

from collections import deque

from typing_extensions import override

from evomol.action import Action
from evomol.action.molecular_graph.action_molecular_graph import ActionMolGraph
from evomol.newmut.matrix_rules import _Graph, to_matrix
from evomol.newmut.rule_filtered import RuleCheckedMixin
from evomol.representation import MolecularGraph, Molecule

# maximum number of steps of the exhaustive search
MAX_STEPS = 200_000


def ring_systems(graph: _Graph) -> list[list[int]]:
    """Fused ring systems (atoms reachable through ring bonds), each one in
    the DFS order of reference_app.js from its smallest atom."""
    seen: set[int] = set()
    systems = []
    for start in range(graph.n):
        if start in seen:
            continue
        system = [start]
        in_system = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in graph.adj[u]:
                if v not in in_system and graph.connected_without(u, v):
                    in_system.add(v)
                    system.append(v)
                    stack.append(v)
        seen |= in_system
        if len(system) >= 3:
            systems.append(system)
    return systems


def kekule_bonds(
    graph: _Graph, system: list[int]
) -> dict[tuple[int, int], int] | None:
    """New orders of the ring bonds of the system (aromatize of
    reference_app.js), or None if no Kekulé structure exists."""
    in_system = set(system)

    def in_ring(x: int, y: int) -> bool:
        return x in in_system and y in in_system and graph.connected_without(x, y)

    neighbours = {x: [y for y in graph.adj[x] if in_ring(x, y)] for x in system}
    ring_bonds = [(x, y) for x in system for y in neighbours[x] if x < y]

    role: dict[int, str] = {}
    for x in system:
        exo = [y for y in graph.adj[x] if y not in neighbours[x]]
        base = len(neighbours[x]) + sum(graph.order[x][y] for y in exo)
        element = graph.types[x]
        if any(graph.order[x][y] >= 2 for y in exo):
            role[x] = "none"
        elif element == "C":
            role[x] = "need" if base + 1 <= 4 else "none"
        elif element == "N" and base + 1 <= 3:
            role[x] = "opt"
        else:
            role[x] = "none"

    def eligible(x: int) -> bool:
        return role[x] in ("need", "opt")

    old_partner: dict[int, int] = {}
    for x, y in ring_bonds:
        if graph.order[x][y] == 2:
            old_partner[x] = y
            old_partner[y] = x

    # 1) keep the existing double bonds, flip the shortest augmenting paths
    mate: dict[int, int] = {
        x: y for x, y in old_partner.items() if eligible(x) and eligible(y)
    }

    def augment(u: int) -> bool:
        prev: dict[int, int | None] = {u: None}
        queue = deque([u])
        while queue:
            x = queue.popleft()
            for y in neighbours[x]:
                if not eligible(y) or y in prev or mate.get(x) == y:
                    continue
                prev[y] = x
                if y not in mate:  # free atom reached: flip the path u ... x - y
                    b: int | None = y
                    c: int | None = x
                    while c is not None and b is not None:
                        next_b = mate.get(c)
                        mate[b] = c
                        mate[c] = b
                        b = next_b
                        c = prev.get(b) if b is not None else None
                    return True
                z = mate[y]
                if z not in prev:
                    prev[z] = y
                    queue.append(z)
        return False

    fast = True
    for x in system:
        if role[x] == "need" and x not in mate and not augment(x):
            fast = False
            break
    if not fast or any(role[x] == "need" and x not in mate for x in system):
        mate.clear()
        fast = False

    # 2) fallback: exhaustive search, preferring the existing double bonds
    steps = 0

    def solve() -> bool:
        nonlocal steps
        steps += 1
        if steps > MAX_STEPS:
            return False
        pick: int | None = None
        options: list[int] = []
        for x in system:
            if role[x] != "need" or x in mate:
                continue
            candidates = [y for y in neighbours[x] if y not in mate and eligible(y)]
            if pick is None or len(candidates) < len(options):
                pick, options = x, candidates
                if not candidates:
                    break
        if pick is None:
            return True
        options.sort(
            key=lambda y: (old_partner.get(pick) != y, role[y] == "opt")  # type: ignore[arg-type]
        )
        for y in options:
            mate[pick] = y
            mate[y] = pick
            if solve():
                return True
            del mate[pick]
            del mate[y]
        return False

    if not fast and not solve():
        return None

    return {(x, y): 2 if mate.get(x) == y else 1 for x, y in ring_bonds}


class AromatizeMG(RuleCheckedMixin, ActionMolGraph):
    """
    Give a Kekulé structure to a fused ring system.
    """

    def __init__(
        self,
        molecule: Molecule,
        atom_idx: int,
        bond_changes: tuple[tuple[int, int, int], ...],
    ) -> None:
        """Initialize the action.

        Args:
            molecule: molecule to modify.
            atom_idx: smallest atom index of the ring system.
            bond_changes: (atom1, atom2, new bond order) of the changed bonds.
        """
        super().__init__(molecule)
        self.atom_idx: int = atom_idx
        self.bond_changes: tuple[tuple[int, int, int], ...] = bond_changes

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AromatizeMG):
            return False
        return self.molecule == other.molecule and self.atom_idx == other.atom_idx

    @override
    def __hash__(self) -> int:
        return hash(self.__repr__())

    @override
    def apply_action(self, new_mol_graph: MolecularGraph) -> None:
        for atom1, atom2, order in self.bond_changes:
            new_mol_graph.set_bond(atom1, atom2, order)

    def __repr__(self) -> str:
        return (
            f"AromatizeMolGraph({self.molecule}, {self.atom_idx}, "
            f"{list(self.bond_changes)})"
        )

    @override
    @classmethod
    def list_unchecked_actions(cls, molecule: Molecule) -> list[Action]:
        """One action per ring system without charged atom that has a Kekulé
        structure different from the current bonds."""
        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)
        types, matrix = to_matrix(mol_graph)
        graph = _Graph(types, matrix)

        action_list: list[Action] = []
        for system in ring_systems(graph):
            if any(mol_graph.atom_charged_or_radical(x) for x in system):
                continue
            bonds = kekule_bonds(graph, system)
            if bonds is None:
                continue
            changes = tuple(
                (x, y, order)
                for (x, y), order in sorted(bonds.items())
                if graph.order[x][y] != order
            )
            if changes:
                action_list.append(AromatizeMG(molecule, min(system), changes))
        return action_list
