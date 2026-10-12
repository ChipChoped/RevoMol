"""
FlipAlternatingPath ("Flip" of the specification): invert the bond orders
along an alternating path (quinoid switch).

For two atoms u < v, the shortest alternating path u -> v starts and ends with
a single bond and alternates strictly 1, 2, 1, ..., 1 (at least one double
bond, so at least 3 bonds, as the "flip" of classify in reference_core.js).
Inverting it (1 <-> 2) keeps the row sums of the inner atoms; u and v each
gain one bond order, so they must each have at least one hydrogen.
Example: CC1=CC=C(C)S1 -> C=C1C=CC(=C)S1.

The result must also respect the matrix rules.
"""

from __future__ import annotations

from collections import deque

from typing_extensions import override

from evomol.action import Action
from evomol.action.molecular_graph.action_molecular_graph import ActionMolGraph
from evomol.newmut.rule_filtered import RuleCheckedMixin
from evomol.representation import MolecularGraph, Molecule


def alternating_paths(
    mol_graph: MolecularGraph, start: int, excluded: set[int]
) -> dict[int, list[int]]:
    """Shortest alternating paths 1, 2, 1, ..., 1 from start, by end atom.

    BFS on the states (atom, order of the next bond, number of bonds capped at
    3): a direct single bond start-v does not hide a longer path to v. A path
    is kept only if it has at least 3 bonds and is simple (the shortest walk
    of the BFS may revisit an atom: the pair is then skipped). Atoms in
    excluded are not crossed.

    Returns:
        dict[int, list[int]]: end atom -> atoms of the path, start included.
    """
    State = tuple[int, int, int]
    first: State = (start, 1, 0)
    parent: dict[State, State | None] = {first: None}
    queue = deque([first])
    paths: dict[int, list[int]] = {}
    while queue:
        state = queue.popleft()
        atom, expected, length = state
        for other in mol_graph.atoms_bonded_to(atom):
            if other in excluded or mol_graph.bond_order(atom, other) != expected:
                continue
            next_state = (other, 3 - expected, min(length + 1, 3))
            if next_state in parent:
                continue
            parent[next_state] = state
            queue.append(next_state)
            # end of a path: a single bond, after at least 3 bonds
            if expected == 1 and next_state[2] == 3 and other not in paths:
                path = []
                current: State | None = next_state
                while current is not None:
                    path.append(current[0])
                    current = parent[current]
                path.reverse()
                if len(set(path)) == len(path):
                    paths[other] = path
    return paths


class FlipAlternatingPathMG(RuleCheckedMixin, ActionMolGraph):
    """
    Invert the bond orders along an alternating path between two atoms.
    """

    def __init__(self, molecule: Molecule, path: tuple[int, ...]) -> None:
        """Initialize the action.

        Args:
            molecule: molecule to modify.
            path: atoms of the alternating path, from u to v (u < v).
        """
        super().__init__(molecule)
        self.path: tuple[int, ...] = path

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlipAlternatingPathMG):
            return False
        return self.molecule == other.molecule and self.path == other.path

    @override
    def __hash__(self) -> int:
        return hash(self.__repr__())

    @override
    def apply_action(self, new_mol_graph: MolecularGraph) -> None:
        for k in range(len(self.path) - 1):
            atom1, atom2 = self.path[k], self.path[k + 1]
            # 1, 2, 1, ..., 1 becomes 2, 1, 2, ..., 2
            new_mol_graph.set_bond(atom1, atom2, 2 if k % 2 == 0 else 1)

    def __repr__(self) -> str:
        return f"FlipAlternatingPathMolGraph({self.molecule}, {list(self.path)})"

    @override
    @classmethod
    def list_unchecked_actions(cls, molecule: Molecule) -> list[Action]:
        """For each pair u < v of uncharged atoms with at least one hydrogen,
        the shortest alternating path between them, if any."""
        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)
        implicit_valences = mol_graph.implicit_valences
        charged = {
            atom
            for atom in range(mol_graph.nb_atoms)
            if mol_graph.atom_charged_or_radical(atom)
        }
        ends = [
            atom
            for atom in range(mol_graph.nb_atoms)
            if atom not in charged and implicit_valences[atom] >= 1
        ]
        ends_set = set(ends)

        action_list: list[Action] = []
        for u in ends:
            for v, path in sorted(alternating_paths(mol_graph, u, charged).items()):
                if v > u and v in ends_set:
                    action_list.append(FlipAlternatingPathMG(molecule, tuple(path)))
        return action_list
