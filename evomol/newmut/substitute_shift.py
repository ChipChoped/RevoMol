"""
SubstituteShift ("SubS" of the specification): substitute an atom, shifting
the double bonds when the new element accepts fewer bonds (substituteWithShift
of reference_app.js).

- If the new element accepts the sum of the row, simple substitution.
- Otherwise (e.g. C= -> O):
  1. the multiple bonds of the atom are lowered, the highest first, until the
     valence holds; the partners that lost a bond order are "holes";
  2. for each hole: candidate states "no move", then the hole moved along an
     alternating path hole -1- x =2= y (shortest first, not through the
     substituted atom), each path being inverted (1 <-> 2);
  3. the first state whose result creates no new violation of the matrix rules
     is kept. If there is none, the action does not exist.

The final state is computed when the actions are listed; apply_action replays
the substitution and the bond changes.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
from typing_extensions import override

from evomol.action import Action
from evomol.action.molecular_graph.action_molecular_graph import ActionMolGraph
from evomol.newmut.matrix_rules import (
    VALENCE_ELECTRONS,
    matrices_allowed,
    max_valence,
    to_matrix,
)
from evomol.newmut.rule_filtered import RuleCheckedMixin
from evomol.representation import MolecularGraph, Molecule

Matrix = list[list[int]]
Path = list[tuple[int, int]]


def _neighbors(matrix: Matrix, x: int) -> list[int]:
    return [y for y, order in enumerate(matrix[x]) if order and y != x]


def _hole_moves(matrix: Matrix, hole: int, atom: int) -> list[Path]:
    """Candidate moves of a hole: no move, then alternating paths
    hole -1- y =2= z -1- ... (BFS, shortest first), avoiding atom."""
    out: list[Path] = [[]]
    seen = {hole}
    queue: deque[tuple[int, Path]] = deque([(hole, [])])
    while queue:
        x, path = queue.popleft()
        for y in _neighbors(matrix, x):
            if y == atom or y in seen or matrix[x][y] != 1:
                continue
            z = next(
                (z for z in _neighbors(matrix, y) if z != x and matrix[y][z] == 2),
                None,
            )
            if z is None or z == atom or z in seen:
                continue
            seen.add(y)
            seen.add(z)
            new_path = [*path, (x, y), (y, z)]
            out.append(new_path)
            queue.append((z, new_path))
    return out


def _apply_path(matrix: Matrix, path: Path) -> Matrix:
    """Copy of the matrix with the path inverted: orders 2, 1, 2, ..."""
    new = [row[:] for row in matrix]
    for k, (u, v) in enumerate(path):
        new[u][v] = new[v][u] = 2 if k % 2 == 0 else 1
    return new


def shift_states(
    types: Sequence[str], matrix: np.ndarray, atom: int, element: str
) -> list[Matrix]:
    """Candidate bond matrices after substituting atom by element, in the order
    of reference_app.js (empty list: the valence cannot hold).

    The diagonal of the returned matrices is that of the input matrix.
    """
    work: Matrix = matrix.tolist()
    n = len(types)
    excess = sum(work[atom][j] for j in range(n) if j != atom) - max_valence(element)
    holes: list[int] = []
    while excess > 0:
        multiple = sorted(
            (j for j in range(n) if j != atom and work[atom][j] > 1),
            key=lambda j: -work[atom][j],
        )
        if not multiple:
            return []
        partner = multiple[0]
        work[atom][partner] -= 1
        work[partner][atom] -= 1
        excess -= 1
        if partner not in holes:
            holes.append(partner)

    states = [work]
    for hole in holes:
        candidates: list[tuple[Matrix, int]] = []
        for state in states:
            for path in _hole_moves(state, hole, atom):
                candidates.append((_apply_path(state, path), len(path)))
        candidates.sort(key=lambda candidate: candidate[1])
        states = [state for state, _ in candidates]
    return states


def substitute_with_shift(
    types: Sequence[str], matrix: np.ndarray, atom: int, element: str
) -> tuple[list[str], np.ndarray] | None:
    """First state (types, matrix) after the substitution that creates no new
    violation, or None."""
    new_types = list(types)
    new_types[atom] = element
    for state in shift_states(types, matrix, atom, element):
        new_matrix = np.array(state, dtype=np.int64)
        new_matrix[atom, atom] = VALENCE_ELECTRONS.get(element, 0)
        if matrices_allowed(types, matrix, new_types, new_matrix):
            return new_types, new_matrix
    return None


class SubstituteShiftMG(RuleCheckedMixin, ActionMolGraph):
    """
    Substitute an atom, shifting the double bonds if needed.
    """

    def __init__(
        self,
        molecule: Molecule,
        atom_idx: int,
        new_type: str,
        bond_changes: tuple[tuple[int, int, int], ...] = (),
    ) -> None:
        """Initialize the action.

        Args:
            molecule: molecule to modify.
            atom_idx: index of the atom to substitute.
            new_type: new element.
            bond_changes: (atom1, atom2, new bond order) of the bonds shifted
                by the substitution.
        """
        super().__init__(molecule)
        self.atom_idx: int = atom_idx
        self.new_type: str = new_type
        self.bond_changes: tuple[tuple[int, int, int], ...] = bond_changes

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubstituteShiftMG):
            return False
        return (
            self.molecule == other.molecule
            and self.atom_idx == other.atom_idx
            and self.new_type == other.new_type
        )

    @override
    def __hash__(self) -> int:
        return hash(self.__repr__())

    @override
    def apply_action(self, new_mol_graph: MolecularGraph) -> None:
        new_mol_graph.replace_atom(self.atom_idx, self.new_type)
        for atom1, atom2, order in self.bond_changes:
            new_mol_graph.set_bond(atom1, atom2, order)

    def __repr__(self) -> str:
        shift = f", {list(self.bond_changes)}" if self.bond_changes else ""
        return (
            f"SubstituteShiftMolGraph({self.molecule}, "
            f"{self.atom_idx}, {self.new_type}{shift})"
        )

    @override
    @classmethod
    def action_is_allowed(
        cls,
        action: Action,
        molecule: Molecule,
        before: tuple[list[str], np.ndarray] | None = None,
    ) -> bool:
        """The final state was chosen with the rules (list_unchecked_actions)."""
        return True

    @override
    @classmethod
    def list_unchecked_actions(cls, molecule: Molecule) -> list[Action]:
        """For each uncharged atom and each other accepted element, the
        substitution with the first allowed shift of the double bonds."""
        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)
        types, matrix = to_matrix(mol_graph)

        action_list: list[Action] = []
        for atom in range(len(types)):
            if mol_graph.atom_charged_or_radical(atom):
                continue
            for element in Molecule.accepted_atoms:
                if element == types[atom]:
                    continue
                result = substitute_with_shift(types, matrix, atom, element)
                if result is None:
                    continue
                new_matrix = result[1]
                changes = tuple(
                    (int(i), int(j), int(new_matrix[i, j]))
                    for i, j in zip(*np.nonzero(np.triu(new_matrix != matrix, 1)))
                )
                action_list.append(SubstituteShiftMG(molecule, atom, element, changes))
        return action_list
