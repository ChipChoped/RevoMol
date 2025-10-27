import random
import copy
import xarray as xr
import numpy as np
from collections import deque
import rdkit.Chem as Chem


class TabuSearchManager:
    """
    Encapsulates the Tabu list logic to make it modular.
    Uses a deque for recency and a set for fast O(1) lookups.
    """

    is_tabu_enabled: bool

    def __init__(self, max_size: int):
        if max_size < 0:
            raise ValueError("The maximum size of the Tabu list cannot be negative.")
        # The deque efficiently manages the maximum length.
        if max_size == 0:
            self.is_tabu_enabled = False
        else:
            self.is_tabu_enabled = True
            self._moves = deque(maxlen=max_size if max_size > 0 else None)
            # The set allows for almost instantaneous presence checking.
            self._move_set = set()

    def add(self, move: tuple):
        """Adds an action (a 'move') to the Tabu list."""
        # If the deque is full and about to eject an old item,
        # we must also remove it from the set to maintain consistency.
        if not self.is_tabu_enabled:
            return
        if self._moves.maxlen is not None and len(self._moves) == self._moves.maxlen:
            oldest_move = self._moves[0]
            if oldest_move in self._move_set:
                self._move_set.remove(oldest_move)

        self._moves.append(move)
        self._move_set.add(move)

    def __contains__(self, move: tuple) -> bool:
        """Checks if an action is in the Tabu list."""
        if not self.is_tabu_enabled:
            return False
        return move in self._move_set

    def get_forbidden_actions(self, action_type: str) -> set:
        """Returns a set of forbidden actions for a given type, for efficient filtering."""
        if not self.is_tabu_enabled:
            return set()
        if action_type == "remove_bond":
            # Action format: ('remove_bond', a1, a2, None)
            return {
                (move[1], move[2]) for move in self._move_set if move[0] == action_type
            }
        elif action_type in ["add_bond", "edit_bond"]:
            # Action format: ('<action>', a1, a2, bond_type)
            return {
                (move[1], move[2], move[3])
                for move in self._move_set
                if move[0] == action_type
            }
        return set()


class GeneticMol:
    """
    Represents an individual molecule within the genetic algorithm.
    Manages its own structure, mutations, and conversion to SMILES.
    """

    score: float = float("inf")
    smiles: str = ""
    bond_type_conversion = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
    ]

    def __init__(
        self,
        atoms: list[tuple[str, int]],
        matrix_bonds: xr.DataArray,
        tabu_manager: TabuSearchManager,
    ):
        self._atoms = [
            (Chem.Atom(atom_symbol), valence) for atom_symbol, valence in atoms
        ]
        self._mol = Chem.RWMol()
        for atom, _ in self._atoms:
            self._mol.AddAtom(atom)

        self._bonds = {}
        self._available_edits = {}
        self._matrix_bonds = matrix_bonds.copy(deep=True)
        # Use the shared Tabu manager instead of creating its own list.
        self.tabu_manager = tabu_manager

    def __deepcopy__(self, memo):
        """
        Implementation of a custom deep copy.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Configuration attributes can be shared.
        result._atoms = self._atoms
        # The Tabu manager is shared by the entire population.
        result.tabu_manager = self.tabu_manager
        # State attributes must be deep copies to ensure independence.
        result.score = self.score
        result.smiles = self.smiles
        result._mol = copy.deepcopy(self._mol, memo)
        result._matrix_bonds = self._matrix_bonds.copy(deep=True)
        result._bonds = copy.deepcopy(self._bonds, memo)
        result._available_edits = copy.deepcopy(self._available_edits, memo)

        return result

    def are_additions_available(self) -> bool:
        """Checks if there are any bonds available for addition."""
        forbidden_adds = self.tabu_manager.get_forbidden_actions("add_bond")
        atoms_to_start = self.get_atoms_with_bonds()
        if atoms_to_start.size == 0:
            return False
        for atom1_idx in atoms_to_start:
            potential_partners = np.where(
                (self._matrix_bonds.values[atom1_idx] > 0).any(axis=1)
            )[0]
            # keep only potential partners that are in the atoms_to_start list
            potential_partners = [
                idx for idx in potential_partners if idx in atoms_to_start
            ]

            for atom2_idx in potential_partners:
                # Cannot bond an atom to itself.
                if atom1_idx == atom2_idx:
                    continue
                # In case the bond already exists, skip it.
                if (
                    min(atom1_idx, atom2_idx),
                    max(atom1_idx, atom2_idx),
                ) in self._bonds:
                    continue

                for n in range(3):
                    if self._matrix_bonds.values[atom1_idx, atom2_idx, n] > 0:
                        bond_type = self.bond_type_conversion[n]
                        action_tuple = (
                            min(atom1_idx, atom2_idx),
                            max(atom1_idx, atom2_idx),
                            bond_type,
                        )
                        if action_tuple not in forbidden_adds:
                            return True
        return False

    def are_edits_available(self) -> bool:
        for atom1_idx, atom2_idx in self._available_edits:
            # if the edit is in the tabu list, skip it
            if (atom1_idx, atom2_idx) in self.tabu_manager.get_forbidden_actions(
                "edit_bond"
            ):
                continue
            return True
        return False

    def are_removal_available(self) -> bool:
        """Checks if there are any bonds available for removal."""
        forbidden_removals = self.tabu_manager.get_forbidden_actions("remove_bond")
        for bond_key in self._bonds.keys():
            if bond_key not in forbidden_removals:
                return True
        return False

    def mutate(self, mutation_type: str):
        """Applies a mutation and records the action in the Tabu list."""
        action = None
        self.smiles = ""
        self.score = float("inf")
        if mutation_type == "add_bond":
            action = self.add_random_bond()
        elif mutation_type == "remove_bond":
            action = self.remove_random_bond()
        elif mutation_type == "edit_bond":
            action = self.edit_random_bond()
        elif mutation_type != "nothing":
            raise ValueError(f"Unknown mutation type: {mutation_type}")
        if action:
            self.tabu_manager.add(action)
        else:
            self.tabu_manager.add((mutation_type, None, None, None))

    def get_atoms_with_bonds(self, raw_value=False, nb_bonds=1) -> np.ndarray:
        """Finds the indices of atoms that can still form bonds."""
        # Vectorized approach to find atoms with potential bonds.
        has_potential_bonds = (self._matrix_bonds.values > 0).any(axis=(1, 2))
        atom_indices = np.where(has_potential_bonds)[0]

        # Additional filter for atoms with available valence.
        if atom_indices.size > 0:
            current_bonds_per_atom = np.zeros(len(self._atoms), dtype=int)
            for (a1, a2), bond in self._bonds.items():
                bond_val = int(bond)
                current_bonds_per_atom[a1] += bond_val
                current_bonds_per_atom[a2] += bond_val

            initial_valences = np.array([v for _, v in self._atoms])
            available_valence_mask = initial_valences - current_bonds_per_atom

            if raw_value:
                # Return the number of bonds available for each atom.
                return (
                    atom_indices[available_valence_mask[atom_indices] >= nb_bonds],
                    available_valence_mask,
                )
            return atom_indices[available_valence_mask[atom_indices] >= nb_bonds]

        return np.array([])

    def add_random_bond(self):
        """Adds a valid random bond."""
        atoms_to_start, max_bond = self.get_atoms_with_bonds(raw_value=True)
        if atoms_to_start.size == 0:
            return None
        random.shuffle(atoms_to_start)

        for atom1_idx in atoms_to_start:
            potential_partners = np.where(
                (self._matrix_bonds.values[atom1_idx] > 0).any(axis=1)
            )[0]
            # keep only potential partners that are in the atoms_to_start list
            potential_partners = [
                idx for idx in potential_partners if idx in atoms_to_start
            ]
            if len(potential_partners) == 0:
                continue

            forbidden_adds = self.tabu_manager.get_forbidden_actions("add_bond")

            available_bonds = []
            for atom2_idx in potential_partners:
                # Cannot bond an atom to itself.
                if atom1_idx == atom2_idx:
                    continue
                # In case the bond already exists, skip it.
                if (
                    min(atom1_idx, atom2_idx),
                    max(atom1_idx, atom2_idx),
                ) in self._bonds:
                    continue
                max_bond_a1_a2 = min(3, max_bond[atom1_idx], max_bond[atom2_idx])
                for n in range(max_bond_a1_a2):
                    if self._matrix_bonds.values[atom1_idx, atom2_idx, n] > 0:
                        bond_type = self.bond_type_conversion[n]
                        action_tuple = (
                            min(atom1_idx, atom2_idx),
                            max(atom1_idx, atom2_idx),
                            bond_type,
                        )
                        if action_tuple not in forbidden_adds:
                            available_bonds.append((atom2_idx, bond_type))

            if available_bonds:
                atom2_idx, bond_type = random.choice(available_bonds)
                return self.add_bond(atom1_idx, atom2_idx, bond_type)
        return None

    def remove_random_bond(self):
        """Removes a valid random bond."""
        if not self._bonds:
            return None

        forbidden_removals = self.tabu_manager.get_forbidden_actions("remove_bond")

        available_to_remove = [
            bond_key
            for bond_key in self._bonds.keys()
            if bond_key not in forbidden_removals
        ]

        if not available_to_remove:
            return None

        atom1_idx, atom2_idx = random.choice(available_to_remove)
        return self.remove_bond(atom1_idx, atom2_idx)

    def edit_random_bond(self):
        """Edits a valid random bond."""
        if not self._available_edits:
            return None

        forbidden_edits = self.tabu_manager.get_forbidden_actions("edit_bond")

        editable_bonds = []
        random.shuffle(editable_bonds)
        bonds, max_bonds = self.get_atoms_with_bonds(raw_value=True)
        every_editable_bond = [
            (atom1_idx, atom2_idx, self._bonds[(atom1_idx, atom2_idx)])
            for atom1_idx, atom2_idx in list(self._available_edits.keys())
        ]
        for atom1_idx, atom2_idx, bond_type in every_editable_bond:
            if atom1_idx not in bonds or atom2_idx not in bonds:
                continue
            editable_bonds.append(
                (
                    atom1_idx,
                    atom2_idx,
                    bond_type + max_bonds[atom1_idx],
                    bond_type + max_bonds[atom2_idx],
                ),
            )
        for atom1_idx, atom2_idx, max_bond_type_1, max_bond_type_2 in editable_bonds:
            possible_types = [
                self.bond_type_conversion[t - 1]
                for t in self._available_edits.get((atom1_idx, atom2_idx), set())
                if (atom1_idx, atom2_idx, self.bond_type_conversion[t - 1])
                not in forbidden_edits
                and t <= min(max_bond_type_1, max_bond_type_2)
            ]
            if possible_types:
                new_bond_type = random.choice(possible_types)
                return self.edit_bond(atom1_idx, atom2_idx, new_bond_type)
        return None

    def add_bond(self, atom1_idx: int, atom2_idx: int, bond_type):
        a1, a2 = min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx)
        bond_type_enum = (
            bond_type
            if isinstance(bond_type, Chem.rdchem.BondType)
            else self.bond_type_conversion[bond_type - 1]
        )
        bond_type_val = int(bond_type_enum)

        self._mol.AddBond(int(a1), int(a2), bond_type_enum)
        self._bonds[(a1, a2)] = bond_type_enum
        self._matrix_bonds.values[a1, a2, bond_type_val - 1] = 0
        self._matrix_bonds.values[a2, a1, bond_type_val - 1] = 0
        available_edits = set()
        for i in range(1, 4):
            if i != bond_type_val and self._matrix_bonds.values[a1, a2, i - 1] > 0:
                available_edits.add(i)
        if (a1, a2) not in self._available_edits and available_edits:
            self._available_edits[(a1, a2)] = available_edits

        return ("add_bond", a1, a2, bond_type_enum)

    def remove_bond(self, atom1_idx: int, atom2_idx: int):
        a1, a2 = min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx)
        bond_to_remove = self._bonds.pop((a1, a2))
        bond_val_removed = int(bond_to_remove)

        # Cast numpy types to standard Python int for RDKit compatibility.
        self._mol.RemoveBond(int(a1), int(a2))
        self._matrix_bonds.values[a1, a2, bond_val_removed - 1] = 1
        self._matrix_bonds.values[a2, a1, bond_val_removed - 1] = 1
        if (a1, a2) in self._available_edits:
            del self._available_edits[(a1, a2)]

        return ("remove_bond", a1, a2, None)

    def edit_bond(self, atom1_idx: int, atom2_idx: int, new_bond_type):
        self.remove_bond(atom1_idx, atom2_idx)
        return self.add_bond(atom1_idx, atom2_idx, new_bond_type)

    def convert_to_smiles(self) -> str:
        """Attempts to convert the RDKit structure to a canonical SMILES."""
        if self.smiles:
            return self.smiles
        try:
            mol = self._mol.GetMol()
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    def __str__(self):
        """Text representation for debugging and tracking diversity."""
        bonds_str = sorted(
            [f"({k[0]}-{k[1]}:{int(v)})" for k, v in self._bonds.items()]
        )
        return f"SMILES:{self.convert_to_smiles()}|Bonds:{''.join(bonds_str)}"
