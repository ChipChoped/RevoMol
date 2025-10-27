from .AtomRebuilder import AtomRebuilder
import pandas as pd
from rdkit import Chem
import xarray as xr
import numpy as np


class DualAtomRebuilder(AtomRebuilder):
    _tree: np.ndarray
    # [valence, periodic_row, rings, single_bonds, double_bonds, triple_bonds]
    _hashes: dict
    _smallest_bond: tuple[int]
    # open and closed positions are tuple with 3 values: (cycle_size/branch, starting_atom, deepness)
    _open_positions: np.ndarray
    _closed_positions: np.ndarray
    _local_validation: xr.DataArray

    _atoms_used: set[int]

    _cycles_size: list[tuple]

    def __init__(self, atom_data_list):
        self._smallest_bond = (-1, -1)
        super().__init__(atom_data_list)
        self._open_positions = np.empty(len(self._atom_data_list), dtype=object)
        self._closed_positions = np.empty(len(self._atom_data_list), dtype=object)
        self._init_hashes()
        self._init_local_validation()
        nb_total_bonds = (
            len(self.atom_descriptors.loc[:, "single_bond"])
            + len(self.atom_descriptors.loc[:, "double_bond"])
            + len(self.atom_descriptors.loc[:, "triple_bond"])
        )
        self._tree = np.empty(nb_total_bonds * 2, dtype=object)
        self._cycles_size = [
            ("small_cycle", (3, 4)),
            ("medium_cycle", (5, 7)),
            ("large_cycle", (8, len(self._atom_data_list))),
        ]

    # ================================
    # Initialization functions
    # ================================

    def _init_hashes(self):
        self._hashes = dict()
        featuress = set()
        nb_nodes = dict()

        self.atom_descriptors = pd.DataFrame(
            columns=[
                "hash",
                "valence",
                "single_bond",
                "double_bond",
                "triple_bond",
                "rings",
            ],
        )

        for i, atom in enumerate(self._atom_data_list):
            # valence", "periodic_row", "rings", "sb", "db", "tb"
            (
                _,
                _,
                triple_bonds,
                double_bonds,
                single_bonds,
                _,
                _,
                rings,
                _,
                valence,
                periodic_row,
            ) = atom

            feature = (
                valence,
                periodic_row,
                rings,
                tuple(single_bonds),
                tuple(double_bonds),
                tuple(triple_bonds),
            )
            featuress.add(feature)

            nb_bonds = len(single_bonds) + len(double_bonds) + len(triple_bonds)

            if self._smallest_bond[0] == -1 or nb_bonds < self._smallest_bond[1]:
                self._smallest_bond = [hash(feature), nb_bonds]

            self.atom_descriptors.loc[i] = {
                "hash": hash(feature),
                "valence": valence,
                "rings": list(rings),
                "single_bond": list(single_bonds),
                "double_bond": list(double_bonds),
                "triple_bond": list(triple_bonds),
            }

            if hash(feature) in nb_nodes:
                nb_nodes[hash(feature)] += 1
            else:
                nb_nodes[hash(feature)] = 1

        for features in featuress:
            self._hashes[hash(features)] = [
                features,
                nb_nodes[hash(features)],
            ]

    def _init_local_validation(self):
        """
        Initializes the local validation data array for the object.

        This method creates a 3-dimensional xarray.DataArray named `local_validation`
        with dimensions "hash_1", "hash_2", and "bond". It is used to store the number
        of bonds between pairs of hashes for different bond types. The array is filled
        by iterating over all combinations of hashes and bond types, and the values
        are computed using the `_get_nb_bonds` method.

        The resulting DataArray is sorted by the "hash_1", "hash_2", and "bond" dimensions.

        Attributes:
            _local_validation (xarray.DataArray): A 3D array containing the number of bonds
                between pairs of hashes for bond types 1, 2, and 3.

        Raises:
            KeyError: If any of the keys in `_hashes` are not found during the computation.
        """
        self._local_validation = xr.DataArray(
            np.zeros((len(self._hashes), len(self._hashes), 3), dtype=int),
            dims=["hash_1", "hash_2", "bond"],
            coords={
                "hash_1": list(self._hashes.keys()),
                "hash_2": list(self._hashes.keys()),
                "bond": [1, 2, 3],
            },
            name="local_validation",
        )
        # Fill the local validation array
        for hash_1 in self._hashes.keys():
            for hash_2 in self._hashes.keys():
                for bond in range(1, 4):
                    nb_bonds_1_2 = self._get_nb_bonds(hash_1, hash_2, bond)
                    nb_bonds_2_1 = self._get_nb_bonds(hash_2, hash_1, bond)

                    self._local_validation.loc[hash_1, hash_2, bond] = nb_bonds_1_2
                    self._local_validation.loc[hash_2, hash_1, bond] = nb_bonds_2_1

        self._local_validation = self._local_validation.sortby("hash_1")
        self._local_validation = self._local_validation.sortby("hash_2")
        self._local_validation = self._local_validation.sortby("bond")

    # ================================
    # Bond functions
    # ================================

    def _get_nb_bonds(self, atom_1: int, atom_2: int, bond: int) -> int:
        """
        Calculate the number of bonds of a specific type between two atoms.

        Args:
            atom_1 (int): The index of the first atom in the hash table.
            atom_2 (int): The index of the second atom in the hash table.
            bond (int): The bond type to check.

        Returns:
            int: The number of bonds of the specified type between the two atoms.
        """
        features1 = self._hashes[atom_1][0]
        features2 = self._hashes[atom_2][0]
        return features1[bond + 2].count(features2[0])

    def _last_element_list(self, np_list: np.ndarray) -> int:
        """
        Finds the index of the last non-None element in a NumPy array.

        Args:
            np_list (np.ndarray): A NumPy array to search for the last non-None element.

        Returns:
            int: The index of the last non-None element in the array.
                 Returns -1 if no such element is found or if the array is empty.
        """
        try:
            return (np.where(np_list != None)[-1])[-1]
        except IndexError:
            return -1

    def move_none_to_end(self, input_array: np.ndarray) -> np.ndarray:
        """
        Rearranges a NumPy so that all non-None elements
        are moved to the beginning of the array, followed by all None elements.

        Args:
            input_array: The input NumPy array (dtype=object) which may contain None.

        Returns:
            A new NumPy array with non-None elements first, followed by Nones.
            Returns an empty array if the input is empty.
            Returns the original array if it contains no None values.
        """
        if input_array.size == 0:
            return np.array([], dtype=object)
        is_none_mask = input_array == None
        is_not_none_mask = ~is_none_mask
        non_none_elements = input_array[is_not_none_mask]
        none_elements = input_array[is_none_mask]
        result_array = np.concatenate((non_none_elements, none_elements))
        return result_array.astype(object)

    def _bonds_available(self, prev_atom_idx: int, atom_idx: int) -> np.ndarray:
        """
        Determine the bonds available for a given atom in the molecular structure.

        This method calculates the possible bonds that can be formed or closed
        for a given atom based on its current state, its descriptors, and the
        molecular structure's open positions and cycles.

        Args:
            prev_atom_idx (int): The index of the previous atom in the structure.
            atom_idx (int): The index of the current atom for which bonds are being evaluated.

        Returns:
            np.ndarray: An array of possible bond actions. Each element in the array
            is a tuple describing a bond action, which can include:
                - Closing a branch or cycle.
                - Opening a new branch or cycle.
                - Forming a single, double, or triple bond with another atom.
            The array may also contain `None` or be empty if no valid actions are available.

        Notes:
            - The method considers compatibility of bonds based on atom descriptors
              (e.g., valence, bond types, and ring structures).
            - It handles special cases such as closing branches or cycles, and opening
              new branches or cycles.
            - The method uses internal data structures like `_local_validation`,
              `_open_positions`, and `_cycles_size` to determine bond availability.
            - The output array is dynamically sized based on the number of valid bond actions.
        """
        # Max bonds available: len(single_bonds) + len(double_bonds) + len(triple_bonds) + 12 (open branch, open cycle, close branch, close cycle)
        hash_checked = np.zeros(len(self._hashes), dtype=int)
        idx_hash_checked = 0
        hash_atom = self.atom_descriptors.loc[atom_idx, "hash"]
        nb_bonds = np.sum(np.where(self._local_validation.loc[hash_atom] > 0, 1, 0)) - 1
        nb_bonds_available = (
            len(self.atom_descriptors.loc[atom_idx, "single_bond"])
            + len(self.atom_descriptors.loc[atom_idx, "double_bond"])
            + len(self.atom_descriptors.loc[atom_idx, "triple_bond"])
        )
        type_bond = ["single_bond", "double_bond", "triple_bond"]
        last_open_pos = self._last_element_list(self._open_positions)
        last_idx_tree = self._last_element_list(self._tree)

        out = np.empty(nb_bonds + 6 + last_open_pos + 1, dtype=object)
        idx = 0

        biggest_cycle = False

        # 0: no cycle, 1: small cycle, 2: medium cycle, 3: large cycle
        cycle_size = 0

        # =========================
        # Close
        # =========================
        if last_open_pos != -1:
            if "small" in self._open_positions[last_open_pos][0]:
                cycle_size = 1
            elif "medium" in self._open_positions[last_open_pos][0]:
                cycle_size = 2
            elif "large" in self._open_positions[last_open_pos][0]:
                cycle_size = 3
            mask = np.array(
                [
                    isinstance(item, tuple)
                    and len(item) > 0
                    and item[0] == "open_branch"
                    for item in self._open_positions
                ]
            )
            last_branch = np.where(mask)[-1]
            if len(last_branch) > 0:
                last_branch = last_branch[0]
            else:
                last_branch = -1

            # ============================
            # Close branch
            # ============================
            if self._open_positions[last_open_pos][0] == "open_branch":
                if nb_bonds_available == 0:
                    return np.array(
                        [(-1, -1, "close_branch"), None],
                        dtype=object,
                    )

            if nb_bonds_available == 0:
                return []
            # ============================
            # Close cycle
            # ============================
            # For each cycle in a branch
            for i in range(last_open_pos, last_branch, -1):
                if self._open_positions[i] is None:
                    continue
                next_cycle = False
                # Check for each size of cycle if the cycle is big enough to be closed
                for cycle_name, cycle_rule in self._cycles_size:
                    if cycle_name in self._open_positions[i][0]:
                        if self._open_positions[i][3] < cycle_rule[0]:
                            next_cycle = True
                            break
                if next_cycle:
                    continue
                # idx of the beginning of the cycle
                first_in_cycle = self._open_positions[i][1]
                hash_first = self.atom_descriptors.loc[first_in_cycle, "hash"]
                are_compatible = (
                    self._local_validation.loc[hash_first, hash_atom, 1] > 0
                )
                # If the first atom in the cycle is compatible with the atom, close the cycle
                if are_compatible:
                    bonds_atom = [
                        j
                        for j in range(1, 4)
                        if self.atom_descriptors.loc[first_in_cycle, "valence"]
                        in self.atom_descriptors.loc[atom_idx, type_bond[j - 1]]
                    ]
                    bonds_first = [
                        j
                        for j in range(1, 4)
                        if self.atom_descriptors.loc[atom_idx, "valence"]
                        in self.atom_descriptors.loc[first_in_cycle, type_bond[j - 1]]
                    ]
                    # Get every bonds available for the combination atom - first_in_cycle
                    bonds = np.intersect1d(bonds_atom, bonds_first)
                    if len(bonds) > 0:
                        for cycle_name, cycle_rule in self._cycles_size:
                            if (
                                cycle_name in self._open_positions[i][0]
                                and atom_idx != self._open_positions[i][1]
                            ):
                                for j in range(len(bonds)):
                                    # add it to the tree
                                    out[idx] = (
                                        prev_atom_idx,
                                        atom_idx,
                                        f"close_cycle_{i}_{bonds[j]}",
                                    )
                                    idx += 1
                                if self._open_positions[i][3] == cycle_rule[1]:
                                    biggest_cycle = True
                                if self._open_positions[i][3] > cycle_rule[1]:
                                    return []

        if nb_bonds_available == 0:
            # In case no bond is available
            if type(self._tree[last_idx_tree][2]) != str:
                return []
            elif prev_atom_idx != None:
                return self._bonds_available(None, prev_atom_idx)
        if prev_atom_idx != None:
            # =========================
            # Open cycle
            # =========================
            for i, cycle_name in enumerate(self._cycles_size):
                cycle_name = cycle_name[0]
                if (
                    self.atom_descriptors.loc[atom_idx, "rings"][i] > 0
                    and self.atom_descriptors.loc[prev_atom_idx, "rings"][i] > 0
                ):
                    out[idx] = (prev_atom_idx, atom_idx, f"open_{cycle_name}")
                    idx += 1

            # =========================
            # Open branch
            # =========================
            if (
                nb_bonds_available > 1
                and self._tree[last_idx_tree] is not None
                and "open" not in str(self._tree[last_idx_tree][2])
                and self.atom_descriptors.loc[atom_idx, "rings"][0] == 0
                and self.atom_descriptors.loc[atom_idx, "rings"][1] == 0
                and self.atom_descriptors.loc[atom_idx, "rings"][2] == 0
            ):
                out[idx] = (prev_atom_idx, atom_idx, "open_branch")
                idx += 1
                return out

            if biggest_cycle:
                return out

        # =========================
        # "Classic" bond
        # =========================

        every_bonds_available = self._local_validation.loc[hash_atom][
            np.where(self._local_validation.loc[hash_atom] > 0)
        ]

        for bonds in every_bonds_available:
            available_hash = int(bonds["hash_2"])
            if available_hash in hash_checked:
                continue
            hash_checked[idx_hash_checked] = available_hash
            idx_hash_checked += 1
            available_bond = set(bonds["bond"].values.tolist())
            if self._hashes[available_hash][1] == 0:
                continue
            for bond in available_bond:
                if (
                    self._hashes[available_hash][0][0]
                    not in self.atom_descriptors.loc[atom_idx, type_bond[bond - 1]]
                ):
                    continue
                if (
                    self.atom_descriptors.loc[atom_idx, "valence"]
                    not in self._hashes[available_hash][0][bond + 2]
                ):
                    continue
                if (
                    cycle_size != 0
                    and self._hashes[available_hash][0][2][cycle_size - 1] == 0
                ):
                    continue
                out[idx] = (
                    int(atom_idx),
                    int(available_hash),
                    int(bond),
                )
                idx += 1

        return out

    def _add_node(self, node: tuple) -> tuple[int]:
        """
        Adds a node to the tree structure based on the provided node tuple.

        This method handles the addition of nodes, including opening and closing
        branches or cycles, and updating the internal tree and position tracking
        structures accordingly.

        Args:
            node (tuple): A tuple containing the following elements:
                - idx_prev (int): The index of the previous node.
                - hash_atom (int): The hash or identifier of the current atom.
                - bond (str or int): The bond type or identifier. Can be a string
                  indicating the type of bond (e.g., "open_branch", "close_cycle"),
                  or an integer representing the bond order.

        Returns:
            tuple[int]: A tuple containing the indices of the previous and current
            nodes after the addition. If the addition fails, returns (None, None).

        Notes:
            - The method updates the `_tree`, `_open_positions`, and `_closed_positions`
              attributes to reflect the changes in the tree structure.
            - Handles special cases for opening and closing branches or cycles.
            - Updates atom descriptors for ring sizes when opening cycles.
            - Adds bonds between atoms and tracks used atoms.
        """
        idx_prev, hash_atom, bond = node
        idx_tree = self._last_element_list(self._tree) + 1
        if idx_tree == -1:
            idx_tree = 0
        out = (None, None)

        in_cycle = self._last_element_list(self._open_positions)
        if in_cycle != -1:
            in_cycle = self._open_positions[in_cycle][0]
        else:
            in_cycle = ""
        if type(bond) == str:
            if "close" in bond:
                last_close_pos = self._last_element_list(self._closed_positions) + 1
                position = None
                last_open_pos = None
                if "branch" in bond:
                    last_open_pos = self._last_element_list(self._open_positions)
                    position = self._open_positions[last_open_pos]
                    self._closed_positions[last_close_pos] = (
                        position[0],
                        position[1],
                        position[2],
                        position[3],
                        last_open_pos,
                    )
                    self._tree[idx_tree] = (None, None, "close_branch")
                    out = (position[1], position[2])

                elif "cycle" in bond:
                    last_open_pos = int(bond[-3])
                    position = self._open_positions[last_open_pos]

                    self._closed_positions[last_close_pos] = (
                        position[0],
                        position[1],
                        position[2],
                        position[3],
                        last_open_pos,
                    )
                    out = self._add_bond(
                        position[1],
                        hash_atom,
                        int(bond[-1]),
                        in_cycle,
                        idx_tree,
                        close_cycle=True,
                    )
                    idx_tree += 1
                    self._tree[idx_tree] = (
                        None,
                        None,
                        "close_cycle",
                    )
                self._open_positions[last_open_pos] = None
            if "open" in bond:
                last_open_pos = self._last_element_list(self._open_positions) + 1
                if "branch" in bond:
                    self._open_positions[last_open_pos] = (
                        "open_branch",
                        idx_prev,
                        hash_atom,
                        0,
                    )
                    self._tree[idx_tree] = (None, None, "open_branch")
                # Open cycle
                elif "cycle" in bond:
                    self._open_positions[last_open_pos] = (
                        bond,
                        idx_prev,
                        hash_atom,
                        2,
                    )
                    if "small" in bond:
                        self.atom_descriptors.loc[idx_prev, "rings"][0] -= 1
                        self.atom_descriptors.loc[hash_atom, "rings"][0] -= 1
                    elif "medium" in bond:
                        self.atom_descriptors.loc[idx_prev, "rings"][1] -= 1
                        self.atom_descriptors.loc[hash_atom, "rings"][1] -= 1
                    elif "large" in bond:
                        self.atom_descriptors.loc[idx_prev, "rings"][2] -= 1
                        self.atom_descriptors.loc[hash_atom, "rings"][2] -= 1
                    self._tree[idx_tree] = (idx_prev, hash_atom, bond)
                out = (idx_prev, hash_atom)
        else:
            idx_atom = self._convert_to_atom(idx_prev, hash_atom)

            if idx_atom == -1:
                return out
            self._atoms_used.add(idx_atom)
            out = self._add_bond(idx_prev, idx_atom, bond, in_cycle, idx_tree)

        return out

    def _add_bond(
        self,
        idx_prev: int,
        idx_atom: int,
        bond: int,
        in_cycle: str,
        idx_tree: int,
        close_cycle: bool = False,
    ) -> tuple[int]:
        """
        Adds a bond between two atoms in the tree structure and updates the atom descriptors.

        Args:
            idx_prev (int): The index of the previous atom in the bond.
            idx_atom (int): The index of the current atom in the bond.
            bond (int): The type of bond to add (1 for single, 2 for double, 3 for triple).
            in_cycle (str): The cycle type the atom is part of ("small", "medium", or "large").
            idx_tree (int): The index of the tree where the bond is being added.
            close_cycle (bool, optional): Whether the bond closes a cycle. Defaults to False.

        Returns:
            tuple[int]: A tuple containing the indices of the two atoms in the bond.
        """
        type_bond = ["single_bond", "double_bond", "triple_bond"]
        if not close_cycle:
            if "small" in in_cycle:
                self.atom_descriptors.loc[idx_atom, "rings"][0] -= 1
            elif "medium" in in_cycle:
                self.atom_descriptors.loc[idx_atom, "rings"][1] -= 1
            elif "large" in in_cycle:
                self.atom_descriptors.loc[idx_atom, "rings"][2] -= 1
            # Inc cycle
            last_open_pos = self._last_element_list(self._open_positions)
            if (
                last_open_pos != -1
                and "cycle" in self._open_positions[last_open_pos][0]
            ):
                self._open_positions[last_open_pos] = (
                    self._open_positions[last_open_pos][0],
                    self._open_positions[last_open_pos][1],
                    self._open_positions[last_open_pos][2],
                    self._open_positions[last_open_pos][3] + 1,
                )
        out = (idx_prev, idx_atom)

        self._tree[idx_tree] = (idx_prev, idx_atom, bond)
        valence_prev = self.atom_descriptors.loc[idx_prev, "valence"]
        valence_atom = self.atom_descriptors.loc[idx_atom, "valence"]
        self.atom_descriptors.loc[idx_atom, type_bond[bond - 1]].remove(valence_prev)
        self.atom_descriptors.loc[idx_prev, type_bond[bond - 1]].remove(valence_atom)

        return out

    def _backtrack_node(self, close_cycle: bool = False) -> None:
        """
        Backtracks a node in the tree structure.

        This method removes a node from the tree and updates the relevant data structures
        to reflect the removal. It also handles the case where a branch or cycle is closed.

        Args:
            node (tuple): The node to be backtracked. It can be either a tuple representing
                          a bond or a string representing a branch or cycle.

        Returns:
            int: The index of the atom that was backtracked.
        """
        last_node_idx = self._last_element_list(self._tree)
        idx_prev, idx_atom, bond = self._tree[last_node_idx]
        self._tree[last_node_idx] = None
        in_cycle = self._last_element_list(self._open_positions)
        if in_cycle != -1:
            in_cycle = self._open_positions[in_cycle][0]
        else:
            in_cycle = ""
        if type(bond) == str:
            if "close" in bond:
                last_close_pos_idx = self._last_element_list(self._closed_positions)
                last_close_pos = self._closed_positions[last_close_pos_idx]
                self._open_positions[last_close_pos[-1]] = (
                    last_close_pos[0],
                    last_close_pos[1],
                    last_close_pos[2],
                    last_close_pos[3],
                )
                self._closed_positions[last_close_pos_idx] = None
                # Close cycle
                if "cycle" in bond:
                    self._backtrack_node(True)
            elif "open" in bond:
                last_open_pos = self._last_element_list(self._open_positions)
                self._open_positions[last_open_pos] = None
                if "cycle" in bond:
                    if "small" in bond:
                        self.atom_descriptors.loc[idx_prev, "rings"][0] += 1
                        self.atom_descriptors.loc[idx_atom, "rings"][0] += 1
                    elif "medium" in bond:
                        self.atom_descriptors.loc[idx_prev, "rings"][1] += 1
                        self.atom_descriptors.loc[idx_atom, "rings"][1] += 1
                    elif "large" in bond:
                        self.atom_descriptors.loc[idx_prev, "rings"][2] += 1
                        self.atom_descriptors.loc[idx_atom, "rings"][2] += 1
        else:
            if not close_cycle:
                self._atoms_used.remove(idx_atom)
                if "small" in in_cycle:
                    self.atom_descriptors.loc[idx_atom, "rings"][0] += 1
                elif "medium" in in_cycle:
                    self.atom_descriptors.loc[idx_atom, "rings"][1] += 1
                elif "large" in in_cycle:
                    self.atom_descriptors.loc[idx_atom, "rings"][2] += 1

                # Decrease cycle
                last_open_pos = self._last_element_list(self._open_positions)
                if (
                    last_open_pos != -1
                    and "cycle" in self._open_positions[last_open_pos][0]
                ):
                    self._open_positions[last_open_pos] = (
                        self._open_positions[last_open_pos][0],
                        self._open_positions[last_open_pos][1],
                        self._open_positions[last_open_pos][2],
                        self._open_positions[last_open_pos][3] - 1,
                    )

            type_bond = ["single_bond", "double_bond", "triple_bond"]
            valence_prev = self.atom_descriptors.loc[idx_prev, "valence"]
            valence_atom = self.atom_descriptors.loc[idx_atom, "valence"]

            self.atom_descriptors.loc[idx_atom, type_bond[bond - 1]].append(
                valence_prev
            )
            self.atom_descriptors.loc[idx_prev, type_bond[bond - 1]].append(
                valence_atom
            )

    # ================================
    # Tree functions
    # ================================

    def _global_check(self) -> bool:
        """
        Check if the current tree structure is valid.

        This method verifies if the current tree structure satisfies the global
        constraints of the molecular structure. It checks if all open branches
        and cycles are properly closed and if the number of bonds is consistent.

        Returns:
            bool: True if the tree structure is valid, False otherwise.
        """
        if not self._is_connected():
            return False
        for atom in self.atom_descriptors.index:
            if len(self.atom_descriptors.loc[atom, "single_bond"]) != 0:
                return False
            if len(self.atom_descriptors.loc[atom, "double_bond"]) != 0:
                return False
            if len(self.atom_descriptors.loc[atom, "triple_bond"]) != 0:
                return False
            if any(cycle != 0 for cycle in self.atom_descriptors.loc[atom, "rings"]):
                return False
        if not self._check_cycles():
            return False
        return True

    def _is_connected(self) -> bool:
        """
        Check if the graph is connected.

        Returns:
            bool: True if the tree is connected, False otherwise.
        """
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for edge in self._tree:
                    if edge is None:
                        break
                    if node in edge:
                        stack.append(edge[0] if edge[1] == node else edge[1])
        return len(visited) == len(self.atom_descriptors)

    def tree_search(self, node: tuple) -> None:
        """
        Perform a depth-first search on a tree structure to find valid solutions.

        This method recursively explores nodes in a tree, attempting to add them
        and validate the resulting structure. If a valid solution is found, it
        generates a SMILES string representation and stores it. The search stops
        when the desired number of solutions is found or all possibilities are
        exhausted.

        Args:
            node (tuple): A tuple representing the current node to explore.
                          It contains three elements: hash_1, hash_2, and bond.
            deep (int, optional): The current depth of the recursive search.
                                  Defaults to 0.

        Returns:
            None
        """
        if not self._all_solutions and self._results_found >= self._n_solutions:
            return
        node = tuple(node)
        # print("Node :", node)
        hash_1, hash_2, bond = node
        hash_1 = int(hash_1)
        hash_2 = int(hash_2)
        bond = str(bond)
        if bond.isnumeric():
            bond = int(bond)
        # Add the node (if possible)
        idx_1, idx_2 = self._add_node(node)
        if idx_1 == None or idx_2 == None:
            return
        # print("Tree: ", self._tree[self._tree != None])
        # print("Atoms used: ", self._atoms_used)
        # print("Open positions: ", self._open_positions[self._open_positions != None])
        # print(
        #     "Closed positions: ", self._closed_positions[self._closed_positions != None]
        # )
        # Check if the node works globally
        if self._global_check():
            # print("Tree: ", self._tree[self._tree != None])
            smiles = self.smiles_from_tree()
            self._results_found += 1
            self._solutions.add(smiles)

        # Try every available bonds
        bonds_available = self._bonds_available(idx_1, idx_2)
        for bond_available in bonds_available:
            if bond_available is None:
                break
            # Recursive call
            self.tree_search(bond_available)

        # Backtrack the node
        self._backtrack_node()

    # ================================
    # Solution functions
    # ================================

    def compute_solutions(self, all_solutions=False, n_solutions=1):
        nb_total_bonds = (
            len(self.atom_descriptors.loc[:, "single_bond"])
            + len(self.atom_descriptors.loc[:, "double_bond"])
            + len(self.atom_descriptors.loc[:, "triple_bond"])
        )
        self._tree = np.empty(nb_total_bonds * 2, dtype=object)
        self._atoms_used = set()
        self._all_solutions = all_solutions
        self._n_solutions = n_solutions
        self._open_positions = np.empty(nb_total_bonds * 2, dtype=object)
        self._closed_positions = np.empty(nb_total_bonds * 2, dtype=object)
        self._init_local_validation()
        atom = self._smallest_bond[0]
        idx_atom = self._convert_to_atom(-1, atom)
        self._atoms_used.add(idx_atom)
        for bonds_available in self._bonds_available(None, idx_atom):
            if bonds_available is None:
                break
            self.tree_search(bonds_available)
        return self._solutions

    def get_mol(self):
        return self._solutions

    def smiles_from_tree(self) -> str:
        """
        Generate a SMILES string from the tree.

        Returns:
            str: The SMILES string of the molecule.
        """
        mol = Chem.RWMol()
        tree_without_none = self._tree[np.where(self._tree != None)[0]]
        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(self._atoms_symbols)):
            a = Chem.Atom(self._atoms_symbols[i])
            # a.SetIsAromatic(self._atoms["is_aromatic"][i])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for bond in tree_without_none:
            atom1, atom2, bond_type = bond
            if type(bond_type) == str:
                continue
            chem_bond_type = None
            if bond_type == 1:
                chem_bond_type = Chem.rdchem.BondType.SINGLE
            elif bond_type == 2:
                chem_bond_type = Chem.rdchem.BondType.DOUBLE
            elif bond_type == 3:
                chem_bond_type = Chem.rdchem.BondType.TRIPLE
            mol.AddBond(node_to_idx[atom1], node_to_idx[atom2], chem_bond_type)
        # Convert RWMol to Mol object
        mol = mol.GetMol()
        return Chem.MolToSmiles(mol, canonical=True)

    # ================================
    # Tree idx functions
    # ================================

    def _check_cycles(self):
        """
        Checks for the presence and count of cycles (rings) in the atom descriptors.

        This method iterates through atoms that have rings and verifies if the number
        of cycles of different lengths (3-4, 5-7, and above 8) matches the expected
        counts specified in the atom descriptors.

        Returns:
            bool: True if the cycles match the expected counts, False otherwise.
        """
        atoms_with_rings = self.atom_descriptors[
            self.atom_descriptors["rings"].apply(lambda x: any(val > 0 for val in x))
        ]
        for atom in atoms_with_rings.index:
            rings = self.atom_descriptors.loc[atom, "rings"]
            cycles = self._all_cycles_from_node(atom)
            ring_between_3_4 = rings[0]
            ring_between_5_7 = rings[1]
            ring_above_8 = rings[2]
            count_3_4 = 0
            count_5_7 = 0
            count_8 = 0
            for cycle in cycles:
                if len(cycle) == 3 or len(cycle) == 4:
                    count_3_4 += 1
                if 5 <= len(cycle) and len(cycle) <= 7:
                    count_5_7 += 1
                if len(cycle) > 7:
                    count_8 += 1
            if ring_between_3_4 != count_3_4:
                return False
            if ring_between_5_7 != count_5_7:
                return False
            if ring_above_8 != count_8:
                return False
        return True

    def _all_cycles_from_node(self, start_node):
        """
        Finds all cycles containing start_node in an undirected graph.

        Args:
            start_node: The node to start searching for cycles from.

        Returns:
            A set of lists, where each inner list represents a cycle containing start_node.
            Returns an empty set if there are no cycles containing start_node.
            Each cycle includes the start node.
        """

        adjacencies = {}
        for u, v, _ in self._tree[self._tree != None]:
            # if sum(self.atom_descriptors.loc[u, "rings"]) or not sum(
            #     self.atom_descriptors.loc[v, "rings"]
            # ):
            adjacencies.setdefault(u, []).append(v)
            adjacencies.setdefault(v, []).append(u)

        if start_node not in adjacencies:
            return set()

        cycles_list = list()
        visited = set()

        def dfs(node, path):
            visited.add(node)
            path.append(node)
            for neighbor in adjacencies[node]:
                if neighbor == start_node and len(path) > 2:
                    if set(path[:]) not in cycles_list:
                        cycles_list.append(set(path[:]))
                elif neighbor not in visited:
                    dfs(neighbor, path)

            path.pop()
            visited.remove(node)

        dfs(start_node, [])
        if len(cycles_list) < 2:
            return cycles_list

        cycles = set()
        edits = True
        while edits:
            edits = False
            for i in range(len(cycles_list) - 1):
                for j in range(i + 1, len(cycles_list)):
                    intersection = cycles_list[i] & cycles_list[j]
                    if intersection == set() or tuple(intersection) in cycles:
                        continue
                    if intersection == cycles_list[i]:
                        cycles.add(tuple(cycles_list[i]))
                        edits = True
                    elif intersection == cycles_list[j]:
                        cycles.add(tuple(cycles_list[j]))
                        edits = True

        return cycles

    def find_indices_by_hash(self, hash_value: int) -> list:
        """
        Finds the indices in a Pandas DataFrame where the 'hash' column
        is equal to the provided value.

        Parameters
        ----------
        df : pd.DataFrame
            The Pandas DataFrame to examine.  It is assumed that 'df' has a
            column named 'hash'.
        hash_value : str
            The hash value to search for.

        Returns
        -------
        list
            A list of the indices where 'hash' == hash_value.
            Returns an empty list if no matches are found.

        Raises
        ------
        TypeError
            If df is not a pd.DataFrame or if hash_value is not a str.
        KeyError
            If df does not have a column named 'hash'.
        """
        if not isinstance(self.atom_descriptors, pd.DataFrame):
            raise TypeError("df must be a pd.DataFrame")
        if not isinstance(hash_value, int):
            raise TypeError("hash_value must be an int")

        if "hash" not in self.atom_descriptors:
            raise KeyError("The DataFrame does not have a column named 'hash'")

        # Use boolean masking to find the indices
        indices = self.atom_descriptors.index[
            self.atom_descriptors["hash"] == hash_value
        ].tolist()
        return indices

    def _convert_to_atom(self, idx_prev, carac_hash_value) -> int:
        """
        Convert a hash value to an atom index.

        Args:
            carac_hash_value (int): The hash value representing the atom.

        Returns:
            int: The index of the atom corresponding to the hash value.
        """
        available_indices = self.find_indices_by_hash(carac_hash_value)
        for i in available_indices:
            if i not in self._atoms_used and i != idx_prev:
                return i
        return -1
