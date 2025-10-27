from .AtomRebuilder import AtomRebuilder
import pandas as pd
from .MoleculeFeatures import MoleculeFeatures
from rdkit import Chem
import numpy as np
import xarray as xr
import numpy as np


class HashAtomRebuilder(AtomRebuilder):
    atom_descriptors: pd.DataFrame
    _tree_carac: list
    _n_solutions: int
    _all_solutions: bool
    _carac_hash: dict
    _carac_comb_available: xr.DataArray
    # 1 : branch, 2 : small cycle, 3 : medium cycle, 4 : large cycle
    _nb_bonds: xr.DataArray
    _actual_position: list[int]
    _nb_caracs: dict
    _nb_cycles: int
    _nb_branches: int
    _atoms_used: list
    _removed_positions: list

    @property
    def hash_keys(self):
        return list(self._carac_hash.keys())

    def __init__(self, atom_data_list: list[tuple]):
        """
        Initializes the Rebuilder class with a list of atom data tuples.

        Args:
            atom_data_list (list[tuple]): A list of tuples where each tuple contains
                                          atom data. The last two elements of each
                                          tuple should be the valence and periodic
                                          row of the atom, respectively.

        Attributes:
            _atom_data_list (list[tuple]): The sorted list of atom data tuples.
            _atoms_symbols (list): A list of atom symbols derived from the atom data.
            _atoms (dict): A dictionary to store atoms.
        """
        super().__init__(atom_data_list)
        self._carac_hash = dict()
        self._solutions = set()
        self._init_atom_descriptor()
        self._init_equal()
        self._init_combinations()
        self._init_bonds()

    # ================================
    # INITIALIZATION METHODS
    # ================================

    def _init_atom_descriptor(self):
        """
        Initializes the `atom_descriptors` attribute as a pandas DataFrame with
        specific columns to store atomic properties. The method iterates over
        `_atom_data_list` to populate the DataFrame with details about each atom,
        including bond types, counts, valence, periodic row, and ring information.

        Attributes:
            atom_descriptors (pd.DataFrame): A DataFrame with the following columns:
                - "valence": The valence of the atom.
                - "periodic_row": The periodic table row of the atom.
                - "rings": Information about the rings the atom is part of.
                - "sb": List of single bonds associated with the atom.
                - "db": List of double bonds associated with the atom.
                - "tb": List of triple bonds associated with the atom.
                - "nb_sb": Number of single bonds.
                - "nb_db": Number of double bonds.
                - "nb_tb": Number of triple bonds.
                - "hash": (Unused in the current implementation).

        Notes:
            - The `_atom_data_list` is expected to be a list of tuples containing
              atomic data, where specific indices correspond to the required
              properties.
            - The method assumes that the structure of `_atom_data_list` matches
              the unpacking pattern used in the loop.
        """
        self.atom_descriptors = pd.DataFrame(
            columns=[
                "valence",
                "periodic_row",
                "rings",
                "sb",
                "db",
                "tb",
                "nb_sb",
                "nb_db",
                "nb_tb",
                "hash",
            ],
        )
        for i, atom in enumerate(self._atom_data_list):
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
            self.atom_descriptors.loc[i] = {
                "tb": list(triple_bonds),
                "nb_tb": len(triple_bonds),
                "db": list(double_bonds),
                "nb_db": len(double_bonds),
                "sb": list(single_bonds),
                "nb_sb": len(single_bonds),
                "rings": rings,
                "valence": valence,
                "periodic_row": periodic_row,
            }

    def _init_equal(self):
        """
        Initializes the equality characteristics for atoms and their descriptors.

        This method processes the atom descriptors to create unique hash values
        for each atom's characteristics and stores them in a dictionary. It also
        initializes a DataArray to track the availability of characteristic
        combinations for different bond types.

        Attributes:
            self._atom_data_list (list): A list containing atom data.
            self.atom_descriptors (pd.DataFrame): A DataFrame containing atom
                descriptors with columns such as "valence", "periodic_row",
                "rings", "sb", "db", and "tb".
            self._carac_hash (dict): A dictionary mapping hash values of atom
                characteristics to their respective tuples.
            self._carac_comb_available (xr.DataArray): A DataArray representing
                the availability of characteristic combinations for bond types.

        Steps:
            1. Iterates through the atom data list and extracts characteristics
               from the atom descriptors.
            2. Converts specific characteristics into tuples for hashing.
            3. Computes a hash for the characteristics tuple and stores it in
               the `_carac_hash` dictionary.
            4. Updates the "hash" column in the `atom_descriptors` DataFrame
               with the computed hash values.
            5. Initializes a DataArray `_carac_comb_available` to track the
               availability of characteristic combinations for bond types.

        Note:
            The bond types are represented as 1, 2, and 3 in the DataArray.
        """
        for i in range(len(self._atom_data_list)):
            caracteristics = list(
                self.atom_descriptors.loc[
                    i, ["valence", "periodic_row", "rings", "sb", "db", "tb"]
                ]
            )
            caracteristics[3] = tuple(caracteristics[3])
            caracteristics[4] = tuple(caracteristics[4])
            caracteristics[5] = tuple(caracteristics[5])
            caracteristics = tuple(caracteristics)
            self._carac_hash[hash(caracteristics)] = caracteristics
            self.atom_descriptors.loc[i, "hash"] = hash(caracteristics)
        self._carac_comb_available = xr.DataArray(
            np.zeros((len(self._carac_hash), len(self._carac_hash), 3), dtype=bool),
            dims=["carac1", "carac2", "bond_type"],
            coords={
                "carac1": list(self._carac_hash.keys()),
                "carac2": list(self._carac_hash.keys()),
                "bond_type": [1, 2, 3],
            },
        )

    def _init_combinations(self):
        """
        Computes the validity of all possible combinations of characteristics (carac1 and carac2)
        and bond types (1, 2, or 3). The validity is determined by the `_verify_carac_comb` method.

        For each pair of characteristics and bond type:
        - Calls `_verify_carac_comb` to check if the combination is valid.
        - Updates the `_carac_comb_available` DataFrame with the validity result for both
          (carac1, carac2) and (carac2, carac1) to ensure symmetry.

        This method assumes:
        - `self._carac_hash` is an iterable containing the characteristics.
        - `self._carac_comb_available` is a DataFrame with a multi-index structure
          (carac1, carac2, bond_type) to store the validity of combinations.

        Returns:
            None
        """
        for carac1 in self._carac_hash:
            for carac2 in self._carac_hash:
                for bond_type in range(1, 4):
                    validity = self._verify_carac_comb(carac1, carac2, bond_type)
                    self._carac_comb_available.loc[carac1, carac2, bond_type] = validity
                    self._carac_comb_available.loc[carac2, carac1, bond_type] = validity

    def _init_bonds(self):
        """
        Initializes the `_nb_bonds` attribute as a DataArray to track the
        availability of bonds between different characteristics.

        Attributes:
            _nb_bonds (xr.DataArray): A DataArray with dimensions (carac1, carac2, bond_type)
                initialized to zero. It will be used to store the number of available bonds
                between different characteristics for each bond type.

        Notes:
            - The dimensions of the DataArray are defined by the number of unique
              characteristics and the bond types (1, 2, 3).
            - The coordinates of the DataArray are set to match the keys of the
              `_carac_hash` dictionary.
        """
        self._nb_caracs = dict()
        self._nb_bonds = xr.DataArray(
            np.zeros((len(self._carac_hash), 3), dtype=int),
            dims=["carac1", "bond_type"],
            coords={
                "carac1": list(self._carac_hash.keys()),
                "bond_type": [1, 2, 3],
            },
        )
        for carac in self._carac_hash:
            nb_caracs = self._nb_caracteristics(carac)
            nb_bond_carac = self._carac_hash[carac]
            self._nb_caracs[carac] = nb_caracs
            self._nb_bonds.loc[carac, 1] = len(nb_bond_carac[3]) * nb_caracs
            self._nb_bonds.loc[carac, 2] = len(nb_bond_carac[4]) * nb_caracs
            self._nb_bonds.loc[carac, 3] = len(nb_bond_carac[5]) * nb_caracs

    # ================================
    # VERIFICATION METHODS
    # ================================

    def _verify_carac_comb(self, carac1: int, carac2: int, bond_type: int):
        """
        Verifies if two chemical characteristics (carac1 and carac2) can form a bond of the specified type.

        Args:
            carac1 (int): The hash key representing the first chemical characteristic.
            carac2 (int): The hash key representing the second chemical characteristic.
            bond_type (int): The type of bond to verify (e.g., single, double, or triple bond).

        Returns:
            bool: True if the two characteristics can form the specified bond type, False otherwise.

        Notes:
            - The `carac1` and `carac2` are looked up in the `_carac_hash` dictionary to retrieve their
              respective properties.
            - Each characteristic contains information such as valence, periodic row, and bond capabilities.
            - The bond type is used as an index to check compatibility between the two characteristics.
        """
        # carac has these informations : [valence, periodic_row, rings, single_bonds, double_bonds, triple_bonds]
        carac1 = self._carac_hash[carac1]
        carac2 = self._carac_hash[carac2]
        if carac1[0] in carac2[bond_type + 2] and carac2[0] in carac1[bond_type + 2]:
            return True
        return False

    def _local_evaluation(self, bond: tuple) -> bool:
        """
        Evaluate a bond by comparing it to the target molecule.

        Args:
            bond (tuple): A tuple containing the bond information.

        Returns:
            bool: True if the bond is correct, False otherwise.
        """
        hash1, hash2, bond_type = bond
        return self._carac_comb_available.loc[hash1, hash2, bond_type]

    def _verify_bond(self, bond):
        """
        Verifies the validity of a bond between two atoms based on their descriptors.

        Args:
            bond (tuple): A tuple containing three elements:
                - atom1 (int): The index of the first atom.
                - atom2 (int): The index of the second atom.
                - bond_type (int): The type of bond (1 for single bond, 2 for double bond, 3 for triple bond).

        Returns:
            bool: True if the bond is valid based on the atom descriptors, False otherwise.

        The method checks the following conditions:
            1. Both atoms must have available bonds of the specified type.
            2. The valence of each atom must be compatible with the bond type of the other atom.
        """
        atom1, atom2, bond_type = bond
        bond_type_str = ""
        if bond_type == 1:
            bond_type_str = "sb"
        elif bond_type == 2:
            bond_type_str = "db"
        elif bond_type == 3:
            bond_type_str = "tb"
        if self.atom_descriptors.loc[atom1, f"nb_{bond_type_str}"] == 0:
            return False
        if self.atom_descriptors.loc[atom2, f"nb_{bond_type_str}"] == 0:
            return False
        if (
            self.atom_descriptors.loc[atom1, "valence"]
            not in self.atom_descriptors.loc[atom2, f"{bond_type_str}"]
        ):
            return False
        if (
            self.atom_descriptors.loc[atom2, "valence"]
            not in self.atom_descriptors.loc[atom1, f"{bond_type_str}"]
        ):
            return False
        return True

    def _global_evaluation(self, tree) -> bool:
        """
        Evaluate the current state of the tree.

        Returns:
            bool: True if the tree is correct, False otherwise.
        """
        if not self._is_connected(tree):
            return False
        return True

    def _is_connected(self, tree) -> bool:
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
                for edge in tree:
                    if node in edge:
                        stack.append(edge[0] if edge[1] == node else edge[1])
        return len(visited) == len(self.atom_descriptors)

    def _check_open_branch(self, carac_hash) -> bool:
        """
        Check if a branch can be opened.

        Args:
            bond (tuple): A tuple containing the bond information.

        Returns:
            bool: True if a branch can be opened, False otherwise.
        """
        carac = self._carac_hash[carac_hash]
        i = 0
        for node in self._tree_carac:
            if node[0] == carac_hash:
                i += 1
        nb_bond_available = len(carac[3]) + len(carac[4]) + len(carac[5]) - i
        if nb_bond_available > sum(carac[2]) + 2:
            return True
        return False

    def _check_open_cycle(self, carac_hash) -> bool:
        """
        Check if a cycle can be opened.

        Args:
            bond (tuple): A tuple containing the bond information.

        Returns:
            bool: True if a cycle can be opened, False otherwise.
        """
        carac = self._carac_hash[carac_hash]
        cycles_carac = carac[2]
        cycles = [0, 0, 0]
        for position in self._actual_position:
            if position[0] == 2:
                cycles[0] += 1
            elif position[0] == 3:
                cycles[1] += 1
            elif position[0] == 4:
                cycles[2] += 1
        if cycles_carac[0] == 1 and cycles[0] == 0:
            self._actual_position.append([2, carac_hash, 0])
            return True
        if cycles_carac[1] == 1 and cycles[1] == 0:
            self._actual_position.append([3, carac_hash, 0])
            return True
        if cycles_carac[2] == 1 and cycles[2] == 0:
            self._actual_position.append([4, carac_hash, 0])
            return True
        if cycles_carac[0] == 2 and cycles[0] < 2:
            self._actual_position.append([2, carac_hash, 0])
            return True
        if cycles_carac[1] == 2 and cycles[1] < 2:
            self._actual_position.append([3, carac_hash, 0])
            return True
        if cycles_carac[2] == 2 and cycles[2] < 2:
            self._actual_position.append([4, carac_hash, 0])
            return True
        return False

    # ================================
    # BOND MANIPULATION METHODS
    # ================================

    def _add_bond_carac(self, bond: tuple) -> bool:
        """
        Adds a bond between two atoms and updates their descriptors.

        Args:
            bond (tuple): A tuple containing two atom indices and the bond type.
                          The bond type is represented as an integer:
                          1 for single bond, 2 for double bond, and 3 for triple bond.
        """
        if bond == "start_branch":
            self._nb_branches += 1
        elif bond == "open_cycle":
            self._nb_cycles += 1
        elif bond == "close_branch":
            removed_bond = self._actual_position.pop()
            self._removed_positions.append(removed_bond)
            self._nb_branches -= 1
        elif bond == "close_cycle":
            removed_bond = self._actual_position.pop()
            self._removed_positions.append(removed_bond)
            self._nb_cycles -= 1
        if type(bond) == str:
            self._tree_carac.append(bond)
            return True
        if bond[0] == bond[1] and self._nb_bonds.loc[bond[0], bond[2]] == 1:
            return False
        self._nb_bonds.loc[bond[0], bond[2]] -= 1
        self._nb_bonds.loc[bond[1], bond[2]] -= 1
        self._tree_carac.append(bond)
        return True

    def _backtrack_node_carac(self) -> None:
        """
        Backtrack a bond.
        Args:
            bond (tuple): A tuple containing the bond information.
        """
        bond = self._tree_carac.pop()
        if bond == "start_branch":
            self._actual_position.pop()
            self._nb_branches -= 1
        elif bond == "open_cycle":
            self._actual_position.pop()
            self._nb_cycles -= 1
        elif bond == "close_branch":
            added_position = self._removed_positions.pop()
            self._actual_position.append(added_position)
            self._nb_branches += 1
            self._backtrack_node_carac()
        elif bond == "close_cycle":
            added_position = self._removed_positions.pop()
            self._actual_position.append(added_position)
            self._nb_cycles += 1
        if type(bond) == str:
            return
        self._nb_bonds.loc[bond[0], bond[2]] += 1
        self._nb_bonds.loc[bond[1], bond[2]] += 1
        self._nb_caracs[bond[0]] += 1

    def _close_position(self, carac_hash_value) -> int:
        """
        Close a position.

        Args:
            bond (tuple): A tuple containing the bond information.

        Returns:
            int: The hash of the caracteristics at the beginning of the position.
        """
        actual_pos = self._actual_position[-1]
        out = None
        if actual_pos[0] == 1:
            out = self._close_branch(carac_hash_value)
        elif actual_pos[0] == 2:
            out = self._close_cycle(carac_hash_value, actual_pos[2], 1)
        elif actual_pos[0] == 3:
            out = self._close_cycle(carac_hash_value, actual_pos[2], 2)
        elif actual_pos[0] == 4:
            out = self._close_cycle(carac_hash_value, actual_pos[2], 3)
        return out

    def _close_branch(self, carac_hash_value) -> int:
        """
        Close a branch.

        Args:
            bond (tuple): A tuple containing the bond information.

        Returns:
            int: The hash of the caracteristics at the beginning of the branch.
        """
        # caracs : [valence, periodic_row, rings, single_bonds, double_bonds, triple_bonds]
        caracs = self._carac_hash[carac_hash_value]
        nb_bond_available = len(caracs[3]) + len(caracs[4]) + len(caracs[5])
        if nb_bond_available == 1:
            hash_last_branch = self._actual_position[-1]
            self._add_bond_carac(("close_branch"))
            return hash_last_branch

    def _close_cycle(self, hash_atom_2, last_cycle_size, objective_size) -> int:
        """
        Close a cycle.

        Args:
            bond (tuple): A tuple containing the bond information.

        Returns:
            int: The hash of the caracteristics at the beginning of the cycle.
        """
        # caracs : [valence, periodic_row, rings, single_bonds, double_bonds, triple_bonds]
        hash_last_cycle = None
        if hash_atom_2 != self._actual_position[-1][1]:
            return None
        if objective_size == 1:
            if last_cycle_size == 3 or last_cycle_size == 4:
                hash_last_cycle = self._actual_position[-1]
                self._add_bond_carac(("close_cycle"))
        elif objective_size == 2:
            if last_cycle_size > 4 and last_cycle_size < 8:
                hash_last_cycle = self._actual_position[-1]
                self._add_bond_carac(("close_cycle"))
        elif objective_size == 3:
            if last_cycle_size > 7:
                hash_last_cycle = self._actual_position[-1]
                self._add_bond_carac(("close_cycle"))
        return hash_last_cycle

    # ================================
    # SEARCHING METHODS
    # ================================

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
            raise TypeError("hash_value must be a str")

        if "hash" not in self.atom_descriptors:
            raise KeyError("The DataFrame does not have a column named 'hash'")

        # Use boolean masking to find the indices
        indices = self.atom_descriptors.index[
            self.atom_descriptors["hash"] == hash_value
        ].tolist()
        return indices

    def _convert_to_atom(self, carac_hash_value) -> int:
        """
        Convert a hash value to an atom index.

        Args:
            carac_hash_value (int): The hash value representing the atom.

        Returns:
            int: The index of the atom corresponding to the hash value.
        """
        available_indices = self.find_indices_by_hash(carac_hash_value)
        for i in available_indices:
            if i not in self._atoms_used:
                return i
        return -1

    def _convert_to_atoms(self) -> None:
        """
        Converts the current tree characteristics into a tree of atoms and evaluates it.

        This method processes the tree characteristics (`_tree_carac`) to construct a tree
        of atoms based on the provided bonds and their types. It verifies the validity of
        each bond and evaluates the resulting tree. If the tree is valid, it generates a
        SMILES representation of the tree and adds it to the set of solutions.

        Returns:
            Optional[str]: The SMILES representation of the valid tree if the evaluation
            is successful, otherwise None.
        """
        tree = list()
        self._atoms_used = list()
        atoms_to_reuse = list()
        next_node = None
        previous_node = None
        atom_to_use = None
        cycle = False
        branch = False
        for i, bond in enumerate(self._tree_carac):
            if type(bond) == str:
                if bond == "start_branch":
                    atoms_to_reuse.append(next_node)
                elif bond == "open_cycle":
                    atoms_to_reuse.append(previous_node)
                elif bond == "close_branch":
                    branch = True
                    atom_to_use = atoms_to_reuse.pop()
                continue
            if (
                i + 1 < len(self._tree_carac)
                and self._tree_carac[i + 1] == "close_cycle"
            ):
                cycle = True
                atom_to_use = atoms_to_reuse.pop()
            carac1, carac2, bond_type = bond
            atom1 = self._convert_to_atom(carac1)
            self._atoms_used.append(atom1)
            if cycle:
                cycle = False
                atom2 = atom_to_use
                atom_to_use = None
            elif branch:
                branch = False
                atom1 = atom_to_use
                atom_to_use = None
                atom2 = self._convert_to_atom(carac2)
            else:
                atom2 = self._convert_to_atom(carac2)
            if atom1 == -1 or atom2 == -1:
                return None
            tree.append((atom1, atom2, bond_type))
            previous_node = atom1
            next_node = atom2

        # Verify the smiles
        if self._global_evaluation(tree):
            smiles = self.smiles_from_tree(tree)
            if not smiles:
                return None
            self._solutions.add(smiles)
            return smiles
        return None

    def get_edits_available(self, carac):
        """
        Retrieves available edits for a given characteristic.

        This method filters the data array for the specified characteristic and identifies
        the indices where the data is marked as True. It then maps these indices to hash keys
        and bond types, ensuring that the number of bonds for each hash key and bond type is
        greater than zero.

        Args:
            carac (str): The characteristic for which to retrieve available edits.

        Returns:
            dict: A dictionary where the keys are hash keys and the values are lists of bond
                  types that are available for editing.
        """
        # Filter the data array for the given characteristic
        filtered_data = self._carac_comb_available.loc[carac]

        # Use numpy to efficiently find indices where the data is True
        true_indices = np.argwhere(filtered_data.values)

        # Map the indices to hash keys and bond types
        result = dict()
        for i, j in true_indices:
            hash_key = self.hash_keys[i]
            bond_type = j + 1
            if self._nb_bonds.loc[hash_key, bond_type] <= 0:
                continue
            if hash_key not in result:
                result[hash_key] = []
            result[hash_key].append(bond_type)

        return result

    def _nb_caracteristics(self, carac):
        """
        Counts the number of occurrences of a specific characteristic in the atom descriptors.

        Args:
            carac (any): The characteristic to search for in the "hash" column of the atom descriptors.

        Returns:
            int: The count of occurrences of the specified characteristic.
        """
        return (
            self.atom_descriptors.where(self.atom_descriptors["hash"] == carac)
            .count()
            .iloc[0]
        )

    # ================================
    # TREE SEARCH METHODS
    # ================================

    def smiles_from_tree(self, tree) -> str:
        """
        Generate a SMILES string from the tree.

        Returns:
            str: The SMILES string of the molecule.
        """
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        try:
            node_to_idx = {}
            for i in range(len(self._atoms_symbols)):
                a = Chem.Atom(self._atoms_symbols[i])
                # a.SetIsAromatic(self._atoms["is_aromatic"][i])
                molIdx = mol.AddAtom(a)
                node_to_idx[i] = molIdx

            for bond in tree:
                atom1, atom2, bond_type = bond
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
            return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
        except Exception as e:
            return None

    def tree_search(self, node_hash) -> None:
        """
        Perform a depth-first tree search to explore possible molecular structures.

        Args:
            node: A tuple representing the current state: (atom_1, atom_2, bond).
                  atom_1: Index of the first atom.
                  atom_2: Index of the second atom.
                  bond: The bond type being considered (1, 2, 3 for single, double, triple).
        """
        # print("Tree: ", self._tree_carac)
        # print("\tTested node: ", node_hash)
        # print("\tActual positions: ", self._actual_position)
        if not self._all_solutions and len(self._solutions) == self._n_solutions:
            return
        # carac hash : [valence, periodic_row, single_bonds, double_bonds, triple_bonds]
        carac1, carac2, bond = node_hash

        if self._nb_caracs[carac2] <= 0 or self._nb_caracs[carac1] <= 0:
            return

        # First, check if the caracteristics are compatible
        if not self._local_evaluation(node_hash):
            return

        # If so, add it in the tree
        if not self._add_bond_carac(node_hash):
            return
        self._nb_caracs[carac1] -= 1
        # If it is not compatible, add a new bond accordidng to this order:
        # 1. Check if there is a branch, and verify if you can close it
        if len(self._actual_position) > 0:
            was_closed = True
            while was_closed and self._actual_position != []:
                was_closed = False
                hash_closed = self._close_position(carac2)
                if hash_closed:
                    was_closed = True
                    carac2 = hash_closed[1]
        if (
            len(self._actual_position) <= 0
            and self._nb_bonds.sum() == 0
            and self._convert_to_atoms()
        ):
            self._results_found += 1
            self._nb_caracs[carac1] += 1
            return

        # 2. Check if you can open a branch
        if self._check_open_branch(carac2):
            self._actual_position.append([1, carac2, 0])
            self._add_bond_carac(("start_branch"))
            self._nb_branches += 1
            self._nb_caracs[carac2] += 1

        # 4. Check if you can open a cycle
        if self._check_open_cycle(carac2):
            self._add_bond_carac(("open_cycle"))
            self._nb_cycles += 1
            self._nb_caracs[carac2] += 1

        for i in range(len(self._actual_position)):
            self._actual_position[i][2] += 1

        # 5. Get every combinations available for carac2
        available_nodes = self.get_edits_available(carac2)
        if self._nb_bonds.loc[carac2].sum() != 0:
            for bond in available_nodes.items():
                for bond_type in bond[1]:
                    node = (carac2, bond[0], bond_type)
                    self.tree_search(node)

        # 6. Backtrack the node
        self._nb_caracs[carac1] += 1
        self._backtrack_node_carac()

    def compute_solutions(self, all_solutions=False, n_solutions=1):
        self._all_solutions = all_solutions
        self._n_solutions = n_solutions

        for carac in self.hash_keys:
            self._actual_position = list()
            self._nb_cycles = 0
            self._nb_branches = 0
            self._tree_carac = list()
            self._removed_positions = list()
            self._init_bonds()

            result = self.get_edits_available(carac)
            for bond in result.items():
                for bond_type in bond[1]:
                    node = (carac, bond[0], bond_type)
                    self.tree_search(node)
        return self._solutions

    # ================================
    # OTHER METHODS
    # ================================

    def evaluate_solution(self, smiles: str) -> bool:
        """
        Evaluate a solution by comparing it to the target molecule.

        Args:
            smiles (str): The SMILES string of the solution molecule.

        Returns:
            bool: True if the solution is correct, False otherwise.
        """
        mol_features = MoleculeFeatures(smiles)
        mol_features.generate_features()
        features_smiles = mol_features.features_as_tuples
        for feature in self._atom_data_list:
            if feature not in features_smiles:
                return False
            features_smiles.remove(feature)
        if features_smiles:
            return False
        return True

    def get_mol(self):
        return self._solutions
