from .AtomRebuilder import AtomRebuilder
import pandas as pd
from .MoleculeFeatures import MoleculeFeatures
from rdkit import Chem


class IndexAtomRebuilder(AtomRebuilder):
    atom_descriptors: pd.DataFrame
    _tree: dict
    _n_solutions: int
    _all_solutions: bool
    _equals: list
    _checked_bonds: list
    _results_found: int

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
        self._tree = dict()

        self._equals = []
        self._init_atom_descriptor()
        self._init_equal()

    def _init_atom_descriptor(self):
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

    def _init_equal(self):
        """
        Initializes the _equals attribute by finding and grouping indices of
        equal elements in the _atom_data_list attribute.

        This method iterates through the _atom_data_list and identifies groups
        of indices where the elements are equal. These groups are then appended
        to the _equals attribute. If an index is already part of an existing
        group in _equals, it is skipped.

        Attributes:
            _atom_data_list (list): A list containing the data elements to be
                                    compared for equality.
            _equals (list): A list of lists, where each sublist contains indices
                            of equal elements in _atom_data_list.
        """
        for i in range(len(self._atom_data_list)):
            if i in [x for y in self._equals for x in y]:
                continue
            equal = [i]
            for j in range(i + 1, len(self._atom_data_list)):
                if self._atom_data_list[i] == self._atom_data_list[j]:
                    equal.append(j)
            if len(equal) > 1:
                self._equals.append(equal)

    def _local_evaluation(self, bond: tuple) -> bool:
        """
        Evaluate a bond by comparing it to the target molecule.

        Args:
            bond (tuple): A tuple containing the bond information.

        Returns:
            bool: True if the bond is correct, False otherwise.
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

    def _add_bond(self, bond: tuple) -> None:
        """
        Adds a bond between two atoms and updates their descriptors.

        Args:
            bond (tuple): A tuple containing two atom indices and the bond type.
                          The bond type is represented as an integer:
                          1 for single bond, 2 for double bond, and 3 for triple bond.

        Returns:
            None
        """
        atom1, atom2, bond_type = bond
        if bond_type == 1:
            bond_type_str = "sb"
        elif bond_type == 2:
            bond_type_str = "db"
        elif bond_type == 3:
            bond_type_str = "tb"
        self.atom_descriptors.loc[atom1, f"nb_{bond_type_str}"] -= 1
        self.atom_descriptors.loc[atom2, f"nb_{bond_type_str}"] -= 1
        self.atom_descriptors.loc[atom1, f"{bond_type_str}"].remove(
            self.atom_descriptors.loc[atom2, "valence"]
        )
        self.atom_descriptors.loc[atom2, f"{bond_type_str}"].remove(
            self.atom_descriptors.loc[atom1, "valence"]
        )
        self._tree[(atom1, atom2)] = bond_type

    def _backtrack_node(self, bond: tuple) -> None:
        """
        Backtrack a bond.

        Args:
            bond (tuple): A tuple containing the bond information.
        """
        if not self._tree:
            return
        atom1, atom2 = bond
        bond_type = self._tree.pop((atom1, atom2))
        if bond_type == 1:
            str_bond_type = "sb"
        elif bond_type == 2:
            str_bond_type = "db"
        elif bond_type == 3:
            str_bond_type = "tb"
        self.atom_descriptors.loc[atom1, f"nb_{str_bond_type}"] += 1
        self.atom_descriptors.loc[atom2, f"nb_{str_bond_type}"] += 1
        self.atom_descriptors.loc[atom1, f"{str_bond_type}"].append(
            self.atom_descriptors.loc[atom2, "valence"]
        )
        self.atom_descriptors.loc[atom2, f"{str_bond_type}"].append(
            self.atom_descriptors.loc[atom1, "valence"]
        )
        connections = [[], []]
        for atoms, bond_t in self._tree.items():
            if atom1 in atoms:
                atom1c = atoms[0] if atoms[1] == atom1 else atoms[1]
                connections[0].append((atom1c, bond_t))
            if atom2 in atoms:
                atom2c = atoms[0] if atoms[1] == atom2 else atoms[1]
                connections[1].append((atom2c, bond_t))
        if connections[0] and connections[1]:
            if (atom1, atom2) not in self._checked_bonds:
                self._checked_bonds[(atom1, atom2)] = [[(bond_type), connections]]
            else:
                self._checked_bonds[(atom1, atom2)].append([bond_type, connections])

    def _global_evaluation(self) -> bool:
        """
        Evaluate the current state of the tree.

        Returns:
            bool: True if the tree is correct, False otherwise.
        """
        for _, row in self.atom_descriptors.iterrows():
            if row["nb_sb"] > 0 or row["nb_db"] > 0 or row["nb_tb"] > 0:
                return False
        if not self._is_connected():
            return False
        if not self._check_cycles():
            return False

        return True

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

            if ring_between_3_4 == 1 and count_3_4 != 1:
                return False
            if ring_between_3_4 == 2 and count_3_4 < 2:
                return False
            if ring_between_5_7 == 1 and count_5_7 != 1:
                return False
            if ring_between_5_7 == 2 and count_5_7 < 2:
                return False
            if ring_above_8 == 1 and count_8 != 1:
                return False
            if ring_above_8 == 2 and count_8 < 2:
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
                    if node in edge:
                        stack.append(edge[0] if edge[1] == node else edge[1])
        return len(visited) == len(self.atom_descriptors)

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
        for u, v in self._tree:
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
                    cycles_list.append(set(path[:]))
                elif neighbor not in visited:
                    dfs(neighbor, path)

            path.pop()
            visited.remove(node)

        dfs(start_node, [])
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

    def _check_equivalent(self, bond: tuple) -> bool:
        """
        Check if a given bond is equivalent to any bond in the tree or checked bonds.

        This method checks if the bond (atom_1, atom_2) or its equivalent bonds
        (based on the equivalence groups in self._equals) are present in either
        self._tree or self._checked_bonds.

        Args:
            bond (tuple): A tuple containing two atoms (atom_1, atom_2, bond_type) representing the bond to check.

        Returns:
            bool: True if the bond or its equivalent is found in self._tree or self._checked_bonds, False otherwise.
        """
        return False
        atom_1, atom_2, bond_type = bond
        equivalent_atoms = []
        for equal in self._equals:
            if atom_2 in equal:
                equivalent_atoms = equal
                break
        equivalent_atoms = [x for x in equivalent_atoms if x > atom_1]
        for eq_atom_2 in equivalent_atoms:
            tested_bond = (atom_1, eq_atom_2)
            if tested_bond in self._checked_bonds:
                equivalents = self._checked_bonds[tested_bond]
                for bond_info in equivalents:  # [bond_type, connections]
                    if bond_info[0] == bond_type:
                        if all(
                            tuple(sorted((atom_1, node[0]))) in self._tree
                            for node in bond_info[1][0]
                        ) and all(
                            tuple(sorted((atom_2, node[0]))) in self._tree
                            for node in bond_info[1][1]
                        ):
                            return True
        return False

    def _increment_bond(self, bond: tuple, change_bond_type: bool = True) -> tuple:
        """
        Increment the bond type of a given bond.

        Args:
            bond (tuple): A tuple containing the bond information.

        Returns:
            tuple: A tuple containing the updated bond information.
        """
        atom_1, atom_2, bond_type = bond
        out_bond = ()

        while True:
            if bond_type + 1 < 4 and change_bond_type:
                out_bond = (atom_1, atom_2, bond_type + 1)
            else:
                if atom_2 + 1 < len(self.atom_descriptors):
                    out_bond = (atom_1, atom_2 + 1, 1)
                elif atom_1 + 1 < len(self.atom_descriptors) - 1:
                    out_bond = (atom_1 + 1, atom_1 + 2, 1)
                else:
                    return ()
            atom_1 = out_bond[0]
            atom_2 = out_bond[1]
            bond_type = out_bond[2]

            # print("\tIncrementing bond: ", out_bond)
            # print("\t\tIs equivalent: ", self._check_equivalent(out_bond))
            if (not self._check_equivalent(out_bond)) and (
                self._local_evaluation(out_bond)
            ):
                break
            change_bond_type = True
        return out_bond

    def smiles_from_tree(self) -> str:
        """
        Generate a SMILES string from the tree.

        Returns:
            str: The SMILES string of the molecule.
        """
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(self._atoms_symbols)):
            a = Chem.Atom(self._atoms_symbols[i])
            # a.SetIsAromatic(self._atoms["is_aromatic"][i])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for bond in self._tree:
            atom1, atom2 = bond
            bond_type = self._tree[bond]
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
        return Chem.MolToSmiles(mol)

    def tree_search(self, node) -> None:
        """
        Perform a depth-first tree search to explore possible molecular structures.

        Args:
            node: A tuple representing the current state: (atom_1, atom_2, bond).
                  atom_1: Index of the first atom.
                  atom_2: Index of the second atom.
                  bond: The bond type being considered (1, 2, 3 for single, double, triple).
        """
        if not self._all_solutions and len(self._solutions) == self._n_solutions:
            return

        atom_1, atom_2, bond = node
        # print("Tree: ", self._tree)
        # print("Current Node: ", node)
        # print("Checked Bonds: ", self._checked_bonds)
        if self._local_evaluation(node):
            self._add_bond(node)
            if self._global_evaluation():
                self._solutions.add(self.smiles_from_tree())
                self._backtrack_node((atom_1, atom_2))
                self._results_found += 1
                return
            increment = self._increment_bond(
                (atom_1, atom_2, bond), change_bond_type=False
            )
            if increment:
                self.tree_search(increment)

            self._backtrack_node((atom_1, atom_2))

        increment = self._increment_bond((atom_1, atom_2, bond))
        if increment:
            self.tree_search(increment)

    def get_mol(self):
        return self._solutions

    def compute_solutions(self, all_solutions=False, n_solutions=1):
        self._all_solutions = all_solutions
        self._n_solutions = n_solutions
        self._checked_bonds = dict()
        self._tree = dict()
        self._init_atom_descriptor()
        self.tree_search((0, 1, 1))
        return self._solutions
