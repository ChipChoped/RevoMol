import os
import minizinc
from rdkit import Chem
from .AtomRebuilder import AtomRebuilder
from dataclasses import asdict
import asyncio


class CSPAtomRebuilder(AtomRebuilder):
    """CSPAtomRebuilder is a class that rebuilds chemical structures from given atomic data using constraint satisfaction programming (CSP).

    Attributes:
        periodic_table_organic (dict): A dictionary mapping tuples of (periodic row, periodic column) to element symbols.
        script_dir (str): The directory of the current script.
        minizinc_path (str): The path to the MiniZinc model file.

    Methods:
        __init__(atom_data_list: list[tuple]):
            Initializes the CSPAtomRebuilder with a list of atom data tuples.

        atoms_symbols:
            Returns the list of atom symbols.

        MolFromGraphs(adjacency_matrix):
            Generates a molecular structure from a list of nodes and an adjacency matrix.

        get_mol() -> set[str]:
            Constructs molecular structures from the provided solutions and returns a set of SMILES strings.

        compute_solutions(all_solutions=False, n_solutions=None) -> list[list[int]]:
            Runs the MiniZinc model to compute solutions and returns a list of solutions.
    """

    minizinc_path: str
    script_dir: str

    def __init__(self, atom_data_list: list[tuple]):
        """
        Initializes the CSPRebuilder class with atom data and sets up necessary paths.

        Args:
            atom_data_list (list[tuple]): A list of tuples where each tuple contains atom data.
                                          Each tuple should have at least two elements:
                                          valence electrons and periodic row.

        Attributes:
            script_dir (str): The directory where the current script is located.
            minizinc_path (str): The absolute path to the MiniZinc model file.
        """
        super().__init__(atom_data_list)
        # Determine absolute paths
        self._solutions_mzn = []
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.minizinc_path = os.path.join(self.script_dir, "../csp/rebuilder.mzn")

    def _convert_atom_data(self, atom_data: tuple) -> dict:
        """
        Converts atom data from a tuple to a dictionary with specific keys.

        Args:
            atom_data (tuple): A tuple containing the following elements:
                - formal_charge (int): The formal charge of the atom.
                - unpaired_electrons (int): The number of unpaired electrons.
                - triple_bonds (iterable): An iterable of triple bond counts.
                - double_bonds (iterable): An iterable of double bond counts.
                - single_bonds (iterable): An iterable of single bond counts.
                - _ (Any): Placeholder for unused data.
                - lone_pairs (int): The number of lone pairs.
                - rings (iterable): An iterable of ring sizes.
                - is_aromatic (bool): Whether the atom is aromatic.
                - valence_electrons (int): The number of valence electrons.
                - periodic_row (int): The periodic table row of the atom.

        Returns:
            dict: A dictionary with the following keys:
                - "formal_charge" (int): The formal charge of the atom.
                - "unpaired_electron" (int): The number of unpaired electrons.
                - "triple_bond" (list): A list of triple bond counts, padded to length 2.
                - "double_bond" (list): A list of double bond counts, padded to length 4.
                - "single_bond" (list): A list of single bond counts, padded to length 7.
                - "lone_pair" (int): The number of lone pairs.
                - "ring_size" (list): A list of ring sizes.
                - "is_aromatic" (bool): Whether the atom is aromatic.
                - "valence" (int): The number of valence electrons.
                - "periodic" (int): The periodic table row of the atom.
        """
        (
            formal_charge,
            unpaired_electrons,
            triple_bonds,
            double_bonds,
            single_bonds,
            _,
            lone_pairs,
            rings,
            is_aromatic,
            valence_electrons,
            periodic_row,
        ) = atom_data

        single_bonds = list(single_bonds)
        size = len(single_bonds)
        single_bonds.extend([0] * (7 - size))

        double_bonds = list(double_bonds)
        size = len(double_bonds)
        double_bonds.extend([0] * (4 - size))

        triple_bonds = list(triple_bonds)
        size = len(triple_bonds)
        triple_bonds.extend([0] * (2 - size))

        return {
            "formal_charge": formal_charge,
            "unpaired_electron": unpaired_electrons,
            "triple_bond": triple_bonds,
            "double_bond": double_bonds,
            "single_bond": single_bonds,
            "lone_pair": lone_pairs,
            "ring_size": list(rings),
            "is_aromatic": is_aromatic,
            "valence": valence_electrons,
            "periodic": periodic_row,
        }

    def _build_minizinc_data(self) -> str:
        """
        Constructs a MiniZinc data string from the atom data list.

        This method processes the atom data stored in `self._atom_data_list` and
        converts it into a dictionary format suitable for MiniZinc. The dictionary
        contains various attributes of atoms such as formal charge, unpaired electrons,
        bond types, lone pairs, ring sizes, aromaticity, valence, and periodicity.

        Returns:
            str: A string representation of the MiniZinc data.
        """

        natoms = len(self._atom_data_list)

        atom_datas = [
            self._convert_atom_data(atom_data) for atom_data in self._atom_data_list
        ]
        self._atoms = {"natoms": natoms}
        self._atoms = {
            "natoms": natoms,
            "nsingle_bonds": 7,
            "ndouble_bonds": 4,
            "ntriple_bonds": 2,
        }
        self._atoms["formal_charge"] = []
        self._atoms["unpaired_electron"] = []
        self._atoms["triple_bond"] = []
        self._atoms["double_bond"] = []
        self._atoms["single_bond"] = []
        self._atoms["lone_pair"] = []
        self._atoms["ring_size"] = []
        self._atoms["is_aromatic"] = []
        self._atoms["valence"] = []
        self._atoms["periodic"] = []

        for atom_data in atom_datas:
            self._atoms["formal_charge"].append(atom_data["formal_charge"])
            self._atoms["unpaired_electron"].append(atom_data["unpaired_electron"])
            self._atoms["triple_bond"].append(atom_data["triple_bond"])
            self._atoms["double_bond"].append(atom_data["double_bond"])
            self._atoms["single_bond"].append(atom_data["single_bond"])
            self._atoms["lone_pair"].append(atom_data["lone_pair"])
            self._atoms["ring_size"].append(atom_data["ring_size"])
            self._atoms["is_aromatic"].append(atom_data["is_aromatic"])
            self._atoms["valence"].append(atom_data["valence"])
            self._atoms["periodic"].append(atom_data["periodic"])

    def MolFromGraphs(self, adjacency_matrix):
        """
        Generate a molecular structure from a list of nodes and an adjacency matrix.

        Args:
            node_list (list): A list of atomic numbers representing the atoms in the molecule.
            adjacency_matrix (list of list of int): A 2D list representing the adjacency matrix of the molecule.
                Each element in the matrix represents the bond type between atoms:
                0 - no bond, 1 - single bond, 2 - double bond, 3 - triple bond.

        Returns:
            Chem.Mol: An RDKit Mol object representing the molecular structure.
        """
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(self._atoms_symbols)):
            a = Chem.Atom(self._atoms_symbols[i])
            # a.SetIsAromatic(self._atoms["is_aromatic"][i])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        # add bonds between adjacent atoms
        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):

                if iy <= ix:
                    continue

                if bond == 0:
                    continue
                elif bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        # Convert RWMol to Mol object
        mol = mol.GetMol()

        return mol

    def get_mol(self) -> set[str]:
        self._solutions = set()
        mols = set()
        for solution in self._solutions_mzn:
            adjacency_matrix = solution["adjacency_matrix"]
            mol = self.MolFromGraphs(adjacency_matrix)
            mols.add(mol)
        for molecule in mols:
            self._solutions.add(Chem.MolToSmiles(molecule))
        return self._solutions

    def compute_solutions(
        self, all_solutions=True, n_solutions=None
    ) -> list[list[int]]:
        gecode = minizinc.Solver.lookup("gecode")
        model = minizinc.Model()
        model.add_file(self.minizinc_path)
        instance = minizinc.Instance(gecode, model)
        self._build_minizinc_data()

        for key, value in self._atoms.items():
            instance[key] = value
        if n_solutions is not None:
            all_solutions = False
        result = asyncio.run(
            instance.solve_async(all_solutions=all_solutions, nr_solutions=n_solutions)
        )
        self._solutions_mzn = result.solution
        if not isinstance(self._solutions_mzn, list):
            self._solutions_mzn = [self._solutions_mzn]
        # Convert every Solution type to dictionary
        self._solutions_mzn = [asdict(solution) for solution in self._solutions_mzn]
        self._results_found = len(self._solutions_mzn)
        return self.get_mol()
