from abc import ABC, abstractmethod


class AtomRebuilder(ABC):
    periodic_table_organic: dict = {
        (1, 1): "H",
        (2, 13): "B",
        (2, 14): "C",
        (2, 15): "N",
        (2, 16): "O",
        (2, 17): "F",
        (3, 13): "Al",
        (3, 14): "Si",
        (3, 15): "P",
        (3, 16): "S",
        (3, 17): "Cl",
        (4, 16): "Se",
        (4, 17): "Br",
        (5, 17): "I",
    }
    _atom_data_list: list[tuple]
    _atoms: dict
    _atoms_symbols: list[str]
    _solutions: set[str]
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
        self._atom_data_list = atom_data_list
        self._atom_data_list = sorted(self._atom_data_list, key=lambda x: x[-2])

        self._atoms_symbols = []
        for atom_data in self._atom_data_list:
            valence, periodic_row = atom_data[-2], atom_data[-1]
            self._atoms_symbols.append(
                self._get_atom_from_features(valence, periodic_row)
            )
        self._atoms = dict()
        self._solutions = set()
        self._results_found = 0

    @property
    def atoms_symbols(self):
        """
        Retrieve the symbols of the atoms.

        Returns:
            list: A list containing the symbols of the atoms.
        """
        return self._atoms_symbols

    def _get_atom_from_features(self, valence, periodic_row):
        """
        Retrieve an atom from the periodic table based on its valence and periodic row.

        Args:
            valence (int): The valence of the atom.
            periodic_row (int): The periodic row of the atom.

        Returns:
            str: The symbol of the atom from the periodic table.

        Raises:
            KeyError: If the atom with the given valence and periodic row is not found in the periodic table.
        """
        periodic_column = valence if valence < 3 else valence + 10
        return self.periodic_table_organic[(periodic_row, periodic_column)]

    @abstractmethod
    def get_mol(self) -> set[str]:
        """
        Retrieves a set of molecular identifiers.

        Returns:
            set[str]: A set containing molecular identifiers as strings.
        """
        pass

    @abstractmethod
    def compute_solutions(self, all_solutions=False, n_solutions=1) -> list[list[int]]:
        """
        Compute solutions based on the given parameters.

        Args:
            all_solutions (bool): If True, compute all possible solutions. If False, compute a limited number of solutions.
            n_solutions (int, optional): The number of solutions to compute if all_solutions is False. Defaults to None.

        Returns:
            list[list[int]]: A list of solutions, where each solution is represented as a list of integers.
        """
        pass
