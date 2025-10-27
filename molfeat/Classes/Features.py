from rdkit import Chem
import json
from functools import singledispatchmethod


class Features:
    """
    The Features class encapsulates various chemical features of an atom within a molecule. It provides methods to compute and access these features, including formal charge, unpaired electrons, bond types, number of hydrogens, lone pairs, ring membership, aromaticity, valence electrons, and periodic table row.

    Attributes:
        periodic_table (PeriodicTable): The periodic table instance from RDKit.

    Methods:
        __init__(self, molecule: Chem.Mol, atom: Chem.Atom):
            Initializes the Features object with the given molecule and atom.

        molecular(self):
            Returns the molecule containing the atom.

        atom(self):
            Returns the atom for which features are computed.

        atomic_number(self):
            Returns the atomic number of the atom.

        formal_charge(self):
            Returns the formal charge of the atom.

        unpaired_electrons(self):
            Returns the number of unpaired electrons of the atom.

        triple_bonds(self):
            Returns the list of valence electrons of atoms bonded with triple bonds.

        double_bonds(self):
            Returns the list of valence electrons of atoms bonded with double bonds.

        single_bonds(self):
            Returns the list of valence electrons of atoms bonded with single bonds.

        num_hydrogens(self):
            Returns the number of hydrogen atoms bonded to the atom.

        lone_pairs(self):
            Returns the number of lone pairs on the atom.

        rings(self):
            Returns the tuple indicating the atom's membership in small, medium, and large cycles.

        is_aromatic(self):
            Returns whether the atom is aromatic.

        valence_electrons(self):
            Returns the number of valence electrons of the atom.

        periodic_row(self):
            Returns the periodic table row of the atom.

        get_tuple(self):
            Returns a tuple of the atom's features.
    """

    periodic_table = Chem.GetPeriodicTable()
    _molecule: Chem.Mol
    _atom: Chem.Atom
    _atomic_number: int

    # Features
    _formal_charge: int
    _unpaired_electrons: int
    _triple_bonds: list[int]
    _double_bonds: list[int]
    _single_bonds: list[int]
    _num_hydrogens: int
    _lone_pair: int
    _rings: tuple[int, int, int]
    _is_aromatic: bool
    _valence: int
    _periodic_row: int
    _weights: int

    # __DEFAULT_WEIGHTS = {
    #     "formal_charge": 0.143,
    #     "unpaired_electrons": 0.143,
    #     "triple_bonds": 0.143,
    #     "double_bonds": 0.114,
    #     "single_bonds": 0.044,
    #     "num_hydrogens": 0.012,
    #     "lone_pairs": 0.012,
    #     "rings": 0,
    #     "is_aromatic": 0.084,
    #     "valence_electrons": 0.29,
    #     "periodic_row": 0.0143,
    # }

    __DEFAULT_WEIGHTS = {
        "formal_charge": 0.0994662061598495,
        "unpaired_electrons": 0.139554083020951,
        "triple_bonds": 0.1193919466018106,
        "double_bonds": 0.22254489036167885,
        "single_bonds": 0.070091945585402,
        "num_hydrogens": 0.040815197413890635,
        "lone_pairs": 0.06939645270993518,
        "rings": 0.01249421765649615,
        "is_aromatic": 0.03657555352938531,
        "valence_electrons": 0.11062217925754168,
        "periodic_row": 0.07904732770305911,
    }

    @singledispatchmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the Features class. This method is a placeholder and should not be called directly.
        Use the appropriate constructor for initializing with a molecule and atom or a tuple of features.
        """
        raise NotImplementedError(
            "Please use the constructor with a molecule and atom or a tuple of features."
        )

    @__init__.register
    def _(self, molecule: Chem.Mol, atom: Chem.Atom, weights: dict = None):
        """
        Initialize the Features class with a molecule and an atom.

        Args:
            molecule (Chem.Mol): The molecule containing the atom.
            atom (Chem.Atom): The atom in the molecule for which features are being calculated.
        """
        if atom is None:
            return
        self._atomic_number = atom.GetAtomicNum()
        self._atom = atom
        self._molecule = molecule
        self._periodic_row = self._get_periodic_row()
        self._formal_charge = atom.GetFormalCharge()
        self._unpaired_electrons = atom.GetNumRadicalElectrons()

        self._num_hydrogens = self.atom.GetTotalNumHs()

        self._valence = Features.periodic_table.GetNOuterElecs(self._atomic_number)

        # Bonds
        self._single_bonds = []
        self._double_bonds = []
        self._triple_bonds = []
        self._get_bonds()

        # Lone pairs
        self._lone_pair = 0
        self._lone_pair_computation()

        # Rings
        ring_info = molecule.GetRingInfo()
        rings_sizes = ring_info.AtomRingSizes(atom.GetIdx())
        self._rings = self._get_cycle_info(rings_sizes)

        # Is the atom aromatic?
        self._is_aromatic = atom.GetIsAromatic()

        # Determine the periodic table row
        self._periodic_row = self._get_periodic_row()

        # If weights are provided, update the weights
        self._weights = Features.__DEFAULT_WEIGHTS.copy()
        if weights is not None:
            if not isinstance(weights, dict):
                raise TypeError("Weights must be a dictionary.")
            for key, value in weights.items():
                if key in Features.__DEFAULT_WEIGHTS:
                    self._weights[key] = value
                else:
                    raise KeyError(f"Invalid weight key: {key}")

    @__init__.register
    def _(self, features_tuple: tuple, weights: dict = None):
        """
        Initialize the Features class from a tuple of features.

        Args:
            features_tuple (tuple): A tuple containing the features of the atom in the following order:
                (atomic_number, formal_charge, unpaired_electrons, triple_bonds, double
                _bonds, single_bonds, num_hydrogens, lone_pairs, rings, is_aromatic,
                valence_electrons, periodic_row).
        """
        if len(features_tuple) != 11:
            raise ValueError(
                "Tuple must contain exactly 11 elements. Not {} elements.".format(
                    len(features_tuple)
                )
            )

        (
            formal_charge,
            unpaired_electrons,
            triple_bonds,
            double_bonds,
            single_bonds,
            num_hydrogens,
            lone_pairs,
            rings,
            is_aromatic,
            valence_electrons,
            periodic_row,
        ) = features_tuple
        self._formal_charge = formal_charge
        self._unpaired_electrons = unpaired_electrons
        self._triple_bonds = list(triple_bonds)
        self._double_bonds = list(double_bonds)
        self._single_bonds = list(single_bonds)
        self._num_hydrogens = num_hydrogens
        self._lone_pair = lone_pairs
        self._rings = rings
        self._is_aromatic = is_aromatic
        self._valence = valence_electrons
        self._periodic_row = periodic_row

        # If weights are provided, update the weights
        self._weights = Features.__DEFAULT_WEIGHTS.copy()
        if weights is not None:
            if not isinstance(weights, dict):
                raise TypeError("Weights must be a dictionary.")
            for key, value in weights.items():
                if key in Features.__DEFAULT_WEIGHTS:
                    self._weights[key] = value
                else:
                    raise KeyError(f"Invalid weight key: {key}")

    # ========================================
    # Features
    # ========================================
    @property
    def molecular(self) -> Chem.Mol:
        """
        Returns the molecule of the object.

        Returns:
            Chem.Mol: The molecule in which the atom is present.
        """
        return self._molecule

    @property
    def atom(self) -> Chem.Atom:
        """
        Returns the atom that the object represents.

        Returns:
            Chem.Atom: The atom for which features are computed.
        """
        return self._atom

    @property
    def atomic_number(self) -> int:
        """
        Returns the atomic number of the atom.

        Returns:
            int: The atomic number of the atom.
        """
        return self._atomic_number

    @property
    def formal_charge(self) -> int:
        """
        Returns the formal charge of the atom.

        Returns:
            int: The formal charge of the atom.
        """
        return self._formal_charge

    @property
    def unpaired_electrons(self) -> int:
        """
        Returns the number of unpaired electrons of the atom.

        Returns:
            int: The number of unpaired electrons of the atom.
        """
        return self._unpaired_electrons

    @property
    def triple_bonds(self) -> list[int]:
        """
        Returns the list of valence electrons of atoms bonded with triple bonds.

        Returns:
            list[int]: The list of valence electrons of atoms bonded with triple bonds.
        """
        return self._triple_bonds

    @property
    def double_bonds(self) -> list[int]:
        """
        Returns the list of valence electrons of atoms bonded with double bonds.

        Returns:
            list[int]: The list of valence electrons of atoms bonded with double bonds.
        """
        return self._double_bonds

    @property
    def single_bonds(self) -> list[int]:
        """
        Returns the list of valence electrons of atoms bonded with single bonds.

        Returns:
            list[int]: The list of valence electrons of atoms bonded with single bonds.
        """
        return self._single_bonds

    @property
    def num_hydrogens(self) -> int:
        """
        Returns the number of hydrogen atoms bonded to the atom.

        Returns:
            int: The number of hydrogen atoms bonded to the atom.
        """
        return self._num_hydrogens

    @property
    def lone_pairs(self) -> int:
        """
        Returns the number of lone pairs on the atom.

        Returns:
            int: The number of lone pairs on the atom.
        """
        return self._lone_pair

    @property
    def rings(self) -> tuple[int, int, int]:
        """
        Returns the tuple indicating the atom's membership in small, medium, and large cycles.

        Returns:
            tuple[int, int, int]: The tuple indicating the atom's membership in small, medium, and large cycles.
        """
        return self._rings

    @property
    def is_aromatic(self) -> bool:
        """
        Returns whether the atom is aromatic.

        Returns:
            bool: True if the atom is aromatic, False otherwise.
        """
        return self._is_aromatic

    @property
    def valence_electrons(self) -> int:
        """
        Returns the number of valence electrons of the atom.

        Returns:
            int: The number of valence electrons of the atom.
        """
        return self._valence

    @property
    def periodic_row(self) -> int:
        """
        Returns the periodic table row of the atom.

        Returns:
            int: The periodic table row of the atom.
        """
        return self._periodic_row

    # ========================================
    # Methods
    # ========================================

    def _get_cycle_info(self, ring_sizes):
        """
        Returns a tuple indicating the atom's membership in different cycle types:
        - First element: Small cycles (size <= 4)
        - Second element: Medium cycles (size between 5 and 7)
        - Third element: Large cycles (size > 7)

        Each value in the tuple is:
        - 0 if the atom belongs to no cycles of this type.
        - 1 if the atom belongs to one cycle of this type.
        - 2 if the atom belongs to two or more cycles of this type.
        """
        small_cycles = 0
        medium_cycles = 0
        large_cycles = 0
        # Count occurrences for each category
        for size in ring_sizes:
            if size <= 4:
                small_cycles += 1
            elif 5 <= size <= 7:
                medium_cycles += 1
            else:
                large_cycles += 1

        return (small_cycles, medium_cycles, large_cycles)

    def _get_bonds(self):
        """
        Analyzes the bonds of the atom and categorizes them into single, double, and triple bonds.

        Iterates over all bonds of the atom, determines the type of each bond, and appends the
        valence of the neighboring atom to the corresponding bond type list. Finally, sorts
        the lists of single, double, and triple bonds.
        """
        for bond in self.atom.GetBonds():
            neighbor = bond.GetOtherAtom(self.atom)
            neighbor_atomic_number = neighbor.GetAtomicNum()
            neighbor_valence = Features.periodic_table.GetNOuterElecs(
                neighbor_atomic_number
            )

            bond_type = bond.GetBondType()
            if bond_type == Chem.rdchem.BondType.SINGLE:
                self._single_bonds.append(neighbor_valence)
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                self._double_bonds.append(neighbor_valence)
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                self._triple_bonds.append(neighbor_valence)

        self._single_bonds.sort()
        self._double_bonds.sort()
        self._triple_bonds.sort()

    def _get_periodic_row(self):
        """
        Returns the periodic table row based on the atomic number.
        """
        if self._atomic_number <= 2:
            return 1
        elif self._atomic_number <= 10:
            return 2
        elif self._atomic_number <= 18:
            return 3
        elif self._atomic_number <= 36:
            return 4
        elif self._atomic_number <= 54:
            return 5
        elif self._atomic_number <= 86:
            return 6
        else:
            return 7

    def _lone_pair_computation(self):
        """
        Computes the number of lone pairs of electrons for an atom.

        This method calculates the lone pairs of electrons based on the atom's
        valence electrons, formal charge, number of hydrogen atoms, and the number
        of single, double, and triple bonds.

        The formula used is:
            lone_pairs = (effective_valence_electrons - bonding_electrons - unpaired_electrons) // 2

        Where:
            - effective_valence_electrons = valence_electrons - formal_charge
            - bonding_electrons = number of hydrogen atoms + single bonds + 2 * double bonds + 3 * triple bonds

        Attributes:
            valence_electrons (int): Number of valence electrons for the atom.
            effective_valence_electrons (int): Valence electrons adjusted for the formal charge.
            bonding_electrons (int): Total number of electrons involved in bonds.
            _lone_pair (int): Computed number of lone pairs of electrons.
        """
        valence_electrons = Features.periodic_table.GetNOuterElecs(self.atomic_number)
        effective_valence_electrons = valence_electrons - self._formal_charge
        bonding_electrons = (
            self._num_hydrogens
            + len(self._single_bonds)
            + 2 * len(self._double_bonds)
            + 3 * len(self._triple_bonds)
        )
        self._lone_pair = (
            effective_valence_electrons - bonding_electrons - self._unpaired_electrons
        ) // 2

    def features_as_tuple(self):
        """
        Converts the features of the object to a tuple.

        Returns:
            tuple: A tuple containing the following features:
                - formal_charge (int): The formal charge of the object.
                - unpaired_electrons (int): The number of unpaired electrons.
                - triple_bonds (tuple): A tuple of triple bonds.
                - double_bonds (tuple): A tuple of double bonds.
                - single_bonds (tuple): A tuple of single bonds.
                - num_hydrogens (int): The number of hydrogen atoms.
                - lone_pairs (int): The number of lone pairs.
                - rings (int): The number of rings.
                - is_aromatic (bool): Whether the object is aromatic.
                - valence_electrons (int): The number of valence electrons.
                - periodic_row (int): The periodic row of the object.
        """
        return (
            self.formal_charge,
            self.unpaired_electrons,
            # Converting lists to tuples to ensure immutability
            tuple(self.triple_bonds),
            tuple(self.double_bonds),
            tuple(self.single_bonds),
            self.num_hydrogens,
            self.lone_pairs,
            self.rings,
            self.is_aromatic,
            self.valence_electrons,
            self.periodic_row,
        )

    def get_distance(self, other: "Features", raw=False) -> float:
        """
        Compute the distance between two features.

        Args:
            other (Features): The other Features object.

        Returns:
            float: The distance between the two Features objects.
        """
        distance = 0
        raw_dist = []
        for key, weight in self._weights.items():
            add_dist = 0
            if type(getattr(self, key)) == list or type(getattr(self, key)) == tuple:
                add_dist = sum(
                    a != b for a, b in zip(getattr(self, key), getattr(other, key))
                )
            else:
                add_dist = abs(getattr(self, key) - getattr(other, key))
            distance += weight * add_dist
            if raw:
                raw_dist.append(add_dist)
        if raw:
            return round(distance, 10), raw_dist
        return round(distance, 10)

    # ========================================
    # Magic methods
    # ========================================

    def __str__(self) -> str:
        """Convert the object to a string."""
        return json.dumps(
            {
                "formal_charge": self.formal_charge,
                "unpaired_electrons": self.unpaired_electrons,
                "triple_bonds": self.triple_bonds,
                "double_bonds": self.double_bonds,
                "single_bonds": self.single_bonds,
                "num_hydrogens": self.num_hydrogens,
                "lone_pairs": self.lone_pairs,
                "rings": self.rings,
                "is_aromatic": self.is_aromatic,
                "valence_electrons": self.valence_electrons,
                "periodic_row": self.periodic_row,
            }
        )

    def __repr__(self):
        """Return the object representation."""
        return self.__str__()
