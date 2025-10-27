from .Features import Features
from rdkit import Chem
import pandas as pd
import numpy as np
import minizinc
import os
import copy
from functools import singledispatchmethod


class MoleculeFeatures:
    """
    A class to represent and manipulate molecular features from a SMILES string.

    Attributes:
    ----------
    _smiles : str
        The SMILES string representation of the molecule.
    _mol : Chem.Mol
        The RDKit molecule object created from the SMILES string.
    _features : list[Features]
        A list of Features objects representing the features of the molecule.
    _features_as_tuples : list[tuple]
        A list of tuples representing the features of the molecule.

    Methods:
    -------
    __init__(smiles: str):
        Initializes the MoleculeFeatures object with a SMILES string.
    smiles:
        Returns the SMILES string of the molecule.
    mol:
        Returns the RDKit molecule object.
    features:
        Returns the list of Features objects.
    features_as_tuples:
        Returns the list of features as tuples.
    generate_features():
        Generates and stores the features of the molecule.
    """

    _periodic_table_organic: dict = {
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

    _smiles: str
    _mol: Chem.Mol
    _features: list[Features]
    _features_as_tuples: list[tuple]

    @singledispatchmethod
    def __init__(self):
        """
        Initialize a Molecule object without any SMILES string.
        This constructor is not intended to be used directly.
        """
        raise NotImplementedError("This constructor should not be used directly.")

    @__init__.register
    def _(self, smiles: str):
        """
        Initialize a Molecule object with a SMILES string.

        Args:
            smiles (str): The SMILES string representing the molecule.

        Raises:
            ValueError: If the SMILES string is invalid and cannot be converted to a molecule.
        """
        self._smiles = smiles
        self._mol = Chem.MolFromSmiles(smiles)
        if not self._mol:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        Chem.Kekulize(self._mol, clearAromaticFlags=False)
        self.generate_features()

    @__init__.register
    def _(self, mol: Chem.Mol):
        """
        Initialize a Molecule object with an RDKit molecule object.

        Args:
            mol (Chem.Mol): The RDKit molecule object.
        """
        self._mol = mol
        self._smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        Chem.Kekulize(self._mol, clearAromaticFlags=False)
        self.generate_features()

    @__init__.register
    def _(self, other: object):
        """
        Initialize a Molecule object by copying features from another object.

        Args:
            other (object): An object that has a `features` attribute containing
                            a list of Features objects.
        """
        if not isinstance(other, MoleculeFeatures):
            raise ValueError("The provided object does not have 'features' attribute.")
        self._smiles = other.smiles
        self._mol = other.mol
        self._features = copy.deepcopy(other.features)
        self._features_as_tuples = copy.deepcopy(other.features_as_tuples)

    @__init__.register
    def _(self, features: list):
        """
        Initialize a Molecule object with a list of Features objects.

        Args:
            features (list[Features]): A list of Features objects representing the features of the molecule.
        """
        if not features:
            raise ValueError("The features list cannot be empty.")
        if isinstance(features[0], Features):
            self._features = features
            self._features_as_tuples = [
                feature.features_as_tuple() for feature in features
            ]
        elif isinstance(features[0], tuple):
            self._features = [Features(feature) for feature in features]
            self._features_as_tuples = features
        else:
            raise TypeError(
                "The features must be a list of Features objects or tuples."
            )

    @property
    def smiles(self) -> str:
        """
        Returns the SMILES (Simplified Molecular Input Line Entry System) representation of the molecule.

        Returns:
            str: The SMILES string of the molecule.
        """
        return self._smiles

    @property
    def mol(self) -> Chem.Mol:
        """
        Returns the RDKit molecule object.

        Returns:
            Chem.Mol: The RDKit molecule object.
        """
        return self._mol

    @property
    def features(self) -> list[Features]:
        """
        Returns the set of features (features) associated with the molecule.

        Returns:
            list[Features]: A set containing the features of the molecule.
        """
        return self._features

    @property
    def features_as_tuples(self) -> list[tuple]:
        """
        Returns the features of the molecule as a set of tuples.

        Returns:
            list[tuple]: A set containing the features of the molecule as tuples.
        """
        return self._features_as_tuples

    def get_atoms_from_features(self):
        """
        Retrieve an atom symbol from the periodic table based on its valence and periodic row.
        """
        for feature in self._features:
            valence, periodic_row = feature.valence_electrons, feature.periodic_row
            periodic_column = valence if valence < 3 else valence + 10
            if (periodic_row, periodic_column) in self._periodic_table_organic:
                yield self._periodic_table_organic[(periodic_row, periodic_column)]
            else:
                raise KeyError(
                    f"Atom with valence {valence} and periodic row {periodic_row} not found in the periodic table."
                )

    def generate_features(self, weights: dict = None):
        """
        Generate features for each atom in the molecule.

        This method iterates over all atoms in the molecule and generates
        features for each atom. The generated features are added to the
        `_features` set, and the features represented as tuples are added
        to the `_features_as_tuples` set.
        """
        self._features = list()
        self._features_as_tuples = list()
        for atom in self._mol.GetAtoms():
            feature = Features(self.mol, atom, weights=weights)
            self._features.append(feature)
            self._features_as_tuples.append(feature.features_as_tuple())

    def evaluate_distance(
        self, molecule: "MoleculeFeatures", method: str = "csp", raw: bool = False
    ) -> tuple:
        """
        Evaluate the distance between the features of this molecule and another molecule.

        Parameters
        ----------
        molecule : MoleculeFeatures
            The other molecule to compare against.
        method : str, optional
            The method to use for distance evaluation. Supported values are:
            - "csp": Use the CSP (custom shortest path) method.
            - "min": Use the minimum distance matching method.
            Default is "csp".

        Returns
        -------
        tuple
            A tuple containing:
            - The total distance (rounded to 3 decimal places).
            - The average distance per feature in the second molecule (rounded to 3 decimal places).

        Raises
        ------
        ValueError
            If an unknown method is provided.

        Notes
        -----
        The function computes a pairwise distance matrix between the features of the two molecules,
        then applies the selected method to aggregate the distances.
        """
        round_value = 3
        if len(self.features) <= len(molecule.features):
            features_1 = self._features.copy()
            features_2 = molecule.features.copy()
        else:
            features_1 = molecule.features.copy()
            features_2 = self._features.copy()
        tab_distance = pd.DataFrame(
            index=[i for i in range(len(features_1))],
            columns=[i for i in range(len(features_2))],
            dtype=np.float64,
        )
        raw_tab_distance = pd.DataFrame(
            index=[i for i in range(len(features_1))],
            columns=[i for i in range(len(features_2))],
            dtype=object,
        )
        # Fill the distance table
        for i in range(len(features_1)):
            for j in range(len(features_2)):
                distance, raw_dist = features_1[i].get_distance(features_2[j], raw=True)
                if raw:
                    raw_tab_distance.iloc[i, j] = raw_dist
                tab_distance.iloc[i, j] = distance
        if method == "min":
            out, diff = self._evaluate_distance_min(
                features_1, features_2, tab_distance, raw_tab_distance, raw
            )
        elif method == "csp":
            out, diff = self._evaluate_distance_csp(
                features_1, features_2, tab_distance, raw_tab_distance, raw
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'min' or 'csp'.")
        if raw:
            return (
                round(out, round_value),
                round(out / len(features_2), round_value),
                diff,
            )
        return round(out, round_value), round(out / len(features_2), round_value)

    def _evaluate_distance_csp(
        self,
        features_1: list[Features],
        features_2: list[Features],
        tab_distance: pd.DataFrame,
        raw_tab: pd.DataFrame = None,
        raw_dist: bool = False,
    ):
        """
        Evaluates the distance between two sets of molecular features using a constraint satisfaction problem (CSP) approach.

        This method leverages a MiniZinc model to find the optimal assignment between features of two molecules that minimizes
        the total distance, as defined by the provided distance table. It uses the 'gecode' solver to solve the CSP.

        Args:
            features_1 (list or array-like): The features of the first molecule.
            features_2 (list or array-like): The features of the second molecule.
            tab_distance (pandas.DataFrame): A DataFrame representing the pairwise distances between features of the two molecules.

        Returns:
            float or None: The computed minimal distance between the two sets of features, possibly adjusted for unmatched features.
                           Returns None if the CSP solver fails to find a solution.
        """
        gecode = minizinc.Solver.lookup("gecode")
        model = minizinc.Model()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        minizinc_path = os.path.join(script_dir, "../csp/smallest_tab.mzn")
        model.add_file(minizinc_path)
        instance = minizinc.Instance(gecode, model)
        diff = np.zeros(11, dtype=float)
        data = {
            "size_mol": len(features_1),
            "size_mol2": len(features_2),
            "differences": tab_distance.values.tolist(),
        }
        for key, value in data.items():
            instance[key] = value
        result = instance.solve()
        if result is None:
            return None
        out = abs(result["smallest_size"])
        idx_used = result["smallest_idx"]
        all_indices = np.arange(len(features_2))
        excluded_indices = np.array(idx_used) - 1
        if raw_dist:
            for index, value in enumerate(excluded_indices):
                if raw_tab is not None:
                    diff += raw_tab.iloc[index, value]

        missing_indices = np.setdiff1d(all_indices, excluded_indices)
        for index in missing_indices:
            # Compute the mean in the difference column of the index
            index_max = np.argmax(tab_distance.iloc[:, index])
            out += 1 + abs(tab_distance.iloc[index_max, index])

            diff += raw_tab.iloc[index_max, index]
        return out, diff

    def _evaluate_distance_min(
        self,
        features_1: list[Features],
        features_2: list[Features],
        tab_distance: pd.DataFrame,
        raw_tab: pd.DataFrame = None,
        raw_dist: bool = False,
    ):
        """
        Computes a custom minimum distance metric between two sets of features using a distance matrix.

        The function iteratively finds the minimum value in the distance matrix, accumulates it,
        and marks the corresponding row and column as used (by setting them to NaN) to avoid reusing
        the same features. After all features in `features_1` are matched, it penalizes unmatched
        features in `features_2` by adding a value based on the maximum distance in their respective columns.

        Args:
            features_1 (list or array-like): The first set of features.
            features_2 (list or array-like): The second set of features.
            tab_distance (pandas.DataFrame): A distance matrix (shape: len(features_1) x len(features_2))
                representing pairwise distances between features in `features_1` and `features_2`.

        Returns:
            float: The computed minimum distance metric between the two feature sets.
        """
        out = 0
        copy_tab_distance = copy.deepcopy(tab_distance)
        idx_used = []
        diff = np.zeros(11, dtype=float)
        for _ in range(len(features_1)):
            min_tab = np.nanmin(copy_tab_distance.values)
            min_index = np.unravel_index(
                np.nanargmin(copy_tab_distance.values), copy_tab_distance.shape
            )
            idx_used.append(min_index[1])
            copy_tab_distance.iloc[min_index[0], :] = np.nan
            copy_tab_distance.iloc[:, min_index[1]] = np.nan
            out += min_tab
            if raw_dist:
                diff += raw_tab.iloc[min_index[0], min_index[1]]
        all_indices = np.arange(len(features_2))
        excluded_indices = np.array(idx_used)
        missing_indices = np.setdiff1d(all_indices, excluded_indices)
        for index in missing_indices:
            arg_max = np.argmax(tab_distance.iloc[:, index])
            out += 1 + abs(tab_distance.iloc[arg_max, index])
            if raw_dist:
                diff += raw_tab.iloc[arg_max, index]
        else:
            diff = None
        return out, diff

    def has_same_features(self, features_generated: tuple) -> bool:
        """
        Check if two MoleculeFeatures objects are equal based on their SMILES strings.

        Args:
            value (MoleculeFeatures): The other MoleculeFeatures object to compare.

        Returns:
            bool: True if the SMILES strings are equal, False otherwise.
        """
        features = self._features_as_tuples.copy()
        if len(features) != len(features_generated):
            return False
        for i in range(len(features)):
            try:
                features.remove(features_generated[i])
            except Exception as e:
                return False
        return True
