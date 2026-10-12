"""
Molecular graph representation of a molecule with RDKit.
"""

from __future__ import annotations

import networkx as nx
from rdkit import Chem
from typing_extensions import override

from evomol.representation.molecule import MoleculeRepresentation

# double bond inside a ring (see MolecularGraph.canonical_smiles)
_RING_DOUBLE_BOND = Chem.MolFromSmarts("*=@*")


# pylint: disable=too-many-public-methods
class MolecularGraph(MoleculeRepresentation):
    """
    Molecule representation using a molecular graph with RDKit.
    """

    # True: canonical_smiles gives one Kekulé form per molecule (see
    # canonical_smiles). False (default): direct writing of self.mol, an
    # aromatic molecule can have two SMILES depending on the mutations that
    # built it. Set by default_parameters.setup_default_parameters.
    unique_kekule: bool = False

    def __init__(self, smiles: str, sanitize: bool = False):
        super().__init__(smiles)
        try:
            self.mol: Chem.rdchem.RWMol = Chem.rdchem.RWMol(Chem.MolFromSmiles(smiles))
        except Exception as e:
            raise ValueError(f"Error with SMILES {smiles}: {e}") from e

        if self.mol is None:
            raise ValueError(f"Error with SMILES {smiles}")

        self.sanitize: bool = sanitize

        self.update_representation()

    def __copy__(self) -> MolecularGraph:
        mol = MolecularGraph("", self.sanitize)
        mol.mol = Chem.RWMol(self.mol, True)
        mol.update_representation()
        return mol

    def raw_copy(self) -> MolecularGraph:
        """Structural copy that skips update_representation.

        self.mol is already sanitized/kekulized, and Chem.RWMol(mol, True)
        preserves that structure, so re-sanitizing the copy just redoes work
        already done on the parent. Instead the parent is sanitized once (at
        its creation) and each action clones from it cheaply.

        Contract: the returned graph is meant to be mutated in place by an
        action and then finalized with update_representation(). It must NOT be
        read as-is (canonical_smiles, validity, ...) before that call, since
        its property cache / ring info are not refreshed.
        """
        mol = MolecularGraph.__new__(MolecularGraph)
        mol.str_id = self.str_id
        mol.sanitize = self.sanitize
        mol.mol = Chem.RWMol(self.mol, True)
        mol._clear_cached_derivatives()
        return mol

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, MolecularGraph):
            return False
        return self.aromatic_canonical_smiles == value.aromatic_canonical_smiles

    @override
    def representation(self) -> str:
        return self.canonical_smiles

    def __repr__(self) -> str:
        return f"MolecularGraph({self.canonical_smiles})"

    def _clear_cached_derivatives(self) -> None:
        """Invalidate the cached SMILES/mol derivatives after any mutation."""
        self._canonical_smiles: str | None = None
        self._aromatic_mol: Chem.rdchem.Mol | None = None
        self._aromatic_smiles: str | None = None

    @property
    def nb_atoms(self) -> int:
        """Return the number of atoms in the molecule."""
        nb_atoms: int = self.mol.GetNumAtoms()
        return nb_atoms

    @property
    def canonical_smiles(self) -> str:
        """
        Return the canonical and non-aromatic SMILES representation of the molecule.

        With MolecularGraph.unique_kekule, unique whatever the placement of
        the double bonds in self.mol: an aromatic molecule can be kekulized in
        several ways (depending on the mutations that built it), so its Kekulé
        form is rebuilt from the aromatic canonical SMILES (one aromatization,
        one kekulization).
        A molecule with several Kekulé forms has an alternating ring, so at
        least two ring double bonds in each form; the others have a single
        Kekulé form and keep the direct writing of self.mol (same string, no
        round trip).
        """
        if self._canonical_smiles is None:
            if (
                MolecularGraph.unique_kekule
                and len(self.mol.GetSubstructMatches(_RING_DOUBLE_BOND, maxMatches=2))
                >= 2
            ):
                self._canonical_smiles = self._unique_kekule_smiles()
            else:
                self._canonical_smiles = Chem.MolToSmiles(
                    self.mol,
                    kekuleSmiles=True,
                    canonical=True,
                )
        return self._canonical_smiles

    def _unique_kekule_smiles(self) -> str:
        """Kekulé SMILES rebuilt from the aromatic canonical SMILES.

        Also fills the aromatic SMILES cache. Falls back to the direct writing
        of self.mol if RDKit cannot aromatize or kekulize the molecule.
        """
        try:
            aromatic = Chem.Mol(self.mol)
            Chem.SanitizeMol(aromatic)
            aromatic_smiles = Chem.MolToSmiles(aromatic, canonical=True)
            kekule = Chem.MolFromSmiles(aromatic_smiles)
            Chem.Kekulize(kekule, clearAromaticFlags=True)
            smiles: str = Chem.MolToSmiles(kekule, kekuleSmiles=True, canonical=True)
        except Exception:  # pylint: disable=broad-except
            return Chem.MolToSmiles(self.mol, kekuleSmiles=True, canonical=True)
        self._aromatic_smiles = aromatic_smiles
        return smiles

    @property
    def aromatic_mol(self) -> Chem.rdchem.Mol:
        """Return a sanitized (aromatic) RDKit mol of the molecule.

        Parsed once from the canonical SMILES and cached; self.mol stays the
        kekulized working graph.
        """
        if self._aromatic_mol is None:
            self._aromatic_mol = Chem.MolFromSmiles(self.canonical_smiles)
        return self._aromatic_mol

    @property
    def aromatic_canonical_smiles(self) -> str:
        """Return the aromatic canonical SMILES representation of the molecule."""
        if self._aromatic_smiles is None:
            self._aromatic_smiles = Chem.MolToSmiles(
                self.aromatic_mol,
                canonical=True,
            )
        return self._aromatic_smiles

    @property
    def adjacency_matrix(self) -> list[list[int]]:
        """Return the adjacency matrix of the molecule."""
        adjacency_matrix: list[list[int]] = Chem.GetAdjacencyMatrix(self.mol)
        return adjacency_matrix

    @property
    def multigraph_adjacency_matrix(self) -> list[list[int]]:
        """Return the adjacency matrix of the molecule."""
        adjacency_matrix: list[list[int]] = Chem.GetAdjacencyMatrix(
            self.mol, useBO=True
        )
        return adjacency_matrix

    @property
    def bonds(self) -> list[tuple[int, int]]:
        """Return the list of bonds in the molecule."""
        return [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in self.mol.GetBonds()
        ]

    @property
    def bridge_bonds(self) -> list[tuple[int, int]]:
        """Return a list of the pairs of atoms in bridge bonds."""
        return list(nx.bridges(nx.Graph(self.adjacency_matrix)))

    @property
    def bridge_bonds_matrix(self) -> list[list[bool]]:
        """Return a boolean matrix of size (nb_atoms, nb_atoms)
        representing whether bonds of the molecule are bridge bonds.
        """
        matrix = [[False] * self.nb_atoms for _ in range(self.nb_atoms)]

        for atom1, atom2 in nx.bridges(nx.Graph(self.adjacency_matrix)):
            matrix[atom1][atom2] = True
            matrix[atom2][atom1] = True

        return matrix

    @property
    def atoms(self) -> list[str]:
        """Return the list of atoms in the molecule."""
        return [atom.GetSymbol() for atom in self.mol.GetAtoms()]

    @property
    def formal_charges(self) -> list[int]:
        """Return the formal charge of each atom in the molecule.

        Returns:
            list[int]: formal charge of each atom in the molecule
        """
        return [
            self.mol.GetAtomWithIdx(atom_idx).GetFormalCharge()
            for atom_idx in range(self.nb_atoms)
        ]

    @property
    def explicit_valences(self) -> list[int]:
        """Return the explicit valence of each atom in the molecule."""
        return [self._explicit_valence(atom_idx) for atom_idx in range(self.nb_atoms)]

    @property
    def implicit_valences(self) -> list[int]:
        """Return the implicit valence of each atom in the molecule."""
        return [self._implicit_valence(atom_idx) for atom_idx in range(self.nb_atoms)]

    def atom_charged_or_radical(self, atom_idx: int) -> bool:
        """Return the true if the atom at the given index is charged or radical."""
        atom = self.mol.GetAtomWithIdx(atom_idx)
        is_radical: bool = atom.GetNumRadicalElectrons() != 0
        is_charged: bool = atom.GetFormalCharge() != 0
        return is_radical or is_charged

    def atom_degree(self, atom_idx: int, as_multigraph: bool) -> int:
        """Return the degree of the atom at the given index.

        Args:
            atom_idx (int):
                atom index
            as_multigraph (bool, optional):
                if True, the degree is the number of bonds (including multiple
                bonds) connected to the atom.
                If False, the degree is the number of atoms connected to the atom.

        Returns:
            int: degree of the atom at the given index
        """
        atom = self.mol.GetAtomWithIdx(atom_idx)
        if as_multigraph:
            atom.UpdatePropertyCache()
            explicit_valence: int = atom.GetExplicitValence()
            return explicit_valence
        degree: int = atom.GetDegree()
        return degree

    def update_representation(self) -> None:
        """Update internal RDKit representation of the molecular graph.
        This method should be called after each action on the molecular graph.
        """
        self._clear_cached_derivatives()

        # sanitize molecule if needed
        # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
        if self.sanitize:
            Chem.SanitizeMol(self.mol)
            # clear computed properties
            self.mol.ClearComputedProps()

        # kekulize the molecular graph
        # https://chemistry.stackexchange.com/questions/116498/what-is-kekulization-in-rdkit
        Chem.Kekulize(self.mol, clearAromaticFlags=True)

        # set all explicit hydrogens to implicit
        # if its not a radical or charged atom
        for atom in self.mol.GetAtoms():
            if atom.GetFormalCharge() == 0 and atom.GetNumRadicalElectrons() == 0:
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)

        # Updating the property cache of atoms
        for atom in self.mol.GetAtoms():
            atom.UpdatePropertyCache()

        # update RDKit representation
        self.mol.UpdatePropertyCache()
        Chem.FastFindRings(self.mol)

    def atom_type(self, atom_idx: int) -> str:
        """Return the atom type of the atom at the given index.

        Args:
            atom_idx (int):
                atom index

        Returns:
            str: atom type of the atom at the given index
        """
        symbol: str = self.mol.GetAtomWithIdx(atom_idx).GetSymbol()
        return symbol

    def bond_order(self, atom1_idx: int, atom2_idx: int) -> int:
        """Return the bond type between the two atoms.

        Args:
            atom1_idx (int):
                index of the first atom
            atom2_idx (int):
                index of the second atom

        Returns:
            int: bond type between the two atoms
        """
        bond: Chem.Bond | None = self.mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is None:
            return 0
        return int(bond.GetBondType())

    def atoms_bonded_to(self, atom_idx: int) -> list[int]:
        """Return the list of atoms bonded to the atom at the given index.

        Args:
            atom_idx (int):
                atom index

        Returns:
            list[int]: list of atoms bonded to the atom at the given index
        """
        return [
            bond.GetOtherAtomIdx(atom_idx)
            for bond in self.mol.GetAtomWithIdx(atom_idx).GetBonds()
        ]

    def _explicit_valence(self, atom_idx: int) -> int:
        """Return the explicit valence of the atom at the given index.

        Args:
            atom_idx (int):
                atom index

        Returns:
            int: explicit valence of the atom at the given index
        """
        atom = self.mol.GetAtomWithIdx(atom_idx)
        atom.UpdatePropertyCache()
        valence: int = atom.GetExplicitValence()
        return valence

    def _implicit_valence(self, atom_idx: int) -> int:
        """Return the implicit valence of the atom at the given index.

        Args:
            atom_idx (int):
                atom index

        Returns:
            int: implicit valence of the atom at the given index
        """
        atom = self.mol.GetAtomWithIdx(atom_idx)
        atom.UpdatePropertyCache()
        valence: int = atom.GetImplicitValence()
        if atom.GetSymbol() == "S":
            total_valence: int = atom.GetTotalValence()
            if total_valence in (2, 4):
                valence += 2
        return valence

    ############################################################
    #                         ACTIONS                          #
    ############################################################

    def add_atom(self, atom_type: str) -> None:
        """Add an atom to the molecular graph.

        Args:
            atom (str):
                atom to add to the molecular graph.
        """
        atom = Chem.Atom(atom_type)
        atom.SetBoolProp("mutability", True)
        self.mol.AddAtom(atom)
        self._clear_cached_derivatives()

    def remove_atom(self, atom_idx: int) -> None:
        """Removing the atom at the given index from the molecular graph.

        Args:
            atom_idx (int):
                atom index
        """
        self.mol.RemoveAtom(atom_idx)
        self._clear_cached_derivatives()

    def replace_atom(self, atom_idx: int, atom_type: str) -> None:
        """Replace the atom at the given index with the given atom type.

        Args:
            atom_idx (int):
                atom index
            atom_type (str):
                atom to replace the atom at the given index.
        """
        # Changing atomic number
        self.mol.GetAtomWithIdx(atom_idx).SetAtomicNum(
            Chem.GetPeriodicTable().GetAtomicNumber(atom_type)
        )

        # Setting formal charge to 0
        self.mol.GetAtomWithIdx(atom_idx).SetFormalCharge(0)
        self._clear_cached_derivatives()

    def set_bond(self, atom1_idx: int, atom2_idx: int, bond_type_num: int) -> None:
        """Set the bond between the two atoms to the given bond type.

        Args:
            atom1_idx (int):
                index of the first atom
            atom2_idx (int):
                index of the second atom
            bond_type_num (int):
                bond type
        """
        # Extracting current bond between given atoms
        bond: Chem.Bond | None = self.mol.GetBondBetweenAtoms(
            int(atom1_idx), int(atom2_idx)
        )

        bond_type = None
        if bond_type_num == 0:
            bond_type = None
        elif bond_type_num == 1:
            bond_type = Chem.BondType.SINGLE
        elif bond_type_num == 2:
            bond_type = Chem.BondType.DOUBLE
        elif bond_type_num == 3:
            bond_type = Chem.BondType.TRIPLE
        else:
            raise ValueError(f"Unknown bond type {bond_type_num} (use 0, 1, 2 or 3)")

        if bond is None:
            # Adding a new bond
            self.mol.AddBond(atom1_idx, atom2_idx, bond_type)
        elif bond_type is None:
            # Removing the bond
            self.mol.RemoveBond(atom1_idx, atom2_idx)
        else:
            # Changing the bond type
            bond.SetBondType(bond_type)
        self._clear_cached_derivatives()

    def add_group(self, atom_to_link: int, smiles: str, added_group_atom: int) -> None:
        """Add a group to the molecular graph.

        Args:
            atom_to_link (int):
                index of the atom to link the group to
            smiles (str):
                SMILES representation of the group to add
            added_group_atom (int):
                index of the atom in the group to link to the atom_to_link
        """
        group = Chem.MolFromSmiles(smiles)
        mol = Chem.EditableMol(Chem.CombineMols(self.mol, group))

        # Adding the bond between the group and the molecule
        mol.AddBond(
            atom_to_link, self.nb_atoms + added_group_atom, Chem.BondType.SINGLE
        )
        self.mol = Chem.rdchem.RWMol(mol.GetMol())

        # Must be done before updating the representation
        # to avoid error with kekulization
        # Updating the property cache of atoms
        for atom in self.mol.GetAtoms():
            atom.UpdatePropertyCache()
        # update RDKit representation
        self.mol.UpdatePropertyCache()
        self._clear_cached_derivatives()

    def change_smiles(self, smiles: str) -> None:
        """Change the SMILES representation of the molecular graph.

        Args:
            smiles (str):
                new SMILES representation of the molecular graph.
        """
        self.mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
        self._clear_cached_derivatives()
