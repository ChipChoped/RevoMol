"""
The action AddAtom consists in adding a new atom (among the list
Molecule.accepted_atoms) to the molecular graph.
If the molecule is empty, the atom becomes the only atom in the molecule;
otherwise, it must be connected to an existing atom.
The atom can only be added if the number of atoms in the molecule is less than
Molecule.max_heavy_atoms and can only be attached to a mutable atom.
"""

from rdkit import Chem
from typing_extensions import override

from evomol.action import Action
from evomol.representation import MolecularGraph, Molecule

from .action_molecular_graph import ActionMolGraph

# Bond types that carry a π system: an atom bearing one of them is sp2/sp (not sp3).
_PI_BOND_TYPES = (
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)


def _atom_is_sp3_carbon(atom: Chem.Atom) -> bool:
    """Return True if the atom is a carbon with no π bond (sp3, type "C2").

    Such a carbon contributes no pz orbital to the Hückel system; attaching a new
    single-bonded carbon to it (which is itself sp3) makes it a non-terminal sp3
    carbon, incompatible with the Hückel model.
    """
    if atom.GetSymbol() != "C":
        return False
    return not any(bond.GetBondType() in _PI_BOND_TYPES for bond in atom.GetBonds())


class AddAtomMG(ActionMolGraph):
    """
    Add an atom to the molecular graph.
    """

    # When True, forbid adding a new sp3 carbon (single-bonded C) onto an atom
    # that is already an sp3 carbon. Prevents building non-terminal sp3 carbons,
    # which are systematically rejected by the Hückel compatibility filter.
    avoid_sp3_on_sp3: bool = False

    # When True, the new atom can also be attached by a double or triple bond
    # (e.g. =CH2, =O, #N), when both atoms have enough free valence.
    allow_multiple_bonds: bool = False

    def __init__(
        self, molecule: Molecule, index_atom: int, atom_type: str, bond_order: int = 1
    ) -> None:
        """Initialize the action.

        Args:
            molecule (Molecule): Molecule to which the atom is added
            index_atom (int): index of the atom to bond to the new atom,
            0 if the molecule is empty
            atom_type (str): type of the new atom
            bond_order (int): order of the bond to the new atom (1, 2 or 3)
        """
        super().__init__(molecule)

        # index of the atom to bond to the new atom
        self.index_atom: int = index_atom
        # type of the new atom
        self.atom_type: str = atom_type
        # order of the bond between the new atom and index_atom
        self.bond_order: int = bond_order

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AddAtomMG):
            return False
        return (
            self.molecule == other.molecule
            and self.index_atom == other.index_atom
            and self.atom_type == other.atom_type
            and self.bond_order == other.bond_order
        )

    @override
    def __hash__(self) -> int:
        return hash(self.__repr__())

    @override
    def apply_action(self, new_mol_graph: MolecularGraph) -> None:
        """Add an atom of type self.atom_type to the molecular graph.
        If the graph is not empty, a bond is created between the new atom and
        the atom at index self.index_atom with a bond order of self.bond_order.

        Args:
            new_mol_graph (MolecularGraph): Molecular graph to modify.
        """

        # adding the atom
        new_mol_graph.add_atom(self.atom_type)

        # if the molecule is not empty, a bond is created
        if new_mol_graph.nb_atoms > 1:
            # creating a bond from the last inserted atom to the existing one
            new_mol_graph.set_bond(
                new_mol_graph.nb_atoms - 1, self.index_atom, self.bond_order
            )

    def __repr__(self) -> str:
        order = "" if self.bond_order == 1 else f", {self.bond_order}"
        return (
            f"AddAtomMolGraph({self.molecule}, {self.index_atom}, {self.atom_type}{order})"
        )

    @override
    @classmethod
    def list_actions(cls, molecule: Molecule) -> list[Action]:
        """List possible actions to add an atom to the molecular graph.

        Three cases are considered:

        - if the molecule size is equal or higher than Molecule.max_heavy_atoms,
          no action is possible

        - if the molecule is empty, any atom can be added

        - otherwise, for each atom, if the implicit valence is greater than 0 an
          action can be performed for each possible atom type

        Args:
            molecule (Molecule): Molecule to which the atom is added

        Returns:
            list[Action]: list of possible actions to add an atom to the molecular graph
        """

        mol_graph: MolecularGraph = molecule.get_representation(MolecularGraph)
        assert mol_graph is not None

        # if the molecule is too big, we cannot add an atom
        if mol_graph.nb_atoms >= Molecule.max_heavy_atoms:
            return []

        # if the molecule is empty, we can add any atom
        if mol_graph.nb_atoms == 0:
            return [
                AddAtomMG(molecule, 0, atom_type)
                for atom_type in Molecule.accepted_atoms
            ]

        # otherwise, for each atom, if the implicit valence is greater than 0
        # an action can be performed for each possible atom type

        implicit_valences = mol_graph.implicit_valences

        # Pre-mutation check: mark atoms that are already sp3 carbons so we can
        # forbid attaching another sp3 carbon to them (see avoid_sp3_on_sp3).
        if cls.avoid_sp3_on_sp3:
            is_sp3_carbon = [
                _atom_is_sp3_carbon(mol_graph.mol.GetAtomWithIdx(atom_idx))
                for atom_idx in range(mol_graph.nb_atoms)
            ]
        else:
            is_sp3_carbon = [False] * mol_graph.nb_atoms

        return [
            AddAtomMG(molecule, atom_idx, atom_type, bond_order)
            for atom_idx in range(mol_graph.nb_atoms)
            if implicit_valences[atom_idx] > 0
            and not mol_graph.atom_charged_or_radical(atom_idx)
            for atom_type in Molecule.accepted_atoms
            for bond_order in (
                range(1, cls._max_bond_order(implicit_valences[atom_idx], atom_type) + 1)
                if cls.allow_multiple_bonds
                else (1,)
            )
            # forbid adding an sp3 carbon ("C" via a single bond) onto an sp3 carbon
            if not (
                cls.avoid_sp3_on_sp3
                and is_sp3_carbon[atom_idx]
                and atom_type == "C"
                and bond_order == 1
            )
        ]

    @staticmethod
    def _max_bond_order(free_valence: int, atom_type: str) -> int:
        """Highest bond order to a new atom of type atom_type: limited by the free
        valence of the atom it is attached to, the usual valence of the new atom
        (C 4, N 3, O 2, S 2, halogens 1) and 3."""
        new_atom_valence = Chem.GetPeriodicTable().GetDefaultValence(atom_type)
        return min(free_valence, new_atom_valence, 3)
