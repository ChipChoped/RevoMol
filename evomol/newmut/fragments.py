"""
Fragment vocabulary of the composite actions (newmut).

A fragment is a Kekulé SMILES whose attachment points are dummy atoms
[*:1] (graft) or [*:1] and [*:2] (annelation): the fragment atom bonded to
[*:k] is bonded to the k-th anchor of the molecule by a single bond.
The dictionaries below are only the defaults: the vocabulary can be given to
setup_newmut_action_space (graft_fragments, annelation_fragments).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem

# name -> Kekulé SMILES with one attachment point [*:1]
GRAFT_FRAGMENTS: dict[str, str] = {
    "vinyl": "C=C[*:1]",
    "ethynyl": "C#C[*:1]",
    "phenyl": "C1=CC=C(C=C1)[*:1]",
    "thien-2-yl": "C1=CSC(=C1)[*:1]",
    "dicyanovinyl": "N#CC(C#N)=C[*:1]",
    "dimethylamino": "CN(C)[*:1]",
    "methoxy": "CO[*:1]",
    "cyano": "N#C[*:1]",
}

# name -> SMILES with two attachment points [*:1] and [*:2], put on the two
# atoms of a bond
ANNELATION_FRAGMENTS: dict[str, str] = {
    "benzo": "[*:1]C=CC=C[*:2]",
    "thieno": "[*:1]C=CS[*:2]",
}


@dataclass(frozen=True)
class Fragment:
    """A fragment ready to be added to a molecular graph.

    smiles: Kekulé SMILES of the fragment without its attachment points, in
        the atom order used by MolecularGraph.add_group;
    attach: index (in smiles) of the atom bonded to [*:1], [*:2], ...;
    nb_atoms: number of heavy atoms of the fragment;
    elements: elements of the fragment;
    symmetric: True if swapping [*:1] and [*:2] gives the same fragment.
    """

    smiles: str
    attach: tuple[int, ...]
    nb_atoms: int
    elements: frozenset[str]
    symmetric: bool


@lru_cache(maxsize=None)
def parse_fragment(smiles: str) -> Fragment:
    """Read a fragment SMILES with attachment points [*:1], [*:2], ...

    Raises:
        ValueError: SMILES not readable, or an attachment point that is not
            bonded to exactly one atom, or attachment numbers that are not
            1, 2, ..., k.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Fragment SMILES not readable: {smiles}")

    attach_by_label: dict[int, int] = {}
    dummies = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        if len(neighbors) != 1:
            raise ValueError(f"Attachment point bonded to {len(neighbors)} atoms: {smiles}")
        attach_by_label[atom.GetAtomMapNum()] = neighbors[0]
        dummies.append(atom.GetIdx())
    labels = sorted(attach_by_label)
    if labels != list(range(1, len(labels) + 1)):
        raise ValueError(f"Attachment points must be [*:1] ... [*:k]: {smiles}")

    # remove the attachment points (RemoveAtom renumbers the following atoms)
    rwmol = Chem.RWMol(mol)
    for idx in sorted(dummies, reverse=True):
        rwmol.RemoveAtom(idx)
    kept = [i for i in range(mol.GetNumAtoms()) if i not in dummies]
    new_index = {old: new for new, old in enumerate(kept)}
    core = rwmol.GetMol()
    Chem.SanitizeMol(core)
    Chem.Kekulize(core, clearAromaticFlags=True)

    # SMILES in a known atom order: MolFromSmiles(fragment_smiles) numbers its
    # atoms in output order
    fragment_smiles = Chem.MolToSmiles(core, kekuleSmiles=True, canonical=False)
    output_order = list(core.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
    attach = tuple(
        output_order.index(new_index[attach_by_label[label]]) for label in labels
    )

    symmetric = False
    if len(labels) == 2:
        swapped = smiles.replace("[*:1]", "[*:9]").replace("[*:2]", "[*:1]")
        swapped = swapped.replace("[*:9]", "[*:2]")
        symmetric = Chem.CanonSmiles(swapped) == Chem.CanonSmiles(smiles)

    return Fragment(
        smiles=fragment_smiles,
        attach=attach,
        nb_atoms=core.GetNumAtoms(),
        elements=frozenset(atom.GetSymbol() for atom in core.GetAtoms()),
        symmetric=symmetric,
    )
