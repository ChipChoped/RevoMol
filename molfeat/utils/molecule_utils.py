import networkx as nx
from rdkit import Chem
from molfeat.Classes.MoleculeFeatures import MoleculeFeatures


def get_graph(mol):
    """
    Converts an RDKit molecule to a NetworkX graph.

    Args:
        mol (Chem.Mol): The RDKit molecule.

    Returns:
        nx.Graph: The NetworkX graph representation.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        raise ValueError("Invalid molecule provided. Cannot convert to graph.")
    Chem.Kekulize(mol)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    am = Chem.GetAdjacencyMatrix(mol, useBO=True)
    for i, atom in enumerate(atoms):
        am[i, i] = atom
    g = nx.from_numpy_array(am)
    return g


def compute_graph_edit_distance(mol1, mol2):
    """
    Computes the graph edit distance between two molecules.

    Args:
        mol1 (Chem.Mol): The first molecule.
        mol2 (Chem.Mol): The second molecule.

    Returns:
        float: The graph edit distance.
    """
    g1 = get_graph(mol1)
    g2 = get_graph(mol2)
    return nx.graph_edit_distance(
        g1, g2, edge_match=lambda a, b: a["weight"] == b["weight"]
    )


def compute_features_distance(mol1, mol2, weights=None, raw=False):
    """
    Computes the distance between two molecules based on their features.

    Args:
        mol1 (Chem.Mol): The first molecule.
        mol2 (Chem.Mol): The second molecule.

    Returns:
        float: The distance based on features.
    """

    features1 = MoleculeFeatures(mol1)
    features2 = MoleculeFeatures(mol2)
    features1.generate_features(weights=weights)
    features2.generate_features(weights=weights)
    if not features1.features or not features2.features:
        raise ValueError("Features could not be generated for one or both molecules.")

    distance_initial, _, raw_diffs_from_method = features1.evaluate_distance(
        features2, raw=True
    )
    if not raw:
        return distance_initial
    return distance_initial, raw_diffs_from_method