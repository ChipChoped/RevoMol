"""
Counting the proportion of bits in the ECFP4 fingerprint that never appear in the ChemBL.
Based on the work of Patrick Walters (https://github.com/PatWalters/silly_walks)

MIT License

Copyright (c) 2020 Patrick Walters

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import json

import rdkit.Chem
from rdkit.Chem import AllChem

from evomol import default_parameters as dp
from evomol.evaluation import ecfp4
from evomol.representation import Molecule
from evomol.evaluation.evaluation import Function


def silly_walks(molecule: Molecule, radius: int=2):
    """
    Calculate the silliness of a molecule (its non-realism)

    Args:
        molecule: Molecule to evaluate
        radius: Radius for the ECFP (ECFP4 by default)

    Returns:
        float: Sillywalk score
    """
    if molecule:
        # smiles =
        molecule = rdkit.Chem.rdmolfiles.MolFromSmiles(molecule.id_representation.smiles)
        fp = AllChem.GetMorganFingerprint(molecule, radius=radius)
        on_bits = fp.GetNonzeroElements().keys()

        with open(os.path.join("external_data", "complete_ChEMBL_ZINC_union_ecfp4_dict.json"), "r") as f:
            ecfp4_dict: list[str] = json.load(f)

        silly_bits: list = [x for x in [ecfp4_dict.get(str(x)) for x in on_bits] if x is None]
        score: float = len(silly_bits) / len(on_bits) if len(on_bits) > 0 else 0

    else:
        score = 1
    return score, [score]

Silly_walk = Function("Silly_walks", silly_walks)
