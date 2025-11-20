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

from evomol.evaluation import UnknownECFP
from evomol.evaluation.evaluation import Function
from evomol.evaluation.unknown_ecfp import list_ecfp
from evomol.representation import Molecule

def silly_walks(molecule: Molecule) -> float:
    """
    Calculate the silliness of a molecule (its non-realism)

    Args:
        molecule: Molecule to evaluate
        radius: Radius for the ECFP (ECFP4 by default)

    Returns:
        float: Sillywalk score
    """
    if molecule:
        unknown_ecfp: UnknownECFP = UnknownECFP()
        return unknown_ecfp.evaluate(molecule) / len(list_ecfp(molecule))
    else:
        return 1


Silly_Walks = Function("Silly_Walks", silly_walks)
