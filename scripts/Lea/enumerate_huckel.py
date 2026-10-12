"""
Enumeration of molecules compatible with the Hückel model,
starting from a given SMILES, with a time limit.

Usage:
    python scripts/Lea/enumerate_huckel.py "C=C" 60
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evomol import default_parameters as dp
from evomol.evaluation.evaluation import Evaluation, EvaluationError
from evomol.evaluation.huckel import HuckelCompatible
from huckel_filters import HeteroatomChainFilter
from evomol.representation import MolecularGraph, Molecule
from evomol.search.enumeration import find_neighbors


def _is_valid(mol: Molecule, filters: list[Evaluation]) -> bool:
    for f in filters:
        try:
            f.evaluate(mol)
        except EvaluationError:
            return False
    return True


def enumerate_huckel_timed(
    starting_smiles: str,
    max_seconds: float,
    depth: int = 1,
    extra_evaluations: list[Evaluation] | None = None,
) -> tuple[set[str], set[str], bool]:
    """Enumerate Hückel-compatible molecules starting from a SMILES.

    Stops when max_seconds is reached. Returns valid molecules, invalid ones,
    and whether the enumeration completed fully.

    Args:
        starting_smiles: SMILES to start from
        max_seconds: wall-clock time limit in seconds
        depth: neighbor search depth per step (default 1)
        extra_evaluations: additional filters applied after HuckelCompatible

    Returns:
        (found_valid, found_invalid, completed)
    """
    all_filters: list[Evaluation] = [HuckelCompatible()] + (extra_evaluations or [])
    deadline = time.time() + max_seconds

    to_explore: set[str] = set()
    found_valid: set[str] = set()
    found_invalid: set[str] = set()

    mol = Molecule(starting_smiles)
    canonical = mol.get_representation(MolecularGraph).canonical_smiles

    if _is_valid(mol, all_filters):
        found_valid.add(canonical)
        to_explore.add(canonical)
    else:
        found_invalid.add(canonical)

    completed = False
    while to_explore:
        if time.time() >= deadline:
            break

        # Process one molecule at a time so the timeout is checked frequently
        current = to_explore.pop()
        new_mols = find_neighbors(Molecule(current), depth)

        for new_smi in new_mols:
            if new_smi in found_valid or new_smi in found_invalid:
                continue
            new_mol = Molecule(new_smi)
            if _is_valid(new_mol, all_filters):
                to_explore.add(new_smi)
                found_valid.add(new_smi)
            else:
                found_invalid.add(new_smi)
    else:
        completed = True

    return found_valid, found_invalid, completed


def main() -> None:
    start_smiles = sys.argv[1] if len(sys.argv) > 1 else "C=C"
    max_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

    dp.setup_default_parameters(
        accepted_atoms=["C", "N", "O", "S", "F", "Cl", "Br"]
    )
    dp.setup_default_action_space()

    # ChEMBL + ZINC realism filters (GCF + ECFP4) + heteroatom chain rules
    extra_filters = dp.setup_filters("chembl_zinc") + [HeteroatomChainFilter(max_ring_n=3)]

    print(f"Starting enumeration from '{start_smiles}' (limit: {max_seconds}s)")
    t0 = time.time()

    found_valid, found_invalid, completed = enumerate_huckel_timed(
        start_smiles, max_seconds, extra_evaluations=extra_filters
    )

    elapsed = time.time() - t0
    status = "completed" if completed else "stopped (time limit)"

    print(f"\nStatus   : {status}")
    print(f"Time     : {elapsed:.1f}s")
    print(f"Valid    : {len(found_valid)}")
    print(f"Invalid  : {len(found_invalid)}")
    print(f"\nFirst 20 valid molecules:")
    for smi in sorted(found_valid)[:20]:
        print(f"  {smi}")


if __name__ == "__main__":
    main()
