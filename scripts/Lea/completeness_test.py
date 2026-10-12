"""
Test d'invariance de complétude (Phase 0).

Question : l'espace d'action PRUNÉ (InsertCarbonMG retiré, avoid_sp3_on_sp3=True)
atteint-il EXACTEMENT le même ensemble de molécules valides que l'espace COMPLET
(toutes les actions, avoid_sp3_on_sp3=False) ?

On énumère À ÉPUISEMENT un petit espace depuis un SMILES de départ, une fois avec
chaque espace d'action, puis on compare les ENSEMBLES de valides.

Gate de validité par défaut : HuckelCompatible + HeteroatomChainFilter (le pruning
est justifié PAR Hückel). --with-chembl-zinc ajoute le filtre de réalisme (lent,
fragmente l'espace ; pour information).

Code retour 0 si les ensembles sont identiques, 1 sinon.

Usage :
    python scripts/Lea/completeness_test.py                       # C=C, 4 atomes
    python scripts/Lea/completeness_test.py --start C --max-atoms 5
    python scripts/Lea/completeness_test.py --with-chembl-zinc
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
os.chdir(ROOT)  # les filtres ChEMBL/ZINC lisent external_data/ en relatif

# pylint: disable=wrong-import-position
from rdkit import RDLogger

from evomol import default_parameters as dp
import huckel_filters as hf
from evomol.action import molecular_graph as mg
from evomol.representation import MolecularGraph
from evomol.search.enumeration import exhaustive_bfs

RDLogger.DisableLog("rdApp.*")


def _run_one(pruned: bool, args: argparse.Namespace) -> tuple[set[str], int, float]:
    """Énumération exhaustive pour un espace d'action (FULL ou PRUNED)."""
    dp.setup_default_parameters(
        accepted_atoms=hf.HUCKEL_ACCEPTED_ATOMS, max_heavy_atoms=args.max_atoms
    )
    dp.setup_default_action_space(insert_carbon=not pruned, avoid_sp3_on_sp3=pruned)
    filters = hf.setup_huckel_filters(
        realism="chembl_zinc" if args.with_chembl_zinc else None,
        max_ring_n=args.max_ring_n,
    )
    print(f"\n=== {'PRUNED' if pruned else 'FULL'} ===")
    print(
        f"  actions={[a.__name__ for a in MolecularGraph.action_space]} "
        f"avoid_bond_breaking={mg.ChangeBondMG.avoid_bond_breaking} "
        f"avoid_sp3_on_sp3={mg.AddAtomMG.avoid_sp3_on_sp3}"
    )
    print(f"  filtres : {[f.name for f in filters]}")

    t0 = time.time()

    def _progress(nb_valid: int, nb_seen: int) -> None:
        print(f"    ... {nb_valid:>6} valides / {nb_seen:>6} vus", end="\r")

    valid, seen = exhaustive_bfs(args.start, filters, args.max_atoms, on_progress=_progress)
    elapsed = time.time() - t0
    print(f"  -> {len(valid):>6} valides / {seen:>6} vus   (épuisé en {elapsed:.1f}s)          ")
    return valid, seen, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="C=C", help="SMILES de départ")
    parser.add_argument("--max-atoms", type=int, default=4, help="plafond d'atomes lourds (5 = plusieurs minutes)")
    parser.add_argument("--max-ring-n", type=int, default=3)
    parser.add_argument("--with-chembl-zinc", action="store_true", help="ajouter le filtre ChEMBL/ZINC (lent)")
    parser.add_argument("--sample", type=int, default=20, help="nb de SMILES divergents affichés en cas d'écart")
    args = parser.parse_args()

    print("=" * 70)
    print("TEST D'INVARIANCE DE COMPLÉTUDE (FULL vs PRUNED)")
    print(
        f"départ={args.start!r}  max_atoms={args.max_atoms}  "
        f"gate={'Huckel+ChEMBL/ZINC+Hetero' if args.with_chembl_zinc else 'Huckel+Hetero'}"
    )
    print("=" * 70)

    valid_full, seen_full, t_full = _run_one(pruned=False, args=args)
    valid_pruned, seen_pruned, t_pruned = _run_one(pruned=True, args=args)

    lost = valid_full - valid_pruned
    gained = valid_pruned - valid_full

    print("\n" + "=" * 70)
    print("RÉSULTAT")
    print("=" * 70)
    print(f"  |valides FULL|   = {len(valid_full)}   ({seen_full} vus, {t_full:.1f}s)")
    print(f"  |valides PRUNED| = {len(valid_pruned)}   ({seen_pruned} vus, {t_pruned:.1f}s)")
    print(f"  perdues par PRUNED : {len(lost)}")
    print(f"  gagnées par PRUNED : {len(gained)}")

    if not lost and not gained:
        print("\n  ✅ INVARIANCE VÉRIFIÉE : même ensemble de molécules valides.")
        return 0

    print("\n  ❌ INVARIANCE VIOLÉE : le pruning change l'ensemble atteint.")
    for title, smiles_set, sign in [("PERDUES", lost, "-"), ("GAGNÉES", gained, "+")]:
        if smiles_set:
            print(f"\n  Molécules {title} par le pruning (échantillon {min(args.sample, len(smiles_set))}) :")
            for smi in sorted(smiles_set)[: args.sample]:
                print(f"    {sign} {smi}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
