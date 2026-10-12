"""
Profilage d'un run d'énumération réaliste (cProfile).

Chiffre la part RÉELLE de chaque fonction sur un run borné par un budget de
molécules, avec la config de PRODUCTION corrigée (espace pruné + filtres
Huckel → ChEMBL/ZINC → Hetero). Toutes les décisions des phases 1-2 s'appuient
sur ces chiffres.

Sortie :
  - top des fonctions par temps cumulé (cumtime) et par temps propre (tottime)
  - une synthèse "hotspots" regroupée (parsing RDKit, écriture SMILES,
    kékulisation, filtres, expansion des voisins)
  - un fichier .prof réutilisable (snakeviz output/prof/run.prof, ou py-spy)

Usage :
    python scripts/Lea/profile_run.py                      # 50k molécules, 10 atomes, filtres complets
    python scripts/Lea/profile_run.py --budget 20000       # plus court
    python scripts/Lea/profile_run.py --no-chembl-zinc     # sans les filtres de réalisme
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
os.chdir(ROOT)  # les filtres ChEMBL/ZINC lisent external_data/ en relatif

# pylint: disable=wrong-import-position
from rdkit import RDLogger

from evomol import default_parameters as dp
import huckel_filters as hf
from evomol.action import molecular_graph as mg
from evomol.representation import MolecularGraph
from evomol.search.enumeration import budget_bfs

RDLogger.DisableLog("rdApp.*")

PROF_DIR = os.path.join(ROOT, "output", "prof")

# Regroupement des hotspots : (label, prédicat sur (filename, funcname)).
# Premier prédicat qui matche gagne (break) -> ordonner du plus spécifique au
# plus général. Aligné sur les leviers de la Phase 1.
HOTSPOTS: list[tuple[str, Callable[[str, str], bool]]] = [
    ("RDKit MolFromSmiles (parse)", lambda f, n: "MolFromSmiles" in n),
    ("RDKit MolToSmiles (écriture SMILES)", lambda f, n: "MolToSmiles" in n),
    ("RDKit Kekulize", lambda f, n: "ekulize" in n),
    ("RDKit Sanitize/FastFindRings", lambda f, n: "anitize" in n or "FindRings" in n),
    ("update_representation (recalcul graphe)", lambda f, n: n == "update_representation"),
    ("RDKit itération atomes/liaisons (wrappers py)", lambda f, n: "rdkit" in f and "Chem" in f),
    ("Filtre Huckel", lambda f, n: f.endswith("huckel.py")),
    ("Filtre HeteroatomChain", lambda f, n: f.endswith("huckel_filters.py")),
    ("Filtre ECFP (ChEMBL/ZINC)", lambda f, n: f.endswith("unknown_ecfp.py")),
    ("Filtre GCF (ChEMBL/ZINC)", lambda f, n: f.endswith("generic_cyclic_features.py")),
    ("MolecularGraph __init__/__copy__/canonical", lambda f, n: f.endswith("molecular_graph.py")),
    ("Expansion voisins (enumeration/action)", lambda f, n: f.endswith("enumeration.py") or f"{os.sep}action{os.sep}" in f),
    ("Molecule.__init__ / evaluation", lambda f, n: f.endswith("molecule.py") or f.endswith("evaluation.py")),
]


def _rollup(stats: pstats.Stats, budget_seen: int) -> None:
    """Synthèse regroupée du temps propre (tottime) par hotspot."""
    total_tt = 0.0
    buckets = {label: [0.0, 0] for label, _ in HOTSPOTS}
    # stats.stats : {(file, line, func): (cc, nc, tt, ct, callers)}
    for (filename, _line, funcname), (_cc, nc, tt, _ct, _callers) in stats.stats.items():
        total_tt += tt
        for label, pred in HOTSPOTS:
            try:
                if pred(filename, funcname):
                    buckets[label][0] += tt
                    buckets[label][1] += nc
                    break
            except Exception:
                pass

    print("\n" + "=" * 78)
    print("SYNTHÈSE HOTSPOTS (temps propre / tottime)")
    print("=" * 78)
    print(f"{'hotspot':<42}{'tottime':>10}{'% total':>9}{'appels':>14}")
    print("-" * 78)
    accounted = 0.0
    for label, _ in HOTSPOTS:
        tt, nc = buckets[label]
        accounted += tt
        pct = 100 * tt / total_tt if total_tt else 0
        print(f"{label:<42}{tt:>10.2f}{pct:>8.1f}%{nc:>14,}")
    other = total_tt - accounted
    print("-" * 78)
    print(f"{'(autre / overhead Python)':<42}{other:>10.2f}{100*other/total_tt if total_tt else 0:>8.1f}%")
    print(f"{'TOTAL tottime':<42}{total_tt:>10.2f}")
    if budget_seen:
        print(f"\n  {budget_seen:,} molécules examinées -> {1e6*total_tt/budget_seen:.0f} µs/molécule (CPU profilé)")
    print(
        "\n  Note : cProfile impute le temps C de MolFromSmiles/MolToSmiles (fonctions\n"
        "  Boost, pas des builtins CPython) à leur APPELANT Python -> lignes dédiées à\n"
        "  0 ; ce temps est compté dans 'MolecularGraph __init__/__copy__/canonical'.\n"
        "  Le parsing/écriture SMILES est donc le gros de ce poste (levier Phase 1)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="C=C", help="SMILES de départ")
    parser.add_argument("--budget", type=int, default=50000, help="nb de molécules à examiner")
    parser.add_argument("--max-atoms", type=int, default=10, help="plafond d'atomes lourds")
    parser.add_argument("--max-ring-n", type=int, default=3)
    parser.add_argument(
        "--no-chembl-zinc",
        action="store_true",
        help="profiler SANS les filtres de réalisme ChEMBL/ZINC",
    )
    parser.add_argument("--top", type=int, default=30, help="nb de lignes des tops")
    args = parser.parse_args()

    use_cz = not args.no_chembl_zinc

    # Config de PRODUCTION corrigée : espace pruné + ordre Huckel→ChEMBL→Hetero.
    dp.setup_default_parameters(accepted_atoms=hf.HUCKEL_ACCEPTED_ATOMS, max_heavy_atoms=args.max_atoms)
    dp.setup_default_action_space(insert_carbon=False, avoid_sp3_on_sp3=True)
    filters = hf.setup_huckel_filters(realism="chembl_zinc" if use_cz else None, max_ring_n=args.max_ring_n)

    print("=" * 78)
    print("PROFILAGE — config de production (pruned + filtres corrigés)")
    print("=" * 78)
    print(
        f"  actions={[a.__name__ for a in MolecularGraph.action_space]} "
        f"avoid_bond_breaking={mg.ChangeBondMG.avoid_bond_breaking} "
        f"avoid_sp3_on_sp3={mg.AddAtomMG.avoid_sp3_on_sp3}"
    )
    print(f"  filtres : {[f.name for f in filters]}")
    print(f"  départ={args.start!r}  budget={args.budget:,}  max_atoms={args.max_atoms}")
    print("  (cProfile ajoute un surcoût ; les RATIOS entre fonctions restent valides)\n")

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    result = budget_bfs(args.start, filters, args.budget)
    profiler.disable()
    wall = time.time() - t0

    print(
        f"Run terminé : {result['n_valid']:,} valides / {result['n_invalid']:,} invalides "
        f"/ {result['seen']:,} vus   en {wall:.1f}s (avec surcoût profilage)"
    )

    stats = pstats.Stats(profiler)
    stats.strip_dirs = False  # garder les chemins pour distinguer les modules

    print("\n" + "=" * 78)
    print(f"TOP {args.top} PAR TEMPS CUMULÉ (cumtime)")
    print("=" * 78)
    stats.sort_stats("cumulative").print_stats(args.top)

    print("\n" + "=" * 78)
    print(f"TOP {args.top} PAR TEMPS PROPRE (tottime)")
    print("=" * 78)
    stats.sort_stats("tottime").print_stats(args.top)

    _rollup(stats, result["seen"])

    os.makedirs(PROF_DIR, exist_ok=True)
    tag = "full" if use_cz else "nocz"
    out = os.path.join(PROF_DIR, f"run_{tag}_{args.budget}.prof")
    stats.dump_stats(out)
    print(f"\nProfil binaire sauvegardé : {out}")
    print(f"  -> visualiser : snakeviz {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
