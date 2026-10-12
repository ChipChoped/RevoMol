"""
Profilage (cProfile) d'une marche aléatoire — scripts/Lea/random_walk_core.py.

Mêmes options que scripts/Lea/ancien/random_walk.py (sauf --burn-in, 0 par défaut ici : le
burn-in, s'il est demandé, n'est pas profilé). Lancé en sous-processus par
notebooks/random_walk_profiling.ipynb : les hooks ipykernel/debugpy empêchent
cProfile.enable() dans le kernel.

Usage :
    python scripts/Lea/profile_random_walk.py --steps 300
    python scripts/Lea/profile_random_walk.py --steps 100 --count-valid-neighbors
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ancien"))  # random_walk

# pylint: disable=wrong-import-position
from random_walk import ROOT, add_walk_arguments, burn_in, setup_from_args  # chdir vers ROOT
from profile_run import HOTSPOTS

from random_walk_core import random_walk

PROF_DIR = os.path.join(ROOT, "output", "prof")


def rollup(stats: pstats.Stats) -> list[dict]:
    """Temps propre (tottime) regroupé par poste, comme profile_run.py."""
    buckets = {label: [0.0, 0] for label, _ in HOTSPOTS}
    total = 0.0
    for (filename, _line, funcname), (_cc, nc, tt, _ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        total += tt
        for label, pred in HOTSPOTS:
            if pred(filename, funcname):
                buckets[label][0] += tt
                buckets[label][1] += nc
                break
    rows = [{"poste": k, "tottime": v[0], "appels": v[1]} for k, v in buckets.items()]
    rows.append({"poste": "autre / overhead Python", "tottime": total - sum(v[0] for v in buckets.values()), "appels": 0})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_walk_arguments(parser)
    parser.set_defaults(burn_in=0, steps=300)
    parser.add_argument("--json", action="store_true", help="résumé JSON sur stdout (pour le notebook)")
    args = parser.parse_args()

    filters, walk_kwargs = setup_from_args(args)
    start = burn_in(args.start, args.burn_in, filters, args.seed, walk_kwargs)

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    rows = random_walk(start, args.steps, filters, seed=args.seed, **walk_kwargs)
    profiler.disable()
    wall = time.time() - t0

    os.makedirs(PROF_DIR, exist_ok=True)
    mode = "eager" if args.count_valid_neighbors else "lazy"
    out = os.path.join(PROF_DIR, f"walk_{mode}_{args.steps}.prof")
    stats = pstats.Stats(profiler)
    stats.dump_stats(out)

    summary = {
        "prof": out,
        "wall": wall,
        "steps": args.steps,
        "n_filter_calls": sum(r["n_filter_calls"] for r in rows),
        "n_neighbors": sum(r["neighborhood_size"] or 0 for r in rows),
        "rollup": rollup(stats),
    }
    if args.json:
        print(json.dumps(summary))
    else:
        print(f"{args.steps} pas en {wall:.1f}s (avec surcoût cProfile) -> {out}")
        for r in sorted(summary["rollup"], key=lambda r: -r["tottime"]):
            print(f"  {r['poste']:<46}{r['tottime']:>8.2f}s{r['appels']:>12,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
