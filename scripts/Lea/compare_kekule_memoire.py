"""
Compare l'algorithme évolutionnaire avec / sans la forme kékulé unique et
avec / sans la mémoire des SMILES invalides (plan 2 × 2, une course par
processus, durée fixe).

Usage :
    python scripts/Lea/compare_kekule_memoire.py              # les 4 courses en parallèle
    python scripts/Lea/compare_kekule_memoire.py --minutes 10 --seed 42
    # fitness par désirabilité (scripts/Lea/fitness_functions.py) + taille ~70 atomes
    python scripts/Lea/compare_kekule_memoire.py --name grandes_molecules --desirability --size-center 70

Résultats : output/compare_kekule_memoire/<name>/<course>/ (populations,
run_summary, journal DEBUG propre à la course) et .../<name>/resume.json.

Fitness :
  - par défaut, somme brute de evolutionary_algorithm.py avec FITNESS_GOALS ;
  - avec --desirability, build_fitness de fitness_functions.py (scores 0–1,
    moyenne pondérée) avec DESIRABILITY_GOALS, plus, si --size-center est
    donné, Gaussian(size_center, size_sigma) sur n_heavy_atoms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Process

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "Lea"))
os.chdir(ROOT)

OUT = os.path.join(ROOT, "output", "compare_kekule_memoire")

# objectif difficile : gap le plus petit ET HOMO proche de 0.4 (poids 3)
FITNESS_GOALS = [
    ("huckel_gap_lambda", "min", 1.0, None),
    ("huckel_homo_lambda", "target", 3.0, 0.4),
]

# même objectif en courbes de désirabilité : gap petit, HOMO proche de 0.4 (poids 3)
DESIRABILITY_GOALS = [
    ("huckel_gap_lambda", ("Sigmoid", 0.3, 15.0, "down"), 1.0),
    ("huckel_homo_lambda", ("Gaussian", 0.4, 0.15), 3.0),
]

RUNS = {
    "kekule_ancien__sans_memoire": (False, False),
    "kekule_ancien__avec_memoire": (False, True),
    "kekule_unique__sans_memoire": (True, False),
    "kekule_unique__avec_memoire": (True, True),
}


def _run(name: str, unique_kekule: bool, memory: bool, args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from rdkit import Chem, RDLogger

    from evomol.representation import MolecularGraph

    RDLogger.DisableLog("rdApp.*")
    minutes, seed = args.minutes, args.seed
    run_dir = os.path.join(OUT, args.name, name)
    os.makedirs(run_dir, exist_ok=True)

    import evolutionary_algorithm as ea_script

    # journal DEBUG propre à la course (4 processus ne peuvent pas partager
    # le même fichier tournant)
    init_logger = ea_script.init_logger

    def _logger(config_file, output_file):  # type: ignore[no-untyped-def]
        return init_logger(config_file=config_file, output_file=os.path.join(run_dir, "evomol.log"))

    ea_script.init_logger = _logger

    goals: list[tuple] = list(FITNESS_GOALS)
    if args.desirability:
        import fitness_functions as ff

        curves = {"Sigmoid": ff.Sigmoid, "Gaussian": ff.Gaussian}
        goals = [(n, curves[c[0]](*c[1:]), w) for n, c, w in DESIRABILITY_GOALS]
        if args.homo_weight is not None:
            goals = [(n, c, args.homo_weight if n == "huckel_homo_lambda" else w) for n, c, w in goals]
        if args.size_center is not None:
            goals.append(
                ("n_heavy_atoms", ff.Gaussian(args.size_center, args.size_sigma), args.size_weight)
            )
        ea_script._build_fitness = lambda g: ff.build_fitness(g, "sum")  # pylint: disable=protected-access
    elif args.size_center is not None:
        raise SystemExit("--size-center demande --desirability")

    trajectory: list[dict] = []
    t0 = time.perf_counter()

    def _callback(population, generation):  # type: ignore[no-untyped-def]
        fitness = [m.value("fitness") for m in population]
        sizes = [m.get_representation(MolecularGraph).nb_atoms for m in population]
        trajectory.append(
            {"generation": generation, "t": time.perf_counter() - t0, "best": max(fitness),
             "mean": sum(fitness) / len(fitness), "mean_atoms": sum(sizes) / len(sizes),
             "max_atoms": max(sizes)}
        )

    ea = ea_script.main(
        start_smiles="c1ccccc1",
        huckel=True,
        realism=None if args.realism == "none" else args.realism,
        max_sp3=None if args.max_sp3 < 0 else args.max_sp3,
        fitness_goals=goals,
        max_generations=10**9,
        max_time=minutes * 60,
        max_heavy_atoms=args.max_heavy_atoms,
        unique_kekule=unique_kekule,
        remember_invalid=memory,
        random_seed=seed,
        generation_callback=_callback,
        output_dir=run_dir,
        save_n_generations=0,
    )
    elapsed = time.perf_counter() - t0
    ns = ea.neighborhood_strategy
    smiles = [str(m) for m in ea.population]
    aromatic = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles}
    best = max(ea.population, key=lambda m: m.value("fitness"))
    result = {
        "course": name,
        "kekule_unique": unique_kekule,
        "memoire": memory,
        "secondes": elapsed,
        "generations": ea.nb_generations,
        "meilleure_fitness": best.value("fitness"),
        "meilleure_molecule": str(best),
        "fitness_moyenne": sum(m.value("fitness") for m in ea.population) / len(smiles),
        "molecules_distinctes": len(aromatic),
        "atomes_moyen": sum(
            m.get_representation(MolecularGraph).nb_atoms for m in ea.population
        ) / len(smiles),
        "atomes_meilleure": best.get_representation(MolecularGraph).nb_atoms,
        "evaluations": ns.n_function_calls_total,
        "valides": ns.n_valid_candidates,
        "invalides": ns.n_invalid_candidates,
        "rejets_memoire": ns.n_known_invalid_candidates,
        "tabou": ns.n_tabu_candidates,
        "trajectoire": trajectory,
    }
    with open(os.path.join(run_dir, "resultat.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="objectif_dur", help="sous-dossier de l'expérience")
    parser.add_argument("--desirability", action="store_true",
                        help="fitness de scripts/Lea/fitness_functions.py (DESIRABILITY_GOALS)")
    parser.add_argument("--homo-weight", type=float, default=None,
                        help="poids de λ HOMO avec --desirability (défaut : celui de DESIRABILITY_GOALS, 3)")
    parser.add_argument("--size-center", type=float, default=None, help="centre de la gaussienne sur la taille")
    parser.add_argument("--size-sigma", type=float, default=25.0)
    parser.add_argument("--size-weight", type=float, default=1.0)
    parser.add_argument("--max-heavy-atoms", type=int, default=100)
    parser.add_argument("--realism", default="chembl_zinc", choices=["chembl_zinc", "chembl", "none"],
                        help="filtre de réalisme ECFP + GCF (none : désactivé)")
    parser.add_argument("--max-sp3", type=int, default=-1,
                        help="FilterSP3 : nb max d'atomes sans liaison π (-1 : désactivé)")
    args = parser.parse_args()

    os.makedirs(os.path.join(OUT, args.name), exist_ok=True)
    procs = [
        Process(target=_run, args=(name, kek, mem, args), name=name)
        for name, (kek, mem) in RUNS.items()
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    summary = []
    for name in RUNS:
        path = os.path.join(OUT, args.name, name, "resultat.json")
        if not os.path.exists(path):
            print(f"{name} : pas de résultat (échec)")
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        r.pop("trajectoire")
        summary.append(r)
        print(
            f"{name:30s} gén. {r['generations']:5d}  meilleure {r['meilleure_fitness']:.4f}  "
            f"moyenne {r['fitness_moyenne']:.4f}  distinctes {r['molecules_distinctes']:3d}  "
            f"évaluations {r['evaluations']:8d}  rejets mémoire {r['rejets_memoire']:8d}  "
            f"atomes moyen {r['atomes_moyen']:.1f}"
        )
    with open(os.path.join(OUT, args.name, "resume.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
