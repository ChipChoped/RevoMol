"""
Cost of the matrix rules (newmut, milestone 2).

Runs the same short QED evolutionary algorithm (max_heavy_atoms = 38, fixed
seed) four times:
- default action space (setup_default_action_space);
- matrix rules without the cache of filtered actions;
- matrix rules with the cache;
- matrix rules tested one at a time (NewMutNeighborhoodStrategy, lazy_rules).
Then times list_actions of each action type on a large molecule, without
cache.

Usage:
    python scripts/newmut/bench_rules.py [--generations 10] [--size 20]
                                         [--replacements 10] [--seed 0]
"""

import argparse
import os
import random
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_ROOT)

# pylint: disable=wrong-import-position
from evomol import default_parameters as dp
from evomol import evaluation as evaluator
from evomol.newmut import rule_filtered as rf
from evomol.newmut.default_parameters import setup_newmut_action_space
from evomol.newmut.neighborhood_strategy import NewMutNeighborhoodStrategy
from evomol.representation import MolecularGraph, Molecule
from evomol.search.evolutionary_algorithm import (
    EvolutionaryAlgorithm,
    ParametersEvolutionaryAlgorithm,
)
from evomol.search.neighborhood_strategy import RandomNeighborhoodStrategy
from evomol.search.parameters import search_parameters
from evomol.search.selection_strategy import BestSelection

INITIAL_POPULATION = [
    "c1ccccc1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "c1ccncc1",
    "Nc1ccccc1",
    "CC(=O)Nc1ccccc1",
    "OC(=O)c1ccccc1",
    "c1ccsc1",
    "c1ccc2ccccc2c1",
    "C=Cc1ccccc1",
]

LARGE_MOLECULE = "CC(=O)Nc1ccc(cc1)C(=O)c1ccc2cc(OC)ccc2c1C=CC(=O)Nc1ccncc1"


def run(mode: str, args: argparse.Namespace) -> dict[str, float]:
    """Run the evolutionary algorithm in one mode and return its counters."""
    if mode == "default":
        dp.setup_default_action_space()
    else:
        setup_newmut_action_space(matrix_rules=True)
        rf.set_cache_size(0 if mode == "rules_no_cache" else 10_000)

    random.seed(args.seed)
    search_parameters.fitness_criteria = "QED"
    evaluations = [evaluator.QED]
    population = [Molecule(smiles) for smiles in INITIAL_POPULATION]
    for molecule in population:
        evaluator.QED.evaluate(molecule)

    neighborhood = (
        NewMutNeighborhoodStrategy(depth=2, lazy_rules=True)
        if mode == "rules_lazy"
        else RandomNeighborhoodStrategy(depth=2)
    )
    algorithm = EvolutionaryAlgorithm(
        population=population,
        parameters=ParametersEvolutionaryAlgorithm(
            size=args.size,
            max_generations=args.generations,
            max_time=10**6,
            nb_replacements=args.replacements,
            save_final_population=False,
        ),
        selection_strategy=BestSelection(),
        evaluations=evaluations,
        neighborhood_strategy=neighborhood,
    )
    start = time.perf_counter()
    algorithm.run()
    elapsed = time.perf_counter() - start
    nb_candidates = (
        neighborhood.n_function_calls_total
        + neighborhood.n_tabu_candidates
        + neighborhood.n_action_errors
    )
    return {
        "time_s": elapsed,
        "generations": algorithm.nb_generations,
        "candidates": nb_candidates,
        "ms_per_candidate": 1000 * elapsed / max(1, nb_candidates),
        "action_errors": neighborhood.n_action_errors,
        "best_qed": algorithm.population[0].value("QED"),
    }


def time_list_actions(smiles: str, repeat: int = 3) -> None:
    """Time list_actions of each action type, with and without the rules."""
    molecule = Molecule(smiles)
    print(
        f"\nlist_actions sur {smiles} "
        f"({molecule.get_representation(MolecularGraph).nb_atoms} atomes), sans cache :"
    )
    rf.set_cache_size(0)
    dp.setup_default_action_space()
    default_space = list(MolecularGraph.action_space)
    setup_newmut_action_space(matrix_rules=True)
    rules_space = list(MolecularGraph.action_space)
    print(f"{'action':<18}{'nb':>6}{'ms':>9}{'nb règles':>11}{'ms règles':>11}{'x':>7}")
    for plain, filtered in zip(default_space, rules_space):
        times = []
        sizes = []
        for action_class in (plain, filtered):
            start = time.perf_counter()
            for _ in range(repeat):
                actions = action_class.list_actions(molecule)
            times.append(1000 * (time.perf_counter() - start) / repeat)
            sizes.append(len(actions))
        ratio = times[1] / times[0] if times[0] > 0 else float("nan")
        print(
            f"{plain.class_name():<18}{sizes[0]:>6}{times[0]:>9.2f}"
            f"{sizes[1]:>11}{times[1]:>11.2f}{ratio:>7.1f}"
        )


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--replacements", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dp.setup_default_parameters(max_heavy_atoms=38)

    results = {
        mode: run(mode, args) for mode in ("default", "rules_no_cache", "rules_cache", "rules_lazy")
    }
    print(
        f"EA QED : {args.generations} générations x {args.replacements} "
        f"remplacements, population {args.size}, graine {args.seed}"
    )
    print(
        f"{'mode':<16}{'temps (s)':>10}{'gén.':>6}{'candidats':>11}"
        f"{'ms/cand.':>10}{'erreurs':>9}{'meilleur QED':>14}{'x temps':>9}"
    )
    reference = results["default"]["time_s"]
    for mode, res in results.items():
        print(
            f"{mode:<16}{res['time_s']:>10.2f}{res['generations']:>6}"
            f"{res['candidates']:>11}{res['ms_per_candidate']:>10.2f}"
            f"{res['action_errors']:>9}{res['best_qed']:>14.4f}"
            f"{res['time_s'] / reference:>9.2f}"
        )

    time_list_actions(LARGE_MOLECULE)


if __name__ == "__main__":
    main()
