"""
Short end-to-end run with all the new mutations (newmut, milestone 6).

QED evolutionary algorithm, atoms C, O, N, max_heavy_atoms = 38, mutation
depth 2, matrix rules and every new action on. Run once with the rules
checked on whole lists, once with lazy_rules. Checks that the run ends without
exception nor ActionError, and counts the mutation types actually applied.

Usage:
    python scripts/newmut/run_all_options.py [--generations 20] [--seed 0]
"""

import argparse
import os
import random
import sys
import time
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_ROOT)

# pylint: disable=wrong-import-position
from evomol import default_parameters as dp
from evomol import evaluation as evaluator
from evomol.action import Action
from evomol.newmut import EvolutionaryAlgorithmNewMut, NewMutNeighborhoodStrategy
from evomol.newmut.default_parameters import setup_newmut_action_space
from evomol.representation import Molecule
from evomol.search.evolutionary_algorithm import ParametersEvolutionaryAlgorithm
from evomol.search.parameters import search_parameters
from evomol.search.selection_strategy import BestSelection

INITIAL_POPULATION = ["c1ccccc1", "c1ccncc1", "C=CC=O", "Nc1ccccc1", "c1ccc2ccccc2c1"]

applied: Counter = Counter()


def count_applied() -> None:
    """Count the mutation types applied (wraps Action.apply)."""
    original_apply = Action.apply

    def apply(self: Action, register_action: bool = False) -> Molecule:
        applied[self.class_name()] += 1
        return original_apply(self, register_action)

    Action.apply = apply  # type: ignore[method-assign]


def run(lazy_rules: bool, args: argparse.Namespace) -> None:
    """One short run with all the options."""
    applied.clear()
    random.seed(args.seed)
    population = [Molecule(smiles) for smiles in INITIAL_POPULATION]
    neighborhood = NewMutNeighborhoodStrategy(depth=2, lazy_rules=lazy_rules)
    algorithm = EvolutionaryAlgorithmNewMut(
        population=population,
        parameters=ParametersEvolutionaryAlgorithm(
            size=20,
            max_generations=args.generations,
            max_time=10**6,
            nb_replacements=5,
            save_final_population=False,
        ),
        selection_strategy=BestSelection(),
        evaluations=[evaluator.QED],
        neighborhood_strategy=neighborhood,
    )
    start = time.perf_counter()
    algorithm.run()
    elapsed = time.perf_counter() - start
    print(
        f"\nlazy_rules={lazy_rules} : {algorithm.nb_generations} générations en "
        f"{elapsed:.1f} s, {neighborhood.n_function_calls_total} candidats évalués, "
        f"{neighborhood.n_action_errors} ActionError"
    )
    print("mutations appliquées :", dict(sorted(applied.items())))
    print("meilleures :", [
        (str(m), round(m.value("QED"), 3)) for m in algorithm.population[:3]
    ])
    if neighborhood.n_action_errors:
        raise RuntimeError(f"{neighborhood.n_action_errors} ActionError")


def main() -> None:
    """Run the end-to-end check."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dp.setup_default_parameters(accepted_atoms=["C", "O", "N"], max_heavy_atoms=38)
    setup_newmut_action_space(
        matrix_rules=True,
        substitution_shift=True,
        graft=True,
        annelate=True,
        flip_path=True,
        aromatize=True,
    )
    search_parameters.fitness_criteria = "QED"
    count_applied()
    for lazy_rules in (False, True):
        run(lazy_rules, args)


if __name__ == "__main__":
    main()
