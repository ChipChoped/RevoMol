"""Evolutionary algorithm whose mutations run on several processes.

One shared population, kept by the main process. At each generation the
slots to fill (nb_replacements) are searched at the same time, one slot per
worker: a worker mutates the molecules in the selection order until it finds
an improver, exactly as EvolutionaryAlgorithm does for one slot.

The main process then merges the candidates slot by slot. A candidate already
accepted for an earlier slot of the same generation is a duplicate: the slot
is searched again with the updated tabu list. The tabu list and the known
invalid SMILES are sent to every worker, so that no worker evaluates again a
molecule already in the population or already rejected by another worker.

Each slot search gets its own random seed, drawn by the main process: the
search only depends on the seed, not on the number of workers.
"""

from __future__ import annotations

import multiprocessing
import random
import traceback
from multiprocessing.connection import Connection, wait
from typing import Any, Callable

import os
import sys

# racine du projet (scripts/Lea/ -> ../..), pour importer evomol
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# pylint: disable=wrong-import-position

from evomol.evaluation import Evaluation
from evomol.logging import get_logger
from evomol.representation import MolecularGraph, Molecule
from evomol.search.evolutionary_algorithm import (
    EvolutionaryAlgorithm,
    ParametersEvolutionaryAlgorithm,
)
from evomol.search.neighborhood_strategy import NeighborhoodStrategy
from evomol.search.parameters import search_parameters
from evomol.search.selection_strategy import SelectionStrategy

# Counters of NeighborhoodStrategy summed over the workers.
_COUNTERS = (
    "n_function_calls_total",
    "n_valid_candidates",
    "n_invalid_candidates",
    "n_tabu_candidates",
    "n_known_invalid_candidates",
    "n_action_errors",
)


class _Fitness:
    """Stands for the molecule to replace: only its fitness is needed by mutate."""

    def __init__(self, fitness: float) -> None:
        self._fitness = fitness

    def value(self, name: str) -> float:
        """Fitness of the molecule to replace."""
        if name != search_parameters.fitness_criteria:
            raise KeyError(name)
        return self._fitness


def _worker(
    conn: Connection,
    setup: Callable[..., list[Evaluation]],
    setup_args: tuple,
    strategy: NeighborhoodStrategy,
) -> None:
    """Worker process: searches one slot per message received."""
    try:
        evaluations = setup(*setup_args)
    except Exception:  # pylint: disable=broad-except
        conn.send(("error", traceback.format_exc()))
        return
    conn.send(("ready", None))

    tabu: set[str] = set()
    parents_cache: dict[str, Molecule] = {}

    while True:
        message = conn.recv()
        if message is None:
            return
        tabu_delta, invalid_delta, parents, slot, seed, to_replace_fitness = message
        try:
            tabu.update(tabu_delta)
            strategy.known_invalid.update(invalid_delta)
            invalid_before = set(strategy.known_invalid)
            counters_before = {name: getattr(strategy, name) for name in _COUNTERS}
            strategy.evaluation_counts = {}

            # parent molecules are kept between messages (their possible
            # actions are computed once), only for the current population
            if len(parents_cache) > 4 * len(parents):
                parents_cache = {s: m for s, m in parents_cache.items() if s in set(parents)}

            random.seed(seed)
            to_replace = None if to_replace_fitness is None else _Fitness(to_replace_fitness)
            child: Molecule | None = None
            parent_smiles: str | None = None
            for parent_smiles in parents:
                parent = parents_cache.get(parent_smiles)
                if parent is None:
                    parent = parents_cache[parent_smiles] = Molecule(parent_smiles)
                child = strategy.mutate(
                    parent, to_replace, evaluations=evaluations, tabu_molecules=tabu
                )
                if child is not None:
                    break

            conn.send((
                "result",
                {
                    "slot": slot,
                    "smiles": None if child is None else str(child),
                    "values": None if child is None else dict(child.values),
                    "parent": parent_smiles,
                    "counters": {
                        name: getattr(strategy, name) - counters_before[name]
                        for name in _COUNTERS
                    },
                    "evaluation_counts": strategy.evaluation_counts,
                    "new_invalid": list(strategy.known_invalid - invalid_before),
                },
            ))
        except Exception:  # pylint: disable=broad-except
            conn.send(("error", traceback.format_exc()))
            return


class ParallelEvolutionaryAlgorithm(EvolutionaryAlgorithm):
    """EvolutionaryAlgorithm whose slots of a generation are searched in parallel.

    Differences with EvolutionaryAlgorithm (same results whatever n_workers):
      - the slots of a generation are searched from the population of the
        start of the generation (or of the start of the round, for a slot
        searched again after a duplicate), not after each replacement;
      - the random draws of a slot come from its own seed.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        population: list[Molecule],
        parameters: ParametersEvolutionaryAlgorithm,
        selection_strategy: SelectionStrategy,
        evaluations: list[Evaluation],
        neighborhood_strategy: NeighborhoodStrategy,
        n_workers: int,
        worker_setup: Callable[..., list[Evaluation]],
        worker_setup_args: tuple = (),
        generation_callback: Callable[[list[Molecule], int], None] | None = None,
        callback_every_n: int = 1,
        log_properties: list[str] | None = None,
    ):
        """
        Args:
            n_workers: number of worker processes.
            worker_setup: function run once in each worker (picklable: defined
                at module level); sets up evomol (parameters, action space) and
                returns the evaluations, the same as `evaluations`.
            worker_setup_args: arguments of worker_setup.
            Other arguments: see EvolutionaryAlgorithm.
        """
        super().__init__(
            population=population,
            parameters=parameters,
            selection_strategy=selection_strategy,
            evaluations=evaluations,
            neighborhood_strategy=neighborhood_strategy,
            generation_callback=generation_callback,
            callback_every_n=callback_every_n,
            log_properties=log_properties,
        )
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}.")
        # An EvolutionaryAlgorithm imported before _fill_slots existed (e.g. a
        # notebook kernel started earlier) never calls it: the search would run
        # in the main process only.
        if "_fill_slots" not in EvolutionaryAlgorithm.__dict__:
            raise RuntimeError(
                "evomol.search.evolutionary_algorithm is an old version without "
                "_fill_slots: restart the Python kernel."
            )
        self.n_workers = n_workers
        self.worker_setup = worker_setup
        self.worker_setup_args = worker_setup_args

        self._conns: list[Connection] = []
        self._processes: list[multiprocessing.process.BaseProcess] = []
        # SMILES added to the tabu list / to the known invalid set, in order;
        # _sent[w] = how many of each were already sent to worker w
        self._tabu_log: list[str] = []
        self._invalid_log: list[str] = []
        self._sent: list[list[int]] = []

    @staticmethod
    def _receive(conn: Connection) -> Any:
        kind, content = conn.recv()
        if kind == "error":
            raise RuntimeError(f"Error in a worker process:\n{content}")
        return content

    def _start_workers(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        for _ in range(self.n_workers):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker,
                args=(
                    child_conn,
                    self.worker_setup,
                    self.worker_setup_args,
                    self.neighborhood_strategy,
                ),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._processes.append(process)
            self._sent.append([0, 0])
        for conn in self._conns:
            self._receive(conn)  # "ready" once the evaluations are set up

    def _stop_workers(self) -> None:
        for conn in self._conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self._conns, self._processes, self._sent = [], [], []

    def run(self) -> None:
        self._tabu_log = list(self.tabu_molecules)
        self._invalid_log = list(self.neighborhood_strategy.known_invalid)
        self._start_workers()
        try:
            super().run()
        finally:
            self._stop_workers()

    def _search_slots(
        self, parents: list[str], tasks: list[tuple[int, int, float | None]]
    ) -> dict[int, dict[str, Any]]:
        """Search the slots (slot, seed, fitness to beat) on the workers."""
        results: dict[int, dict[str, Any]] = {}
        todo = list(tasks)
        idle = list(range(len(self._conns)))
        busy: dict[Connection, int] = {}
        while todo or busy:
            while todo and idle:
                worker = idle.pop(0)
                slot, seed, to_replace_fitness = todo.pop(0)
                sent = self._sent[worker]
                self._conns[worker].send((
                    self._tabu_log[sent[0]:],
                    self._invalid_log[sent[1]:],
                    parents,
                    slot,
                    seed,
                    to_replace_fitness,
                ))
                sent[0], sent[1] = len(self._tabu_log), len(self._invalid_log)
                busy[self._conns[worker]] = worker
            for conn in wait(list(busy)):
                worker = busy.pop(conn)  # type: ignore[arg-type]
                result = self._receive(conn)  # type: ignore[arg-type]
                results[result["slot"]] = result
                idle.append(worker)
                # invalid SMILES found by this worker: sent to the others
                for smiles in result["new_invalid"]:
                    if smiles not in self.neighborhood_strategy.known_invalid:
                        self.neighborhood_strategy.known_invalid.add(smiles)
                        self._invalid_log.append(smiles)
        return results

    def _fill_slots(
        self, order_selection: list[int], indexes_to_replace: list[int | None]
    ) -> None:
        logger = get_logger()
        fitness_name = search_parameters.fitness_criteria
        strategy = self.neighborhood_strategy

        pending = list(enumerate(indexes_to_replace))
        while pending:
            parents = [str(self.population[i]) for i in order_selection]
            tasks = [
                (
                    slot,
                    random.getrandbits(63),
                    None if index is None else self.population[index].value(fitness_name),
                )
                for slot, index in pending
            ]
            results = self._search_slots(parents, tasks)

            searched_again = []
            for slot, index_to_replace in pending:
                result = results[slot]
                for name, delta in result["counters"].items():
                    setattr(strategy, name, getattr(strategy, name) + delta)
                for smiles, count in result["evaluation_counts"].items():
                    strategy.evaluation_counts[smiles] = (
                        strategy.evaluation_counts.get(smiles, 0) + count
                    )

                smiles_to_replace = (
                    "<empty>" if index_to_replace is None
                    else str(self.population[index_to_replace])
                )
                if result["smiles"] is None:
                    logger.debug(
                        "No improvement found to replace %s",
                        smiles_to_replace,
                        extra={"event": "no_improvement", "to_replace": smiles_to_replace},
                    )
                    continue

                new_molecule = Molecule(result["smiles"])
                for name, value in result["values"].items():
                    new_molecule.set_value(name, value)
                new_smiles = new_molecule.get_representation(MolecularGraph).canonical_smiles

                # found for an earlier slot of the same generation: search again
                if new_smiles in self.tabu_molecules:
                    strategy.n_tabu_candidates += 1
                    searched_again.append((slot, index_to_replace))
                    continue

                score = new_molecule.value(fitness_name)
                logger.debug(
                    "New molecule : %s (%1.4f) from %s to replace %s",
                    new_smiles,
                    score,
                    result["parent"],
                    smiles_to_replace,
                    extra={
                        "event": "new_molecule",
                        "new_molecule": new_smiles,
                        "score": score,
                        "to_mutate": result["parent"],
                        "to_replace": smiles_to_replace,
                    },
                )
                if index_to_replace is None:
                    self.population.append(new_molecule)
                else:
                    self.population[index_to_replace] = new_molecule
                self.tabu_molecules.add(new_smiles)
                self._tabu_log.append(new_smiles)

            pending = searched_again
