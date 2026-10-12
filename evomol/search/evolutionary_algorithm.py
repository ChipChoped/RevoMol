"""Module for the evolutionary algorithm."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evomol.evaluation import Evaluation, is_valid_molecule
from evomol.logging import get_logger
from evomol.representation import MolecularGraph, Molecule
from evomol.search.neighborhood_strategy import NeighborhoodStrategy
from evomol.search.parameters import search_parameters
from evomol.search.selection_strategy import SelectionStrategy


@dataclass
class ParametersEvolutionaryAlgorithm:
    """Parameters for the evolutionary algorithm."""

    size: int = 1000

    # Stop conditions
    max_generations: int = 1000
    max_time: int = 3600
    nb_replacements: int = 10

    # Output
    output_dir: str | None = None
    save_n_generations: int = 0
    save_final_population: bool = True


def _serialize_value(value: Any) -> str | int | float | bool | None:
    """Serialize evaluation values for TSV output.

    Scalars are kept as scalars. Lists, dicts and objects such as HuckelResult
    are serialized as JSON strings.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "__dict__"):
        return json.dumps(value.__dict__, ensure_ascii=False, default=str)

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)

    return str(value)


def _canonical_smiles(molecule: Molecule) -> str:
    """Return the canonical SMILES stored in the MolecularGraph representation."""
    return molecule.get_representation(MolecularGraph).canonical_smiles


def _population_rows(
    population: list[Molecule],
    generation: int,
    fitness_name: str,
    evaluation_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Convert a population into flat rows for TSV export."""
    rows: list[dict[str, Any]] = []

    for rank, molecule in enumerate(population, start=1):
        smiles = _canonical_smiles(molecule)
        values = molecule.values

        row: dict[str, Any] = {
            "rank": rank,
            "generation": generation,
            "smiles": smiles,
            "fitness_name": fitness_name,
            "fitness": _serialize_value(values.get(fitness_name)),
            "n_function_calls": (
                None
                if evaluation_counts is None
                else evaluation_counts.get(smiles, 0)
            ),
        }

        for name in sorted(values):
            row[f"eval:{name}"] = _serialize_value(values[name])

        rows.append(row)

    return rows


def _write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write flat rows to a TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    preferred_columns = [
        "rank",
        "generation",
        "smiles",
        "fitness_name",
        "fitness",
        "n_function_calls",
    ]

    columns: list[str] = []

    for col in preferred_columns:
        if any(col in row for row in rows):
            columns.append(col)

    for row in rows:
        for col in row:
            if col not in columns:
                columns.append(col)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    tmp_path.replace(path)


def _write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False, default=str)

    tmp_path.replace(path)


# pylint: disable=too-many-instance-attributes
class EvolutionaryAlgorithm:
    """Base class for evolutionary algorithms."""

    def __init__(
        self,
        population: list[Molecule],
        parameters: ParametersEvolutionaryAlgorithm,
        selection_strategy: SelectionStrategy,
        evaluations: list[Evaluation],
        neighborhood_strategy: NeighborhoodStrategy,
        generation_callback: Callable[[list[Molecule], int], None] | None = None,
        callback_every_n: int = 1,
        log_properties: list[str] | None = None,
    ):
        self.population: list[Molecule] = population
        self.parameters: ParametersEvolutionaryAlgorithm = parameters
        self.selection_strategy: SelectionStrategy = selection_strategy
        self.evaluations: list[Evaluation] = evaluations
        self.neighborhood_strategy: NeighborhoodStrategy = neighborhood_strategy
        self.generation_callback = generation_callback
        self.callback_every_n = callback_every_n
        # Properties of the best molecule shown in the generation log line
        self.log_properties: list[str] = log_properties or []

        self.tabu_molecules: set[str] = set()

        # WARNING — differs from the original EvoMol (RevoMol-correlation):
        # 1. A population smaller than parameters.size is no longer padded with
        #    Molecule("") (empty molecules, rejected by HuckelCompatible, so the
        #    original crashes with the Hückel filters). The missing slots stay
        #    empty and are filled first by any valid mutant, without fitness
        #    comparison, nb_replacements per generation (see run/_fill_slots):
        #    no molecule is replaced until the population is full. A run
        #    started from a small population does not follow the original
        #    trajectory.
        # 2. An empty population or nb_replacements > population size now
        #    raises ValueError (the original padded with empty molecules, or
        #    replaced the same slots twice through negative indices).
        # 3. The initial population is sorted by fitness (below): the original
        #    only sorted it at the end of each generation, so in generation 1
        #    the "worst" replaced were the last given, and BestSelection
        #    started from the first given.
        # A population smaller than parameters.size is completed during the
        # search: the missing slots are empty and filled first (see run).
        if not self.population:
            raise ValueError("The initial population must contain at least one molecule.")

        if self.parameters.nb_replacements > max(self.parameters.size, len(self.population)):
            raise ValueError(
                "nb_replacements cannot be larger than the population size. "
                f"Got nb_replacements={self.parameters.nb_replacements} and "
                f"population size={self.parameters.size}."
            )

        for molecule in self.population:
            smiles = _canonical_smiles(molecule)
            self.tabu_molecules.add(smiles)

            if not is_valid_molecule(molecule, evaluations):
                get_logger().error("Invalid molecule in the population : %s", molecule)
                raise ValueError("Invalid molecule in the population")

        # Sort initial population by fitness.
        self.population.sort(
            key=lambda x: x.value(search_parameters.fitness_criteria),
            reverse=True,
        )

        # Stop conditions
        self.nb_generations: int = 0
        self.start_time: float = 0
        self.stop_time: float = 0

    def _output_dir(self) -> Path | None:
        """Return output directory if configured."""
        if self.parameters.output_dir is None:
            return None
        return Path(self.parameters.output_dir)

    def _save_population(self, filename: str) -> None:
        """Save the current population as TSV."""
        output_dir = self._output_dir()
        if output_dir is None:
            return

        rows = _population_rows(
            population=self.population,
            generation=self.nb_generations,
            fitness_name=search_parameters.fitness_criteria,
            evaluation_counts=self.neighborhood_strategy.evaluation_counts,
        )

        _write_tsv(rows, output_dir / filename)

    def _save_metadata(self) -> None:
        """Save initial run metadata."""
        output_dir = self._output_dir()
        if output_dir is None:
            return

        metadata = {
            "format_version": "revomol-output-v1",
            "fitness_name": search_parameters.fitness_criteria,
            "optimization_direction": "maximize",
            "evaluation_names": [
                evaluation.name for evaluation in self.evaluations
            ],
            "parameters": {
                "size": self.parameters.size,
                "max_generations": self.parameters.max_generations,
                "max_time": self.parameters.max_time,
                "nb_replacements": self.parameters.nb_replacements,
                "output_dir": self.parameters.output_dir,
                "save_n_generations": self.parameters.save_n_generations,
                "save_final_population": self.parameters.save_final_population,
            },
            "smiles": {
                "canonical": True,
                "source": "MolecularGraph.canonical_smiles",
            },
        }

        _write_json(metadata, output_dir / "run_metadata.json")

    def _save_summary(self, elapsed_time: float) -> None:
        """Save final run summary."""
        output_dir = self._output_dir()
        if output_dir is None:
            return

        best_molecule = self.population[0]
        worst_molecule = self.population[-1]

        summary = {
            "n_generations": self.nb_generations,
            "elapsed_time": elapsed_time,
            "fitness_name": search_parameters.fitness_criteria,
            "best_smiles": _canonical_smiles(best_molecule),
            "best_fitness": best_molecule.value(search_parameters.fitness_criteria),
            "worst_smiles": _canonical_smiles(worst_molecule),
            "worst_fitness": worst_molecule.value(search_parameters.fitness_criteria),
            "n_function_calls_total": (
                self.neighborhood_strategy.n_function_calls_total
            ),
            "n_valid_candidates": self.neighborhood_strategy.n_valid_candidates,
            "n_invalid_candidates": self.neighborhood_strategy.n_invalid_candidates,
            "n_tabu_candidates": self.neighborhood_strategy.n_tabu_candidates,
            "n_known_invalid_candidates": (
                self.neighborhood_strategy.n_known_invalid_candidates
            ),
            "n_action_errors": self.neighborhood_strategy.n_action_errors,
        }

        _write_json(summary, output_dir / "run_summary.json")

    def run(self) -> None:
        """Run the evolutionary algorithm."""
        logger = get_logger()

        self.start_time = time.time()
        self.stop_time = self.start_time + self.parameters.max_time

        logger.info("Start the evolutionary algorithm.")
        self._save_metadata()

        while self.continue_search():
            # Sort the molecules for the selection.
            # The order is used to select the molecules to mutate.
            # The new molecules will replace the worst molecules.
            order_selection = self.selection_strategy.select(self.population)

            # Empty slots (population smaller than parameters.size) are worse
            # than any molecule: they are filled first (None), then the worst
            # molecules are replaced.
            nb_empty = max(0, self.parameters.size - len(self.population))
            nb_fill = min(nb_empty, self.parameters.nb_replacements)
            last_index = len(self.population) - 1
            indexes_to_replace: list[int | None] = [None] * nb_fill + list(
                range(
                    last_index,
                    last_index - (self.parameters.nb_replacements - nb_fill),
                    -1,
                )
            )

            self._fill_slots(order_selection, indexes_to_replace)

            # Sort the population by fitness.
            self.population.sort(
                key=lambda x: x.value(search_parameters.fitness_criteria),
                reverse=True,
            )

            self.nb_generations += 1

            fitness_values = [
                molecule.value(search_parameters.fitness_criteria)
                for molecule in self.population
            ]

            mean_value = sum(fitness_values) / len(fitness_values)
            max_value = fitness_values[0]
            min_value = fitness_values[-1]
            best_smiles = _canonical_smiles(self.population[0])

            best_properties: dict[str, Any] = {}
            for name in self.log_properties:
                try:
                    best_properties[name] = self.population[0].value(name)
                except KeyError:
                    pass
            best_properties_str = "".join(
                f" - {name}: {value:1.4f}" if isinstance(value, (int, float))
                else f" - {name}: {value}"
                for name, value in best_properties.items()
            )

            logger.info(
                "Generation %6d - Mean: %1.4f - Max: %1.4f - Min: %1.4f - %s%s",
                self.nb_generations,
                mean_value,
                max_value,
                min_value,
                best_smiles,
                best_properties_str,
                extra={
                    "best_properties": best_properties,
                    "event": "generation",
                    "generation": self.nb_generations,
                    "mean": mean_value,
                    "max": max_value,
                    "min": min_value,
                    "best_smiles": best_smiles,
                    "n_function_calls_total": (
                        self.neighborhood_strategy.n_function_calls_total
                    ),
                    "n_valid_candidates": (
                        self.neighborhood_strategy.n_valid_candidates
                    ),
                    "n_invalid_candidates": (
                        self.neighborhood_strategy.n_invalid_candidates
                    ),
                    "n_tabu_candidates": (
                        self.neighborhood_strategy.n_tabu_candidates
                    ),
                    "n_known_invalid_candidates": (
                        self.neighborhood_strategy.n_known_invalid_candidates
                    ),
                    "n_action_errors": self.neighborhood_strategy.n_action_errors,
                },
            )

            if (
                self.parameters.output_dir is not None
                and self.parameters.save_n_generations > 0
                and self.nb_generations % self.parameters.save_n_generations == 0
            ):
                self._save_population(
                    f"population_generation_{self.nb_generations:06d}.tsv"
                )

            if (
                self.generation_callback is not None
                and self.nb_generations % self.callback_every_n == 0
            ):
                self.generation_callback(self.population, self.nb_generations)

        elapsed_time = time.time() - self.start_time

        logger.info(
            "End of search - elapsed time: %ss",
            elapsed_time,
            extra={
                "event": "end_search",
                "elapsed_time": elapsed_time,
                "n_function_calls_total": (
                    self.neighborhood_strategy.n_function_calls_total
                ),
                "n_valid_candidates": self.neighborhood_strategy.n_valid_candidates,
                "n_invalid_candidates": (
                    self.neighborhood_strategy.n_invalid_candidates
                ),
                "n_tabu_candidates": self.neighborhood_strategy.n_tabu_candidates,
                "n_known_invalid_candidates": (
                    self.neighborhood_strategy.n_known_invalid_candidates
                ),
                "n_action_errors": self.neighborhood_strategy.n_action_errors,
            },
        )

        bests = [
            (
                _canonical_smiles(molecule),
                molecule.value(search_parameters.fitness_criteria),
            )
            for molecule in self.population[:5]
        ]

        logger.info(
            "Bests: %s",
            bests,
            extra={
                "event": "bests",
                "bests": {str(i): str(best) for i, best in enumerate(bests)},
            },
        )

        if self.parameters.output_dir is not None and self.parameters.save_final_population:
            self._save_population("population_final.tsv")

        self._save_summary(elapsed_time)

    def _fill_slots(
        self, order_selection: list[int], indexes_to_replace: list[int | None]
    ) -> None:
        """Fill the slots of one generation, one after the other.

        For each slot (None: empty slot, else index of the molecule to replace),
        the molecules are mutated in the selection order until an improver is
        found.
        """
        logger = get_logger()

        for index_to_replace in indexes_to_replace:
            found_improver: bool = False
            current_selected: int = 0
            to_replace: Molecule | None = (
                None if index_to_replace is None
                else self.population[index_to_replace]
            )

            # While no improver is found and there are still molecules to select.
            while current_selected < len(order_selection) and not found_improver:
                index_selected = order_selection[current_selected]

                new_molecule: Molecule | None = self.neighborhood_strategy.mutate(
                    self.population[index_selected],
                    to_replace,
                    evaluations=self.evaluations,
                    tabu_molecules=self.tabu_molecules,
                )

                smiles_to_mutate = _canonical_smiles(
                    self.population[index_selected]
                )
                smiles_to_replace = (
                    "<empty>" if to_replace is None
                    else _canonical_smiles(to_replace)
                )

                if new_molecule is not None:
                    new_smiles = _canonical_smiles(new_molecule)
                    score = new_molecule.value(
                        search_parameters.fitness_criteria
                    )

                    logger.debug(
                        "New molecule : %s (%1.4f) from %s to replace %s",
                        new_smiles,
                        score,
                        smiles_to_mutate,
                        smiles_to_replace,
                        extra={
                            "event": "new_molecule",
                            "new_molecule": new_smiles,
                            "score": score,
                            "to_mutate": smiles_to_mutate,
                            "to_replace": smiles_to_replace,
                        },
                    )

                    if index_to_replace is None:
                        self.population.append(new_molecule)
                    else:
                        self.population[index_to_replace] = new_molecule
                    self.tabu_molecules.add(new_smiles)
                    found_improver = True

                else:
                    logger.debug(
                        "No improvement found from %s to replace %s",
                        smiles_to_mutate,
                        smiles_to_replace,
                        extra={
                            "event": "no_improvement",
                            "to_mutate": smiles_to_mutate,
                            "to_replace": smiles_to_replace,
                        },
                    )

                current_selected += 1

    def continue_search(self) -> bool:
        """Check if the search should continue."""
        return (
            self.nb_generations < self.parameters.max_generations
            and self.stop_time > time.time()
        )

