"""Module to define the neighborhood strategy for a molecule representation."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from typing_extensions import override

import evomol.action.action as action_module
from evomol.action import Action, ActionError
from evomol.evaluation import Evaluation, is_valid_molecule
from evomol.logging import get_logger
from evomol.representation import MolecularGraph, Molecule
from evomol.search.parameters import search_parameters


class NoImproverError(RuntimeError):
    """Error raised when no neighbor is better than the current molecule."""


class NeighborhoodStrategy(ABC):
    """Strategy to generate neighbors of a molecule representation."""

    def __init__(
        self, depth: int = 1, nb_max_tries: int = 50, remember_invalid: bool = False
    ):
        self.depth = depth
        self.nb_max_tries = nb_max_tries
        # True: SMILES already rejected by is_valid_molecule are not evaluated
        # again (same answer, saves time). False: every candidate is evaluated.
        self.remember_invalid = remember_invalid

        # Candidate evaluation counters.
        # These are updated only for non-tabu candidates that are actually
        # evaluated by is_valid_molecule.
        self.n_function_calls_total: int = 0
        self.n_valid_candidates: int = 0
        self.n_invalid_candidates: int = 0

        # Candidates rejected before evaluation because they are already tabu.
        self.n_tabu_candidates: int = 0

        # Failed mutations due to action errors.
        self.n_action_errors: int = 0

        # Number of evaluations per canonical SMILES.
        self.evaluation_counts: dict[str, int] = {}

        # Canonical SMILES already rejected by is_valid_molecule. Validity
        # does not depend on the population, so they are not evaluated again.
        self.known_invalid: set[str] = set()

        # Candidates rejected without evaluation because already known invalid.
        self.n_known_invalid_candidates: int = 0

    @abstractmethod
    def mutate(
        self,
        molecule: Molecule,
        to_replace: Molecule | None,
        evaluations: list[Evaluation],
        tabu_molecules: set[str],
    ) -> Molecule | None:
        """Takes a molecule and returns a new molecule with a mutation.

        to_replace is None for an empty slot: any valid mutant is accepted.
        """


class RandomNeighborhoodStrategy(NeighborhoodStrategy):
    """Random strategy to generate neighbors of a molecule representation."""

    def __init__(
        self,
        depth: int = 1,
        nb_max_tries: int = 50,
        preselect_actions: bool = True,
        random_depth: bool = False,
        uniform_over_actions: bool = False,
        remember_invalid: bool = False,
    ):
        """Initialize the random neighborhood strategy.

        Args:
            depth: Depth of the mutation (max depth if random_depth is True).
            nb_max_tries: Number of tries before giving up on a mutation.
            preselect_actions: Whether one action space is selected before
                generating the actions. If False, all action spaces are computed
                and then one action is randomly selected.
            random_depth: If True, each mutation applies a number of actions
                drawn uniformly in [1, depth]. If False (original behaviour), exactly
                depth actions.
            uniform_over_actions: If True, each action is drawn uniformly among
                all possible actions of all types (types with many actions are
                drawn more often); preselect_actions is then ignored. If False,
                the action type is drawn first, then an action of that type.
            remember_invalid: If True, SMILES already rejected by the filters
                are not evaluated again. If False (original behaviour), every candidate
                is evaluated.
        """
        super().__init__(depth, nb_max_tries, remember_invalid)
        self.preselect_actions: bool = preselect_actions
        self.random_depth: bool = random_depth
        self.uniform_over_actions: bool = uniform_over_actions

    @override
    def mutate(
        self,
        molecule: Molecule,
        to_replace: Molecule | None,
        evaluations: list[Evaluation],
        tabu_molecules: set[str],
    ) -> Molecule | None:
        nb_attempts: int = 0

        while nb_attempts < self.nb_max_tries:
            try:
                new_molecule: Molecule = self.random_mutation(molecule)

            except ActionError as exc:
                # This probably means that an action implementation failed.
                get_logger().critical(exc)
                self.n_action_errors += 1
                nb_attempts += 1
                continue

            new_smiles: str = new_molecule.get_representation(
                MolecularGraph
            ).canonical_smiles

            # Check if the new molecule is in the tabu list.
            # Tabu candidates are not counted as function calls because they are
            # not evaluated by is_valid_molecule.
            if new_smiles in tabu_molecules:
                self.n_tabu_candidates += 1
                nb_attempts += 1
                continue

            # Already evaluated and invalid: same answer, no new evaluation.
            if self.remember_invalid and new_smiles in self.known_invalid:
                self.n_known_invalid_candidates += 1
                nb_attempts += 1
                get_logger().debug(
                    "Candidate already known invalid: %s",
                    new_smiles,
                    extra={
                        "event": "candidate_known_invalid",
                        "smiles": new_smiles,
                        "n_known_invalid_candidates": self.n_known_invalid_candidates,
                    },
                )
                continue

            evaluation_ok = is_valid_molecule(new_molecule, evaluations)

            self.n_function_calls_total += 1
            self.evaluation_counts[new_smiles] = (
                self.evaluation_counts.get(new_smiles, 0) + 1
            )

            if evaluation_ok:
                self.n_valid_candidates += 1
            else:
                self.n_invalid_candidates += 1
                if self.remember_invalid:
                    self.known_invalid.add(new_smiles)

            get_logger().debug(
                "Candidate evaluated: %s - valid=%s",
                new_smiles,
                evaluation_ok,
                extra={
                    "event": "candidate_evaluated",
                    "smiles": new_smiles,
                    "valid": evaluation_ok,
                    "n_function_calls_total": self.n_function_calls_total,
                    "n_valid_candidates": self.n_valid_candidates,
                    "n_invalid_candidates": self.n_invalid_candidates,
                    "n_tabu_candidates": self.n_tabu_candidates,
                    "n_action_errors": self.n_action_errors,
                },
            )

            # Check if the new molecule is better than the molecule to replace.
            if evaluation_ok and (
                not to_replace
                or new_molecule.value(search_parameters.fitness_criteria)
                > to_replace.value(search_parameters.fitness_criteria)
            ):
                return new_molecule

            nb_attempts += 1

        return None

    def random_mutation(self, new_molecule: Molecule) -> Molecule:
        """Choose a random mutation and apply it to the molecule.

        Args:
            new_molecule: Molecule to mutate.

        Returns:
            New molecule after mutation.
        """
        nb_actions = random.randint(1, self.depth) if self.random_depth else self.depth

        for _ in range(nb_actions):
            action: Action

            if self.uniform_over_actions:
                all_actions: list[action_module.Action] = [
                    a
                    for actions_by_type in new_molecule.compute_possible_actions().values()
                    for actions_of_type in actions_by_type.values()
                    for a in actions_of_type
                ]

                if not all_actions:
                    return new_molecule

                action = random.choice(all_actions)

            elif self.preselect_actions:
                representation_dict: dict[str, list[type[action_module.Action]]] = (
                    new_molecule.list_available_actions_space()
                )

                representation_choice: str = random.choice(
                    list(representation_dict.keys())
                )

                action_choice: type[action_module.Action] = random.choice(
                    representation_dict[representation_choice]
                )

                actions: list[action_module.Action] = action_choice.list_actions(
                    new_molecule
                )

                if not actions:
                    continue

                action = random.choice(actions)

            else:
                possible_actions: dict[str, dict[str, list[action_module.Action]]] = (
                    new_molecule.compute_possible_actions()
                )

                if not possible_actions:
                    return new_molecule

                representation: str = random.choice(list(possible_actions.keys()))

                action_space: str = random.choice(
                    list(possible_actions[representation].keys())
                )

                action = random.choice(possible_actions[representation][action_space])

            new_molecule = action.apply()

        return new_molecule


