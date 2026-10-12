"""
Random walk in chemical space, adapted from RevoMol-correlation (scripts/walk.py,
strategy="random").

At each step the walk moves to a neighbor (one mutation away) drawn uniformly
among the distinct neighbors of the current molecule — among the valid ones
only when only_valid=True. No fitness is involved.

Neighbors are keyed by Action.apply_smiles(): a Molecule is built only for the
candidates that are actually examined.

Each difference with the original implementation is an option; the original
behaviour is obtained with:
- count_valid_neighbors=True: validate the whole neighborhood, then draw among
  the valid neighbors (default: validate lazily in random order and take the
  first valid one — same uniform law, far fewer filter calls);
- exclude_current=False: the current molecule may be drawn as its own neighbor.

uniform_action_type=True (off by default) draws in two stages: first a mutation
type uniformly among the types that still have an eligible neighbor, then a
neighbor uniformly among those produced by that type. Mutation types that
produce many neighbors are then no longer over-represented in the walk.

action_types restricts which mutation types are considered, independently of
the globally configured MolecularGraph.action_space (set once for the process
by config.apply_action_space / default_parameters.setup_default_action_space).
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable

import os
import sys

# racine du projet (scripts/Lea/ -> ../..), pour importer evomol
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# pylint: disable=wrong-import-position

from evomol.action import Action, ActionError
from evomol.evaluation import Evaluation, EvaluationError, is_valid_molecule
from evomol.logging import get_logger
from evomol.representation import MolecularGraph, Molecule


def neighbor_actions(
    molecule: Molecule, action_types: list[type[Action]] | None = None
) -> dict[str, Action]:
    """Distinct neighbors of a molecule, keyed by canonical SMILES.

    Args:
        molecule: Molecule whose neighbors are listed.
        action_types: only consider mutations whose type is in this list.
            None: every mutation type in the molecule's action space.

    Returns:
        dict[str, Action]: canonical SMILES -> first action producing it.
    """
    neighbors: dict[str, Action] = {}
    for representation in molecule.compute_possible_actions().values():
        for action_list in representation.values():
            for action in action_list:
                if action_types is not None and type(action) not in action_types:
                    continue
                try:
                    smiles = action.apply_smiles()  # type: ignore[attr-defined]
                except ActionError as exc:
                    get_logger().critical(exc)
                    continue
                neighbors.setdefault(smiles, action)
    return neighbors


def neighbor_actions_by_type(
    molecule: Molecule, action_types: list[type[Action]] | None = None
) -> dict[str, dict[str, Action]]:
    """Distinct neighbors of a molecule, grouped by mutation type.

    A neighbor produced by several mutation types appears in each of them.

    Returns:
        dict[str, dict[str, Action]]: mutation type name -> (canonical SMILES
        -> first action of that type producing it).
    """
    by_type: dict[str, dict[str, Action]] = {}
    for representation in molecule.compute_possible_actions().values():
        for action_list in representation.values():
            for action in action_list:
                if action_types is not None and type(action) not in action_types:
                    continue
                try:
                    smiles = action.apply_smiles()  # type: ignore[attr-defined]
                except ActionError as exc:
                    get_logger().critical(exc)
                    continue
                by_type.setdefault(action.class_name(), {}).setdefault(smiles, action)
    return by_type


def _choose_by_type(
    by_type: dict[str, dict[str, Action]],
    filters: list[Evaluation],
    only_valid: bool,
    count_valid_neighbors: bool,
    rng: random.Random,
) -> tuple[tuple[str, Molecule] | None, str | None, int, int | None, dict[str, int]]:
    """Two-stage draw: a mutation type uniformly among the types having an
    eligible neighbor, then a neighbor uniformly among those of that type.

    Returns:
        (chosen (SMILES, Molecule) or None, mutation type, number of filter
        calls, number of distinct valid neighbors (count_valid_neighbors
        only), rejected neighbors per mutation type).
    """
    validity: dict[str, tuple[bool, Molecule]] = {}  # each SMILES validated once
    rejected: dict[str, int] = {}

    def check(smiles: str) -> tuple[bool, Molecule]:
        if smiles not in validity:
            mol = Molecule(smiles)
            validity[smiles] = (is_valid_molecule(mol, filters), mol)
        return validity[smiles]

    types = sorted(by_type)  # sorted: independent of PYTHONHASHSEED
    if not only_valid:
        if not types:
            return None, None, 0, None, rejected
        name = rng.choice(types)
        smiles = rng.choice(sorted(by_type[name]))
        return (smiles, Molecule(smiles)), name, 0, None, rejected

    if count_valid_neighbors:
        valid: dict[str, list[str]] = {}
        for name in types:
            for smiles in sorted(by_type[name]):
                if check(smiles)[0]:
                    valid.setdefault(name, []).append(smiles)
                else:
                    rejected[name] = rejected.get(name, 0) + 1
        n_valid = sum(ok for ok, _ in validity.values())
        if not valid:
            return None, None, len(validity), n_valid, rejected
        name = rng.choice(sorted(valid))
        smiles = rng.choice(valid[name])
        return (smiles, validity[smiles][1]), name, len(validity), n_valid, rejected

    # lazy: types in random order, neighbors of each type in random order, the
    # first valid one wins (same law as above, far fewer filter calls)
    rng.shuffle(types)
    for name in types:
        keys = sorted(by_type[name])
        rng.shuffle(keys)
        for smiles in keys:
            ok, mol = check(smiles)
            if ok:
                return (smiles, mol), name, len(validity), None, rejected
            rejected[name] = rejected.get(name, 0) + 1
    return None, None, len(validity), None, rejected


def _describe(molecule: Molecule, evaluations: list[Evaluation]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for evaluation in evaluations:
        try:
            values[evaluation.name] = evaluation.evaluate(molecule)
        except EvaluationError:
            values[evaluation.name] = None
    return values


# pylint: disable=too-many-arguments,too-many-locals
def random_walk(
    start_smiles: str,
    n_steps: int,
    filters: list[Evaluation],
    only_valid: bool = True,
    count_valid_neighbors: bool = False,
    evaluations: list[Evaluation] | None = None,
    seed: int | None = 0,
    exclude_current: bool = True,
    action_types: list[type[Action]] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    uniform_action_type: bool = False,
) -> list[dict[str, Any]]:
    """Perform a random walk of n_steps mutations from start_smiles.

    When the current molecule has no eligible neighbor, the walk restarts from
    the starting molecule (as in the original implementation).

    Args:
        start_smiles: SMILES of the starting molecule.
        n_steps: number of moves.
        filters: validity filters, in the order they are applied.
        only_valid: only move to neighbors that pass the filters.
        count_valid_neighbors: validate every neighbor at each step
            (slower, fills n_valid_neighbors). Only used with only_valid.
        evaluations: descriptors computed on every visited molecule (e.g.
            [Huckel(), HuckelHOMO()]); a failing evaluation gives None.
        seed: seed of the walk's random generator (None: a different walk
            at every run).
        exclude_current: never propose the current molecule as its own
            neighbor. False: original behaviour.
        action_types: restrict the mutations allowed during the walk (e.g.
            [mg.AddAtomMG, mg.SubstituteAtomMG]). None: every mutation type
            in the configured action space.
        on_step: called with each row as soon as it is recorded (step 0
            included), e.g. to update a progress bar. None: not used.
        uniform_action_type: draw a mutation type uniformly first, then a
            neighbor of that type (no over-representation of the types
            producing many neighbors). False: original behaviour.

    Returns:
        list[dict]: one row per visited molecule (step 0 = start) with the
        SMILES, its validity, the action that led to it, the neighborhood
        sizes, the mutation types of the neighbors rejected by the filters
        ("rejected_actions", e.g. "AddAtomMG:2;ChangeBondMG:1"), the time
        spent per phase and the evaluations.
    """
    rng = random.Random(seed)
    evaluations = evaluations or []

    start_canonical = Molecule(start_smiles).get_representation(
        MolecularGraph
    ).canonical_smiles
    current = Molecule(start_canonical)

    rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "smiles": str(current),
            "valid": is_valid_molecule(current, filters),
            "action": None,
            "neighborhood_size": None,
            "n_valid_neighbors": None,
            "n_filter_calls": 0,
            "rejected_actions": "",
            "t_neighbors": 0.0,
            "t_filters": 0.0,
            "t_evaluations": 0.0,
            **_describe(current, evaluations),
        }
    ]
    if on_step is not None:
        on_step(rows[-1])

    for step in range(1, n_steps + 1):
        if uniform_action_type:
            t0 = time.perf_counter()
            by_type = neighbor_actions_by_type(current, action_types)
            if exclude_current:
                for group in by_type.values():
                    group.pop(str(current), None)
            by_type = {name: group for name, group in by_type.items() if group}
            neighborhood_size = len(set().union(*by_type.values())) if by_type else 0
            t1 = time.perf_counter()
            chosen, chosen_type, n_filter_calls, n_valid_neighbors, rejected = _choose_by_type(
                by_type, filters, only_valid, count_valid_neighbors, rng
            )
            if chosen is None:
                action_name = "restart"
                current = Molecule(start_canonical)
            else:
                action_name = str(chosen_type)
                current = chosen[1]
            rows.append(_step_row(step, current, filters, evaluations, action_name,
                                  neighborhood_size, n_valid_neighbors, n_filter_calls,
                                  rejected, t0, t1))
            if on_step is not None:
                on_step(rows[-1])
            continue

        t0 = time.perf_counter()
        neighbors = neighbor_actions(current, action_types)
        if exclude_current:
            neighbors.pop(str(current), None)
        keys = sorted(neighbors)  # sorted: independent of PYTHONHASHSEED
        rng.shuffle(keys)
        t1 = time.perf_counter()

        chosen: tuple[str, Molecule] | None = None
        n_filter_calls = 0
        n_valid_neighbors: int | None = None
        rejected: dict[str, int] = {}  # mutation type -> neighbors rejected by the filters

        if only_valid and count_valid_neighbors:
            valid: list[tuple[str, Molecule]] = []
            for smiles in keys:
                mol = Molecule(smiles)
                n_filter_calls += 1
                if is_valid_molecule(mol, filters):
                    valid.append((smiles, mol))
                else:
                    name = neighbors[smiles].class_name()
                    rejected[name] = rejected.get(name, 0) + 1
            n_valid_neighbors = len(valid)
            if valid:
                chosen = rng.choice(valid)
        elif only_valid:
            for smiles in keys:
                mol = Molecule(smiles)
                n_filter_calls += 1
                if is_valid_molecule(mol, filters):
                    chosen = (smiles, mol)
                    break
                name = neighbors[smiles].class_name()
                rejected[name] = rejected.get(name, 0) + 1
        elif keys:
            chosen = (keys[0], Molecule(keys[0]))

        if chosen is None:
            action_name = "restart"
            current = Molecule(start_canonical)
        else:
            action_name = neighbors[chosen[0]].class_name()
            current = chosen[1]

        is_valid = is_valid_molecule(current, filters)
        t2 = time.perf_counter()
        values = _describe(current, evaluations)
        t3 = time.perf_counter()

        rows.append(
            {
                "step": step,
                "smiles": str(current),
                "valid": is_valid,
                "action": action_name,
                "neighborhood_size": len(keys),
                "n_valid_neighbors": n_valid_neighbors,
                "n_filter_calls": n_filter_calls,
                "rejected_actions": ";".join(f"{k}:{v}" for k, v in sorted(rejected.items())),
                "t_neighbors": t1 - t0,
                "t_filters": t2 - t1,
                "t_evaluations": t3 - t2,
                **values,
            }
        )
        if on_step is not None:
            on_step(rows[-1])

    return rows


# pylint: disable=too-many-arguments
def _step_row(
    step: int,
    current: Molecule,
    filters: list[Evaluation],
    evaluations: list[Evaluation],
    action_name: str,
    neighborhood_size: int,
    n_valid_neighbors: int | None,
    n_filter_calls: int,
    rejected: dict[str, int],
    t0: float,
    t1: float,
) -> dict[str, Any]:
    """Row of a step drawn with uniform_action_type (same columns as the others)."""
    is_valid = is_valid_molecule(current, filters)
    t2 = time.perf_counter()
    values = _describe(current, evaluations)
    t3 = time.perf_counter()
    return {
        "step": step,
        "smiles": str(current),
        "valid": is_valid,
        "action": action_name,
        "neighborhood_size": neighborhood_size,
        "n_valid_neighbors": n_valid_neighbors,
        "n_filter_calls": n_filter_calls,
        "rejected_actions": ";".join(f"{k}:{v}" for k, v in sorted(rejected.items())),
        "t_neighbors": t1 - t0,
        "t_filters": t2 - t1,
        "t_evaluations": t3 - t2,
        **values,
    }
