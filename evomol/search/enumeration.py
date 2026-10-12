"""
Those functions are used to explore the neighbors of a molecule up to a certain depth.
Either by listing all the neighbors for one molecule and printing the valid ones
with list_up_to_depth_from_smiles or by exploring the complete neighborhood of
a molecule and the neighborhood of its neighbors and so on with
enumerate_from_smiles.

Those functions may be be optimized.

find_neighbors could take the set of valid neighbors already found and only
consider the neighbors that are not in this set, I tried it but it was slower
and didn't lead to the same results. Maybe I did something wrong.

enumerate_from_smiles_parallel is optimized by using multiprocessing to explore
the neighbors of the molecules in parallel.
I tried to use multiprocessing to check the validity of the molecules but it was
slower :
# to_check: set[str] = set()
# for result in neighbors_results:
#     for new_smi in result:
#         # if the SMILES is already in the set of valid or invalid SMILES
#         # it has already been explored or is in the queue to be explored
#         if new_smi in found_valid or new_smi in found_invalid:
#             continue

#         to_check.add(new_smi)

# validity_results = pool.starmap(
#     parallel_is_valid_molecule, [(smi, evaluations) for smi in to_check]
# )
# for new_smi, is_valid in validity_results:
#     if is_valid:
#         to_explore.add(new_smi)
#         found_valid.add(new_smi)
#     else:
#         found_invalid.add(new_smi)
"""

from collections import deque
from multiprocessing import Manager, Pool, Queue, cpu_count
from multiprocessing.managers import ValueProxy
from threading import Lock
from typing import Callable, Optional

from evomol import evaluation as evaluator
from evomol.representation import MolecularGraph, Molecule


def set_of_neighbors(molecule: Molecule) -> set[Molecule]:
    """List the neighbors of a molecule and compute them.

    Args:
        molecule (Molecule): Molecule to explore

    Returns:
        set[Molecule]: set of molecules found
    """
    # fill the possible actions of the molecule
    molecule.compute_possible_actions()
    # for each representation, for each action list, for each action, apply it
    # and return the set of new molecules
    return {
        action.apply()
        for representation in molecule.possible_actions.values()
        for action_list in representation.values()
        for action in action_list
    }


def list_of_neighbors(molecule: Molecule) -> list[Molecule]:
    """List the neighbors of a molecule and compute them.

    Args:
        molecule (Molecule): Molecule to explore

    Returns:
        list[Molecule]: list of molecules found
    """
    # fill the possible actions of the molecule
    molecule.compute_possible_actions()
    # for each representation, for each action list, for each action, apply it
    # and return the list of new molecules
    return [
        action.apply()
        for representation in molecule.possible_actions.values()
        for action_list in representation.values()
        for action in action_list
    ]


def dict_of_neighbors(molecule: Molecule) -> dict[str, dict[str, list[Molecule]]]:
    """List the neighbors of a molecule and compute them.

    Args:
        molecule (Molecule): Molecule to explore

    Returns:
        dict[str, list[Molecule]]: dict of molecules found
    """
    # fill the possible actions of the molecule
    molecule.compute_possible_actions()
    # for each representation, for each action list, for each action, apply it
    # and return the dict of new molecules
    return {
        representation_name: {
            action_name: [action.apply() for action in action_list]
            for action_name, action_list in action_dict.items()
        }
        for representation_name, action_dict in molecule.possible_actions.items()
    }


def find_neighbors_and_filter_with_duplicates(
    molecule: Molecule,
    max_depth: int,
    evaluations: list[evaluator.Evaluation],
    apply_evaluation: bool,
) -> list[str]:
    """Iteratively find the neighbors of a molecule up to a certain depth.
    Explore all neighbors at level 1 then all neighbors at level 2 and so on.
    Duplicate molecules are kept.
    Molecules can be filtered with the evaluations between each depth.

    Args:
        molecule (Molecule): Molecule to explore
        max_depth (int): Maximum depth to explore

    Returns:
        list[str]: list of molecules found
    """
    # list of neighbors found
    neighbors: list[str] = []

    # queue of molecules to explore
    queue: list[Molecule] = [molecule]

    # start at depth 1
    depth = 1

    # explore the neighbors of the molecule up to the maximum depth
    while depth <= max_depth:
        # set of neighbors found in this depth
        next_queue: list[Molecule] = []
        # for each molecule in the queue, find the neighbors
        for current_mol in queue:
            for new_mol in list_of_neighbors(current_mol):
                new_smi = str(new_mol)
                if apply_evaluation and not evaluator.is_valid_molecule(
                    new_mol, evaluations
                ):
                    continue
                next_queue.append(new_mol)
                neighbors.append(new_smi)
        # update the queue with the new neighbors and increase the depth
        queue = next_queue
        depth += 1

    return neighbors


def find_neighbors_and_filter_without_duplicates(
    molecule: Molecule,
    max_depth: int,
    evaluations: list[evaluator.Evaluation],
    apply_evaluation: bool,
) -> set[str]:
    """Iteratively find the neighbors of a molecule up to a certain depth.
    Explore all neighbors at level 1 then all neighbors at level 2 and so on.
    Duplicate molecules are removed.
    Molecules can be filtered with the evaluations between each depth.

    Args:
        molecule (Molecule): Molecule to explore
        max_depth (int): Maximum depth to explore

    Returns:
        list[str]: list of molecules found
    """
    # set of neighbors found
    neighbors: set[str] = set()

    # queue of molecules to explore
    queue: set[Molecule] = {molecule}

    # start at depth 1
    depth = 1

    # explore the neighbors of the molecule up to the maximum depth
    while depth <= max_depth:
        # set of neighbors found in this depth
        next_queue: set[Molecule] = set()
        # for each molecule in the queue, find the neighbors
        for current_mol in queue:
            for new_mol in list_of_neighbors(current_mol):
                new_smi = str(new_mol)
                if apply_evaluation and not evaluator.is_valid_molecule(
                    new_mol, evaluations
                ):
                    continue
                if new_smi not in neighbors:
                    next_queue.add(new_mol)
                    neighbors.add(new_smi)
        # update the queue with the new neighbors and increase the depth
        queue = next_queue
        depth += 1

    return neighbors


def find_neighbors(molecule: Molecule, max_depth: int) -> set[str]:
    """Iteratively find the neighbors of a molecule up to a certain depth.
    Explore all neighbors at level 1 then all neighbors at level 2 and so on.

    Args:
        molecule (Molecule): Molecule to explore
        max_depth (int): Maximum depth to explore

    Returns:
        set[str]: set of molecules found
    """
    # set of neighbors found
    neighbors: set[str] = set()

    # queue of molecules to explore
    queue: set[Molecule] = {molecule}

    # start at depth 1
    depth = 1

    # explore the neighbors of the molecule up to the maximum depth
    while depth <= max_depth:
        # set of neighbors found in this depth
        next_queue = set()
        # for each molecule in the queue, find the neighbors
        for current_mol in queue:
            for new_mol in set_of_neighbors(current_mol):
                new_smi = str(new_mol)
                # add the new molecule to the set of neighbors if it is not
                # already in it, don't add it if it is already in the set
                # as it has already been explored
                if new_smi not in neighbors:
                    next_queue.add(new_mol)
                    neighbors.add(new_smi)
        # update the queue with the new neighbors and increase the depth
        queue = next_queue
        depth += 1

    return neighbors


def parallel_find_neighbors(smiles: str, depth: int) -> set[str]:
    """Find the neighbors of a molecule up to a certain depth for parallel use.

    Args:
        smiles (str): SMILES to explore
        depth (int): Maximum depth to explore

    Returns:
        set[str]: set of molecules found
    """
    return find_neighbors(Molecule(smiles), depth)


def parallel_is_valid_molecule(
    smiles: str, evaluations: list[evaluator.Evaluation]
) -> tuple[str, bool]:
    """Check if a molecule is valid according to a list of evaluations for
    parallel use.

    Args:
        smiles (str): SMILES to check
        evaluations (list[evaluator.Evaluation]): filter to apply

    Returns:
        tuple[str, bool]: _description_
    """
    return smiles, evaluator.is_valid_molecule(Molecule(smiles), evaluations)


def worker_find_neighbors(
    explore_queue: Queue,
    neighbor_queue: Queue,
    depth: int,
    task_counter: ValueProxy[int],
    lock: Lock,
) -> None:
    """Worker to explore the neighbors of a molecule up to a certain depth.

    Args:
        explore_queue (Queue): queue of SMILES to explore
        neighbor_queue (Queue): queue of neighbors found
        depth (int): Maximum depth to explore
        task_counter (Value): shared counter of tasks
    """
    while True:
        smiles = explore_queue.get()

        neighbors = find_neighbors(Molecule(smiles), depth)
        neighbor_queue.put(neighbors)
        with lock:
            task_counter.value -= 1


def enumerate_from_smiles(
    starting_smiles: str, depth: int, evaluations: list[evaluator.Evaluation]
) -> tuple[set[str], set[str]]:
    """Enumerate all reachable molecules starting from a SMILES up to a certain
    depth and check their validity before exploring them.

    Args:
        starting_smiles (str): SMILES to explore
        depth (int): Maximum depth to explore
        evaluations (list[evaluator.Evaluation]): filter to apply

    Returns:
        tuple[set[str], set[str]]: sets of valid and invalid SMILES
    """

    # set of SMILES to explore
    to_explore: set[str] = {starting_smiles}

    # set of valid and invalid SMILES found
    found_valid: set[str] = set()
    found_invalid: set[str] = set()

    # add the starting SMILES to the set of valid or invalid SMILES
    if evaluator.is_valid_molecule(Molecule(starting_smiles), evaluations):
        found_valid.add(starting_smiles)
    else:
        found_invalid.add(starting_smiles)

    while to_explore:

        to_check: set[str] = set()

        while to_explore:
            # get the next SMILES to explore
            current_smiles: str = to_explore.pop()

            # find the neighbors of the current SMILES
            new_mols = find_neighbors(Molecule(current_smiles), depth)

            to_check.update(new_mols)

        while to_check:
            new_smi = to_check.pop()

            # if the SMILES is already in the set of valid or invalid SMILES
            # it has already been explored or is in the queue to be explored
            if new_smi in found_valid or new_smi in found_invalid:
                continue

            # check if the new molecule is valid
            if evaluator.is_valid_molecule(Molecule(new_smi), evaluations):
                # if it is valid, add it to the set of valid SMILES and to the
                # set of SMILES to explore
                to_explore.add(new_smi)
                found_valid.add(new_smi)
            else:
                # if it is invalid, add it to the set of invalid SMILES
                # to avoid checking it again
                found_invalid.add(new_smi)

    return found_valid, found_invalid


def enumerate_from_smiles_multiprocessing(
    starting_smiles: str,
    depth: int,
    evaluations: list[evaluator.Evaluation],
    nb_cpu: int,
) -> tuple[set[str], set[str]]:
    """Enumerate all reachable molecules starting from a SMILES up to a certain
    depth and check their validity before exploring them in parallel.

    Args:
        starting_smiles (str): SMILES to explore
        depth (int): Maximum depth to explore
        evaluations (list[evaluator.Evaluation]): filter to apply
        nb_cpu (int): number of CPU to use

    Returns:
        tuple[set[str], set[str]]: sets of valid and invalid SMILES
    """

    # set of SMILES to explore
    to_explore: set[str] = {starting_smiles}

    # set of valid and invalid SMILES found
    found_valid: set[str] = set()
    found_invalid: set[str] = set()

    # add the starting SMILES to the set of valid or invalid SMILES
    if evaluator.is_valid_molecule(Molecule(starting_smiles), evaluations):
        found_valid.add(starting_smiles)
    else:
        found_invalid.add(starting_smiles)

    if nb_cpu == 0:
        nb_cpu = cpu_count()

    with Pool(nb_cpu) as pool:
        while to_explore:

            # find the neighbors of the SMILES to explore in parallel
            neighbors_results = pool.starmap(
                parallel_find_neighbors,
                [(smiles, depth) for smiles in to_explore],
            )
            to_explore.clear()

            for result in neighbors_results:
                for new_smi in result:
                    # if the SMILES is already in the set of valid or invalid SMILES
                    # it has already been explored or is in the queue to be explored
                    if new_smi in found_valid or new_smi in found_invalid:
                        continue

                    # check if the new molecule is valid
                    if evaluator.is_valid_molecule(Molecule(new_smi), evaluations):
                        # if it is valid, add it to the set of valid SMILES and to the
                        # set of SMILES to explore
                        to_explore.add(new_smi)
                        found_valid.add(new_smi)
                    else:
                        # if it is invalid, add it to the set of invalid SMILES
                        # to avoid checking it again
                        found_invalid.add(new_smi)

    return found_valid, found_invalid


def enumerate_from_smiles_workers(
    starting_smiles: str,
    depth: int,
    evaluations: list[evaluator.Evaluation],
    nb_cpu: int,
) -> tuple[set[str], set[str]]:
    """Enumerate all reachable molecules starting from a SMILES up to a certain
    depth and check their validity before exploring them in parallel.

    Args:
        starting_smiles (str): SMILES to explore
        depth (int): Maximum depth to explore
        evaluations (list[evaluator.Evaluation]): filter to apply
        nb_cpu (int): number of CPU to use

    Returns:
        tuple[set[str], set[str]]: sets of valid and invalid SMILES
    """

    if nb_cpu == 0:
        nb_cpu = cpu_count()

    manager = Manager()
    explore_queue = manager.Queue()
    neighbor_queue = manager.Queue()
    lock = manager.Lock()
    task_counter = manager.Value(int, 0)
    found_valid = set()
    found_invalid = set()

    # add the starting SMILES to the set of valid or invalid SMILES
    if evaluator.is_valid_molecule(Molecule(starting_smiles), evaluations):
        found_valid.add(starting_smiles)
    else:
        found_invalid.add(starting_smiles)

    with lock:
        task_counter.value += 1
    # starting SMILES is the first molecule to explore
    explore_queue.put(starting_smiles)

    # start the workers
    with Pool(nb_cpu) as pool:
        for _ in range(nb_cpu):
            pool.apply_async(
                worker_find_neighbors,
                (explore_queue, neighbor_queue, depth, task_counter, lock),
            )

        while True:
            while not neighbor_queue.empty():
                new_smis = neighbor_queue.get()
                for new_smi in new_smis:
                    if new_smi in found_valid or new_smi in found_invalid:
                        continue

                    if evaluator.is_valid_molecule(Molecule(new_smi), evaluations):
                        explore_queue.put(new_smi)
                        with lock:
                            task_counter.value += 1
                        found_valid.add(new_smi)
                    else:
                        found_invalid.add(new_smi)

            # if all tasks are done, break the loop
            if task_counter.value == 0:
                break

    return found_valid, found_invalid


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic BFS enumeration (Phase 0 / Phase 1).
#
# Deterministic (independent of PYTHONHASHSEED): neighbors are iterated in
# sorted order and the budget versions stop on a number of molecules, not on
# time. Deduplication is done on the canonical (kekulized) SMILES.
#
# exhaustive_bfs / budget_bfs are the reference versions (one Molecule rebuilt
# at every validity test and expansion). The *_fast versions give the same
# results without the redundant re-parsing: neighbors are produced as SMILES
# (Action.apply_smiles), a new neighbor is parsed once, and the queue holds
# Molecule objects.
# ─────────────────────────────────────────────────────────────────────────────


def canonical(smiles: str) -> str:
    """Canonical kekulized SMILES (same key as find_neighbors)."""
    return Molecule(smiles).get_representation(MolecularGraph).canonical_smiles


def exhaustive_bfs(
    start_smiles: str,
    filters: list[evaluator.Evaluation],
    max_atoms: int,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[set[str], int]:
    """Enumerate exhaustively the space reachable from start_smiles, crossing
    only valid molecules.

    Args:
        start_smiles: starting SMILES.
        filters: validity filters, in the order they are applied.
        max_atoms: max number of heavy atoms (bounds the space).
        on_progress: callback(nb_valid, nb_seen) called periodically.

    Returns:
        (valid_set, seen_count): valid canonical SMILES reached, and number of
        distinct SMILES examined.
    """
    start = canonical(start_smiles)
    seen: set[str] = {start}
    valid: set[str] = set()
    queue: deque[str] = deque()

    if evaluator.is_valid_molecule(Molecule(start), filters):
        valid.add(start)
        queue.append(start)

    while queue:
        current = queue.popleft()
        for neighbor in sorted(find_neighbors(Molecule(current), 1)):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            if Molecule(neighbor).get_representation(MolecularGraph).nb_atoms > max_atoms:
                continue
            if evaluator.is_valid_molecule(Molecule(neighbor), filters):
                valid.add(neighbor)
                queue.append(neighbor)
        if on_progress and len(seen) % 500 == 0:
            on_progress(len(valid), len(seen))

    return valid, len(seen)


def budget_bfs(
    start_smiles: str,
    filters: list[evaluator.Evaluation],
    budget: int,
) -> dict[str, int]:
    """BFS stopped after `budget` distinct molecules have been examined.

    Args:
        start_smiles: starting SMILES.
        filters: validity filters, in the order they are applied.
        budget: number of distinct molecules to examine.

    Returns:
        dict(n_valid, n_invalid, seen).
    """
    start = canonical(start_smiles)
    seen: set[str] = {start}
    n_valid = 0
    n_invalid = 0
    queue: deque[str] = deque()

    if evaluator.is_valid_molecule(Molecule(start), filters):
        n_valid += 1
        queue.append(start)
    else:
        n_invalid += 1

    while queue and len(seen) < budget:
        current = queue.popleft()
        for neighbor in sorted(find_neighbors(Molecule(current), 1)):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            if evaluator.is_valid_molecule(Molecule(neighbor), filters):
                n_valid += 1
                queue.append(neighbor)
            else:
                n_invalid += 1

    return {"n_valid": n_valid, "n_invalid": n_invalid, "seen": len(seen)}


def _neighbor_keys(molecule: Molecule) -> list[str]:
    """Sorted canonical SMILES of the neighbors at one mutation, without
    building a Molecule for each action product."""
    molecule.compute_possible_actions()
    keys = {
        action.apply_smiles()  # type: ignore[attr-defined]
        for representation in molecule.possible_actions.values()
        for action_list in representation.values()
        for action in action_list
    }
    return sorted(keys)


def exhaustive_bfs_fast(
    start_smiles: str,
    filters: list[evaluator.Evaluation],
    max_atoms: int,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[set[str], int]:
    """Same results as exhaustive_bfs, without redundant re-parsing."""
    start_mol = Molecule(canonical(start_smiles))
    start = str(start_mol)
    seen: set[str] = {start}
    valid: set[str] = set()
    queue: deque[Molecule] = deque()

    if evaluator.is_valid_molecule(start_mol, filters):
        valid.add(start)
        queue.append(start_mol)

    while queue:
        current = queue.popleft()
        for smi in _neighbor_keys(current):
            if smi in seen:
                continue
            seen.add(smi)
            mol = Molecule(smi)
            if mol.get_representation(MolecularGraph).nb_atoms > max_atoms:
                continue
            if evaluator.is_valid_molecule(mol, filters):
                valid.add(smi)
                queue.append(mol)
        if on_progress and len(seen) % 500 == 0:
            on_progress(len(valid), len(seen))

    return valid, len(seen)


def budget_bfs_fast(
    start_smiles: str,
    filters: list[evaluator.Evaluation],
    budget: int,
) -> dict[str, int]:
    """Same counts as budget_bfs, without redundant re-parsing."""
    start_mol = Molecule(canonical(start_smiles))
    start = str(start_mol)
    seen: set[str] = {start}
    n_valid = 0
    n_invalid = 0
    queue: deque[Molecule] = deque()

    if evaluator.is_valid_molecule(start_mol, filters):
        n_valid += 1
        queue.append(start_mol)
    else:
        n_invalid += 1

    while queue and len(seen) < budget:
        current = queue.popleft()
        for smi in _neighbor_keys(current):
            if smi in seen:
                continue
            seen.add(smi)
            mol = Molecule(smi)
            if evaluator.is_valid_molecule(mol, filters):
                n_valid += 1
                queue.append(mol)
            else:
                n_invalid += 1

    return {"n_valid": n_valid, "n_invalid": n_invalid, "seen": len(seen)}
