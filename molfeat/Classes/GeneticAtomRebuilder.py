from molfeat.Classes.AtomRebuilder import AtomRebuilder
from molfeat.Classes.MoleculeFeatures import MoleculeFeatures
from molfeat.Classes.GeneticMol import GeneticMol, TabuSearchManager
from molfeat.utils.data_utils import ensure_nested_path

from math import ceil
from tqdm import tqdm
import random
import copy
import json
import xarray as xr
import numpy as np


class GeneticAtomRebuilder(AtomRebuilder):
    """
    Orchestrates the genetic algorithm to reconstruct molecular structures.
    """

    DEFAULT_MUTATION_ORDER = {"edit_bond": 0.2, "add_bond": 0.5, "remove_bond": 0.3}
    population: list[tuple[GeneticMol, list[str]]]
    n_mutations: int

    def __init__(
        self,
        atom_data_list: list[tuple],
        population_size=50,
        max_iterations=3000,
        keep_best=10,
        mutation_operators=None,
        tabu_size=10,
        n_mutations=2,
        iteration_best=100,
        save_history=False,
    ):
        super().__init__(atom_data_list)
        self._eval_call = 0
        self.history = dict()
        self.iteration_best = iteration_best
        self.best_solution = None
        self.molfeat = MoleculeFeatures(self._atom_data_list)
        self.save_history = save_history
        self.mutation_operators = (
            mutation_operators or self.DEFAULT_MUTATION_ORDER.copy()
        )

        # The Tabu manager is instantiated here and shared with the entire population.
        self.search_heuristic = TabuSearchManager(max_size=tabu_size)

        number_bonds = [
            3 * len(data[2]) + 2 * len(data[3]) + len(data[4])
            for data in self._atom_data_list
        ]
        self.atoms = list(self.molfeat.get_atoms_from_features())
        self.atoms_with_valence = list(zip(self.atoms, number_bonds))

        self.population_size = population_size
        self.max_iterations = max_iterations
        self.keep_best = keep_best
        self.n_mutations = n_mutations

        matrix_bond = self._initialize_bond_matrix()

        self.population = []
        for _ in range(self.population_size):
            genetic_mol = GeneticMol(
                self.atoms_with_valence, matrix_bond, self.search_heuristic
            )
            genetic_mol.add_random_bond()
            self.population.append((genetic_mol, []))

    def _initialize_bond_matrix(self) -> xr.DataArray:
        """Prepares the matrix of possible bonds, as in the original code."""
        num_atoms = len(self.atoms)
        matrix_bond = xr.DataArray(
            np.zeros((num_atoms, num_atoms, 3), dtype=np.int8),
            dims=["atom1", "atom2", "bond"],
            coords={
                "atom1": self.atoms,
                "atom2": self.atoms,
                "bond": ["single", "double", "triple"],
            },
        )
        key_map = [data[9] for data in self._atom_data_list]
        bond_targets_sets = [
            [set(data[4]), set(data[3]), set(data[2])] for data in self._atom_data_list
        ]

        i_coords, j_coords, bond_type_coords = [], [], []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                for bond_type_idx in range(3):
                    if (
                        key_map[j] in bond_targets_sets[i][bond_type_idx]
                        and key_map[i] in bond_targets_sets[j][bond_type_idx]
                    ):
                        i_coords.extend([i, j])
                        j_coords.extend([j, i])
                        bond_type_coords.extend([bond_type_idx, bond_type_idx])
        if i_coords:
            matrix_bond.values[i_coords, j_coords, bond_type_coords] = 1
        return matrix_bond

    def evaluate_solution(self, smiles: str) -> float:
        """Evaluates a solution using the MoleculeFeatures class, as requested."""
        mol = MoleculeFeatures(smiles)
        if mol is None:
            return float("inf")
        distance, _ = self.molfeat.evaluate_distance(mol, method="min")
        self._eval_call += 1
        return distance

    def _select_mutation_operator(self, element: GeneticMol) -> str:
        """Selects a mutation operator based on possible actions."""
        current_weights = self.mutation_operators.copy()
        key_to_add_weight = "add_bond"
        if not element.are_additions_available():
            key_to_add_weight = "edit_bond"
            current_weights[key_to_add_weight] += current_weights.pop("add_bond", 0.0)

        if not element.are_edits_available():
            if key_to_add_weight == "edit_bond":
                key_to_add_weight = "remove_bond"
            current_weights[key_to_add_weight] += current_weights.pop("edit_bond", 0.0)
        if not element.are_removal_available():
            if key_to_add_weight == "remove_bond":
                key_to_add_weight = "None"
            else:
                current_weights[key_to_add_weight] += current_weights.pop(
                    "remove_bond", 0.0
                )
        if key_to_add_weight == "None":
            return "nothing"
        operators = list(current_weights.keys())
        weights = list(current_weights.values())

        if sum(weights) == 0:
            return "nothing"
        return random.choices(operators, weights=weights, k=1)[0]

    def compute_solutions(
        self, all_solutions=False, n_solutions=1, verbose: bool = False
    ):
        """Runs the main loop of the genetic algorithm."""
        progress_bar = tqdm(
            range(self.max_iterations),
            desc="Compute solutions",
            disable=not verbose,
            leave=True,
        )
        i_best = 0
        for i in progress_bar:
            if i % 100 == 0 and self.population:
                diversity = len({str(mol) for mol, _ in self.population})
                progress_bar.set_postfix(
                    {
                        "diversity": f"{diversity}/{len(self.population)}",
                        "best_score": (
                            f"{self.best_solution[1]:.4f}"
                            if self.best_solution
                            else "N/A"
                        ),
                    }
                )

            new_population = []

            # Process the current population to create the next generation
            for element, parents in self.population:
                if self.best_solution and self.best_solution[1] == 0:
                    i_best += 1
                    if i_best >= self.iteration_best:
                        progress_bar.close()
                        # Save the evolution history
                        if self.save_history:
                            with open(
                                "molfeat_history.json", "w", encoding="utf-8"
                            ) as f:
                                json.dump(self.history, f, indent=4)
                        return self.best_solution, list(self._solutions)
                child = copy.deepcopy(element)

                for i in range(self.n_mutations):
                    mutation_operator = self._select_mutation_operator(child)
                    child.mutate(mutation_operator)
                smiles = child.convert_to_smiles()
                if not smiles:
                    continue

                score = self.evaluate_solution(smiles)
                if score == float("inf"):
                    continue

                child.score = score
                child.smiles = smiles

                element_smiles = element.convert_to_smiles()
                new_parent_history = parents + (
                    [element_smiles] if element_smiles else []
                )
                new_population.append((child, new_parent_history))
                ensure_nested_path(self.history, new_parent_history + [smiles])

                if self.best_solution is None or score < self.best_solution[1]:
                    self.best_solution = (smiles, score)
                    if verbose:
                        tqdm.write(
                            f"New best solution found: {smiles} (Score: {score:.4f})"
                        )

                if score == 0 and smiles not in self._solutions:
                    self._solutions.add(smiles)
                    if (
                        not all_solutions
                        and n_solutions is not None
                        and len(self._solutions) >= n_solutions
                    ):
                        progress_bar.close()

                        # Save the evolution history
                        if self.save_history:
                            with open(
                                "molfeat_history.json", "w", encoding="utf-8"
                            ) as f:
                                json.dump(self.history, f, indent=4)
                        return self.best_solution, list(self._solutions)

            # keep the best from the previous population.
            # if self.population:
            #     new_population.extend(
            #         sorted(self.population, key=lambda x: x[0].score)[: self.keep_best]
            #     )

            # Sort and select the new population
            # A small percetage to keep less best solutions and randomly others
            new_population.sort(key=lambda x: x[0].score)
            if random.random() < 0.1:
                half_best = new_population[: int(ceil(len(new_population) / 2))]
                random.shuffle(new_population)
                half_random = new_population[: int(ceil(len(new_population) / 2))]
                new_population = half_best + half_random
                new_population = new_population[: self.population_size]
            self.population = new_population[: self.population_size]
            if n_solutions is not None and not all_solutions:
                nb_best = 0
                for mol, _ in self.population:
                    if mol.score == 0:
                        nb_best += 1
                    if nb_best >= 10:

                        progress_bar.close()
                        # Save the evolution history
                        if self.save_history:
                            with open(
                                "molfeat_history.json", "w", encoding="utf-8"
                            ) as f:
                                json.dump(self.history, f, indent=4)
                        return self.best_solution, list(self._solutions)

            if not self.population:
                break

        progress_bar.close()
        # Save the evolution history
        if self.save_history:
            with open("molfeat_history.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4)
        return self.best_solution, list(self._solutions)

    def get_mol(self) -> set[str]:
        return self._solutions
