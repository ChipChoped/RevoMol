"""
Evolutionary algorithm with the new mutations (newmut).

The initial population should respect the matrix rules. If a molecule does
not, a warning is logged when the algorithm is created, without blocking: the
rules compare the number of violations before and after each mutation, so a
molecule already outside the rules stays mutable.
"""

from __future__ import annotations

from typing import Any

from evomol.logging import get_logger
from evomol.newmut.matrix_rules import molecular_graph_violations
from evomol.representation import MolecularGraph, Molecule
from evomol.search.evolutionary_algorithm import EvolutionaryAlgorithm


def warn_rule_violations(population: list[Molecule]) -> int:
    """Log a warning for each molecule that breaks the matrix rules.

    Returns:
        int: number of molecules outside the rules.
    """
    logger = get_logger()
    nb_outside = 0
    for molecule in population:
        mol_graph = molecule.get_representation(MolecularGraph)
        violations = molecular_graph_violations(mol_graph)
        if not violations:
            continue
        nb_outside += 1
        logger.warning(
            "Initial molecule outside the matrix rules: %s (%s)",
            mol_graph.canonical_smiles,
            "; ".join(f"{v.rule_id}: {v.message}" for v in violations),
            extra={
                "event": "initial_rule_violation",
                "smiles": mol_graph.canonical_smiles,
                "rules": [v.rule_id for v in violations],
            },
        )
    return nb_outside


class EvolutionaryAlgorithmNewMut(EvolutionaryAlgorithm):
    """EvolutionaryAlgorithm that warns about the initial molecules outside
    the matrix rules (same behaviour otherwise)."""

    def __init__(self, population: list[Molecule], *args: Any, **kwargs: Any):
        warn_rule_violations(population)
        super().__init__(population, *args, **kwargs)
