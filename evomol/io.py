from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from evomol.representation import MolecularGraph, Molecule
from evomol.search.parameters import search_parameters


def canonical_smiles(molecule: Molecule) -> str:
    mol_graph = molecule.get_representation(MolecularGraph)
    return Chem.MolToSmiles(mol_graph.mol, canonical=True)


def serialize_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "__dict__"):
        return json.dumps(value.__dict__, ensure_ascii=False, default=str)

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)

    return str(value)


def population_rows(
    population: list[Molecule],
    generation: int,
    fitness_name: str,
    evaluation_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows = []

    for rank, molecule in enumerate(population, start=1):
        smi = canonical_smiles(molecule)
        values = molecule.values

        row = {
            "rank": rank,
            "generation": generation,
            "smiles": smi,
            "fitness": serialize_value(values.get(fitness_name)),
            "n_function_calls": (
                None if evaluation_counts is None else evaluation_counts.get(smi, 0)
            ),
        }

        for name in sorted(values):
            row[f"eval:{name}"] = serialize_value(values[name])

        rows.append(row)

    return rows


def write_tsv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    preferred = [
        "rank",
        "generation",
        "smiles",
        "fitness",
        "n_function_calls",
    ]

    columns = []
    for col in preferred:
        if any(col in row for row in rows):
            columns.append(col)

    for row in rows:
        for col in row:
            if col not in columns:
                columns.append(col)

    with path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_population_tsv(
    population: list[Molecule],
    path: str | Path,
    generation: int,
    fitness_name: str | None = None,
    evaluation_counts: dict[str, int] | None = None,
) -> None:
    if fitness_name is None:
        fitness_name = search_parameters.fitness_criteria

    rows = population_rows(
        population=population,
        generation=generation,
        fitness_name=fitness_name,
        evaluation_counts=evaluation_counts,
    )
    write_tsv(rows, path)


def save_run_metadata_json(path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f_out:
        json.dump(metadata, f_out, indent=2, ensure_ascii=False, default=str)

