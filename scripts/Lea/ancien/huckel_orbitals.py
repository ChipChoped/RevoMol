"""
Orbitales de Hückel (λ, occupation, coefficients) des molécules visitées par les marches d'un plan.

Recalcule Hückel à partir des SMILES enregistrés dans OUT_DIR/runs/*/rep*_walk.csv.gz : aucune marche
n'est relancée. Une ligne par (molécule, orbitale), de HOMO-3 à LUMO+3 par défaut ; une orbitale qui
n'existe pas pour une molécule (trop petite) est absente. Les coefficients cₖᵢ sont donnés pour tous
les atomes lourds, dans l'ordre des atomes du SMILES (colonnes `atoms` et `atom_types` dans le même
ordre). Pour des orbitales dégénérées, les coefficients ne sont définis qu'à une rotation près dans le
sous-espace dégénéré.

Usage :
    python scripts/Lea/ancien/huckel_orbitals.py notebooks/random_walk/ancien/results/plan_regles_filtres
    -> OUT_DIR/orbitales.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts", "Lea"))  # huckel_filters

# pylint: disable=wrong-import-position
import pandas as pd
from rdkit import RDLogger

from evomol import default_parameters as dp
import huckel_filters as hf
from evomol.evaluation import EvaluationError, Huckel
from evomol.evaluation.huckel import resolve_mo_index
from evomol.representation import MolecularGraph, Molecule

RDLogger.DisableLog("rdApp.*")
# SMILES -> Molecule (à l'import : aussi dans chaque processus) ; pas de limite de taille, les molécules
# lues ont déjà passé MAX_ATOMS dans leur marche
dp.setup_default_parameters(accepted_atoms=hf.HUCKEL_ACCEPTED_ATOMS, max_heavy_atoms=10_000)

ORBITALS = ["HOMO-3", "HOMO-2", "HOMO-1", "HOMO", "LUMO", "LUMO+1", "LUMO+2", "LUMO+3"]


def molecule_orbitals(smiles: str, orbitals: list[str] = ORBITALS) -> list[dict]:
    """Une ligne par orbitale demandée qui existe pour cette molécule ([] si Hückel impossible)."""
    molecule = Molecule(smiles)
    try:
        result = Huckel().evaluate(molecule)
    except EvaluationError:
        return []
    atoms = [a.GetSymbol() for a in molecule.get_representation(MolecularGraph).mol.GetAtoms()]
    rows = []
    for orbital in orbitals:
        try:
            idx = resolve_mo_index(result, orbital)
        except EvaluationError:
            continue
        rows.append({
            "smiles": smiles, "orbital": orbital, "index": idx,
            "lambda": result.eigenvalues_lambda[idx], "occupation": result.occupations[idx],
            "coefficients": result.eigenvectors[idx], "atoms": atoms, "atom_types": result.atom_types,
            "n_pi_electrons": result.n_pi_electrons,
        })
    return rows


def visited_smiles(out_dir: str | Path) -> list[str]:
    """SMILES distincts de toutes les marches enregistrées dans out_dir/runs."""
    smiles: set[str] = set()
    for path in sorted(Path(out_dir).glob("runs/*/rep*_walk.csv.gz")):
        smiles.update(pd.read_csv(path, usecols=["smiles"]).smiles.dropna())
    return sorted(smiles)


def orbitals_table(smiles: list[str], orbitals: list[str] = ORBITALS, n_jobs: int | None = None) -> pd.DataFrame:
    """Orbitales de toutes ces molécules (en parallèle : n_jobs processus, None : nb de cœurs − 1)."""
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(n_jobs) as pool:
        rows = [r for mol_rows in pool.map(molecule_orbitals, smiles, [orbitals] * len(smiles), chunksize=200)
                for r in mol_rows]
    return pd.DataFrame(rows)


def lambda_table(orbitals: pd.DataFrame) -> pd.DataFrame:
    """λ par molécule (lignes) et orbitale (colonnes, de HOMO-3 à LUMO+3)."""
    order = [o for o in ORBITALS if o in set(orbitals.orbital)]
    return orbitals.pivot(index="smiles", columns="orbital", values="lambda")[order]


def extract(out_dir: str | Path, orbitals: list[str] = ORBITALS, n_jobs: int | None = None) -> pd.DataFrame:
    """Calcule les orbitales des molécules visitées et les écrit dans out_dir/orbitales.parquet."""
    smiles = visited_smiles(out_dir)
    table = orbitals_table(smiles, orbitals, n_jobs)
    table.to_parquet(Path(out_dir) / "orbitales.parquet", index=False)
    print(f"{len(smiles)} molécules distinctes, {len(table)} orbitales -> {Path(out_dir) / 'orbitales.parquet'}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", help="dossier de résultats d'un plan (contient runs/)")
    parser.add_argument("--orbitals", nargs="+", default=ORBITALS)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()
    extract(Path(args.out_dir).resolve(), args.orbitals, args.jobs)
