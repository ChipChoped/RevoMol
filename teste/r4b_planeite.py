"""
Validation de la règle R4b par la géométrie 3D (teste/r4b_planeite.ipynb).

R4b (evomol/newmut/matrix_rules.py, r4b_ring) refuse un grand cycle (> 8 atomes) s'il contient un carbone s, ou si
la somme de ses virages s'écarte de 360° de plus de FLAT_TOLERANCE = 40° (un atome libre peut tourner son H vers
l'intérieur, −120°, si le cycle a au moins INWARD_H_MIN_RING = 14 atomes). Aucune géométrie n'est calculée : on
regarde ici si les cycles acceptés sont plans en 3D et les refusés non plans.

1. Lot de molécules :
   - référence : annulènes, porphine, phtalocyanine, cycloparaphénylènes, cyclophane, éther-couronne… ;
   - mutations réelles : toutes les mutations candidates (nouvelles mutations, règles R1-R5) des molécules de
     notebooks/random_walk/results/newmut_time/temps_par_molecule.csv ; on garde les résultats d'au plus
     MAX_ATOMES atomes lourds qui ont un grand cycle et aucune autre violation que R4b, puis on en tire au hasard
     PAR_CLASSE par classe (acceptée, refusée angle, refusée carbone s).
2. Détail de R4b pour chaque grand cycle (r4b_details, même calcul que r4b_ring) : somme des virages, atomes libres,
   écart à 360° ; le verdict est comparé à celui de r4b_ring (erreur s'ils diffèrent).
3. Géométrie 3D : molécule RDKit construite sur la matrice (mêmes numéros d'atomes), sans perception de
   l'aromaticité (sinon les annulènes 4n+2 n'ont pas d'isomère E/Z), isomères E/Z et R/S (au plus ISOMERES, tirés au
   hasard au-delà), conformères ETKDGv3 (CONFORMERES par isomère, au moins STRUCTURES_MIN par molécule ;
   coordonnées de départ aléatoires, R/S non imposés, EMBED_ATTEMPTS essais par conformère), optimisation MMFF94
   (UFF si MMFF n'a pas les paramètres).
   ETKDG ne respecte pas toujours la configuration E/Z demandée dans un grand cycle (vérifié sur le [18]annulène) :
   l'énumération sert seulement à varier les départs, et on note la configuration cis/trans réellement obtenue des
   doubles liaisons du grand cycle.
4. Planéité de chaque grand cycle dans chaque structure :
   - rms_plan (Å) : écart quadratique moyen des atomes du cycle à leur plan moyen ; ecart_max (Å) : le plus grand ;
   - torsion (°) : écart moyen des dièdres du cycle à 0° ou 180° (dièdres passant par un atome linéaire exclus) ;
   - angle_p (°) : angle moyen entre la direction p de chaque atome du cycle (normale au plan de ses voisins ;
     atomes à 4 voisins ou linéaires exclus) et la normale du plan moyen : 0° = orbitales p parallèles (Hückel).
   Résumé par cycle : valeurs de la structure d'énergie minimale (_min_E) et meilleure valeur sur toutes les
   structures (_mieux), avec l'énergie de la structure la plus plane au-dessus du minimum (dE_mieux, kcal/mol).

Résultats : teste/resultats/ (teste/resultats/rapide/ avec --rapide) : lot.csv (molécules et classes),
cycles.csv (une ligne par grand cycle), molecules.csv (une ligne par molécule).

Usage :
    python teste/r4b_planeite.py [--rapide] [--par-classe 150] [--max-atomes 60] [--isomeres 64]
                                 [--isomeres-reference 1024] [--conformeres 4] [--structures-min 32]
                                 [--workers 8] [--graine 0]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "newmut"))

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    GetStereoisomerCount,
    StereoEnumerationOptions,
)

from evomol.newmut import matrix_rules as mr

OUT = ROOT / "teste" / "resultats"
PARENTS = ROOT / "notebooks" / "random_walk" / "results" / "newmut_time" / "temps_par_molecule.csv"
CLASSES = ("acceptée", "refusée angle", "refusée carbone s")
BOND_TYPES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def _annulene(n: int) -> str:
    return "C1=C" + "C=C" * (n // 2 - 1) + "1"


def _cpp(n: int) -> str:
    """[n]cycloparaphénylène : n benzènes liés en para."""
    return "c1%99ccc(cc1)" + "c1ccc(cc1)" * (n - 2) + "c1ccc%99cc1"


# nom -> (SMILES, nombre d'atomes lourds attendu)
REFERENCE: dict[str, tuple[str, int]] = {
    **{f"[{n}]annulène": (_annulene(n), n) for n in (10, 12, 14, 16, 18, 20, 22, 24)},
    "porphine": ("C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=C2)N5", 24),
    "phtalocyanine": ("C1=CC=C2C(=C1)C3=NC4=C5C=CC=CC5=C(N4)N=C6C7=CC=CC=C7C(=N6)N=C8C9=CC=CC=C9C(=N8)N=C2N3", 40),
    **{f"[{n}]CPP": (_cpp(n), 6 * n) for n in (5, 6, 7, 8)},
    "1,5,9-tridéshydro[12]annulène": ("C1#CC=CC#CC=CC#CC=C1", 12),
    "cyclo[18]carbone": ("C1#C" + "C#C" * 8 + "1", 18),
    "[2.2]paracyclophane": ("C1Cc2ccc(cc2)CCc2ccc1cc2", 16),
    "18-couronne-6": ("C1COCCOCCOCCOCCOCCO1", 18),
    "cyclododécane": ("C1CCCCCCCCCCC1", 12),
}


############################################################
#                     R4B, CYCLE PAR CYCLE                 #
############################################################


def r4b_details(graph: Any, ring: list[int], small: list[list[int]], small_sets: list[set[int]]) -> dict[str, Any]:
    """Même calcul que mr.r4b_ring, avec le détail : verdict, somme des virages, atomes libres, H tournés vers
    l'intérieur, écart à 360° (k permis par la règle) et écart sans le seuil de 14 atomes."""

    def rings_with(*atoms: int) -> list[list[int]]:
        return [r for r, r_set in zip(small, small_sets) if all(x in r_set for x in atoms)]

    n = len(ring)
    row: dict[str, Any] = {"taille": n, "cycle": " ".join(map(str, ring)), "carbone_s": None, "virages": np.nan,
                           "libres": np.nan, "h_dedans": np.nan, "ecart": np.nan, "ecart_sans_seuil14": np.nan}
    total = 0.0
    free = 0
    for k in range(n):
        x, a, b = ring[k], ring[(k + n - 1) % n], ring[(k + 1) % n]
        others = [y for y in graph.adj[x] if y not in (a, b)]
        if graph.is_s_carbon(x):
            return {**row, "verdict": "refusée carbone s", "carbone_s": x}
        if any(graph.order[x][y] == 3 for y in graph.adj[x]):
            continue
        ring_ab = rings_with(x, a, b)
        if ring_ab:
            phi = 360 - mr._ring_angle(ring_ab[0])  # pylint: disable=protected-access
        else:
            t = others[0] if others else None
            ring_a = rings_with(x, a, t) if t is not None else []
            ring_b = rings_with(x, b, t) if t is not None else []
            if ring_a and ring_b:
                phi = 360 - mr._ring_angle(ring_a[0]) - mr._ring_angle(ring_b[0])  # pylint: disable=protected-access
            elif ring_a:
                phi = (360 - mr._ring_angle(ring_a[0])) / 2  # pylint: disable=protected-access
            elif ring_b:
                phi = (360 - mr._ring_angle(ring_b[0])) / 2  # pylint: disable=protected-access
            else:
                phi = 120
                if not others:
                    free += 1
        total += 180 - phi
    max_inward = free if n >= mr.INWARD_H_MIN_RING else 0
    deviations = [abs(total - 120 * k - 360) for k in range(free + 1)]
    inward = min(range(max_inward + 1), key=deviations.__getitem__)
    return {**row, "verdict": "acceptée" if deviations[inward] <= mr.FLAT_TOLERANCE else "refusée angle",
            "virages": total, "libres": free, "h_dedans": inward, "ecart": deviations[inward],
            "ecart_sans_seuil14": min(deviations)}


def ring_details(types: list[str], matrix: np.ndarray) -> list[dict[str, Any]]:
    """Détail de R4b pour chaque grand cycle ; vérifie que le verdict est celui de mr.r4b_ring."""
    graph = mr._graph(types, matrix)  # pylint: disable=protected-access
    small, large, small_sets = mr.split_rings(graph.rings())
    rows = []
    for ring in large:
        row = r4b_details(graph, ring, small, small_sets)
        violation = mr.r4b_ring(graph, ring, small, small_sets)
        expected = "acceptée" if violation is None else (
            "refusée carbone s" if len(violation.atoms) == 1 else "refusée angle")
        if row["verdict"] != expected:
            raise AssertionError(f"r4b_details ({row['verdict']}) et r4b_ring ({expected}) diffèrent : {row}")
        rows.append(row)
    return rows


def molecule_class(rings: list[dict[str, Any]]) -> str:
    """Classe d'une molécule : refusée carbone s si un cycle l'est, sinon refusée angle, sinon acceptée."""
    verdicts = {row["verdict"] for row in rings}
    for verdict in ("refusée carbone s", "refusée angle"):
        if verdict in verdicts:
            return verdict
    return "acceptée"


############################################################
#                         LOT                              #
############################################################


def encode(types: list[str], matrix: np.ndarray) -> tuple[str, str]:
    """(atomes, liaisons) en texte : « C C N », « 0-1:2 1-2:1 »."""
    n = len(types)
    bonds = [f"{i}-{j}:{int(matrix[i][j])}" for i in range(n) for j in range(i + 1, n) if matrix[i][j]]
    return " ".join(types), " ".join(bonds)


def decode(atoms: str, bonds: str) -> tuple[list[str], np.ndarray]:
    """Inverse de encode (diagonale : électrons de valence, comme mr.to_matrix)."""
    types = atoms.split()
    matrix = np.zeros((len(types), len(types)), dtype=np.int64)
    for item in bonds.split():
        pair, order = item.split(":")
        i, j = map(int, pair.split("-"))
        matrix[i][j] = matrix[j][i] = int(order)
    for i, element in enumerate(types):
        matrix[i][i] = mr.VALENCE_ELECTRONS.get(element, 0)
    return types, matrix


def smiles_matrix(smiles: str) -> tuple[list[str], np.ndarray]:
    """(types, matrice) d'un SMILES, sur sa structure de Kekulé (comme mr.to_matrix)."""
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    matrix = np.asarray(Chem.GetAdjacencyMatrix(mol, useBO=True), dtype=np.int64)
    for i, element in enumerate(types):
        matrix[i][i] = mr.VALENCE_ELECTRONS.get(element, 0)
    return types, matrix


def reference_lot() -> list[dict[str, Any]]:
    """Molécules de référence ; vérifie le nombre d'atomes et la présence d'un grand cycle."""
    rows = []
    for name, (smiles, n_atoms) in REFERENCE.items():
        types, matrix = smiles_matrix(smiles)
        if len(types) != n_atoms:
            raise ValueError(f"{name} : {len(types)} atomes au lieu de {n_atoms} ({smiles})")
        rings = ring_details(types, matrix)
        if not rings:
            raise ValueError(f"{name} : aucun grand cycle ({smiles})")
        atoms, bonds = encode(types, matrix)
        rows.append({"nom": name, "source": "référence", "classe": molecule_class(rings),
                     "smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smiles)), "n_atomes": len(types),
                     "mutation": "", "parent": "", "atomes": atoms, "liaisons": bonds})
    return rows


def mutation_lot(per_class: int, max_atoms: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Résultats des mutations candidates qui ont un grand cycle et aucune autre violation que R4b, tirés au
    hasard par classe. Renvoie aussi le nombre de molécules différentes trouvées par classe."""
    # pylint: disable=import-outside-toplevel
    import newmut_time as ntime  # se place à la racine du projet
    import newmut_walk as nw
    from evomol.newmut import rule_filtered as rf
    from evomol.representation import MolecularGraph, Molecule

    nw.configure(ntime.configs(ntime.MAX_ATOMS)["nouvelles"])
    parents = pd.read_csv(PARENTS).drop_duplicates("smiles")
    parents = parents[parents.n_atoms <= max_atoms + 1]  # une mutation retire au plus un atome
    found: dict[str, dict[str, Any]] = {}
    n_candidates = 0
    t0 = time.perf_counter()
    for smiles in parents.smiles:
        molecule = Molecule(smiles)
        before = molecule.get_representation(MolecularGraph)
        for action_class in MolecularGraph.action_space:
            for action in nw._candidates(action_class, molecule):  # pylint: disable=protected-access
                n_candidates += 1
                try:
                    after = rf.result_graph(action, before)  # type: ignore[arg-type]
                except Exception:  # pylint: disable=broad-except
                    continue
                types, matrix = mr.to_matrix(after)
                if len(types) > max_atoms:
                    continue
                if any(v.rule_id != "R4b" for v in mr.rule_violations(types, matrix)):
                    continue
                rings = ring_details(types, matrix)
                if not rings:
                    continue
                key = after.aromatic_canonical_smiles
                if key in found:
                    continue
                atoms, bonds = encode(types, matrix)
                found[key] = {"source": "mutation", "classe": molecule_class(rings), "smiles": key,
                              "n_atomes": len(types), "mutation": action_class.class_name(), "parent": smiles,
                              "atomes": atoms, "liaisons": bonds}
    print(f"{len(parents)} molécules de départ, {n_candidates} mutations candidates, {len(found)} résultats "
          f"différents avec un grand cycle et R4b seule en jeu ({time.perf_counter() - t0:.0f} s)")
    rng = random.Random(seed)
    counts = {}
    rows = []
    for label in CLASSES:
        group = sorted((row for row in found.values() if row["classe"] == label), key=lambda row: row["smiles"])
        counts[label] = len(group)
        rows += rng.sample(group, min(per_class, len(group)))
    for k, row in enumerate(rows):
        row["nom"] = f"m{k:04d}"
    return rows, counts


############################################################
#                      GÉOMÉTRIE 3D                        #
############################################################


def rdkit_mol(types: list[str], matrix: np.ndarray) -> Chem.Mol:
    """Molécule RDKit de la matrice (mêmes numéros d'atomes), sanitisée sans perception de l'aromaticité."""
    rwmol = Chem.RWMol()
    for element in types:
        rwmol.AddAtom(Chem.Atom(element))
    n = len(types)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j]:
                rwmol.AddBond(i, j, BOND_TYPES[int(matrix[i][j])])
    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    return mol


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


LINEAR = 170  # angle (°) au-delà duquel trois atomes sont considérés alignés


def ring_measures(pos: np.ndarray, ring: list[int], neighbors: list[list[int]],
                  double_bonds: list[int]) -> dict[str, Any]:
    """Planéité d'un cycle dans une structure (voir le docstring du module), et configuration cis / trans (c / t)
    des doubles liaisons du cycle qui ne sont dans aucun petit cycle (double_bonds : positions k des liaisons
    ring[k]-ring[k+1])."""
    n = len(ring)
    points = pos[ring]
    centered = points - points.mean(axis=0)
    normal = np.linalg.svd(centered)[2][-1]
    distances = centered @ normal

    torsions = []
    for k in range(n):
        a, b, c, d = (pos[ring[(k + i) % n]] for i in range(4))
        if _angle(a - b, c - b) > LINEAR or _angle(b - c, d - c) > LINEAR:
            continue
        tau = abs(_dihedral(a, b, c, d))
        torsions.append(min(tau, 180 - tau))

    p_angles = []
    for x in ring:
        nb = neighbors[x]
        if len(nb) == 3:
            direction = np.cross(pos[nb[1]] - pos[nb[0]], pos[nb[2]] - pos[nb[0]])
        elif len(nb) == 2 and _angle(pos[nb[0]] - pos[x], pos[nb[1]] - pos[x]) <= LINEAR:
            direction = np.cross(pos[nb[0]] - pos[x], pos[nb[1]] - pos[x])
        else:
            continue
        p_angle = _angle(direction, normal)
        p_angles.append(min(p_angle, 180 - p_angle))

    config = "".join(
        "c" if abs(_dihedral(*(pos[ring[(k + i) % n]] for i in (-1, 0, 1, 2)))) < 90 else "t"
        for k in double_bonds
    )
    return {"rms_plan": float(np.sqrt(np.mean(distances**2))), "ecart_max": float(np.max(np.abs(distances))),
            "torsion": float(np.mean(torsions)) if torsions else np.nan,
            "angle_p": float(np.mean(p_angles)) if p_angles else np.nan, "config": config}


def _free_double_bonds(mol: Chem.Mol, ring: list[int]) -> list[int]:
    """Positions k des doubles liaisons ring[k]-ring[k+1] qui ne sont dans aucun cycle de moins de 8 atomes."""
    info = mol.GetRingInfo()
    n = len(ring)
    positions = []
    for k in range(n):
        bond = mol.GetBondBetweenAtoms(ring[k], ring[(k + 1) % n])
        if bond.GetBondType() == Chem.BondType.DOUBLE and not any(
                info.IsBondInRingOfSize(bond.GetIdx(), size) for size in range(3, 8)):
            positions.append(k)
    return positions


def stereoisomers(mol: Chem.Mol, options: StereoEnumerationOptions) -> tuple[int, list[Chem.Mol], str]:
    """(nombre de combinaisons, isomères, énumération). RDKit 2024.03 échoue sur certaines doubles liaisons dont un
    atome ne porte qu'un H implicite (imine C=NH) : on réessaie avec les H explicites, sinon on garde la molécule."""
    for method, start in (("normale", mol), ("H explicites", Chem.AddHs(mol))):
        try:
            return GetStereoisomerCount(start, options), list(EnumerateStereoisomers(start, options)), method
        except RuntimeError:
            continue
    return 1, [mol], "aucune"


MEASURES = ("rms_plan", "ecart_max", "torsion", "angle_p")
# Plongement : essais par conformère. Avec les réglages par défaut (10 × le nombre d'atomes, chiralité imposée),
# certains isomères tirés au hasard, impossibles dans des cycles fusionnés, bloquaient ETKDG plus de 9 minutes
# (m0000 de l'essai rapide) ; sans imposer R/S et avec des coordonnées de départ aléatoires, 4 conformères en 1 s.
EMBED_ATTEMPTS = 20


def geometry(task: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Structures 3D d'une molécule et planéité de chacun de ses grands cycles."""
    RDLogger.DisableLog("rdApp.*")
    t0 = time.perf_counter()
    types, matrix = decode(task["atomes"], task["liaisons"])
    rings = [list(map(int, cycle.split())) for cycle in task["cycles"]]
    mol = rdkit_mol(types, matrix)

    options = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=task["isomeres"],
                                       rand=task["graine"])
    n_total, isomers, enumeration = stereoisomers(mol, options)
    n_conformers = max(task["conformeres"], math.ceil(task["structures_min"] / len(isomers)))
    # lues avant MMFF, qui marque les cycles aromatiques sur la molécule elle-même
    double_bonds = [_free_double_bonds(mol, ring) for ring in rings]

    structures: list[tuple[float, list[dict[str, Any]]]] = []
    failures = not_converged = 0
    force_field = ""
    for k, isomer in enumerate(isomers):
        with_h = Chem.AddHs(isomer)
        params = AllChem.ETKDGv3()
        params.randomSeed = 1000 * task["graine"] + k
        params.useRandomCoords = True
        params.enforceChirality = False  # voir EMBED_ATTEMPTS
        params.maxIterations = EMBED_ATTEMPTS
        ids = list(AllChem.EmbedMultipleConfs(with_h, n_conformers, params))
        if not ids:
            failures += 1
            continue
        if AllChem.MMFFHasAllMoleculeParams(with_h):
            force_field = "MMFF94"
            results = AllChem.MMFFOptimizeMoleculeConfs(with_h, maxIters=2000)
        else:
            force_field = "UFF"
            results = AllChem.UFFOptimizeMoleculeConfs(with_h, maxIters=2000)
        neighbors = [[nb.GetIdx() for nb in atom.GetNeighbors()] for atom in with_h.GetAtoms()]
        for conf_id, (flag, energy) in zip(ids, results):
            not_converged += flag != 0
            pos = with_h.GetConformer(conf_id).GetPositions()
            structures.append((energy, [ring_measures(pos, ring, neighbors, bonds)
                                        for ring, bonds in zip(rings, double_bonds)]))

    molecule_row = {"nom": task["nom"], "enumeration": enumeration, "n_isomeres_total": n_total,
                    "n_isomeres": len(isomers),
                    "conformeres_par_isomere": n_conformers, "n_structures": len(structures),
                    "isomeres_echoues": failures, "non_convergees": not_converged, "champ": force_field,
                    "secondes": time.perf_counter() - t0}
    ring_rows = []
    for r, cycle in enumerate(task["cycles"]):
        row: dict[str, Any] = {"nom": task["nom"], "cycle": cycle}
        if structures:
            energies = np.array([energy for energy, _ in structures])
            values = {m: np.array([measures[r][m] for _, measures in structures]) for m in MEASURES}
            low = int(np.argmin(energies))
            flattest = int(np.argmin(values["rms_plan"]))
            for m in MEASURES:
                row[f"{m}_min_E"] = values[m][low]
                row[f"{m}_mieux"] = np.nanmin(values[m]) if not np.all(np.isnan(values[m])) else np.nan
            row["dE_mieux"] = energies[flattest] - energies[low]
            row["config_min_E"] = structures[low][1][r]["config"]
            row["config_mieux"] = structures[flattest][1][r]["config"]
            row["configs_vues"] = len({measures[r]["config"] for _, measures in structures})
        ring_rows.append(row)
    return molecule_row, ring_rows


############################################################
#                         PROGRAMME                        #
############################################################


def main() -> None:
    """Construit le lot, calcule les géométries, écrit les tableaux."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rapide", action="store_true",
                        help="essai : 20 molécules par classe, 16 isomères (64 pour la référence)")
    parser.add_argument("--par-classe", type=int, default=150)
    parser.add_argument("--max-atomes", type=int, default=60)
    parser.add_argument("--isomeres", type=int, default=64)
    parser.add_argument("--isomeres-reference", type=int, default=1024)
    parser.add_argument("--conformeres", type=int, default=4)
    parser.add_argument("--structures-min", type=int, default=32)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--graine", type=int, default=0)
    args = parser.parse_args()
    out = OUT
    if args.rapide:
        args.par_classe, args.isomeres, args.isomeres_reference = 20, 16, 64
        out = OUT / "rapide"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    mutations, counts = mutation_lot(args.par_classe, args.max_atomes, args.graine)
    print("résultats différents par classe :", counts)
    pd.Series(counts, name="molecules_trouvees").rename_axis("classe").to_csv(out / "classes_trouvees.csv")
    lot =pd.DataFrame(reference_lot() + mutations)
    lot = lot[["nom", "source", "classe", "smiles", "n_atomes", "mutation", "parent", "atomes", "liaisons"]]
    lot.to_csv(out / "lot.csv", index=False)
    print("lot :", lot.groupby(["source", "classe"]).size().to_dict())

    details = []
    tasks = []
    for k, row in enumerate(lot.itertuples()):
        types, matrix = decode(row.atomes, row.liaisons)
        rings = ring_details(types, matrix)
        details += [{"nom": row.nom, **ring} for ring in rings]
        tasks.append({"nom": row.nom, "atomes": row.atomes, "liaisons": row.liaisons,
                      "cycles": [ring["cycle"] for ring in rings], "graine": args.graine + k,
                      "isomeres": args.isomeres_reference if row.source == "référence" else args.isomeres,
                      "conformeres": args.conformeres, "structures_min": args.structures_min})
    tasks.sort(key=lambda task: -len(task["atomes"].split()))  # les plus grosses d'abord

    molecule_rows, ring_rows = [], []
    with get_context("spawn").Pool(args.workers) as pool:
        for done, (molecule_row, rows) in enumerate(pool.imap_unordered(geometry, tasks), start=1):
            molecule_rows.append(molecule_row)
            ring_rows += rows
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} molécules ({time.perf_counter() - t0:.0f} s)", flush=True)

    molecules = lot.drop(columns=["atomes", "liaisons"]).merge(pd.DataFrame(molecule_rows), on="nom")
    cycles = (lot[["nom", "source", "classe", "smiles", "n_atomes", "mutation"]]
              .merge(pd.DataFrame(details), on="nom")
              .merge(pd.DataFrame(ring_rows), on=["nom", "cycle"]))
    molecules.to_csv(out / "molecules.csv", index=False)
    cycles.to_csv(out / "cycles.csv", index=False)
    print(f"{len(molecules)} molécules, {len(cycles)} grands cycles -> {out} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
