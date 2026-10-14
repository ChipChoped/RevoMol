"""Filtre le fichier CNO par R1..R5 dans l'ordre R5 > R2 > R1 > R3 > R4b > R4a (arrêt à la 1re violation)."""
import sys
import time
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, "/Users/LeaClautour/Desktop/RevoMol")
from evomol.newmut import matrix_rules as mr  # noqa: E402

RDLogger.DisableLog("rdApp.*")
SULFUR = 6
SRC = "/Users/LeaClautour/Downloads/chembl34_CNO.tsv"
DST = "/Users/LeaClautour/Downloads/chembl34_CNO_R1-R5.tsv"


def first_violation(g):
    if any(mr.r5_atom(g, i) for i in range(g.n)):
        return "R5"
    bridges = g.bridges()
    if any(mr.r2_bond(g, a, b, (a, b) not in bridges) for a in range(g.n) for b in g.adj[a] if a < b):
        return "R2"
    if any(mr.r1_atom(g, i, SULFUR) for i in range(g.n)):
        return "R1"
    if any(mr.r3_atom(g, i) for i in range(g.n)):
        return "R3"
    small, large, small_sets = mr.split_rings(g.rings(bridges))
    if any(mr.r4b_ring(g, ring, small, small_sets) for ring in large):
        return "R4b"
    if mr.r4a_violations(g, small, small_sets):
        return "R4a"
    return None


lines = open(SRC).read().splitlines()
header, rows = lines[0], lines[1:]
stats = Counter()
r1_charged = 0
kept = [header]
t0 = time.perf_counter()
for line in rows:
    mol = Chem.MolFromSmiles(line.split("\t")[1])
    if mol is None:
        stats["illisible"] += 1
        continue
    Chem.Kekulize(mol, clearAromaticFlags=True)
    types = [a.GetSymbol() for a in mol.GetAtoms()]
    g = mr._Graph(types, np.asarray(Chem.GetAdjacencyMatrix(mol, useBO=True), dtype=np.int64))
    v = first_violation(g)
    stats[v or "garde"] += 1
    if v == "R1" and any(a.GetFormalCharge() for a in mol.GetAtoms()):
        r1_charged += 1
    if v is None:
        kept.append(line)
elapsed = time.perf_counter() - t0
open(DST, "w").write("\n".join(kept) + "\n")
print(f"{len(rows)} molécules en {elapsed:.0f} s ({1e6*elapsed/len(rows):.0f} µs/molécule)")
for k in ["R5", "R2", "R1", "R3", "R4b", "R4a", "illisible", "garde"]:
    print(f"  {k:10} {stats[k]:7} ({100*stats[k]/len(rows):.1f} %)")
print(f"  dont R1 sur une molécule chargée : {r1_charged}")
