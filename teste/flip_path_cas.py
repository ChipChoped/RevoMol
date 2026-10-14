"""
Exemples de path flip (FlipAlternatingPathMG) permis et refusés par les règles, dessinés avant / après.

Molécules : celles de results/newmut_walk/refus/candidates.csv.gz (classes 1-10 à 31-50) plus quelques petites
molécules choisies à la main. Pour chacune, toutes les inversions candidates (newmut_walk._rows), avec la règle
qui refuse. Chemin inversé surligné : vert si permis, rouge si refusé.

Sorties dans notebooks/random_walk/results/newmut_walk/test/flip_path/ :
  liste.csv               toutes les inversions (molécule, chemin, résultat, permise, règle, cas) ;
  permises_page_XX.png    exemples permis ;
  refusees_page_XX.png    exemples refusés (d'abord le cas D : une extrémité porte déjà une double liaison).

Lancer depuis la racine : /Applications/anaconda3/envs/revomol/bin/python teste/flip_path_cas.py
"""

import os
import sys
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "scripts", "newmut"))
import newmut_walk as nw  # noqa: E402  (se place à la racine du projet)

OUT = os.path.join(ROOT, "notebooks/random_walk/results/newmut_walk/test/flip_path")
SOURCE = os.path.join(ROOT, "notebooks/random_walk/results/newmut_walk/refus/candidates.csv.gz")
ACTION = "FlipAlternatingPathMG"
CLASSES = ["1-10", "11-20", "21-30", "31-50"]   # au-delà, les dessins sont illisibles
HAND = ["CC=CC=CC", "CC=CC=CC=CC", "CC1=CC=C(C)O1", "CC1=CC=C(C)N1", "Cc1ccccc1C", "Cc1ccc(C)cc1",
        "Cc1cccc(C)c1", "CC=CC(C)=O", "NC=CC=CN", "OC=CC=CO", "C=CC=CC", "CC1=CC=CC1",
        "Cc1ccc2ccccc2c1", "Cc1cccc2ccccc12", "CC=CC=CC=O", "C=C1C=CC(C)=C1"]
PER_MOLECULE = 2     # au plus n exemples de chaque sorte par molécule, pour varier
PAIRS_PER_PAGE = 8   # 2 paires par ligne
MAX_PAGES = 6        # par sorte


def end_with_double(graph, path) -> bool:
    """Une extrémité du chemin porte déjà une double (ou triple) liaison hors du chemin."""
    for end, nxt in ((path[0], path[1]), (path[-1], path[-2])):
        if any(graph.bond_order(end, o) >= 2 for o in graph.atoms_bonded_to(end) if o != nxt):
            return True
    return False


def path_bonds(mol, path):
    return [mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in zip(path, path[1:])]


def main() -> None:
    nw.configure(nw.make_config())
    os.makedirs(OUT, exist_ok=True)
    src = pd.read_csv(SOURCE)
    smiles_list = HAND + list(dict.fromkeys(src[(src["mutation"] == ACTION) & src["taille"].isin(CLASSES)]["smiles"]))

    rows = []
    for smiles in smiles_list:
        molecule, cands = nw._rows(smiles, ACTION)  # pylint: disable=protected-access
        before = molecule.get_representation(nw.MolecularGraph)
        for c in cands:
            if c["graph"] is None:
                continue
            path = list(c["action"].path)
            if c["permise"]:
                cas = "permise"
            elif end_with_double(before, path):
                cas = "D : extrémité déjà doublement liée"
            else:
                cas = "autre refus"
            rows.append({"smiles": before.canonical_smiles, "n°": c["n°"], "chemin": path,
                         "mutation": c["mutation"], "résultat": c["résultat"], "permise": c["permise"],
                         "règle": c["raison"], "cas": cas, "_before": before, "_after": c["graph"]})
    table = pd.DataFrame(rows)
    table.drop(columns=["_before", "_after"]).to_csv(os.path.join(OUT, "liste.csv"), index=False)
    print(f"{len(smiles_list)} molécules, {len(table)} inversions candidates")
    print(table.groupby(["cas", "règle"]).size().to_string())

    def pages(selected, color, prefix):
        per_mol = defaultdict(int)
        kept = []
        for r in selected:
            if per_mol[r["smiles"]] < PER_MOLECULE:
                per_mol[r["smiles"]] += 1
                kept.append(r)
        kept = kept[:PAIRS_PER_PAGE * MAX_PAGES]
        for p in range(0, len(kept), PAIRS_PER_PAGE):
            mols, legends, atoms, bonds, acol, bcol = [], [], [], [], [], []
            for r in kept[p:p + PAIRS_PER_PAGE]:
                path = r["chemin"]
                verdict = "permise" if r["permise"] else f"refusee ({r['règle']})"
                for g, legend in ((r["_before"], f"avant : {r['smiles']}\nchemin {path}"),
                                  (r["_after"], f"apres : {r['résultat']}\n{verdict}")):
                    mol = Chem.Mol(g.mol)
                    mols.append(mol)
                    legends.append(legend)
                    atoms.append(path)
                    bonds.append(path_bonds(mol, path))
                    acol.append({a: color for a in path})
                    bcol.append({b: color for b in bonds[-1]})
            img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(380, 300), legends=legends,
                                       highlightAtomLists=atoms, highlightBondLists=bonds,
                                       highlightAtomColors=acol, highlightBondColors=bcol,
                                       drawOptions=nw._options())  # pylint: disable=protected-access
            name = os.path.join(OUT, f"{prefix}_page_{p // PAIRS_PER_PAGE + 1:02d}.png")
            img.save(name)
        print(f"{prefix} : {len(kept)} exemples dessinés")

    recs = table.to_dict("records")
    pages([r for r in recs if r["permise"]], nw.GREEN, "permises")
    refused = ([r for r in recs if r["cas"].startswith("D")] + [r for r in recs if r["cas"] == "autre refus"])
    pages(refused, nw.RED, "refusees")


if __name__ == "__main__":
    main()
