"""
Toutes les mutations possibles sur l'énumération exhaustive V2 (toutes les molécules de 1 à 8 atomes lourds, calculées
sur le serveur : Dataset/Enumeration_rule_V2/resultats), comme l'exhaustif de V1 (teste_newmut_V1/contextes_mutations
.ipynb : verdict_classes, all_smallest_examples), avec les réglages, les règles et les sous-familles de
mutations_V2.ipynb (cellules lues dans le notebook).

Molécules : celles de l'énumération sans S ni P (atomes des marches V2 : C, O, N, B, F ; filtrées en lecture sur le
serveur, exhaustif/enumeration_V2_N1-8_sans_SP.tsv.gz), puis seulement celles qui respectent toutes les règles de
mutations_V2 (V2 + RB : l'énumération a été faite sans RB, ses B# sont écartés), liste gardée dans
exhaustif/molecules_conformes_V2.tsv.gz avec le bilan par taille (bilan_conformes_V2.csv).

L'énumération est trop grande pour garder toutes les mutations en mémoire (comme le fait V1) : les molécules sont
traitées par paquets et seuls sont gardés, par sous-famille, les comptes (mêmes colonnes que verdict_classes de V1)
et les plus petits exemples (mêmes choix que smallest_examples de V1 : atomes, puis SMILES, puis n°), ce qui donne le
même résultat.

Écrit dans parametre_30000/exhaustif/ (lu par synthese_familles_V2.py) : {Famille}.csv et {Famille}/{sous-famille}.png.
    /Applications/anaconda3/envs/revomol/bin/python teste_newmut_V2/Mutation_V2/exhaustif_V2.py [familles...]
        [--essai N] (N molécules seulement, pour mesurer le temps) [--paquet K] [--sortie dossier]
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTEBOOK = HERE / "mutations_V2.ipynb"
DATA = HERE / "exhaustif"
SOURCE = DATA / "enumeration_V2_N1-8_sans_SP.tsv.gz"
CONFORMES = DATA / "molecules_conformes_V2.tsv.gz"
N_PETITS = 10                 # plus petits exemples par colonne d'image (V1 : N_PETITS = 10)
FAMILLES = ["AddAtomMG", "ChangeBondMG", "CutAtomMG", "InsertCarbonMG", "RemoveAtomMG", "SubstituteShiftMG"]

NS: dict[str, Any] = {}


def setup() -> dict[str, Any]:
    """Cellules d'imports, de réglages et règles, et SOUS_FAMILLE de mutations_V2.ipynb (dans ce processus)."""
    os.chdir(ROOT)
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    NS.update({"__name__": "__main__", "display": lambda *a, **k: None})
    wanted = ("import marches_V2", "mv2.install(", "def SOUS_FAMILLE")
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and any(w in source for w in wanted):
            exec(compile(source, str(NOTEBOOK), "exec"), NS)  # pylint: disable=exec-used
            NS["display"] = lambda *a, **k: None
    return NS


# ------------------------------------------------------------------------------ molécules conformes
def _conformity_chunk(items: list[tuple[int, str]]) -> list[tuple[int, str, str]]:
    regles = NS["REGLES"]
    out = []
    for n, smiles in items:
        try:
            broken = sorted({v.rule_id for v in regles.of_smiles(smiles)})
        except ValueError:
            broken = ["illisible"]
        out.append((n, smiles, ", ".join(broken)))
    return out


def conformant_molecules(n_jobs: int) -> list[tuple[int, str]]:
    """(atomes, SMILES) des molécules de l'énumération qui respectent toutes les règles (fichier gardé)."""
    if CONFORMES.exists():
        with gzip.open(CONFORMES, "rt", encoding="utf-8") as f:
            return [(int(n), s) for n, s in (line.rstrip("\n").split("\t") for line in f)]
    with gzip.open(SOURCE, "rt", encoding="utf-8") as f:
        items = [(int(n), s) for n, s in (line.rstrip("\n").split("\t") for line in f)]
    chunks = [items[k:k + 5000] for k in range(0, len(items), 5000)]
    start = time.perf_counter()
    with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
        results = [row for part in pool.imap(_conformity_chunk, chunks) for row in part]
    bilan: dict[int, Counter] = {}
    for n, _, broken in results:
        bilan.setdefault(n, Counter())["énumérées"] += 1
        bilan[n]["conformes" if not broken else f"écartées : {broken}"] += 1
    import pandas as pd  # pylint: disable=import-outside-toplevel
    table = pd.DataFrame(bilan).T.fillna(0).astype(int).sort_index()
    table.index.name = "atomes"
    table.to_csv(DATA / "bilan_conformes_V2.csv")
    print(table.to_string())
    kept = [(n, s) for n, s, broken in results if not broken]
    with gzip.open(CONFORMES, "wt", encoding="utf-8") as f:
        f.writelines(f"{n}\t{s}\n" for n, s in kept)
    print(f"{len(kept)} molécules conformes sur {len(items)} ({time.perf_counter() - start:.0f} s)")
    return kept


# ------------------------------------------------------------------------------ test par paquets
def _key(row: dict) -> tuple:
    return (row["atomes"], row["SMILES"], row["n°"])


def _keep_smallest(rows: list[dict], new: list[dict], n: int) -> list[dict]:
    return sorted(rows + new, key=_key)[:n]


class Aggregate:
    """Comptes d'une sous-famille (colonnes de verdict_classes de V1) et ses plus petits exemples."""

    def __init__(self, family: str, name: str):
        self.family, self.name = family, name
        self.n = self.acc = self.molecules = 0
        self.broken: Counter = Counter()
        self.alone: Counter = Counter()
        self.min_acc: int | None = None
        self.min_ref: int | None = None
        self.max_atoms = 0
        self.accepted: list[dict] = []
        self.refused: dict[str, list[dict]] = {}

    def add(self, group: Any, pr: Any) -> None:
        accepted = group[group.permise]
        refused = group[~group.permise]
        self.n += len(group)
        self.acc += len(accepted)
        self.molecules += group["molécule"].nunique()
        self.max_atoms = max(self.max_atoms, int(group.atomes.max()))
        if len(accepted):
            m = int(accepted.atomes.min())
            self.min_acc = m if self.min_acc is None else min(self.min_acc, m)
            self.accepted = _keep_smallest(self.accepted, _smallest(accepted, N_PETITS), N_PETITS)
        if len(refused):
            m = int(refused.atomes.min())
            self.min_ref = m if self.min_ref is None else min(self.min_ref, m)
            for text, count in refused["règles cassées"].value_counts().items():
                rules = pr._broken(text)  # pylint: disable=protected-access
                for rule in rules:
                    self.broken[rule] += int(count)
                if len(rules) == 1:
                    self.alone[rules[0]] += int(count)
            for rule in {r for text in refused["règles cassées"].unique() for r in pr._broken(text)}:  # pylint: disable=W0212
                part = refused[pr.rule_mask(refused["règles cassées"], rule)]
                self.refused[rule] = _keep_smallest(self.refused.get(rule, []), _smallest(part, N_PETITS), N_PETITS)

    def row(self) -> dict:
        def text(counter: Counter) -> str:
            return ", ".join(name if c == 1 else f"{name} ×{c}" for name, c in counter.most_common()) or "—"

        verdict = "toujours autorisée" if self.acc == self.n else "toujours interdite" if self.acc == 0 else "dépend"
        return {"famille": self.family, "sous-famille": self.name, "verdict": verdict, "candidates": self.n,
                "molécules": self.molecules, "acceptées": self.acc, "refusées": self.n - self.acc,
                "% acceptées": round(100 * self.acc / self.n, 1),
                "% refusées": round(100 * (self.n - self.acc) / self.n, 1),
                "règles qui refusent": text(self.broken), "règle seule (refus)": text(self.alone),
                "plus petite acceptée (atomes)": self.min_acc, "plus petite refusée (atomes)": self.min_ref}


def _smallest(rows: Any, n: int) -> list[dict]:
    """Comme _smallest de V1 : les n plus petites lignes (atomes, puis SMILES, puis n°)."""
    if len(rows) > n:
        rows = rows[rows.atomes <= rows.atomes.nsmallest(n).max()]
    columns = ["molécule", "SMILES", "atomes", "famille", "sous-famille", "n°", "mutation", "permise", "règles cassées"]
    return rows.sort_values(["atomes", "SMILES", "n°"]).head(n)[columns].to_dict("records")


def run_family(family: str, molecules: list[tuple[int, str]], out: Path, chunk: int, n_jobs: int) -> None:
    pr, regles = NS["pr"], NS["REGLES"]
    start = time.perf_counter()
    aggregates: dict[str, Aggregate] = {}
    removed = 0
    smiles = [s for _, s in molecules]
    for k in range(0, len(smiles), chunk):
        essai = pr.mutation_test(smiles[k:k + chunk], regles, NS["SOUS_FAMILLE"], None, NS["AJOUT_COMPLET"],
                                 familles=[family], progress_every=0, n_jobs=n_jobs)
        if essai.empty:   # paquet sans aucune mutation de la famille (ex. RemoveAtomMG : aucun atome en bout de chaîne)
            continue
        mask = essai["règles cassées"].fillna("").str.split(", ").apply(lambda r: "RB" in r)
        removed += int(mask.sum())          # 0 attendu : les B# ne sont pas proposés (mv2.hide_forbidden_bore)
        essai = essai[~mask]
        essai["molécule"] += k
        for name, group in essai.groupby("sous-famille", sort=False):
            aggregates.setdefault(name, Aggregate(family, name)).add(group, pr)
        done = min(k + chunk, len(smiles))
        print(f"  {family} : {done}/{len(smiles)} molécules, {sum(a.n for a in aggregates.values())} mutations, "
              f"{time.perf_counter() - start:.0f} s", flush=True)
    import pandas as pd  # pylint: disable=import-outside-toplevel
    table = pd.DataFrame([a.row() for a in aggregates.values()]).set_index(["famille", "sous-famille"])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{family}.csv")
    folder = out / family
    folder.mkdir(exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    for a in aggregates.values():
        columns = [("Acceptées", pr.CHANGED, a.accepted, f" ({a.acc})")]
        for rule in regles.names:
            if a.broken.get(rule):
                columns.append((f"Refusées par {rule}", regles.colors[rule], a.refused[rule], f" ({a.broken[rule]})"))
        if len(columns) == 1:
            columns.append(("Refusées", pr.PALETTE[0], [], " (0)"))
        title = f"{a.name} · plus petits exemples sur {len(molecules)} molécules (≤ {a.max_atoms} atomes)"
        pr.examples_page(columns, regles, title, NS["AJOUT_COMPLET"], [family],
                         str(folder / f"{pr.subfamily_file_name(a.name)}.png"), (230, 190))
    print(f"{family} : {len(aggregates)} sous-familles, {sum(a.n for a in aggregates.values())} mutations, "
          f"{removed} B# retirées, {time.perf_counter() - start:.0f} s → {out / (family + '.csv')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("familles", nargs="*", default=FAMILLES)
    parser.add_argument("--essai", type=int, default=None, help="seulement les N premières molécules de 8 atomes")
    parser.add_argument("--paquet", type=int, default=20000)
    parser.add_argument("--coeurs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--sortie", default=str(HERE / "parametre_30000" / "exhaustif"))
    args = parser.parse_args()
    setup()
    molecules = conformant_molecules(args.coeurs)
    if args.essai:
        molecules = [m for m in molecules if m[0] == 8][:args.essai]
    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_info.json").write_text(json.dumps({"molécules": len(molecules), "source": str(SOURCE.name),
                                                    "atomes": "1 à 8", "éléments": "C N O B F"}, ensure_ascii=False))
    for family in args.familles:
        run_family(family, molecules, out, args.paquet, args.coeurs)
    print("FINI", flush=True)


if __name__ == "__main__":
    sys.exit(main())
