"""
États s / d / t des atomes de ChEMBL (teste_newmut_V2/ChEMBL_comparaison/etats_chembl.ipynb).

Point de départ pour réécrire les règles de newmut : on regarde comment les atomes des vraies molécules se répartissent
entre les trois états de matrix_rules._tag, sur les 1 163 333 molécules de ChEMBL 34 qui n'ont que des atomes
C, N, O, F, B, en un seul fragment et sans atome chargé (donnees/chembl34_CNOFSB.tsv, extrait sur le serveur le
2026-10-20 avec C N O F S B ; en ont été retirées ensuite les 97 590 molécules chargées, nitro [N+](=O)[O-] compris,
puis les 525 713 qui ont un S). L'option sans_charges refait le tri des charges sur un autre fichier.

État d'un atome, lu sur la structure de Kekulé (comme matrix_rules._tag, ici pour tous les éléments) :
« t » s'il porte une triple, « d » s'il porte une double, « s » s'il n'a que des simples.

  - load_chembl : identifiants et SMILES ;
  - compute : parcourt toutes les molécules (plusieurs cœurs), renvoie
      * par_molecule : une ligne par molécule, nombre d'atomes lourds et nombre d'atomes de chaque élément × état ;
      * contextes : nombre d'atomes par (élément, état, voisins lourds, H, charge, cycle, aromatique, doubles) ;
    gardés en cache dans resultats/ (refait si le fichier de données change) ;
  - tables de synthèse et tracés pour le notebook.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from multiprocessing import get_context
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "donnees", "chembl34_CNOFSB.tsv")
OUT_DIR = os.path.join(HERE, "resultats")

ELEMENTS = ("C", "N", "O", "F", "S", "B")
ETATS = ("s", "d", "t")
COLONNES = [e + t for e in ELEMENTS for t in ETATS]           # "Cs", "Cd", "Ct", "Ns", …
CONTEXTE = ["element", "etat", "voisins", "H", "charge", "cycle", "aromatique", "doubles"]

# Couleurs des trois états (palette catégorielle de référence, ordre fixe).
COULEURS = {"s": "#2a78d6", "d": "#eb6834", "t": "#1baf7a"}


# ==============================================================================
#  ÉTAT D'UN ATOME
# ==============================================================================
def etat(atom: Chem.Atom) -> str:
    """« t » (une triple), « d » (une double), « s » (que des simples), sur la structure de Kekulé."""
    orders = [b.GetBondType() for b in atom.GetBonds()]
    if Chem.BondType.TRIPLE in orders:
        return "t"
    if Chem.BondType.DOUBLE in orders:
        return "d"
    return "s"


def has_charge(mol: Chem.Mol) -> bool:
    """True si un atome porte une charge formelle (le groupe nitro [N+](=O)[O-] compte comme chargé)."""
    return any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())


def molecule_atoms(smiles: str, sans_charges: bool = False) -> list[tuple] | str | None:
    """Une ligne par atome lourd (les H explicites, dont [2H], ne comptent pas) : élément, état, voisins lourds, H,
    charge, dans un cycle, aromatique (SMILES de départ), nombre de doubles. None si le SMILES ne se lit pas ;
    « chargée » si sans_charges et qu'un atome est chargé."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if sans_charges and has_charge(mol):
        return "chargée"
    try:
        Chem.Kekulize(mol)                       # types de liaison de Kekulé, drapeaux aromatiques gardés
    except Chem.KekulizeException:
        return None
    rows = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        orders = [b.GetBondType() for b in atom.GetBonds()]
        rows.append((
            atom.GetSymbol(),
            etat(atom),
            sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() != 1),
            atom.GetTotalNumHs(includeNeighbors=True),
            atom.GetFormalCharge(),
            atom.IsInRing(),
            atom.GetIsAromatic(),
            orders.count(Chem.BondType.DOUBLE),
        ))
    return rows


# ==============================================================================
#  CALCUL SUR TOUT ChEMBL
# ==============================================================================
def load_chembl(path: str = DATA) -> pd.DataFrame:
    """chembl_id, canonical_smiles."""
    return pd.read_csv(path, sep="\t")


def _chunk(args: tuple[list[str], bool]) -> tuple[np.ndarray, Counter, int, int]:
    """Comptes par molécule (n_lourds + COLONNES ; -1 si illisible, -2 si chargée) et contextes d'un morceau."""
    smiles, sans_charges = args
    index = {c: k for k, c in enumerate(COLONNES)}
    counts = np.zeros((len(smiles), 1 + len(COLONNES)), dtype=np.int32)
    contextes: Counter = Counter()
    bad = charged = 0
    for i, smi in enumerate(smiles):
        rows = molecule_atoms(smi, sans_charges)
        if rows is None:
            counts[i, 0] = -1
            bad += 1
            continue
        if rows == "chargée":
            counts[i, 0] = -2
            charged += 1
            continue
        counts[i, 0] = len(rows)
        for row in rows:
            counts[i, 1 + index[row[0] + row[1]]] += 1
        contextes.update(rows)
    return counts, contextes, bad, charged


def _cache_key(path: str, sans_charges: bool) -> str:
    st = os.stat(path)
    return f"{st.st_size}_{int(st.st_mtime)}" + ("_sans_charges" if sans_charges else "")


def compute(df: pd.DataFrame, n_jobs: int = 1, chunk: int = 20000, path: str = DATA, sans_charges: bool = False,
            refaire: bool = False, id_col: str = "chembl_id", smiles_col: str = "canonical_smiles",
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """par_molecule (id_col, n_lourds, COLONNES) et contextes (CONTEXTE + « atomes »), en cache dans resultats/
    (un cache par fichier `path` d'où vient df). Les molécules que RDKit ne lit pas ou ne kekulise pas sont écartées,
    et avec sans_charges celles qui ont un atome chargé (leurs nombres sont affichés)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    key = _cache_key(path, sans_charges)
    nom = os.path.splitext(os.path.basename(path))[0]
    f_mol = os.path.join(OUT_DIR, f"par_molecule_{nom}_{key}.parquet")
    f_ctx = os.path.join(OUT_DIR, f"contextes_{nom}_{key}.parquet")
    if not refaire and os.path.exists(f_mol) and os.path.exists(f_ctx):
        return pd.read_parquet(f_mol), pd.read_parquet(f_ctx)

    t0 = time.time()
    smiles = df[smiles_col].tolist()
    chunks = [(smiles[k:k + chunk], sans_charges) for k in range(0, len(smiles), chunk)]
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs > 1:
        with get_context("spawn").Pool(n_jobs) as pool:
            results = pool.map(_chunk, chunks)
    else:
        results = [_chunk(c) for c in chunks]

    counts = np.vstack([r[0] for r in results])
    contextes: Counter = Counter()
    for r in results:
        contextes.update(r[1])
    bad = sum(r[2] for r in results)
    charged = sum(r[3] for r in results)

    par_molecule = pd.DataFrame(counts, columns=["n_lourds", *COLONNES])
    par_molecule.insert(0, id_col, df[id_col].to_numpy())
    par_molecule = par_molecule[par_molecule["n_lourds"] >= 0].reset_index(drop=True)
    ctx = pd.DataFrame([(*k, v) for k, v in contextes.items()], columns=[*CONTEXTE, "atomes"])
    ctx = ctx.sort_values("atomes", ascending=False).reset_index(drop=True)

    par_molecule.to_parquet(f_mol)
    ctx.to_parquet(f_ctx)
    print(f"{len(df)} molécules lues en {time.time() - t0:.0f} s ; écartées : {bad} (SMILES illisible ou non "
          f"kekulisable), {charged} (atome chargé) ; gardées : {len(par_molecule)} ; cache : {f_mol}")
    return par_molecule, ctx


# ==============================================================================
#  SYNTHÈSES
# ==============================================================================
def atomes_par_etat(ctx: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'atomes par élément × état, et part de chaque état dans l'élément (%)."""
    t = ctx.pivot_table(index="element", columns="etat", values="atomes", aggfunc="sum", fill_value=0)
    t = t.reindex(index=[e for e in ELEMENTS if e in t.index], columns=list(ETATS), fill_value=0)
    t["total"] = t.sum(axis=1)
    t.loc["tous"] = t.sum()
    for e in ETATS:
        t[f"% {e}"] = 100 * t[e] / t["total"]
    return t


def molecules_avec(par_molecule: pd.DataFrame) -> pd.DataFrame:
    """Nombre et part des molécules qui ont au moins un atome de chaque élément × état."""
    n = len(par_molecule)
    rows = []
    for e in ELEMENTS:
        for t in ETATS:
            k = int((par_molecule[e + t] > 0).sum())
            rows.append((e, t, k, 100 * k / n))
    out = pd.DataFrame(rows, columns=["element", "etat", "molecules", "%"])
    return out.pivot(index="element", columns="etat", values=["molecules", "%"]).reindex(list(ELEMENTS))


def nombre_par_molecule(par_molecule: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'atomes s, d, t par molécule (tous éléments) : colonnes s, d, t, n_lourds, et parts (%)."""
    out = pd.DataFrame({t: par_molecule[[e + t for e in ELEMENTS]].sum(axis=1) for t in ETATS})
    out["n_lourds"] = par_molecule["n_lourds"]
    for t in ETATS:
        out[f"% {t}"] = 100 * out[t] / out["n_lourds"].where(out["n_lourds"] > 0)
    return out


def resume(nb: pd.DataFrame) -> pd.DataFrame:
    """Moyenne, médiane, quantiles et maximum du nombre et de la part d'atomes s, d, t par molécule."""
    cols = ["n_lourds", *ETATS, *(f"% {t}" for t in ETATS)]
    return nb[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T


def parts_par_taille(nb: pd.DataFrame, max_atomes: int = 80) -> pd.DataFrame:
    """Part moyenne (%) des atomes s, d, t selon le nombre d'atomes lourds (jusqu'à max_atomes), et nombre de
    molécules de chaque taille."""
    sub = nb[nb["n_lourds"] <= max_atomes]
    g = sub.groupby("n_lourds")
    out = g[[f"% {t}" for t in ETATS]].mean()
    out["molecules"] = g.size()
    return out


def contextes_de(ctx: pd.DataFrame, element: str, etat: str | None = None,
                 par: Iterable[str] = ("voisins", "H"), top: int | None = None) -> pd.DataFrame:
    """Atomes d'un élément (et d'un état) regroupés selon les colonnes `par`, avec leur part (%)."""
    sub = ctx[ctx["element"] == element]
    if etat is not None:
        sub = sub[sub["etat"] == etat]
    keys = ["etat", *par] if etat is None else list(par)
    out = sub.groupby(keys)["atomes"].sum().sort_values(ascending=False).to_frame()
    out["%"] = 100 * out["atomes"] / out["atomes"].sum()
    return out.head(top) if top else out


def compare_par_etat(t_mol: pd.DataFrame, t_pi: pd.DataFrame) -> pd.DataFrame:
    """Part (%) de s, d, t par élément dans les molécules et dans les systèmes π (tables de atomes_par_etat),
    côte à côte, avec l'écart (points de %, systèmes π moins molécules)."""
    cols = [f"% {e}" for e in ETATS]
    out = pd.concat({"molécules": t_mol[cols + ["total"]], "systèmes π": t_pi[cols + ["total"]]}, axis=1)
    for c in cols:
        out[("écart", c)] = out[("systèmes π", c)] - out[("molécules", c)]
    return out


# ==============================================================================
#  TRACÉS
# ==============================================================================
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_etats_par_element(t: pd.DataFrame):
    """Barres empilées : part de s, d, t dans chaque élément (100 %), nombre d'atomes en étiquette."""
    import matplotlib.pyplot as plt
    rows = [e for e in t.index if e != "tous"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bottom = np.zeros(len(rows))
    for e in ETATS:
        vals = t.loc[rows, f"% {e}"].to_numpy()
        ax.bar(rows, vals, bottom=bottom, color=COULEURS[e], label=e, width=0.6,
               edgecolor="white", linewidth=1.5)
        bottom += vals
    for k, r in enumerate(rows):
        ax.text(k, 101, f"{int(t.loc[r, 'total']):,}".replace(",", " "), ha="center", va="bottom",
                fontsize=8, color="#555555")
    ax.set_ylabel("part des atomes de l'élément (%)")
    ax.set_ylim(0, 110)
    ax.legend(title="état", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_distribution(nb: pd.DataFrame, max_atomes: int = 60):
    """Histogrammes du nombre d'atomes s, d et t par molécule (un panneau par état)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    for ax, e in zip(axes, ETATS):
        vals = nb[e].clip(upper=max_atomes)
        counts = vals.value_counts().sort_index()
        ax.bar(counts.index, counts.to_numpy(), color=COULEURS[e], width=0.85)
        ax.set_title(f"atomes {e} par molécule", fontsize=10)
        ax.set_xlabel(f"nombre (≥ {max_atomes} regroupés)" if vals.max() >= max_atomes else "nombre")
        _style(ax)
    axes[0].set_ylabel("molécules")
    fig.tight_layout()
    return fig


def plot_parts_par_taille(pt: pd.DataFrame):
    """Part moyenne des atomes s, d, t selon la taille de la molécule."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for e in ETATS:
        ax.plot(pt.index, pt[f"% {e}"], color=COULEURS[e], linewidth=2, label=e)
        ax.text(pt.index[-1] + 0.8, pt[f"% {e}"].iloc[-1], e, color="#333333", va="center", fontsize=9)
    ax.set_xlabel("atomes lourds")
    ax.set_ylabel("part moyenne des atomes (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="état", frameon=False, bbox_to_anchor=(1.04, 1), loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_compare_etats(t_mol: pd.DataFrame, t_pi: pd.DataFrame):
    """Deux panneaux de barres empilées (molécules, systèmes π) : part de s, d, t dans chaque élément."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (titre, t) in zip(axes, (("molécules", t_mol), ("systèmes π (distincts)", t_pi))):
        rows = [e for e in t.index if e != "tous" and t.loc[e, "total"] > 0] + ["tous"]
        bottom = np.zeros(len(rows))
        for e in ETATS:
            vals = t.loc[rows, f"% {e}"].to_numpy()
            ax.bar(rows, vals, bottom=bottom, color=COULEURS[e], label=e, width=0.6, edgecolor="white", linewidth=1.5)
            bottom += vals
        for k, r in enumerate(rows):
            ax.text(k, 101, f"{int(t.loc[r, 'total']):,}".replace(",", " "), ha="center", va="bottom",
                    fontsize=7, color="#555555")
        ax.set_title(titre, fontsize=10)
        ax.set_ylim(0, 110)
        _style(ax)
    axes[0].set_ylabel("part des atomes de l'élément (%)")
    axes[1].legend(title="état", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_compare_distribution(nb_mol: pd.DataFrame, nb_pi: pd.DataFrame):
    """Part des atomes s, d, t par objet : histogrammes (en % des objets) pour les molécules et les systèmes π."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    bins = np.linspace(0, 100, 21)
    for ax, e in zip(axes, ETATS):
        for nb, label, style in ((nb_mol, "molécules", "-"), (nb_pi, "systèmes π", "--")):
            vals = nb[f"% {e}"].dropna()
            h, _ = np.histogram(vals, bins=bins)
            ax.step(bins[:-1], 100 * h / max(len(vals), 1), where="post", color=COULEURS[e], linestyle=style,
                    linewidth=2, label=label)
        ax.set_title(f"part d'atomes {e}", fontsize=10)
        ax.set_xlabel("% des atomes de l'objet")
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("% des objets")
    fig.tight_layout()
    return fig
