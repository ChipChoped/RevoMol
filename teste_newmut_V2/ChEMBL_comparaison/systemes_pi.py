"""
Systèmes π des molécules ChEMBL (teste_newmut_V2/ChEMBL_comparaison/etats_chembl.ipynb).

Tous les systèmes π d'au moins taille_min atomes de chaque molécule (pas seulement le plus grand), avec la définition
de evomol/evaluation/pi_systems.py : atome π = aromatique ou hybridé SP2 / SP (RDKit, règle du soufre comprise),
système π = composante connexe de ces atomes. pi_components n'y renvoie que les tailles ; pi_component_atoms
ci-dessous fait le même parcours et renvoie les atomes.

Un système est écrit comme le fragment de ses seuls atomes π et des liaisons entre eux (les substituants sont perdus,
les atomes qui les portaient prennent des H) : en Kekulé pour qu'il se relise toujours (un n aromatique qui perd son
substituant devient NH), puis relu et réécrit en SMILES canonique pour qu'un même système venu de deux molécules ait
le même SMILES.

  - molecule_systems : SMILES des systèmes π d'une molécule ;
  - extract : toutes les molécules (plusieurs cœurs) ; écrit donnees/systemes_pi_chembl34.tsv (systèmes distincts,
    sans doublons : smiles_pi, taille, occurrences, molecules, exemple) et renvoie aussi un tableau par molécule ;
  - synthèses et tracés pour le notebook.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from multiprocessing import get_context

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                                       # premier dossier parent qui contient evomol
while not os.path.isdir(os.path.join(ROOT, "evomol")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
for d in (ROOT, HERE):
    if d not in sys.path:
        sys.path.insert(0, d)

from evomol.evaluation.pi_systems import pi_atoms  # noqa: E402

import etats_chembl as ec  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PI_TSV = os.path.join(HERE, "donnees", "systemes_pi_chembl34.tsv")
COULEUR = "#2a78d6"


# ==============================================================================
#  SYSTÈMES π D'UNE MOLÉCULE
# ==============================================================================
def pi_component_atoms(mol: Chem.Mol, pi_set: set[int]) -> list[set[int]]:
    """Atomes de chaque système π (composante connexe des atomes π) ; même parcours que pi_systems.pi_components."""
    visited, components = set(), []
    for start in pi_set:
        if start in visited:
            continue
        comp, stack = {start}, [start]
        visited.add(start)
        while stack:
            u = stack.pop()
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                v = nb.GetIdx()
                if v in pi_set and v not in visited:
                    visited.add(v)
                    comp.add(v)
                    stack.append(v)
        components.append(comp)
    return components


def fragment_smiles(mol_kekule: Chem.Mol, atoms: set[int]) -> str | None:
    """SMILES canonique du fragment formé des atomes `atoms` et des liaisons entre eux (None s'il ne se relit pas)."""
    smi = Chem.MolFragmentToSmiles(mol_kekule, atomsToUse=sorted(atoms), kekuleSmiles=True, canonical=False)
    frag = Chem.MolFromSmiles(smi)
    return None if frag is None else Chem.MolToSmiles(frag)


def molecule_systems(smiles: str, taille_min: int = 4) -> tuple[int, list[tuple[str | None, int]]] | None:
    """(atomes lourds, [(SMILES du fragment, taille), …] pour chaque système π d'au moins taille_min atomes).
    None si le SMILES ne se lit pas."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    comps = [c for c in pi_component_atoms(mol, pi_atoms(mol)) if len(c) >= taille_min]
    if not comps:
        return mol.GetNumHeavyAtoms(), []
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    return mol.GetNumHeavyAtoms(), [(fragment_smiles(kek, c), len(c)) for c in comps]


# ==============================================================================
#  EXTRACTION SUR TOUT ChEMBL
# ==============================================================================
def _chunk(args: tuple[list[str], list[str], int]):
    ids, smiles, taille_min = args
    rows = np.zeros((len(smiles), 4), dtype=np.int32)       # n_lourds, n_systemes, plus_grand, atomes_pi
    occ: Counter = Counter()
    mols: Counter = Counter()
    taille: dict[str, int] = {}
    exemple: dict[str, str] = {}
    bad = unreadable = 0
    for i, (cid, smi) in enumerate(zip(ids, smiles)):
        res = molecule_systems(smi, taille_min)
        if res is None:
            rows[i, 0] = -1
            bad += 1
            continue
        n_heavy, systems = res
        sizes = [n for _, n in systems]
        rows[i] = (n_heavy, len(systems), max(sizes, default=0), sum(sizes))
        for frag, n in systems:
            if frag is None:
                unreadable += 1
                continue
            occ[frag] += 1
            taille[frag] = n
            exemple.setdefault(frag, cid)
        mols.update({f for f, _ in systems if f is not None})
    return rows, occ, mols, taille, exemple, bad, unreadable


def _cache_key(path: str, taille_min: int) -> str:
    st = os.stat(path)
    return f"{st.st_size}_{int(st.st_mtime)}_min{taille_min}"


def extract(df: pd.DataFrame, taille_min: int = 4, n_jobs: int = 1, chunk: int = 20000, path: str = ec.DATA,
            refaire: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(systemes, par_molecule).
      - systemes : un système π distinct par ligne (smiles_pi, taille, occurrences, molecules, exemple), trié par
        nombre de molécules ; écrit dans donnees/systemes_pi_chembl34.tsv ;
      - par_molecule : chembl_id, n_lourds, n_systemes, plus_grand, atomes_pi (systèmes d'au moins taille_min
        atomes), en cache dans resultats/.
    Refait si le fichier de données ou taille_min change (ou avec refaire)."""
    os.makedirs(ec.OUT_DIR, exist_ok=True)
    f_mol = os.path.join(ec.OUT_DIR, f"pi_par_molecule_{_cache_key(path, taille_min)}.parquet")
    f_sys = os.path.join(ec.OUT_DIR, f"pi_systemes_{_cache_key(path, taille_min)}.parquet")
    if not refaire and os.path.exists(f_mol) and os.path.exists(f_sys) and os.path.exists(PI_TSV):
        return pd.read_parquet(f_sys), pd.read_parquet(f_mol)

    t0 = time.time()
    ids, smiles = df["chembl_id"].tolist(), df["canonical_smiles"].tolist()
    chunks = [(ids[k:k + chunk], smiles[k:k + chunk], taille_min) for k in range(0, len(smiles), chunk)]
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs > 1:
        with get_context("spawn").Pool(n_jobs) as pool:
            results = pool.map(_chunk, chunks)
    else:
        results = [_chunk(c) for c in chunks]

    rows = np.vstack([r[0] for r in results])
    occ: Counter = Counter()
    mols: Counter = Counter()
    taille: dict[str, int] = {}
    exemple: dict[str, str] = {}
    for r in results:
        occ.update(r[1])
        mols.update(r[2])
        taille.update(r[3])
        for k, v in r[4].items():
            exemple.setdefault(k, v)
    bad = sum(r[5] for r in results)
    unreadable = sum(r[6] for r in results)

    par_molecule = pd.DataFrame(rows, columns=["n_lourds", "n_systemes", "plus_grand", "atomes_pi"])
    par_molecule.insert(0, "chembl_id", ids)
    par_molecule = par_molecule[par_molecule["n_lourds"] >= 0].reset_index(drop=True)
    systemes = pd.DataFrame({"smiles_pi": list(occ)})
    systemes["taille"] = systemes["smiles_pi"].map(taille)
    systemes["occurrences"] = systemes["smiles_pi"].map(occ)
    systemes["molecules"] = systemes["smiles_pi"].map(mols)
    systemes["exemple"] = systemes["smiles_pi"].map(exemple)
    systemes = systemes.sort_values(["molecules", "occurrences"], ascending=False).reset_index(drop=True)

    systemes.to_csv(PI_TSV, sep="\t", index=False)
    systemes.to_parquet(f_sys)
    par_molecule.to_parquet(f_mol)
    print(f"{len(df)} molécules lues en {time.time() - t0:.0f} s ({bad} illisibles) ; "
          f"{int(par_molecule['n_systemes'].sum())} systèmes π d'au moins {taille_min} atomes, "
          f"{len(systemes)} distincts ({unreadable} fragments illisibles écartés) ; écrits dans {PI_TSV}")
    return systemes, par_molecule


# ==============================================================================
#  SYNTHÈSES
# ==============================================================================
def systemes_par_molecule(par_molecule: pd.DataFrame, max_n: int = 6) -> pd.DataFrame:
    """Nombre et part des molécules selon leur nombre de systèmes π (max_n et plus regroupés)."""
    n = par_molecule["n_systemes"].clip(upper=max_n)
    out = n.value_counts().sort_index().to_frame("molecules")
    out.index = [str(k) if k < max_n else f"≥ {max_n}" for k in out.index]
    out.index.name = "systèmes π"
    out["%"] = 100 * out["molecules"] / len(par_molecule)
    return out


def resume_tailles(systemes: pd.DataFrame, par_molecule: pd.DataFrame) -> pd.DataFrame:
    """Résumé du nombre d'atomes : systèmes distincts, toutes les occurrences (pondérées), plus grand système de chaque
    molécule qui en a un, et part (%) des atomes de la molécule qui sont dans un système π."""
    pct = 100 * par_molecule["atomes_pi"] / par_molecule["n_lourds"].where(par_molecule["n_lourds"] > 0)
    series = {
        "taille (systèmes distincts)": systemes["taille"],
        "taille (toutes les occurrences)": systemes["taille"].repeat(systemes["occurrences"]),
        "plus grand système par molécule": par_molecule.loc[par_molecule["n_systemes"] > 0, "plus_grand"],
        "% des atomes dans un système π": pct,
    }
    return pd.DataFrame({k: v.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]) for k, v in series.items()}).T


def composition(par_pi: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'atomes de chaque élément dans les systèmes π distincts, et part des systèmes qui en ont."""
    rows = []
    total = int(par_pi["n_lourds"].sum())
    for e in ec.ELEMENTS:
        n = par_pi[[e + t for t in ec.ETATS]].sum(axis=1)
        if n.sum() == 0:
            continue
        rows.append((e, int(n.sum()), 100 * n.sum() / total, 100 * (n > 0).mean()))
    return pd.DataFrame(rows, columns=["element", "atomes", "% des atomes", "% des systèmes qui en ont"]).set_index("element")


# ==============================================================================
#  TRACÉS
# ==============================================================================
def plot_systemes_par_molecule(t: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(t.index, t["molecules"], color=COULEUR, width=0.6)
    for k, v in enumerate(t["%"]):
        ax.text(k, t["molecules"].iloc[k], f"{v:.1f} %", ha="center", va="bottom", fontsize=8, color="#555555")
    ax.set_xlabel("systèmes π d'au moins 4 atomes dans la molécule")
    ax.set_ylabel("molécules")
    ec._style(ax)
    fig.tight_layout()
    return fig


def plot_tailles(systemes: pd.DataFrame, par_molecule: pd.DataFrame, max_taille: int = 60):
    """Histogrammes du nombre d'atomes : systèmes distincts, toutes les occurrences, plus grand par molécule."""
    import matplotlib.pyplot as plt
    data = (
        ("systèmes distincts", systemes["taille"], None),
        ("toutes les occurrences", systemes["taille"], systemes["occurrences"]),
        ("plus grand système par molécule", par_molecule.loc[par_molecule["n_systemes"] > 0, "plus_grand"], None),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (titre, vals, poids) in zip(axes, data):
        v = vals.clip(upper=max_taille)
        counts = (v.value_counts() if poids is None else poids.groupby(v).sum()).sort_index()
        ax.bar(counts.index, counts.to_numpy(), color=COULEUR, width=0.85)
        ax.set_title(titre, fontsize=10)
        ax.set_xlabel(f"atomes du système (≥ {max_taille} regroupés)")
        ec._style(ax)
    axes[0].set_ylabel("nombre")
    fig.tight_layout()
    return fig


def plot_plus_grand_vs_taille(par_molecule: pd.DataFrame, max_atomes: int = 80):
    """Taille moyenne (et quartiles) du plus grand système π selon le nombre d'atomes lourds de la molécule."""
    import matplotlib.pyplot as plt
    sub = par_molecule[par_molecule["n_lourds"] <= max_atomes]
    g = sub.groupby("n_lourds")["plus_grand"]
    q1, med, q3 = g.quantile(0.25), g.median(), g.quantile(0.75)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.fill_between(med.index, q1, q3, color=COULEUR, alpha=0.2, linewidth=0, label="quartiles")
    ax.plot(med.index, med, color=COULEUR, linewidth=2, label="médiane")
    ax.plot(med.index, med.index, color="#999999", linewidth=1, linestyle=":", label="toute la molécule")
    ax.set_xlabel("atomes lourds de la molécule")
    ax.set_ylabel("atomes du plus grand système π")
    ax.legend(frameon=False)
    ec._style(ax)
    fig.tight_layout()
    return fig


def draw_frequents(systemes: pd.DataFrame, n: int = 20, par_ligne: int = 5):
    """Dessin des n systèmes π les plus fréquents (par nombre de molécules)."""
    from rdkit.Chem import Draw
    top = systemes.head(n)
    mols = [Chem.MolFromSmiles(s) for s in top["smiles_pi"]]
    legends = [f"{m:,} mol. · {t} at.".replace(",", " ") for m, t in zip(top["molecules"], top["taille"])]
    return Draw.MolsToGridImage(mols, molsPerRow=par_ligne, subImgSize=(200, 160), legends=legends)


# ==============================================================================
#  ATOMES PONTÉS ET PLUS GRAND CYCLE
# ==============================================================================
def ponts(smiles: str) -> tuple[int, int, int]:
    """(plus grand cycle, têtes de pont, atomes lourds) d'un SMILES ; (-1, -1, -1) s'il ne se lit pas.
    Têtes de pont : rdMolDescriptors.CalcNumBridgeheadAtoms (cycles qui partagent au moins deux liaisons), comme
    num_bridgehead_atoms de teste_newmut_V1/pontage_smiles.ipynb. Plus grand cycle : parmi les cycles minimaux
    (SSSR) de RDKit, 0 sans cycle."""
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1, -1, -1
    rings = mol.GetRingInfo().AtomRings()
    return max((len(r) for r in rings), default=0), rdMolDescriptors.CalcNumBridgeheadAtoms(mol), mol.GetNumHeavyAtoms()


def _ponts_chunk(smiles: list[str]) -> np.ndarray:
    return np.array([ponts(s) for s in smiles], dtype=np.int32).reshape(-1, 3)


def compute_ponts(df: pd.DataFrame, smiles_col: str, path: str, n_jobs: int = 1, chunk: int = 20000,
                  refaire: bool = False) -> pd.DataFrame:
    """plus_grand_cycle, tetes_de_pont, n_lourds pour chaque ligne de df (même ordre ; -1 si illisible), en cache dans
    resultats/ (un cache par fichier `path` d'où vient df)."""
    os.makedirs(ec.OUT_DIR, exist_ok=True)
    st = os.stat(path)
    nom = os.path.splitext(os.path.basename(path))[0]
    f = os.path.join(ec.OUT_DIR, f"ponts_{nom}_{st.st_size}_{int(st.st_mtime)}.parquet")
    if not refaire and os.path.exists(f):
        return pd.read_parquet(f)
    t0 = time.time()
    smiles = df[smiles_col].tolist()
    chunks = [smiles[k:k + chunk] for k in range(0, len(smiles), chunk)]
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs > 1:
        with get_context("spawn").Pool(n_jobs) as pool:
            results = pool.map(_ponts_chunk, chunks)
    else:
        results = [_ponts_chunk(c) for c in chunks]
    out = pd.DataFrame(np.vstack(results), columns=["plus_grand_cycle", "tetes_de_pont", "n_lourds"])
    out.to_parquet(f)
    print(f"{len(df)} SMILES en {time.time() - t0:.0f} s ({int((out['n_lourds'] < 0).sum())} illisibles)")
    return out


def ponts_par_cycle(p: pd.DataFrame, max_cycle: int = 12) -> pd.DataFrame:
    """Selon la taille du plus grand cycle (max_cycle et plus regroupés ; 0 : sans cycle) : nombre d'objets, têtes de
    pont (total, moyenne, par atome lourd en %), part des objets qui en ont au moins une, et nombre d'objets à 1, 2, 3,
    ≥ 4 têtes de pont."""
    p = p[p["n_lourds"] >= 0]
    cycle = p["plus_grand_cycle"].clip(upper=max_cycle)
    g = p.groupby(cycle)
    out = pd.DataFrame({
        "objets": g.size(),
        "têtes de pont": g["tetes_de_pont"].sum(),
        "moyenne": g["tetes_de_pont"].mean(),
        "% des atomes": 100 * g["tetes_de_pont"].sum() / g["n_lourds"].sum(),
        "% avec ≥ 1": 100 * g["tetes_de_pont"].apply(lambda s: (s > 0).mean()),
    })
    n = p["tetes_de_pont"].clip(upper=4)
    counts = pd.crosstab(cycle, n).reindex(columns=[1, 2, 3, 4], fill_value=0)
    counts.columns = ["1", "2", "3", "≥ 4"]
    out = out.join(counts)
    out.index = [str(k) if k < max_cycle else f"≥ {max_cycle}" for k in out.index]
    out.index.name = "plus grand cycle"
    return out


def plot_ponts(t_mol: pd.DataFrame, t_pi: pd.DataFrame):
    """Part des objets avec au moins une tête de pont, et nombre moyen de têtes de pont, selon le plus grand cycle
    (molécules en trait plein, systèmes π en tirets)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for t, label, style in ((t_mol, "molécules", "-"), (t_pi, "systèmes π", "--")):
        for ax, col in zip(axes, ("% avec ≥ 1", "moyenne")):
            ax.plot(t.index, t[col], color=COULEUR, linestyle=style, linewidth=2, marker="o", markersize=5, label=label)
    axes[0].set_ylabel("% avec au moins une tête de pont")
    axes[1].set_ylabel("têtes de pont par objet (moyenne)")
    for ax in axes:
        ax.set_xlabel("taille du plus grand cycle (0 : sans cycle)")
        ax.legend(frameon=False)
        ec._style(ax)
    fig.tight_layout()
    return fig


# ==============================================================================
#  EXPLORATEUR : MOLÉCULES OU SYSTÈMES π SELON LE PLUS GRAND CYCLE
# ==============================================================================
BLEU_CYCLE = (0.62, 0.77, 0.96)   # atomes et liaisons du plus grand cycle
ORANGE_PONT = (0.96, 0.62, 0.42)  # têtes de pont (par-dessus le bleu)


def _dessin(smiles: str) -> tuple[Chem.Mol, dict[int, tuple], list[int]]:
    """Molécule, couleurs des atomes et liaisons à colorer : en bleu le plus grand cycle (tous les cycles de cette
    taille parmi les cycles minimaux SSSR, comme compute_ponts), en orange les têtes de pont (RDKit)."""
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    ri = mol.GetRingInfo()
    n_max = max((len(r) for r in ri.AtomRings()), default=0)
    couleurs = {a: BLEU_CYCLE for r in ri.AtomRings() if len(r) == n_max for a in r}
    liaisons = sorted({b for r in ri.BondRings() if len(r) == n_max for b in r})
    tetes: list[int] = []
    rdMolDescriptors.CalcNumBridgeheadAtoms(mol, atoms=tetes)
    couleurs.update({a: ORANGE_PONT for a in tetes})
    return mol, couleurs, liaisons


def grille(mols: list[Chem.Mol], couleurs: list[dict[int, tuple]], liaisons: list[list[int]], legends: list[str],
           par_ligne: int = 4, taille: tuple[int, int] = (320, 260), svg: bool = True):
    """Grille de dessins : atomes colorés selon `couleurs`, liaisons `liaisons` en bleu ; chaque molécule à sa propre
    échelle."""
    from rdkit.Chem.Draw import rdMolDraw2D
    lignes = (len(mols) + par_ligne - 1) // par_ligne
    W, H = taille[0] * min(par_ligne, len(mols)), taille[1] * lignes
    drawer = rdMolDraw2D.MolDraw2DSVG(W, H, *taille) if svg else rdMolDraw2D.MolDraw2DCairo(W, H, *taille)
    opts = drawer.drawOptions()
    opts.highlightBondWidthMultiplier = 12
    opts.highlightRadius = 0.4
    opts.legendFontSize = 14
    opts.drawMolsSameScale = False
    prepared = [rdMolDraw2D.PrepareMolForDrawing(m) for m in mols]
    drawer.DrawMolecules(prepared, highlightAtoms=[list(c) for c in couleurs], highlightAtomColors=couleurs,
                         highlightBonds=liaisons, highlightBondColors=[{b: BLEU_CYCLE for b in bl} for bl in liaisons],
                         legends=legends)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def explorer(chembl: pd.DataFrame, ponts_mol: pd.DataFrame, systemes: pd.DataFrame, ponts_pi: pd.DataFrame,
             par_ligne: int = 4):
    """Widget : choisir « molécules » ou « systèmes π », la taille du plus grand cycle et le nombre d'exemples ;
    « Tirer » dessine autant d'exemples tirés au hasard (bleu : plus grand cycle ; orange : têtes de pont). Option : seulement ceux qui ont au moins une tête de pont."""
    import ipywidgets as w
    from IPython.display import SVG, display

    jeux = {
        "molécules": pd.concat([chembl[["chembl_id", "canonical_smiles"]].reset_index(drop=True), ponts_mol], axis=1)
                       .rename(columns={"canonical_smiles": "smiles"}),
        "systèmes π": pd.concat([systemes[["smiles_pi", "molecules"]].reset_index(drop=True), ponts_pi], axis=1)
                        .rename(columns={"smiles_pi": "smiles"}),
    }
    jeu = w.ToggleButtons(options=list(jeux), description="Jeu")
    cycle = w.Dropdown(description="Plus grand cycle")
    nombre = w.BoundedIntText(value=12, min=1, max=200, description="Nombre")
    seulement = w.Checkbox(value=False, description="seulement avec têtes de pont")
    tirer = w.Button(description="Tirer", button_style="primary")
    sortie = w.Output()

    def options(*_):
        df = jeux[jeu.value]
        if seulement.value:
            df = df[df["tetes_de_pont"] > 0]
        n = df["plus_grand_cycle"].value_counts().sort_index()
        old = cycle.value
        fmt = lambda v: f"{v:,}".replace(",", " ")
        cycle.options = [(f"{k} ({fmt(v)} objets)" if k else f"0, sans cycle ({fmt(v)} objets)", k) for k, v in n.items()]
        if old in n.index:
            cycle.value = old

    def montrer(*_):
        sortie.clear_output()
        df = jeux[jeu.value]
        sub = df[df["plus_grand_cycle"] == cycle.value]
        if seulement.value:
            sub = sub[sub["tetes_de_pont"] > 0]
        sub = sub.sample(min(nombre.value, len(sub))) if len(sub) else sub
        mols, colors, bonds, legends = [], [], [], []
        for _, r in sub.iterrows():
            mol, c, b = _dessin(r["smiles"])
            mols.append(mol)
            colors.append(c)
            bonds.append(b)
            nom = r["chembl_id"] if jeu.value == "molécules" else f"{int(r['molecules'])} mol."
            legends.append(f"{nom} · cycle {r['plus_grand_cycle']} · {r['tetes_de_pont']} t.d.p.")
        with sortie:
            print(f"{len(sub)} sur {int((df['plus_grand_cycle'] == cycle.value).sum())} {jeu.value} dont le plus grand "
                  f"cycle a {cycle.value} atomes — bleu : plus grand cycle ; orange : têtes de pont")
            if mols:
                display(SVG(grille(mols, colors, bonds, legends, par_ligne)))

    jeu.observe(options, "value")
    seulement.observe(options, "value")
    tirer.on_click(montrer)
    options()
    cycle.value = 6 if 6 in [v for _, v in cycle.options] else cycle.options[0][1]
    return w.VBox([jeu, w.HBox([cycle, nombre]), w.HBox([seulement, tirer]), sortie])


# ==============================================================================
#  SYSTÈMES π AVEC DES C SP3 ENCLOS
# ==============================================================================
# Un C qui n'est pas π (sp3) est ajouté à un système π si au moins deux de ses voisins lourds sont des atomes de ce
# même système (C9 du fluorène et du 9,9-diméthylfluorène, CH2 du cyclopentadiène, de l'indane-1,3-dione) ; il peut
# porter d'autres substituants non π, perdus dans le fragment. Le CH2 du diphénylméthane ou du 9,10-dihydroanthracène
# relie deux systèmes différents (un voisin dans chacun) et n'est pas ajouté.
PI_SP3_TSV = os.path.join(HERE, "donnees", "systemes_pi_sp3_chembl34.tsv")


def sp3_enclos(mol: Chem.Mol, comps: list[set[int]]) -> dict[int, list[int]]:
    """{numéro du système π : [C sp3 enclos]} (règle ci-dessus ; un C qui a au moins deux voisins dans deux systèmes
    va dans celui où il en a le plus)."""
    which = {a: k for k, c in enumerate(comps) for a in c}
    out: dict[int, list[int]] = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetIdx() in which:
            continue
        ks = Counter(which[n.GetIdx()] for n in atom.GetNeighbors() if n.GetIdx() in which)
        if ks:
            k, n = ks.most_common(1)[0]
            if n >= 2:
                out.setdefault(k, []).append(atom.GetIdx())
    return out


def h_sp3_fragment(smiles_frag: str) -> list[int]:
    """H de chaque C sp3 d'un fragment (C non aromatique sans liaison multiple : dans un fragment, ce sont les C sp3
    enclos, les C π portant toujours une liaison multiple ou étant aromatiques)."""
    frag = Chem.MolFromSmiles(smiles_frag)
    return sorted(a.GetTotalNumHs() for a in frag.GetAtoms() if a.GetSymbol() == "C" and not a.GetIsAromatic()
                  and all(b.GetBondType() == Chem.BondType.SINGLE for b in a.GetBonds()))


def molecule_systems_sp3(smiles: str, taille_min: int = 4) -> list[tuple[str | None, int, list[int]]]:
    """Pour chaque système π (atomes π + C sp3 enclos, au moins taille_min atomes) qui a au moins un C sp3 enclos :
    (SMILES du fragment, taille, H de ses C sp3 dans la molécule)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    comps = pi_component_atoms(mol, pi_atoms(mol))
    enclos = sp3_enclos(mol, comps)
    if not enclos:
        return []
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    out = []
    for k, sp3 in enclos.items():
        atoms = comps[k] | set(sp3)
        if len(atoms) >= taille_min:
            out.append((fragment_smiles(kek, atoms), len(atoms),
                        sorted(mol.GetAtomWithIdx(a).GetTotalNumHs() for a in sp3)))
    return out


def _sp3_chunk(args: tuple[list[str], list[str], int]):
    ids, smiles, taille_min = args
    occ: Counter = Counter()
    mols: Counter = Counter()
    taille: dict[str, int] = {}
    exemple: dict[str, str] = {}
    h_mol: dict[str, Counter] = {}
    for cid, smi in zip(ids, smiles):
        systems = [s for s in molecule_systems_sp3(smi, taille_min) if s[0] is not None]
        for frag, n, hs in systems:
            occ[frag] += 1
            taille[frag] = n
            exemple.setdefault(frag, cid)
            h_mol.setdefault(frag, Counter()).update(hs)
        mols.update({s[0] for s in systems})
    return occ, mols, taille, exemple, h_mol


def extract_sp3(df: pd.DataFrame, n: int = 20000, graine: int = 0, taille_min: int = 4, n_jobs: int = 1,
                chunk: int = 20000, path: str = ec.DATA, refaire: bool = False) -> pd.DataFrame:
    """Systèmes π distincts avec au moins un C sp3 enclos ; n d'entre eux tirés au hasard (graine), écrits dans
    donnees/systemes_pi_sp3_chembl34.tsv : smiles_pi, taille, n_sp3, h_sp3 (H de chaque C sp3 dans le fragment,
    séparés par des virgules), occurrences, molecules, exemple, h_sp3_molecules (H des C sp3 dans les molécules,
    sur toutes les occurrences : « H:nombre » séparés par des virgules). En cache dans resultats/."""
    st = os.stat(path)
    f = os.path.join(ec.OUT_DIR, f"pi_sp3_{st.st_size}_{int(st.st_mtime)}_min{taille_min}_n{n}_g{graine}.parquet")
    if not refaire and os.path.exists(f) and os.path.exists(PI_SP3_TSV):
        return pd.read_parquet(f)
    t0 = time.time()
    ids, smiles = df["chembl_id"].tolist(), df["canonical_smiles"].tolist()
    chunks = [(ids[k:k + chunk], smiles[k:k + chunk], taille_min) for k in range(0, len(smiles), chunk)]
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs > 1:
        with get_context("spawn").Pool(n_jobs) as pool:
            results = pool.map(_sp3_chunk, chunks)
    else:
        results = [_sp3_chunk(c) for c in chunks]
    occ: Counter = Counter()
    mols: Counter = Counter()
    taille: dict[str, int] = {}
    exemple: dict[str, str] = {}
    h_mol: dict[str, Counter] = {}
    for r in results:
        occ.update(r[0])
        mols.update(r[1])
        taille.update(r[2])
        for k, v in r[3].items():
            exemple.setdefault(k, v)
        for k, v in r[4].items():
            h_mol.setdefault(k, Counter()).update(v)
    tous = pd.DataFrame({"smiles_pi": list(occ)})
    sel = tous.sample(min(n, len(tous)), random_state=graine) if len(tous) > n else tous
    sel = sel.copy()
    sel["taille"] = sel["smiles_pi"].map(taille)
    h_frag = sel["smiles_pi"].map(h_sp3_fragment)
    sel["n_sp3"] = h_frag.map(len)
    sel["h_sp3"] = h_frag.map(lambda hs: ",".join(map(str, hs)))
    sel["occurrences"] = sel["smiles_pi"].map(occ)
    sel["molecules"] = sel["smiles_pi"].map(mols)
    sel["exemple"] = sel["smiles_pi"].map(exemple)
    sel["h_sp3_molecules"] = sel["smiles_pi"].map(
        lambda s: ",".join(f"{h}:{c}" for h, c in sorted(h_mol[s].items())))
    sel = sel.sort_values(["molecules", "occurrences"], ascending=False).reset_index(drop=True)
    sel.to_csv(PI_SP3_TSV, sep="\t", index=False)
    sel.to_parquet(f)
    print(f"{len(df)} molécules lues en {time.time() - t0:.0f} s ; {len(tous)} systèmes π distincts avec un C sp3 "
          f"enclos, {len(sel)} gardés → {PI_SP3_TSV}")
    return sel


def h_sp3_comptes(sel: pd.DataFrame) -> pd.DataFrame:
    """Nombre de C sp3 enclos selon leur nombre de H : dans les fragments (une fois par système distinct) et dans
    les molécules (toutes les occurrences des systèmes gardés)."""
    frag = Counter(int(h) for hs in sel["h_sp3"] for h in hs.split(",") if h != "")
    mol: Counter = Counter()
    for item in sel["h_sp3_molecules"]:
        for hc in item.split(","):
            h, c = hc.split(":")
            mol[int(h)] += int(c)
    out = pd.DataFrame({"fragments": pd.Series(frag), "molécules": pd.Series(mol)}).fillna(0).astype(int).sort_index()
    out.index.name = "H sur le C sp3"
    for c in ("fragments", "molécules"):
        out[f"% {c}"] = 100 * out[c] / out[c].sum()
    return out


def plot_h_sp3(t: pd.DataFrame):
    """Histogrammes du nombre de H des C sp3 enclos : dans les fragments, et dans les molécules."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, col, titre in zip(axes, ("fragments", "molécules"),
                              ("dans les systèmes π distincts (fragments)", "dans les molécules (toutes les occurrences)")):
        ax.bar(t.index.astype(str), t[f"% {col}"], color=COULEUR, width=0.6)
        for k, (v, n) in enumerate(zip(t[f"% {col}"], t[col])):
            ax.text(k, v, f"{v:.1f} %\n({n:,})".replace(",", " "), ha="center", va="bottom", fontsize=8,
                    color="#555555")
        ax.set_title(titre, fontsize=10)
        ax.set_xlabel("H sur le C sp3")
        ax.set_ylim(0, 110)
        ec._style(ax)
    axes[0].set_ylabel("% des C sp3 enclos")
    fig.tight_layout()
    return fig


def exemples_h_sp3(systemes_sp3: pd.DataFrame, h: int, n: int = 12, graine: int = 0, par_ligne: int = 4):
    """Dessin de n fragments (systèmes π avec C sp3 enclos, tirés au hasard, graine) dont un C sp3 porte h H dans le
    fragment ; ces C sont en orange. Sous chaque fragment : nombre de molécules qui le contiennent. Renvoie le SVG."""
    from IPython.display import SVG
    sel = systemes_sp3[systemes_sp3["h_sp3"].map(lambda hs: str(h) in hs.split(","))]
    sel = sel.sample(min(n, len(sel)), random_state=graine)
    mols, couleurs, legends = [], [], []
    for smi, m in zip(sel["smiles_pi"], sel["molecules"]):
        frag = Chem.MolFromSmiles(smi)
        cibles = [a.GetIdx() for a in frag.GetAtoms() if a.GetSymbol() == "C" and not a.GetIsAromatic()
                  and all(b.GetBondType() == Chem.BondType.SINGLE for b in a.GetBonds()) and a.GetTotalNumHs() == h]
        mols.append(frag)
        couleurs.append({a: ORANGE_PONT for a in cibles})
        legends.append(f"{m:,} mol.".replace(",", " "))
    print(f"{len(sel)} fragments sur {int(systemes_sp3['h_sp3'].map(lambda hs: str(h) in hs.split(',')).sum())}")
    return SVG(grille(mols, couleurs, [[] for _ in mols], legends, par_ligne)) if mols else None
