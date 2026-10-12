"""
Analyse des marches aléatoires écrites par random_walk.run_many (notebooks/random_walk/ancien/).

Mesures du paysage, sur la suite des valeurs d'une observable le long de la marche :
  - ρ(k) : autocorrélation (fitness_auto_correlation de RevoMol-correlation/scripts/correlations.py) ;
  - ℓ = −1 / ln ρ(1) : longueur de corrélation (pas au-delà desquels les valeurs ne sont plus corrélées) ;
  - |Δ| moyen par pas et % de pas neutres (|Δ| <= eps), retours au départ (restart) exclus ;
  - contenu d'information (Vassilev, Fogarty & Miller 2000) : H(ε) entropie des changements de sens,
    M(ε) contenu partiel (densité d'optima le long de la marche), ε* = max |Δ|.
Mutations : fréquence de chaque type, effet moyen Δ sur chaque observable, % de hausse / baisse de fitness.
Évolvabilité : % de voisins meilleurs, % d'optima locaux (molécules sans voisin meilleur).
Sans l'effet taille (suffixe « _taille ») : ρ(1), ℓ et M sur le résidu de l'observable après régression
linéaire sur n_atoms dans la marche ; % de voisins meilleurs parmi ceux de même taille que le parent.
Autre fitness : rescore_fitness / rescore_plan la recalculent sur les marches et les voisinages enregistrés.
Front de Pareto des configurations : pareto_configs.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import doe

MAX_LAG = 100        # plus grand décalage k de ρ(k)
NEUTRAL_EPS = 1e-3   # un pas est neutre si |Δ| <= NEUTRAL_EPS

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8c8c8c"
DIVERGING = LinearSegmentedColormap.from_list("effet", [ORANGE, "#f4f4f2", BLUE])
STYLE = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e5", "axes.axisbelow": True, "font.size": 10,
}
plt.rcParams.update(STYLE)


# ==============================================================================
#  MESURES SUR UNE MARCHE
# ==============================================================================
def autocorrelation(values: pd.Series, max_lag: int) -> pd.Series:
    """ρ(k) = corrélation de Pearson entre values[t] et values[t + k], k = 1 … max_lag."""
    values = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    # série constante ou trop courte : ρ non défini (NaN), sans avertissement numpy
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pd.Series({k: values.autocorr(k) for k in range(1, max_lag + 1)}, dtype=float)


def correlation_length(rho1: float) -> float:
    if not np.isfinite(rho1) or rho1 <= 0:
        return math.nan
    return math.inf if rho1 >= 1 else -1 / math.log(rho1)


def information_content(deltas: np.ndarray, eps: float) -> tuple[float, float]:
    """(H(ε), M(ε)) d'une suite de Δ : symboles −1 / 0 / +1 selon Δ < −ε, |Δ| <= ε, Δ > ε.
    H : entropie des paires de symboles consécutifs différents (log base 6) ;
    M : longueur de la suite sans 0 ni répétitions / nb de symboles."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return math.nan, math.nan
    s = np.where(d > eps, 1, np.where(d < -eps, -1, 0))
    a, b = s[:-1], s[1:]
    codes = (a + 1) * 3 + (b + 1)
    p = np.bincount(codes[a != b], minlength=9) / len(a)
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum() / np.log(6)) + 0.0
    nonzero = s[s != 0]
    mu = 0 if len(nonzero) == 0 else 1 + int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
    return h, mu / len(s)


def eps_grid(eps_star: float, n: int = 41) -> np.ndarray:
    """ε = 0 puis grille log de ε*·10⁻⁴ à ε*."""
    if not np.isfinite(eps_star) or eps_star <= 0:
        return np.array([0.0])
    return np.concatenate([[0.0], eps_star * np.logspace(-4, 0, n)])


def walk_deltas(walk: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(valeurs numériques, Δ par pas sans les restarts). Δ du pas t = valeur(t) − valeur(t−1)."""
    numeric = walk[columns].apply(pd.to_numeric, errors="coerce")
    deltas = numeric.diff().iloc[1:]
    moves = deltas[(walk.action.iloc[1:] != "restart").to_numpy()]
    return numeric, moves


def size_residual(values: pd.Series, n_atoms: pd.Series) -> pd.Series:
    """Résidu de values après régression linéaire sur n_atoms (valeurs définies seulement)."""
    y, x = pd.to_numeric(values, errors="coerce"), pd.to_numeric(n_atoms, errors="coerce")
    ok = y.notna() & x.notna()
    residual = pd.Series(math.nan, index=values.index)
    if ok.sum() < 3:
        return residual
    if np.ptp(x[ok]) == 0:
        residual[ok] = y[ok] - y[ok].mean()
    else:
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        residual[ok] = y[ok] - (intercept + slope * x[ok])
    return residual


def landscape(walk: pd.DataFrame, columns: list[str], max_lag: int = MAX_LAG,
              eps: float = NEUTRAL_EPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mesures du paysage par observable (une ligne par colonne) et ρ(k) (une colonne par observable)."""
    numeric, moves = walk_deltas(walk, columns)
    max_lag = max(1, min(max_lag, len(walk) - 2))
    rho = pd.DataFrame({c: autocorrelation(numeric[c], max_lag) for c in columns})
    rho.index.name = "k"
    rows = {}
    for c in columns:
        d = moves[c].to_numpy()
        eps_star = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else math.nan
        grid = eps_grid(eps_star)
        hs = [information_content(d, e)[0] for e in grid]
        h, m = information_content(d, eps)
        defined = np.isfinite(d)
        rows[c] = {
            "mean": numeric[c].mean(), "std": numeric[c].std(), "min": numeric[c].min(), "max": numeric[c].max(),
            "undefined_pct": 100 * numeric[c].isna().mean(),
            "rho1": rho[c].iloc[0], "corr_length": correlation_length(rho[c].iloc[0]),
            "abs_delta_mean": np.nanmean(np.abs(d)) if defined.any() else math.nan,
            "neutral_pct": 100 * (np.abs(d[defined]) <= eps).mean() if defined.any() else math.nan,
            "H": h, "M": m, "H_max": np.nanmax(hs) if np.isfinite(hs).any() else math.nan,
            "eps_star": eps_star,
        }
        if c != "n_atoms" and "n_atoms" in walk:  # mêmes mesures sans l'effet taille
            residual = size_residual(numeric[c], walk.n_atoms)
            rho1 = autocorrelation(residual, 1).iloc[0]
            d_res = residual.diff().iloc[1:][(walk.action.iloc[1:] != "restart").to_numpy()].to_numpy()
            rows[c].update({"rho1_taille": rho1, "corr_length_taille": correlation_length(rho1),
                            "M_taille": information_content(d_res, eps)[1]})
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "observable"
    return table, rho


def mutation_stats(walk: pd.DataFrame, columns: list[str], eps: float = NEUTRAL_EPS) -> pd.DataFrame:
    """Par type de mutation choisi : nb de pas, %, % valides, ms/pas, Δ et |Δ| moyens par observable,
    % de pas qui font monter / ne changent pas / font baisser la fitness."""
    steps = walk.iloc[1:]
    deltas = walk[columns].apply(pd.to_numeric, errors="coerce").diff().iloc[1:]
    stats = {}
    for action, idx in steps.groupby("action").groups.items():
        d = deltas.loc[idx]
        row = {
            "n": len(idx), "pct": 100 * len(idx) / len(steps),
            "valid_pct": 100 * steps.valid.loc[idx].mean(),
            "ms_per_step": 1000 * steps.loc[idx, ["t_neighbors", "t_filters", "t_evaluations"]].sum(axis=1).mean(),
        }
        for c in columns:
            row[f"Δ {c}"] = d[c].mean()
            row[f"|Δ| {c}"] = d[c].abs().mean()
        if "fitness" in columns:
            f = d["fitness"].dropna()
            row["fitness up_pct"] = 100 * (f > eps).mean() if len(f) else math.nan
            row["fitness neutral_pct"] = 100 * (f.abs() <= eps).mean() if len(f) else math.nan
            row["fitness down_pct"] = 100 * (f < -eps).mean() if len(f) else math.nan
        stats[action] = row
    table = pd.DataFrame.from_dict(stats, orient="index").sort_values("n", ascending=False)
    table.index.name = "mutation"
    return table


def rejection_stats(walk: pd.DataFrame) -> dict[str, float]:
    """Par type de mutation, voisins tirés puis rejetés par les filtres (colonne rejected_actions) :
    "invalid_pct <mutation>" = rejetés / (rejetés + choisis), "rejected_per_step <mutation>" = rejetés
    par pas (leur somme sur les types = voisins rejetés par pas). {} si la colonne est absente."""
    if "rejected_actions" not in walk:
        return {}
    steps = walk.iloc[1:]
    rejected: dict[str, int] = {}
    for text in steps.rejected_actions.dropna():
        for item in str(text).split(";"):
            if item:
                name, n = item.split(":")
                rejected[name] = rejected.get(name, 0) + int(n)
    chosen = steps.action[steps.action != "restart"].value_counts()
    stats = {}
    for name in sorted(set(rejected) | set(chosen.index)):
        r, c = rejected.get(name, 0), int(chosen.get(name, 0))
        stats[f"invalid_pct {name}"] = 100 * r / (r + c)
        stats[f"rejected_per_step {name}"] = r / len(steps)
    return stats


def _shares(delta: pd.Series, eps: float) -> dict[str, float]:
    d = pd.to_numeric(delta, errors="coerce").dropna()
    if not len(d):
        return {"better_pct": math.nan, "neutral_pct": math.nan, "worse_pct": math.nan}
    return {"better_pct": 100 * (d > eps).mean(), "neutral_pct": 100 * (d.abs() <= eps).mean(),
            "worse_pct": 100 * (d < -eps).mean()}


def evolvability(neighborhoods: pd.DataFrame, neighbors: pd.DataFrame,
                 eps: float = NEUTRAL_EPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Par molécule analysée : % voisins meilleurs / neutres / moins bons, nb de voisins meilleurs,
    optimum local ;
    par type de mutation : les mêmes parts sur tous les voisins qu'il produit."""
    molecules = neighborhoods.copy()
    rows = []
    for _, m in molecules.iterrows():
        d = neighbors.loc[neighbors.step == m.step, "delta"]
        defined = d.dropna()
        rows.append({**_shares(d, eps),
                     "better_neighbors": int((defined > eps).sum()),  # nombre (et non part) de voisins meilleurs
                     "best_neighbor_fitness": m.fitness + defined.max() if len(defined) else math.nan,
                     "local_optimum": float((defined <= eps).all()) if len(defined) and np.isfinite(m.fitness)
                     else math.nan})
    molecules = pd.concat([molecules.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    per_mutation = pd.DataFrame.from_dict({
        mutation: {"neighbors": len(d), **_shares(d.delta, eps), "delta_mean": d.delta.mean()}
        for mutation, d in neighbors.groupby("mutation")
    }, orient="index")
    per_mutation.index.name = "mutation"
    return molecules, per_mutation


def same_size_better_pct(values: pd.DataFrame, eps: float = NEUTRAL_EPS) -> float:
    """% de voisins meilleurs (fitness) parmi ceux qui ont le même n_atoms que leur parent, moyenne sur
    les parents qui ont au moins un tel voisin (values : format rep<k>_voisins.parquet)."""
    parents = values[values.is_parent].set_index("step")[["fitness", "n_atoms"]]
    neighbors = values[~values.is_parent].join(parents, on="step", rsuffix="_parent")
    same = neighbors[neighbors.n_atoms == neighbors.n_atoms_parent]
    delta = same.fitness - same.fitness_parent
    shares = (delta[delta.notna()] > eps).groupby(same.step[delta.notna()]).mean()
    return 100 * shares.mean() if len(shares) else math.nan


# ==============================================================================
#  CHARGEMENT DES RÉSULTATS ET RÉPONSES
# ==============================================================================
def load_run(out_dir: str | Path, config_id: str, rep: int, values: bool | str | Path = False) -> dict[str, Any]:
    """Une marche terminée : config, meta, marche, voisinages.

    values : charger aussi toutes les variables du voisinage ("values", None si absentes) —
    True : rep<k>_voisins.parquet de la marche ; dossier : <dossier>/<config>/rep<k>_voisins.parquet
    (ex. results/evolvabilite/<plan> pour les marches écrites avant le 2026-10-07).
    """
    run_dir = Path(out_dir) / "runs" / config_id
    neighborhoods = run_dir / f"rep{rep}_voisinages.csv.gz"
    run = {
        "config_id": config_id, "rep": rep,
        "config": json.loads((run_dir / "config.json").read_text()),
        "meta": json.loads((run_dir / f"rep{rep}_meta.json").read_text()),
        "walk": pd.read_csv(run_dir / f"rep{rep}_walk.csv.gz"),
        "neighborhoods": pd.read_csv(neighborhoods) if neighborhoods.exists() else None,
        "neighbors": pd.read_csv(run_dir / f"rep{rep}_voisins.csv.gz") if neighborhoods.exists() else None,
    }
    if values is not False:
        path = (run_dir if values is True else Path(values) / config_id) / f"rep{rep}_voisins.parquet"
        run["values"] = pd.read_parquet(path) if path.exists() else None
    return run


def finished_runs(out_dir: str | Path) -> list[tuple[str, int]]:
    """(config_id, rep) des marches terminées (meta.json écrit), dans l'ordre."""
    runs_dir = Path(out_dir) / "runs"
    if not runs_dir.exists():
        return []
    return [(run_dir.name, rep)
            for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir())
            for rep in sorted(int(m.name[3:].split("_")[0]) for m in run_dir.glob("rep*_meta.json"))]


def load_runs(out_dir: str | Path, values: bool | str | Path = False) -> list[dict[str, Any]]:
    """Toutes les marches terminées de out_dir/runs/<config>/ (ordre : config, répétition) ;
    values : voir load_run."""
    return [load_run(out_dir, config_id, rep, values) for config_id, rep in finished_runs(out_dir)]


class LiveView:
    """Affichage en direct pendant run_many (passer l'objet en on_done) : avancement, réponses clés
    par configuration (moyenne ± écart-type des marches déjà terminées) et une figure de ces
    réponses, mis à jour à chaque marche terminée. Les marches déjà faites sont reprises au départ."""

    def __init__(self, out_dir: str | Path, columns: list[str], keys: list[str], total: int,
                 design_table: pd.DataFrame | None = None, plot_keys: list[str] | None = None) -> None:
        from IPython.display import HTML, display  # pylint: disable=import-outside-toplevel

        self.out_dir, self.columns, self.keys, self.total = out_dir, columns, keys, total
        self.design_table = design_table
        self.plot_keys = plot_keys or keys[:6]
        self.rows: list[dict[str, Any]] = []
        self._html = HTML
        self.status = display(HTML(""), display_id=True)
        self.figure = display(HTML(""), display_id=True)
        self.table = display(HTML(""), display_id=True)
        for config_id, rep in finished_runs(out_dir):
            self._add(config_id, rep)
        self._refresh()

    def __call__(self, config_id: str, rep: int) -> None:
        self._add(config_id, rep)
        self._refresh()

    def _add(self, config_id: str, rep: int) -> None:
        run = load_run(self.out_dir, config_id, rep)
        self.rows.append(analyse_runs([run], self.columns)["responses"].iloc[0].to_dict())

    def _refresh(self) -> None:
        if not self.rows:
            self.status.update(self._html(f"<b>0 / {self.total}</b> marches terminées"))
            return
        responses = pd.DataFrame(self.rows)
        keys = [k for k in self.keys if k in responses]
        counts = responses.groupby("config_id").size().rename("marches")
        table = summary_table(responses, keys).join(counts)
        if self.design_table is not None:
            table = self.design_table.join(table, how="inner")
        self.status.update(self._html(
            f"<b>{len(responses)} / {self.total}</b> marches terminées, "
            f"{responses.config_id.nunique()} configurations commencées"))
        self.table.update(table)
        plot_keys = [k for k in self.plot_keys if k in responses]
        fig, axes = _grid(len(plot_keys), ncols=3, width=5.0, height=2.4)
        order = sorted(responses.config_id.unique())
        x = {c: i for i, c in enumerate(order)}
        for ax, k in zip(axes.flat, plot_keys):
            ax.scatter(responses.config_id.map(x), responses[k], s=14, color=BLUE, alpha=0.7)
            means = responses.groupby("config_id")[k].mean()
            ax.scatter([x[c] for c in means.index], means, marker="_", s=160, color="#222222")
            ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
            ax.set_title(k, loc="left", fontsize=9)
            ax.grid(axis="x", visible=False)
        fig.suptitle("réponses des marches terminées (points) et moyenne par configuration (trait)",
                     x=0.01, ha="left", fontsize=9)
        fig.tight_layout()
        self.figure.update(fig)
        plt.close(fig)


def analyse_runs(runs: list[dict[str, Any]], columns: list[str], out_dir: str | Path | None = None,
                 max_lag: int = MAX_LAG, eps: float = NEUTRAL_EPS) -> dict[str, pd.DataFrame]:
    """Mesures de toutes les marches.

    Returns (et écrit dans out_dir si donné) :
        responses : une ligne par marche — config_id, rep, "<observable> <mesure>", exploration,
                    "freq <mutation>", fitness up/down, évolvabilité ;
        actions   : une ligne par marche et par mutation (mutation_stats) ;
        neighbor_actions : une ligne par marche et par mutation (évolvabilité des voisins) ;
        rho       : ρ(k) en format long (config_id, rep, observable, k, rho) ;
        landscape : une ligne par marche et par observable.
    """
    responses, actions, neighbor_actions, rhos, landscapes = [], [], [], [], []
    for run in runs:
        key = {"config_id": run["config_id"], "rep": run["rep"]}
        walk = run["walk"]
        steps = walk.iloc[1:]
        table, rho = landscape(walk, columns, max_lag, eps)
        mutations = mutation_stats(walk, columns, eps)
        row = {**key}
        for c, measures in table.iterrows():
            row.update({f"{c} {m}": v for m, v in measures.items()})
        row.update({
            "distinct_pct": 100 * walk.smiles.nunique() / len(walk),
            "restarts": int((steps.action == "restart").sum()),
            "valid_pct": 100 * walk.valid.mean(),
            "neighborhood_size_mean": steps.neighborhood_size.mean(),
            "filter_calls_mean": steps.n_filter_calls.mean(),
            # voisins tirés et rejetés par les filtres avant le premier valide (pas sans restart ;
            # validation paresseuse seulement : sinon tout le voisinage est validé)
            "invalid_before_valid_mean": (
                (steps.n_filter_calls[steps.action != "restart"] - 1).mean()
                if run["config"].get("ONLY_VALID", True) and not run["config"].get("COUNT_VALID_NEIGHBORS")
                else math.nan),
            "ms_per_step": 1000 * run["meta"]["t_walk"] / max(1, len(steps)),
            "walk_start_atoms": walk.n_atoms.iloc[0],
            **{f"freq {m}": v for m, v in mutations.pct.items()},
        })
        if run["config"].get("ONLY_VALID", True) and not run["config"].get("COUNT_VALID_NEIGHBORS"):
            row.update(rejection_stats(walk))  # validation paresseuse seulement (comme invalid_before_valid_mean)
        if "fitness" in columns:
            _, moves = walk_deltas(walk, ["fitness"])
            f = moves.fitness.dropna()
            row["fitness up_pct"] = 100 * (f > eps).mean() if len(f) else math.nan
            row["fitness down_pct"] = 100 * (f < -eps).mean() if len(f) else math.nan
        if run["neighborhoods"] is not None and len(run["neighborhoods"]):
            molecules, per_mutation = evolvability(run["neighborhoods"], run["neighbors"], eps)
            row.update({
                "evolvability better_pct": molecules.better_pct.mean(),
                "evolvability better_neighbors": molecules.better_neighbors.mean(),
                "evolvability local_optima_pct": 100 * molecules.local_optimum.mean(),
                "evolvability valid_neighbors": molecules.n_evaluated.mean(),
            })
            neighbor_actions.append(per_mutation.reset_index().assign(**key))
        if run.get("values") is not None and "fitness" in run["values"]:
            row["evolvability better_pct_meme_taille"] = same_size_better_pct(run["values"], eps)
        responses.append(row)
        actions.append(mutations.reset_index().assign(**key))
        rhos.append(rho.reset_index().melt(id_vars="k", var_name="observable", value_name="rho").assign(**key))
        landscapes.append(table.reset_index().assign(**key))

    def _front(df: pd.DataFrame) -> pd.DataFrame:
        return df[["config_id", "rep", *[c for c in df.columns if c not in ("config_id", "rep")]]]

    responses = pd.DataFrame(responses)
    freq = [c for c in responses.columns if c.startswith("freq ")]
    responses[freq] = responses[freq].fillna(0.0)  # mutation jamais réalisée (ou absente) : 0 %
    results = {
        "responses": responses,
        "actions": _front(pd.concat(actions, ignore_index=True)),
        "neighbor_actions": _front(pd.concat(neighbor_actions, ignore_index=True)) if neighbor_actions else pd.DataFrame(),
        "rho": _front(pd.concat(rhos, ignore_index=True)),
        "landscape": _front(pd.concat(landscapes, ignore_index=True)),
    }
    if out_dir is not None:
        for name, df in results.items():
            df.to_csv(Path(out_dir) / f"{name}.csv", index=False)
    return results


# ==============================================================================
#  FIGURES (chaque fonction renvoie la figure ; save=chemin l'enregistre en PNG)
# ==============================================================================
def _save(fig: plt.Figure, save: str | Path | None) -> plt.Figure:
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=120, bbox_inches="tight")
    return fig


def _grid(n: int, ncols: int = 3, width: float = 4.0, height: float = 2.8, **kwargs):
    ncols = min(n, ncols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False, **kwargs)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    return fig, axes


def plot_series(walk: pd.DataFrame, columns: list[str], units: dict[str, str], title: str = "",
                save=None) -> plt.Figure:
    """Valeur de chaque observable le long d'une marche."""
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 1.6 * len(columns)), sharex=True, squeeze=False)
    for ax, c in zip(axes[:, 0], columns):
        ax.plot(walk.step, pd.to_numeric(walk[c], errors="coerce"), color=BLUE, linewidth=1.2)
        ax.set_title(c, loc="left", fontsize=10)
        ax.set_ylabel(units.get(c, ""))
        ax.grid(axis="x", visible=False)
    axes[-1, 0].set_xlabel("pas")
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, save)


def plot_histograms(walks: list[pd.DataFrame], columns: list[str], units: dict[str, str],
                    save=None) -> plt.Figure:
    """Distribution de chaque observable sur toutes les molécules visitées (marches regroupées)."""
    fig, axes = _grid(len(columns))
    for ax, c in zip(axes.flat, columns):
        values = pd.concat([pd.to_numeric(w[c], errors="coerce") for w in walks]).dropna()
        ax.hist(values, bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
        ax.set_title(c, loc="left", fontsize=10)
        ax.set_xlabel(units.get(c, ""))
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, save)


def plot_histograms_by_level(walks_minus: list[pd.DataFrame], walks_plus: list[pd.DataFrame],
                             columns: list[str], labels: tuple[str, str], title: str,
                             save=None) -> plt.Figure:
    """Distribution de chaque observable au niveau − et + d'un facteur (densités superposées)."""
    fig, axes = _grid(len(columns), ncols=5, width=3.2, height=2.4)
    for ax, c in zip(axes.flat, columns):
        minus = pd.concat([pd.to_numeric(w[c], errors="coerce") for w in walks_minus]).dropna()
        plus = pd.concat([pd.to_numeric(w[c], errors="coerce") for w in walks_plus]).dropna()
        both = pd.concat([minus, plus])
        if both.empty:
            continue
        lo, hi = both.quantile(0.005), both.quantile(0.995)
        bins = np.linspace(lo, hi, 35) if hi > lo else 10
        ax.hist(minus, bins=bins, density=True, histtype="step", color=GRAY, linewidth=1.8, label=labels[0])
        ax.hist(plus, bins=bins, density=True, histtype="step", color=BLUE, linewidth=1.8, label=labels[1])
        ax.set_title(c, loc="left", fontsize=9)
        ax.set_yticks([])
        ax.grid(axis="x", visible=False)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, save)


def plot_autocorrelation(rho: pd.DataFrame, columns: list[str], save=None) -> plt.Figure:
    """ρ(k) moyen ± écart-type entre marches (rho en format long : observable, k, rho)."""
    fig, axes = _grid(len(columns), sharey=True)
    for ax, c in zip(axes.flat, columns):
        r = rho[rho.observable == c].groupby("k").rho.agg(["mean", "std"])
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.fill_between(r.index, r["mean"] - r["std"], r["mean"] + r["std"], color=BLUE, alpha=0.2, linewidth=0)
        ax.plot(r.index, r["mean"], color=BLUE, linewidth=1.8)
        rho1 = r["mean"].iloc[0] if len(r) else math.nan
        ax.set_title(f"{c}   ρ(1) = {rho1:.2f}   ℓ = {correlation_length(rho1):.1f}", loc="left", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("décalage k (pas)")
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("ρ(k)")
    fig.tight_layout()
    return _save(fig, save)


def plot_information(walks: list[pd.DataFrame], columns: list[str], eps: float = NEUTRAL_EPS,
                     save=None) -> plt.Figure:
    """H(ε) et M(ε) moyens sur les marches, ε de ε*·10⁻⁴ à ε* (ε* = max |Δ| sur toutes les marches)."""
    fig, axes = _grid(len(columns), sharey=True)
    for ax, c in zip(axes.flat, columns):
        deltas = [walk_deltas(w, [c])[1][c].to_numpy() for w in walks]
        all_d = np.concatenate(deltas)
        if not np.isfinite(all_d).any() or np.nanmax(np.abs(all_d)) == 0:
            ax.set_title(c, loc="left", fontsize=9)
            continue
        grid = eps_grid(float(np.nanmax(np.abs(all_d))))[1:]
        values = np.array([[information_content(d, e) for e in grid] for d in deltas])  # marche × ε × (H, M)
        ax.plot(grid, np.nanmean(values[:, :, 0], axis=0), color=BLUE, linewidth=2, label="H(ε)")
        ax.plot(grid, np.nanmean(values[:, :, 1], axis=0), color=ORANGE, linewidth=2, label="M(ε)")
        ax.set_xscale("log")
        if grid[0] < eps < grid[-1]:
            ax.axvline(eps, color=GRAY, linewidth=1, linestyle="--")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(c, loc="left", fontsize=9)
        ax.set_xlabel("ε")
    for ax in axes[:, 0]:
        ax.set_ylabel("contenu d'information")
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, save)


def plot_mutation_frequencies(actions: pd.DataFrame, save=None) -> plt.Figure:
    """Part (%) de chaque type de mutation dans les pas, par configuration (moyenne des répétitions)."""
    table = actions.pivot_table(index="config_id", columns="mutation", values="pct", aggfunc="mean").fillna(0)
    table = table[table.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.0 * len(table.columns) + 2, 0.32 * len(table) + 1.5))
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap=LinearSegmentedColormap.from_list("seq", ["#f4f4f2", BLUE]),
                   vmin=0, vmax=100)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(table)), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 60 else "#333333")
    ax.grid(False)
    ax.set_title("part des pas par type de mutation (%)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    return _save(fig, save)


def mutation_effects(actions: pd.DataFrame, landscape_long: pd.DataFrame,
                     columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Effet moyen Δ de chaque type de mutation sur chaque observable (toutes marches, pondéré par
    le nb de pas) : brut, et en unités d'écart-type de l'observable (pour comparer des échelles)."""
    rows = {}
    for mutation, a in actions.groupby("mutation"):
        rows[mutation] = {}
        for c in columns:
            ok = a[f"Δ {c}"].notna()
            rows[mutation][c] = np.average(a.loc[ok, f"Δ {c}"], weights=a.n[ok]) if a.n[ok].sum() else math.nan
        rows[mutation]["n"] = a.n.sum()
    raw = pd.DataFrame.from_dict(rows, orient="index").sort_values("n", ascending=False)
    raw.index.name = "mutation"
    scale = landscape_long.groupby("observable")["std"].mean()
    standardized = raw[columns] / scale.reindex(columns).to_numpy()
    return raw, standardized


def plot_mutation_effects(standardized: pd.DataFrame, raw: pd.DataFrame, save=None) -> plt.Figure:
    """Heatmap mutation × observable de l'effet moyen Δ (couleur : en écarts-types ; texte : Δ brut)."""
    fig, ax = plt.subplots(figsize=(0.75 * standardized.shape[1] + 2.5, 0.45 * len(standardized) + 1.8))
    lim = np.nanmax(np.abs(standardized.to_numpy())) or 1
    im = ax.imshow(standardized.to_numpy(), aspect="auto", cmap=DIVERGING, vmin=-lim, vmax=lim)
    ax.set_xticks(range(standardized.shape[1]), standardized.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(standardized)), [f"{m}  (n={int(raw.n[m])})" for m in standardized.index])
    for i, m in enumerate(standardized.index):
        for j, c in enumerate(standardized.columns):
            v = raw.at[m, c]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2g}", ha="center", va="center", fontsize=7, color="#222222")
    ax.grid(False)
    ax.set_title("effet moyen d'une mutation : Δ brut (texte), Δ / écart-type de l'observable (couleur)",
                 loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="Δ / écart-type")
    fig.tight_layout()
    return _save(fig, save)


def plot_delta_histograms(walks: list[pd.DataFrame], columns: list[str], save=None) -> plt.Figure:
    """Distribution des Δ par type de mutation (lignes) pour chaque observable (colonnes) ;
    les 1 % de Δ les plus extrêmes de chaque côté sont hors du cadre."""
    frames = []
    for w in walks:
        d = w[columns].apply(pd.to_numeric, errors="coerce").diff().iloc[1:]
        d["action"] = w.action.iloc[1:].to_numpy()
        frames.append(d)
    d = pd.concat(frames)
    mutations = d.action.value_counts().index.tolist()
    fig, axes = plt.subplots(len(mutations), len(columns), figsize=(2.3 * len(columns), 1.5 * len(mutations) + 0.6),
                             squeeze=False, sharex="col")
    for j, c in enumerate(columns):
        values = d[c].dropna()
        lo, hi = (values.quantile(0.01), values.quantile(0.99)) if len(values) else (0, 1)
        bins = np.linspace(lo, hi, 31) if hi > lo else 10
        for i, m in enumerate(mutations):
            ax = axes[i, j]
            values_m = d.loc[d.action == m, c].dropna()
            ax.hist(values_m[values_m.between(lo, hi)], bins=bins, color=BLUE, edgecolor="white", linewidth=0.4)
            ax.axvline(0, color=GRAY, linewidth=0.8)
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(f"Δ {c}", fontsize=9)
            if j == 0:
                ax.set_ylabel(m, rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, save)


def plot_effects_heatmap(effects: pd.DataFrame, responses: list[str], term_labels: dict[str, str],
                         title: str = "", save=None) -> plt.Figure:
    """Termes (effets principaux + interactions) × réponses : couleur = t (effet / erreur, borné à ±10),
    texte = étoiles de significativité. Une ligne blanche = le terme n'a d'effet sur aucune réponse."""
    t = effects.pivot(index="term", columns="response", values="t").reindex(columns=responses)
    p = effects.pivot(index="term", columns="response", values="p").reindex(columns=responses)
    order = [term for term in term_labels if term in t.index]
    t, p = t.loc[order], p.loc[order]
    fig, ax = plt.subplots(figsize=(0.62 * len(responses) + 3.5, 0.32 * len(order) + 2.2))
    im = ax.imshow(t.clip(-10, 10).to_numpy(), aspect="auto", cmap=DIVERGING, vmin=-10, vmax=10)
    ax.set_xticks(range(len(responses)), responses, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), [term_labels[term] for term in order])
    for i in range(len(order)):
        for j in range(len(responses)):
            s = doe.stars(p.iat[i, j])
            if s:
                ax.text(j, i, s, ha="center", va="center", fontsize=7, color="#111111")
    ax.grid(False)
    ax.set_title((f"{title}\n" if title else "")
                 + "effets standardisés t (bleu : la réponse augmente au niveau +) — * p<0.05, ** p<0.01, *** p<0.001",
                 loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="t")
    fig.tight_layout()
    return _save(fig, save)


def plot_pareto(effects: pd.DataFrame, responses: list[str], term_labels: dict[str, str],
                top: int = 12, save=None) -> plt.Figure:
    """Pour chaque réponse : |t| des termes les plus forts ; trait = seuil p = 0.05."""
    from scipy import stats  # pylint: disable=import-outside-toplevel

    fig, axes = _grid(len(responses), ncols=3, width=4.6, height=3.2)
    for ax, response in zip(axes.flat, responses):
        e = effects[effects.response == response].assign(abs_t=lambda d: d.t.abs()).nlargest(top, "abs_t")[::-1]
        if e.empty:
            continue
        colors = [BLUE if v > 0 else ORANGE for v in e.effect]
        ax.barh([term_labels.get(term, term) for term in e.term], e.abs_t, color=colors)
        threshold = stats.t.ppf(0.975, max(1, int(e.n.iloc[0]) - len(term_labels) - 1))
        ax.axvline(threshold, color=GRAY, linewidth=1, linestyle="--")
        ax.set_title(response, loc="left", fontsize=9)
        ax.set_xlabel("|t|  (bleu : hausse, orange : baisse)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, save)


def plot_main_effects(design: pd.DataFrame, data: pd.DataFrame, responses: list[str],
                      factor_labels: dict[str, str], save=None) -> plt.Figure:
    """Moyenne de chaque réponse au niveau − et + de chaque facteur (un segment par facteur)."""
    merged = data.join(design, on="config_id")
    factors = list(design.columns)
    fig, axes = _grid(len(responses), ncols=3, width=5.2, height=2.5)
    for ax, response in zip(axes.flat, responses):
        overall = merged[response].mean()
        ax.axhline(overall, color=GRAY, linewidth=0.8)
        for i, f in enumerate(factors):
            means = merged.groupby(f)[response].mean()
            ax.plot([3 * i, 3 * i + 1.6], [means.get(-1, np.nan), means.get(1, np.nan)], color=BLUE,
                    linewidth=2, marker="o", markersize=4)
        ax.set_xticks([3 * i + 0.8 for i in range(len(factors))], factors)
        ax.set_title(response, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.suptitle("effets principaux : niveau − (gauche) → + (droite)\n"
                 + ", ".join(f"{k} = {v}" for k, v in factor_labels.items()), x=0.01, ha="left", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, save)


def plot_interactions(design: pd.DataFrame, data: pd.DataFrame, pairs: list[tuple[str, str, str]],
                      factor_labels: dict[str, str], save=None) -> plt.Figure:
    """Pour chaque (réponse, facteur 1, facteur 2) : moyenne vs niveau du facteur 1, une ligne par
    niveau du facteur 2. Lignes non parallèles = interaction (l'effet de l'un dépend de l'autre)."""
    merged = data.join(design, on="config_id")
    fig, axes = _grid(len(pairs), ncols=4, width=3.4, height=2.7)
    for ax, (response, f1, f2) in zip(axes.flat, pairs):
        for level, color in [(-1, GRAY), (1, BLUE)]:
            means = merged[merged[f2] == level].groupby(f1)[response].mean()
            ax.plot([-1, 1], [means.get(-1, np.nan), means.get(1, np.nan)], color=color, marker="o",
                    linewidth=2, label=f"{f2} {'−' if level < 0 else '+'}")
        ax.set_xticks([-1, 1], [f"{f1} −", f"{f1} +"])
        ax.set_title(f"{response}\n{factor_labels[f1]} × {factor_labels[f2]}", loc="left", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, save)


def plot_fitness_curves(goals: list[tuple], walks: list[pd.DataFrame], properties: dict[str, str],
                        save=None) -> plt.Figure:
    """Courbes de fitness avec les molécules visitées (x = valeur brute, y = score)."""
    import fitness_functions as ff  # pylint: disable=import-outside-toplevel

    column_of = {prop: col for col, prop in properties.items()}
    points, ranges = {}, {}
    for name, curve, _ in goals:
        col = column_of.get(name)
        values = (pd.concat([pd.to_numeric(w[col], errors="coerce") for w in walks]).dropna()
                  if col else pd.Series(dtype=float))
        lo, hi = ff.auto_range(curve)
        if len(values):
            pad = 0.05 * ((values.max() - values.min()) or 1.0)
            lo, hi = min(lo, values.min() - pad), max(hi, values.max() + pad)
        points[name], ranges[name] = values.tolist(), (lo, hi)
    fig, _ = ff.plot_curves(goals, ranges, points=points)
    return _save(fig, save)


def slug(text: str) -> str:
    """Texte utilisable dans un nom de fichier."""
    import re  # pylint: disable=import-outside-toplevel
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(text), flags=re.ASCII).strip("-")


def save_image(image: Any, path: str | Path) -> None:
    """Enregistre une grille de molécules RDKit (image PIL ou PNG IPython)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image.save(path)
    else:
        Path(path).write_bytes(image.data)


def walks_by_config(runs: list[dict[str, Any]]) -> dict[str, list[pd.DataFrame]]:
    """config_id -> marches de ses répétitions."""
    walks: dict[str, list[pd.DataFrame]] = {}
    for run in runs:
        walks.setdefault(run["config_id"], []).append(run["walk"])
    return walks


def summary_table(responses: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Par configuration : « moyenne ± écart-type » des réponses sur les répétitions."""
    grouped = responses.groupby("config_id")[keys]
    mean, std = grouped.mean(), grouped.std()
    return pd.DataFrame({k: mean[k].map("{:.3g}".format) + " ± " + std[k].map("{:.2g}".format) for k in keys})


def significant_interactions(effects: pd.DataFrame, chains: list[str], responses: list[str],
                             top: int = 12, alpha: float = 0.05) -> pd.DataFrame:
    """Interactions (chaînes d'alias) significatives sur ces réponses, les plus fortes d'abord."""
    e = effects[effects.term.isin(chains) & (effects.p < alpha) & effects.response.isin(responses)]
    return e.assign(abs_t=e.t.abs()).nlargest(top, "abs_t").drop(columns="abs_t")


def _fitness_column(table: pd.DataFrame, goals: list[tuple], column_of: dict[str, str],
                    weights: np.ndarray, aggregation: str) -> np.ndarray:
    """Fitness de chaque ligne de table à partir des colonnes enregistrées (NaN si une valeur manque)."""
    import fitness_functions as ff  # pylint: disable=import-outside-toplevel

    def score(curve, x):
        return curve(x) if np.isfinite(x) else math.nan

    scores = np.column_stack([
        pd.to_numeric(table[column_of[name]], errors="coerce").map(lambda x, c=curve: score(c, x)).to_numpy()
        for name, curve, _ in goals
    ])
    fitness = np.full(len(table), math.nan)
    for i in np.flatnonzero(np.isfinite(scores).all(axis=1)):
        fitness[i] = ff.aggregate(scores[i], weights, aggregation)
    return fitness


def rescore_fitness(runs: list[dict[str, Any]], goals: list[tuple], aggregation: str,
                    properties: dict[str, str]) -> list[dict[str, Any]]:
    """Copies des marches où la colonne "fitness" est recalculée pour une autre fonction de fitness,
    à partir des variables enregistrées sur chaque molécule (aucune marche relancée).

    properties : colonne -> propriété evomol (random_walk.recorded_properties(config), ou
    observable_properties pour les seules observables) ; n_heavy_atoms (colonne n_atoms) s'y ajoute.
    Si la marche a ses voisinages complets (clé "values", load_runs(…, values=…)), leur fitness est
    recalculée aussi et l'évolvabilité suit ; sinon les voisinages des copies sont retirés.
    """
    import fitness_functions as ff  # pylint: disable=import-outside-toplevel

    ff.check_goals(goals, aggregation)
    column_of = {prop: col for col, prop in properties.items()}
    column_of["n_heavy_atoms"] = "n_atoms"
    unknown = [name for name, *_ in goals if name not in column_of]
    if unknown:
        raise ValueError(f"Propriétés non enregistrées dans les marches : {unknown}. "
                         f"Disponibles : {sorted(column_of)}. Sinon, les recalculer depuis les SMILES "
                         "(colonne smiles des marches et des voisinages).")
    needed = {column_of[name] for name, *_ in goals}
    weights = np.array([w for *_, w in goals], dtype=float)

    new_runs = []
    for run in runs:
        missing = needed - set(run["walk"].columns)
        if missing:
            raise ValueError(f"{run['config_id']} rep{run['rep']} : colonnes absentes de la marche {sorted(missing)}")
        walk = run["walk"].copy()
        walk["fitness"] = _fitness_column(walk, goals, column_of, weights, aggregation)
        new_run = {**run, "walk": walk, "neighborhoods": None, "neighbors": None}
        values = run.get("values")
        if values is not None and not needed - set(values.columns):
            values = values.copy()
            values["fitness"] = _fitness_column(values, goals, column_of, weights, aggregation)
            parents = values[values.is_parent]
            f0 = parents.set_index("step").fitness
            neighbors = values[~values.is_parent]
            n_neighbors = (run["neighborhoods"].set_index("step").n_neighbors
                           if run.get("neighborhoods") is not None else pd.Series(dtype=float))
            new_run.update({
                "values": values,
                "neighborhoods": pd.DataFrame({
                    "step": parents.step.to_numpy(), "smiles": parents.smiles.to_numpy(),
                    "fitness": parents.fitness.to_numpy(),
                    "n_neighbors": parents.step.map(n_neighbors).to_numpy(),
                    "n_evaluated": parents.step.map(neighbors.step.value_counts()).fillna(0).astype(int).to_numpy(),
                }),
                "neighbors": pd.DataFrame({
                    "step": neighbors.step.to_numpy(), "mutation": neighbors.mutation.to_numpy(),
                    "fitness": neighbors.fitness.to_numpy(),
                    "delta": (neighbors.fitness - neighbors.step.map(f0)).to_numpy(),
                }),
            })
        else:
            new_run["values"] = None
        new_runs.append(new_run)
    return new_runs


def rescore_plan(out_dir: str | Path, label: str, goals: list[tuple], aggregation: str,
                 design: pd.DataFrame, chains: list[str], properties: dict[str, str],
                 values: bool | str | Path = True, runs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Toutes les réponses d'un plan pour une autre fitness, sans relancer de marche : paysage de la
    fitness, évolvabilité (si les voisinages complets sont là) et effets des facteurs.

    Écrit out_dir/fitness_<label>/ : responses.csv, actions.csv, …, effects.csv, fitness.json.
    values : où trouver les voisinages complets (voir load_run) ; runs : marches déjà chargées.

    Returns:
        {"runs", "responses", "effects", "dir"}.
    """
    import random_walk as rw  # pylint: disable=import-outside-toplevel

    new_dir = Path(out_dir) / f"fitness_{label}"
    new_dir.mkdir(parents=True, exist_ok=True)
    runs = runs if runs is not None else load_runs(out_dir, values)
    new_runs = rescore_fitness(runs, goals, aggregation, properties)
    n_without = sum(r["neighborhoods"] is None for r in new_runs)
    if n_without:
        print(f"{n_without} marches sans voisinages complets : pas d'évolvabilité pour elles")
    results = analyse_runs(new_runs, ["fitness", "n_atoms"], new_dir)
    responses = results["responses"]
    numeric = [c for c in responses.columns if c not in ("config_id", "rep")]
    effects = doe.estimate_effects(design, chains, responses, numeric)
    effects.to_csv(new_dir / "effects.csv", index=False)
    (new_dir / "fitness.json").write_text(json.dumps(
        {"FITNESS_GOALS": rw.goals_to_json(goals), "AGGREGATION": aggregation}, indent=1, ensure_ascii=False))
    return {"runs": new_runs, "responses": responses, "effects": effects, "dir": new_dir}


FITNESS_CRITERIA = {  # réponse -> sens souhaité pour accélérer la recherche (+1 : plus haut, −1 : plus bas)
    "fitness rho1": +1,          # paysage plus lisse
    "fitness corr_length": +1,   # corrélation sur plus de pas
    "fitness M": -1,             # moins de changements de sens (moins d'optima le long du chemin)
    "fitness neutral_pct": -1,   # moins de plateaux
    "fitness up_pct": +1,        # plus de pas qui améliorent
    "fitness mean": +1,          # zone visitée meilleure
    "fitness max": +1,           # meilleure molécule rencontrée
}


def recommendations(effects: pd.DataFrame, factor_labels: dict[str, str],
                    criteria: dict[str, int] | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Pour chaque facteur et chaque critère : niveau à préférer (+ ou −) si son effet est
    significatif, « · » sinon ; colonne bilan = nb de critères en faveur de + et de −."""
    criteria = criteria or FITNESS_CRITERIA
    rows = {}
    for f, label in factor_labels.items():
        row, plus, minus = {}, 0, 0
        for response, direction in criteria.items():
            e = effects[(effects.term == f) & (effects.response == response)]
            if e.empty or not np.isfinite(e.p.iloc[0]) or e.p.iloc[0] >= alpha:
                row[response] = "·"
                continue
            better_plus = np.sign(e.effect.iloc[0]) == direction
            row[response] = f"{'+' if better_plus else '−'} {doe.stars(e.p.iloc[0])}"
            plus, minus = plus + better_plus, minus + (not better_plus)
        row["bilan"] = f"+ : {plus}   − : {minus}"
        rows[f"{f}  {label}"] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def synthesis(effects: pd.DataFrame, term_labels: dict[str, str], alpha: float = 0.05) -> pd.DataFrame:
    """Par terme : réponses sur lesquelles l'effet est significatif (p < alpha), avec le sens."""
    rows = []
    for term in term_labels:
        e = effects[(effects.term == term) & (effects.p < alpha)].sort_values("p")
        rows.append({
            "terme": term_labels[term],
            "nb de réponses touchées": len(e),
            "augmente": ", ".join(e.response[e.effect > 0]),
            "diminue": ", ".join(e.response[e.effect < 0]),
        })
    return pd.DataFrame(rows).set_index("terme")


# ==============================================================================
#  FRONT DE PARETO DES CONFIGURATIONS
# ==============================================================================
EXPLORABILITY_CRITERIA = {  # réponse -> sens souhaité (+1 : plus haut, −1 : plus bas)
    "evolvability better_pct": +1,        # beaucoup de voisins meilleurs
    "evolvability local_optima_pct": -1,  # peu d'optima locaux
    "distinct_pct": +1,                   # pas de perte de diversité
    "fitness rho1_taille": +1,            # paysage lisse, sans l'effet taille
}


def pareto_configs(responses: pd.DataFrame, criteria: dict[str, int] | None = None) -> pd.DataFrame:
    """Moyenne ± intervalle de confiance à 95 % de chaque critère par configuration, et si la
    configuration est sur le front de Pareto (aucune autre n'est au moins aussi bonne sur tous les
    critères et meilleure sur un, d'après les moyennes). Triée : front d'abord."""
    criteria = criteria or EXPLORABILITY_CRITERIA
    keys = list(criteria)
    grouped = responses.groupby("config_id")[keys]
    means, sem = grouped.mean(), grouped.sem()
    oriented = means * pd.Series(criteria)  # tout en « plus haut = mieux »
    values = oriented.to_numpy()
    ok = np.isfinite(values).all(axis=1)
    front = []
    for i, row in enumerate(values):
        dominated = ok[i] and any(
            ok[j] and (values[j] >= row).all() and (values[j] > row).any() for j in range(len(values)) if j != i)
        front.append(bool(ok[i] and not dominated))
    table = means.round(3).astype(str) + " ± " + (1.96 * sem).round(3).astype(str)
    table["front"] = front
    table["marches"] = responses.groupby("config_id").size()
    return table.sort_values("front", ascending=False)


def plot_front(responses: pd.DataFrame, x: str, y: str, criteria: dict[str, int] | None = None,
               save=None) -> plt.Figure:
    """Moyenne de chaque configuration sur deux critères (barres : IC 95 %), front de Pareto (tous les
    critères) en bleu, étiqueté."""
    criteria = criteria or EXPLORABILITY_CRITERIA
    front = pareto_configs(responses, criteria).front
    grouped = responses.groupby("config_id")[[x, y]]
    means, sem = grouped.mean(), grouped.sem()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for config_id, row in means.iterrows():
        on_front = bool(front.get(config_id, False))
        ax.errorbar(row[x], row[y], xerr=1.96 * sem.loc[config_id, x], yerr=1.96 * sem.loc[config_id, y],
                    fmt="o", ms=5, color=BLUE if on_front else GRAY, alpha=1 if on_front else 0.5, lw=0.8)
        if on_front:
            ax.annotate(config_id, (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    arrow = {1: "↑", -1: "↓"}
    ax.set_xlabel(f"{x} {arrow.get(criteria.get(x), '')}")
    ax.set_ylabel(f"{y} {arrow.get(criteria.get(y), '')}")
    ax.set_title("configurations (bleu : front de Pareto sur tous les critères)", loc="left", fontsize=9)
    fig.tight_layout()
    return _save(fig, save)
