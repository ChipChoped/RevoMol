"""
Plans d'expérience factoriels fractionnaires à 2 niveaux et estimation des effets.

Un plan 2^(k−p) étudie k facteurs à 2 niveaux (codés −1 / +1) en 2^(k−p) configurations :
k − p facteurs de base prennent toutes leurs combinaisons, les p autres sont générés comme
produits de facteurs de base (générateurs, ex. F = ABCD). Le prix : certains effets sont
confondus (aliasés). La résolution dit lesquels :
  - résolution III : effets principaux aliasés avec des interactions d'ordre 2 ;
  - résolution IV  : effets principaux propres ; interactions d'ordre 2 aliasées entre elles ;
  - résolution V   : effets principaux et interactions d'ordre 2 propres.

Usage :
    design, info = fractional_factorial(7, {"F": "ABCD", "G": "ABDE"})
    effects = estimate_effects(design, info["chains"], data, ["rho1", "n_distinct"])
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

# lettres des facteurs (sans I, réservé à l'identité de la relation de définition)
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def _word(letters: frozenset[str]) -> str:
    return "".join(sorted(letters, key=LETTERS.index))


def fractional_factorial(
    n_factors: int, generators: dict[str, str], min_resolution: int = 4
) -> tuple[pd.DataFrame, dict]:
    """Plan 2^(k−p) codé −1 / +1, en ordre standard (le premier facteur de base varie le plus vite).

    Args:
        n_factors: nb de facteurs k (lettres A, B, C, … sans I).
        generators: facteur généré -> mot de facteurs de base, ex. {"F": "ABCD", "G": "ABDE"}.
        min_resolution: résolution minimale exigée (ValueError sinon).

    Returns:
        design: une ligne par configuration (index c01, c02, …), une colonne par facteur.
        info: "resolution", "relation" (mots de la relation de définition), "chains"
            (chaînes d'alias des interactions d'ordre 2, ex. ["AB", "CE=FG", …]).
    """
    letters = list(LETTERS[:n_factors])
    unknown = set(generators) - set(letters) | {c for w in generators.values() for c in w} - set(letters)
    if unknown:
        raise ValueError(f"Lettres inconnues dans les générateurs : {sorted(unknown)}")
    base = [l for l in letters if l not in generators]
    n_runs = 2 ** len(base)
    levels = np.array(list(itertools.product([-1, 1], repeat=len(base))))[:, ::-1]
    columns = {l: levels[:, i] for i, l in enumerate(base)}
    for l, word in generators.items():
        if any(c in generators for c in word):
            raise ValueError(f"{l} = {word} : un générateur ne doit contenir que des facteurs de base")
        columns[l] = np.prod([columns[c] for c in word], axis=0)
    design = pd.DataFrame({l: columns[l] for l in letters},
                          index=pd.Index([f"c{i:02d}" for i in range(1, n_runs + 1)], name="config"))

    # relation de définition : tous les produits de mots générateurs (I = ABCDF = …)
    words = [frozenset(word + l) for l, word in generators.items()]
    relation: set[frozenset[str]] = set()
    for r in range(1, len(words) + 1):
        for combination in itertools.combinations(words, r):
            product: frozenset[str] = frozenset()
            for w in combination:
                product = product ^ w
            relation.add(product)
    resolution = min((len(w) for w in relation), default=math.inf)

    # contrôles : équilibre, orthogonalité, résolution
    x = design.to_numpy()
    if not (x.sum(axis=0) == 0).all():
        raise ValueError("Plan non équilibré")
    if not (x.T @ x == n_runs * np.eye(n_factors)).all():
        raise ValueError("Plan non orthogonal")
    if resolution < min_resolution:
        raise ValueError(f"Résolution {resolution} < {min_resolution} : changer les générateurs")

    # chaînes d'alias des interactions d'ordre 2 (seules les interactions d'ordre 2 sont listées)
    chains, seen = [], set()
    for pair in itertools.combinations(letters, 2):
        pair = frozenset(pair)
        if pair in seen:
            continue
        aliases = sorted((_word(pair ^ w) for w in relation if len(pair ^ w) == 2), key=lambda s: [LETTERS.index(c) for c in s])
        chain = [_word(pair), *aliases]
        seen |= {frozenset(t) for t in chain}
        chains.append("=".join(chain))

    info = {
        "resolution": resolution,
        "relation": ["I=" + _word(w) for w in sorted(relation, key=lambda w: (len(w), _word(w)))],
        "chains": chains,
    }
    return design, info


def model_matrix(design: pd.DataFrame, chains: list[str]) -> pd.DataFrame:
    """Colonnes du modèle : effets principaux + une colonne par chaîne d'interactions (codées)."""
    columns = {l: design[l] for l in design.columns}
    for chain in chains:
        first = chain.split("=")[0]
        columns[chain] = design[first[0]] * design[first[1]]
    return pd.DataFrame(columns, index=design.index)


def estimate_effects(
    design: pd.DataFrame, chains: list[str], data: pd.DataFrame, responses: list[str],
    config_col: str = "config_id",
) -> pd.DataFrame:
    """Effets des facteurs et des interactions sur chaque réponse, par moindres carrés.

    Modèle : y = b0 + Σ bᵢ xᵢ + Σ b_chaîne x_chaîne, sur toutes les marches (répétitions comprises).
    effet = 2b = moyenne(y | +) − moyenne(y | −). L'erreur vient de la variabilité entre
    répétitions et des termes non modélisés ; t = effet / erreur, p bilatéral.

    Args:
        design: plan codé (index = identifiants de configuration).
        chains: chaînes d'alias des interactions (info["chains"]).
        data: une ligne par marche, colonne config_col + colonnes de réponses.
        responses: réponses à analyser.

    Returns:
        DataFrame long : response, term, effect, se, t, p, q, pct_variance, n. q : p corrigé de
        Benjamini-Hochberg sur tous les tests du tableau (toutes réponses et tous termes) — garder
        q < 0.05 limite la part attendue de faux positifs à 5 % des effets retenus.
    """
    terms = model_matrix(design, chains)
    rows = []
    for response in responses:
        y_all = pd.to_numeric(data[response], errors="coerce")
        mask = np.isfinite(y_all.to_numpy())
        y = y_all.to_numpy()[mask]
        x_terms = terms.loc[data[config_col].to_numpy()[mask]].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(y)), x_terms])
        n, p = x.shape
        if n <= p or np.ptp(y) == 0:
            for term in terms.columns:
                rows.append({"response": response, "term": term, "effect": 0.0 if n > 0 else math.nan,
                             "se": math.nan, "t": math.nan, "p": math.nan, "pct_variance": math.nan, "n": n})
            continue
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ coef
        df = n - p
        sigma2 = residual @ residual / df
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        se = np.sqrt(np.diag(cov))
        ss_total = ((y - y.mean()) ** 2).sum()
        for j, term in enumerate(terms.columns, start=1):
            t = coef[j] / se[j] if se[j] > 0 else math.nan
            rows.append({
                "response": response, "term": term,
                "effect": 2 * coef[j], "se": 2 * se[j], "t": t,
                "p": 2 * stats.t.sf(abs(t), df) if np.isfinite(t) else math.nan,
                # part de la variance totale expliquée par le terme (plan orthogonal : n·b²)
                "pct_variance": 100 * n * coef[j] ** 2 / ss_total,
                "n": n,
            })
    effects = pd.DataFrame(rows)
    effects.insert(effects.columns.get_loc("p") + 1, "q", benjamini_hochberg(effects.p.to_numpy()))
    return effects


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """q-valeurs de Benjamini-Hochberg (NaN conservés)."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), math.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return q
    order = ok[np.argsort(p[ok])]
    ranked = p[order] * len(ok) / np.arange(1, len(ok) + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def stars(p: float) -> str:
    """* p < 0.05, ** p < 0.01, *** p < 0.001."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
