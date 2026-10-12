"""
================================================================================
 Fitness par fonctions de désirabilité (scores entre 0 et 1)
================================================================================

Remplace les modes "max" / "min" / "target" de evolutionary_algorithm.py
(somme brute poids × valeur) par des COURBES paramétrables bornées dans [0, 1].

Chaque objectif est un tuple :

    (nom_propriété, courbe, poids)

  - nom_propriété : une propriété calculée par evolutionary_algorithm.main()
                    (ex. "huckel_homo_lambda", "SAScore", …)
  - courbe        : transforme la valeur brute en score entre 0 (mauvais) et 1 (parfait)
  - poids         : importance de l'objectif (>= 0)

COURBES DISPONIBLES
  Sigmoid(center, steepness, "up")   : 0 avant center, 1 après   (≈ maximiser)
  Sigmoid(center, steepness, "down") : 1 avant center, 0 après   (≈ minimiser)
  Gaussian(center, sigma)            : 1 en center, décroît des deux côtés (≈ cible)
  courbe_a * courbe_b * …            : produit de courbes dans une même courbe
      ex. Sigmoid(0.2, 30, "up") * Sigmoid(0.6, 30, "down") = fenêtre [0.2, 0.6]

  Le score d'une courbe est toujours ramené dans [0, 1].
  Ajouter une courbe : sous-classer Curve, définir PARAMS et _raw(x).

COMBINAISON DES OBJECTIFS (aggregation)
  "sum"       : Σ wᵢ·sᵢ / Σ wᵢ          → moyenne pondérée, entre 0 et 1
  "geometric" : ∏ sᵢ^(wᵢ / Σ wᵢ)        → un seul score à 0 met la fitness à 0

UTILISATION AVEC evolutionary_algorithm.py (sans modifier ce script)
    ea._build_fitness = lambda goals: build_fitness(goals, "sum")
    ea.main(fitness_goals=[("huckel_homo_lambda", Sigmoid(0.5, 20, "down"), 1.0)], …)
================================================================================
"""

from __future__ import annotations

import copy
import math
import os
import sys
from typing import Sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# pylint: disable=wrong-import-position, import-error

from evomol import evaluation as evaluator
from evomol.representation import Molecule


# ==============================================================================
#  COURBES
# ==============================================================================
class Curve:
    """Courbe de base : valeur brute → score dans [0, 1].

    Une sous-classe définit :
      - PARAMS : noms des paramètres numériques réglables (dans l'ordre d'affichage)
      - _raw(x) : la formule (le résultat est ensuite ramené dans [0, 1])
      - _check() : (optionnel) validation des paramètres
    """

    PARAMS: tuple[str, ...] = ()

    def __call__(self, x: float) -> float:
        try:
            x = float(x)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{self!r} : valeur non numérique {x!r}") from e
        if math.isnan(x):
            raise ValueError(f"{self!r} : valeur NaN")
        return min(1.0, max(0.0, self._raw(x)))

    def _raw(self, x: float) -> float:
        raise NotImplementedError

    def _check(self) -> None:
        """Lève ValueError si les paramètres sont invalides."""

    # --- Paramètres (utilisés par l'éditeur interactif du notebook) ---------
    def params(self) -> dict[str, float]:
        return {p: getattr(self, p) for p in self.PARAMS}

    def leaves(self) -> list[Curve]:
        """Courbes élémentaires qui composent cette courbe (elle-même ici)."""
        return [self]

    def set_param(self, name: str, value: float) -> None:
        if name not in self.PARAMS:
            raise KeyError(f"{type(self).__name__} n'a pas de paramètre {name!r}")
        old = getattr(self, name)
        setattr(self, name, float(value))
        try:
            self._check()
        except ValueError:
            setattr(self, name, old)
            raise

    def copy(self) -> Curve:
        return copy.deepcopy(self)

    # --- Composition --------------------------------------------------------
    def __mul__(self, other: Curve) -> Product:
        if not isinstance(other, Curve):
            return NotImplemented
        return Product(self, other)

    def __repr__(self) -> str:
        args = ", ".join(f"{p}={_fmt(v)}" for p, v in self.params().items())
        return f"{type(self).__name__}({args})"


class Sigmoid(Curve):
    """Courbe en S : 1 / (1 + exp(−k·(x − center))).

    direction "up"   : monte de 0 à 1 autour de center (maximiser)
    direction "down" : descend de 1 à 0 autour de center (minimiser)
    steepness (k > 0) : raideur ; la transition 0.12 → 0.88 a une largeur ≈ 4 / k.
    """

    PARAMS = ("center", "steepness")

    def __init__(self, center: float, steepness: float, direction: str = "up") -> None:
        self.center = float(center)
        self.steepness = float(steepness)
        self.direction = direction
        self._check()

    def _check(self) -> None:
        if not math.isfinite(self.center):
            raise ValueError(f"Sigmoid : center doit être fini (reçu {self.center})")
        if not (self.steepness > 0 and math.isfinite(self.steepness)):
            raise ValueError(f"Sigmoid : steepness doit être > 0 (reçu {self.steepness})")
        if self.direction not in ("up", "down"):
            raise ValueError(
                f"Sigmoid : direction {self.direction!r} inconnue (utiliser 'up' ou 'down')"
            )

    def _raw(self, x: float) -> float:
        sign = 1.0 if self.direction == "up" else -1.0
        z = sign * self.steepness * (x - self.center)
        # forme stable : jamais d'exp d'un grand nombre positif
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def __repr__(self) -> str:
        return (f"Sigmoid(center={_fmt(self.center)}, steepness={_fmt(self.steepness)}, "
                f"direction={self.direction!r})")


class Gaussian(Curve):
    """Cloche : exp(−(x − center)² / (2·sigma²)), vaut 1 en center."""

    PARAMS = ("center", "sigma")

    def __init__(self, center: float, sigma: float) -> None:
        self.center = float(center)
        self.sigma = float(sigma)
        self._check()

    def _check(self) -> None:
        if not math.isfinite(self.center):
            raise ValueError(f"Gaussian : center doit être fini (reçu {self.center})")
        if not (self.sigma > 0 and math.isfinite(self.sigma)):
            raise ValueError(f"Gaussian : sigma doit être > 0 (reçu {self.sigma})")

    def _raw(self, x: float) -> float:
        d = (x - self.center) / self.sigma
        return math.exp(-0.5 * d * d)


class Product(Curve):
    """Produit de courbes (chacune dans [0, 1], donc le produit aussi)."""

    def __init__(self, *curves: Curve) -> None:
        if not curves:
            raise ValueError("Product : au moins une courbe est nécessaire")
        flat: list[Curve] = []
        for c in curves:
            if not isinstance(c, Curve):
                raise TypeError(f"Product : {c!r} n'est pas une Curve")
            flat.extend(c.curves if isinstance(c, Product) else [c])
        self.curves = flat

    def _raw(self, x: float) -> float:
        result = 1.0
        for c in self.curves:
            result *= c(x)
        return result

    def params(self) -> dict[str, float]:
        return {}

    def leaves(self) -> list[Curve]:
        return [leaf for c in self.curves for leaf in c.leaves()]

    def __repr__(self) -> str:
        return " * ".join(repr(c) for c in self.curves)


def _fmt(v: float) -> str:
    return f"{v:.6g}"


# ==============================================================================
#  COMBINAISON DES OBJECTIFS
# ==============================================================================
AGGREGATIONS = ("sum", "geometric")


def aggregate(scores: Sequence[float], weights: Sequence[float], mode: str = "sum") -> float:
    """Combine des scores dans [0, 1] en une fitness dans [0, 1]."""
    if len(scores) != len(weights):
        raise ValueError("aggregate : autant de scores que de poids sont nécessaires")
    _check_weights(weights)
    total_w = float(sum(weights))
    if mode == "sum":
        return sum(w * s for s, w in zip(scores, weights)) / total_w
    if mode == "geometric":
        log_sum = 0.0
        for s, w in zip(scores, weights):
            if w == 0:
                continue
            if s <= 0:
                return 0.0
            log_sum += w * math.log(s)
        return math.exp(log_sum / total_w)
    raise ValueError(f"aggregate : mode {mode!r} inconnu (utiliser {AGGREGATIONS})")


def _check_weights(weights: Sequence[float]) -> None:
    for w in weights:
        if not (w >= 0 and math.isfinite(w)):
            raise ValueError(f"Poids invalide : {w} (doit être fini et >= 0)")
    if sum(weights) <= 0:
        raise ValueError("La somme des poids doit être > 0")


# ==============================================================================
#  CONSTRUCTION DE LA FITNESS
# ==============================================================================
def check_goals(goals: Sequence[tuple], aggregation: str = "sum") -> None:
    """Vérifie le format des objectifs ; lève une erreur explicite sinon."""
    if not goals:
        raise ValueError("fitness_goals est vide")
    for goal in goals:
        if len(goal) != 3:
            raise ValueError(
                f"Objectif {goal!r} : format attendu (nom, courbe, poids). "
                "L'ancien format (nom, 'max'|'min'|'target', poids, cible) n'est pas accepté ici."
            )
        name, curve, _ = goal
        if not isinstance(name, str):
            raise TypeError(f"Objectif {goal!r} : le nom doit être une chaîne")
        if not isinstance(curve, Curve):
            raise TypeError(f"Objectif {name!r} : {curve!r} n'est pas une Curve")
    _check_weights([w for *_, w in goals])
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation {aggregation!r} inconnue (utiliser {AGGREGATIONS})")


def goal_scores(mol: Molecule, goals: Sequence[tuple]) -> list[tuple[str, float, float]]:
    """[(nom, valeur brute, score 0–1)] pour chaque objectif."""
    out = []
    for name, curve, _ in goals:
        value = mol.value(name)
        out.append((name, value, curve(value)))
    return out


def build_fitness(goals: Sequence[tuple], aggregation: str = "sum") -> evaluator.Function:
    """Transforme la liste d'objectifs en UN score « fitness » dans [0, 1].

    Même rôle que _build_fitness de evolutionary_algorithm.py.
    Une propriété NaN rend la molécule invalide (EvaluationError) au lieu
    d'arrêter la recherche.
    """
    check_goals(goals, aggregation)
    goals = list(goals)
    weights = [w for *_, w in goals]

    def _fitness(mol: Molecule) -> float:
        try:
            scores = [s for _, _, s in goal_scores(mol, goals)]
        except ValueError as e:
            raise evaluator.EvaluationError(str(e)) from e
        return aggregate(scores, weights, aggregation)

    return evaluator.Function(name="fitness", function=_fitness)


# ==============================================================================
#  VALEURS D'UNE MOLÉCULE DE RÉFÉRENCE (centres des courbes)
# ==============================================================================
class _HuckelOnly:
    """Fausse molécule qui ne porte que le résultat Hückel (lu par les évaluateurs orbitalaires)."""

    def __init__(self, result) -> None:
        self._result = result

    def value(self, name: str):
        if name != "huckel":
            raise KeyError(name)
        return self._result


def reference_values(smiles: str, orbitals: Sequence = (), overlaps: Sequence[tuple] = ()) -> dict[str, float]:
    """Propriétés Hückel et taille d'une molécule, sous les mêmes noms que dans la recherche.

    orbitals : sélecteurs dont on lit λ ("HOMO-3", …, "LUMO+3") → "huckel_orbital_lambda_<orb>"
    overlaps : tuples de 2+ sélecteurs → recouvrement spatial "huckel_lambda_<orb1>_<orb2>…"
    """
    from rdkit import Chem  # pylint: disable=import-outside-toplevel
    from evomol.evaluation.huckel import kekulized_mol, solve_huckel_mol  # pylint: disable=import-outside-toplevel

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES illisible : {smiles!r}")
    result = solve_huckel_mol(kekulized_mol(mol))
    ref = _HuckelOnly(result)
    values = {
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "huckel_homo_lambda": result.homo_lambda,
        "huckel_lumo_lambda": result.lumo_lambda,
        "huckel_gap_lambda": result.gap_lambda,
    }
    for orbital in orbitals:
        e = evaluator.HuckelOrbitalLambda(orbital)
        values[e.name] = e._evaluate(ref)  # pylint: disable=protected-access
    for selectors in overlaps:
        e = evaluator.HuckelOrbitalLambdaOverlap(list(selectors))
        values[e.name] = e._evaluate(ref)  # pylint: disable=protected-access
    return values


# ==============================================================================
#  TRACÉ
# ==============================================================================
def plot_curves(
    goals: Sequence[tuple],
    ranges: dict[str, tuple[float, float]] | None = None,
    points: dict[str, Sequence[float]] | None = None,
    n: int = 400,
    axes=None,
):
    """Trace la courbe de chaque objectif (+ points optionnels : valeurs de molécules).

    ranges : {nom: (xmin, xmax)} ; sinon intervalle déduit des paramètres de la courbe.
    points : {nom: [valeurs brutes]} posées sur la courbe.
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    ranges = ranges or {}
    if axes is None:
        fig, axes = plt.subplots(1, len(goals), figsize=(4.5 * len(goals), 3.2), squeeze=False)
        axes = axes[0]
    else:
        fig = axes[0].figure
    for ax, (name, curve, weight) in zip(axes, goals):
        xmin, xmax = ranges.get(name) or auto_range(curve)
        xs = [xmin + (xmax - xmin) * i / (n - 1) for i in range(n)]
        ax.plot(xs, [curve(x) for x in xs], color="black", lw=2)
        if points and points.get(name):
            vals = [v for v in points[name] if v is not None and math.isfinite(v)]
            ax.scatter(vals, [curve(v) for v in vals], s=18, alpha=0.6, zorder=3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name}  (poids {weight:g})", fontsize=10)
        ax.set_xlabel("valeur brute")
        ax.set_ylabel("score")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


def auto_range(curve: Curve) -> tuple[float, float]:
    """Intervalle de tracé déduit des paramètres de la courbe."""
    lo, hi = math.inf, -math.inf
    for leaf in curve.leaves():
        c = getattr(leaf, "center", 0.0)
        if isinstance(leaf, Sigmoid):
            half = 6.0 / leaf.steepness
        elif isinstance(leaf, Gaussian):
            half = 4.0 * leaf.sigma
        else:
            half = 1.0
        lo, hi = min(lo, c - half), max(hi, c + half)
    if not math.isfinite(lo):
        lo, hi = 0.0, 1.0
    return lo, hi
