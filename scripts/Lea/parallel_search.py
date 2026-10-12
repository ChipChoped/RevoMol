"""
Plusieurs recherches évolutionnaires en parallèle, une par processus.

Chaque course part des mêmes paramètres communs (arguments de
scripts/Lea/evolutionary_algorithm.main(), dont "unique_kekule" et
"remember_invalid") et peut en changer n'importe lequel.

La fitness vient de scripts/Lea/fitness_functions.py (courbes de
désirabilité). Utilisé par notebooks/search_parallel.ipynb.

Résultats : <output_dir>/<course>/ (population finale, run_summary,
journal DEBUG propre à la course, resultat.json avec la trajectoire).
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LEA = ROOT / "scripts" / "Lea"
for _p in (str(ROOT), str(LEA)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def trajectory_point(population, generation: int, t: float) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Meilleure et moyenne fitness, taille moyenne et maximale de la population."""
    # pylint: disable=import-outside-toplevel
    from evomol.representation import MolecularGraph

    fitness = [m.value("fitness") for m in population]
    sizes = [m.get_representation(MolecularGraph).nb_atoms for m in population]
    return {
        "generation": generation,
        "t": t,
        "best": max(fitness),
        "mean": sum(fitness) / len(fitness),
        "mean_atoms": sum(sizes) / len(sizes),
        "max_atoms": max(sizes),
    }


def summarize(
    name: str,
    ea,  # type: ignore[no-untyped-def]
    goal_names: list[str],
    elapsed: float,
    trajectory: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Résultat d'une course terminée (contenu de resultat.json)."""
    # pylint: disable=import-outside-toplevel
    from rdkit import Chem

    from evomol.representation import MolecularGraph

    ns = ea.neighborhood_strategy
    population = sorted(ea.population, key=lambda m: m.value("fitness"), reverse=True)
    smiles = [str(m) for m in population]
    return {
        "course": name,
        "parametres": {k: v for k, v in (params or {}).items() if _jsonable(v)},
        "secondes": elapsed,
        "generations": ea.nb_generations,
        "meilleure_fitness": population[0].value("fitness"),
        "meilleure_molecule": smiles[0],
        "fitness_moyenne": sum(m.value("fitness") for m in population) / len(population),
        "molecules_distinctes": len({Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles}),
        "atomes_moyen": sum(m.get_representation(MolecularGraph).nb_atoms for m in population)
        / len(population),
        "evaluations": ns.n_function_calls_total,
        "valides": ns.n_valid_candidates,
        "invalides": ns.n_invalid_candidates,
        "rejets_memoire": ns.n_known_invalid_candidates,
        "tabou": ns.n_tabu_candidates,
        "candidats_par_s": (
            ns.n_function_calls_total + ns.n_known_invalid_candidates + ns.n_tabu_candidates
        ) / elapsed,
        "top": [
            {"smiles": s, "fitness": m.value("fitness"),
             **{goal: m.value(goal) for goal in goal_names}}
            for s, m in zip(smiles, population)
        ],
        "trajectoire": trajectory,
    }


def _run(
    name: str,
    params: dict[str, Any],
    fitness_goals: list[tuple],
    aggregation: str,
    run_dir: str,
    history_every: int,
) -> None:
    """Une course (exécutée dans son propre processus)."""
    # pylint: disable=import-outside-toplevel
    os.chdir(ROOT)  # les filtres ChEMBL/ZINC lisent external_data/ en relatif
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    os.makedirs(run_dir, exist_ok=True)
    # messages console (une ligne par génération) dans un fichier de la course,
    # pour ne pas mélanger les sorties des processus dans le notebook
    console = open(  # pylint: disable=consider-using-with
        os.path.join(run_dir, "console.txt"), "w", encoding="utf-8", buffering=1
    )
    sys.stdout = sys.stderr = console
    params = dict(params)

    import evolutionary_algorithm as ea_script
    import fitness_functions as ff

    ea_script._build_fitness = (  # pylint: disable=protected-access
        lambda goals: ff.build_fitness(goals, aggregation)
    )

    # journal DEBUG propre à la course (plusieurs processus ne peuvent pas
    # partager le même fichier tournant)
    init_logger = ea_script.init_logger
    ea_script.init_logger = lambda config_file, output_file: init_logger(
        config_file=config_file, output_file=os.path.join(run_dir, "evomol.log")
    )

    goal_names = [goal_name for goal_name, *_ in fitness_goals]
    trajectory: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def _callback(population, generation):  # type: ignore[no-untyped-def]
        point = trajectory_point(population, generation, time.perf_counter() - t0)
        if generation % history_every == 0:
            point["population"] = {
                goal: [m.value(goal) for m in population] for goal in goal_names
            }
        trajectory.append(point)

    ea = ea_script.main(
        **params,
        fitness_goals=fitness_goals,
        generation_callback=_callback,
        callback_every_n=1,
        output_dir=run_dir,
    )
    elapsed = time.perf_counter() - t0
    result = summarize(name, ea, goal_names, elapsed, trajectory, params)
    with open(os.path.join(run_dir, "resultat.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


def run_parallel(
    common: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    fitness_goals: list[tuple],
    aggregation: str,
    output_dir: str | Path,
    history_every: int = 10,
) -> dict[str, dict[str, Any]]:
    """Lance une course par entrée de `runs`, chacune dans son processus.

    Args:
        common: arguments communs de evolutionary_algorithm.main().
        runs: {nom de la course: paramètres qui remplacent ceux de common}.
        fitness_goals: [(propriété, courbe de fitness_functions, poids)].
        aggregation: "sum" ou "geometric" (fitness_functions.build_fitness).
        output_dir: dossier de l'expérience ; une course = un sous-dossier.
        history_every: garder les valeurs de la population toutes les N
            générations (tracés sur les courbes).

    Returns:
        {nom de la course: contenu de resultat.json} pour les courses terminées.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = get_context("spawn")
    procs = []
    for name, overrides in runs.items():
        unknown = set(overrides) - set(common)
        if unknown:
            raise ValueError(f"Course {name!r} : paramètres inconnus {sorted(unknown)}")
        params = {**common, **overrides}
        proc = ctx.Process(
            target=_run,
            args=(name, params, fitness_goals, aggregation, str(output_dir / name), history_every),
            name=name,
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
    return load_results(output_dir, runs)


def load_results(output_dir: str | Path, runs: dict[str, Any] | None = None) -> dict[str, dict]:
    """Relit les resultat.json d'une expérience (toutes les courses si runs=None)."""
    output_dir = Path(output_dir)
    names = list(runs) if runs is not None else sorted(
        p.name for p in output_dir.iterdir() if (p / "resultat.json").exists()
    )
    results = {}
    for name in names:
        path = output_dir / name / "resultat.json"
        if path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"{name} : pas de résultat (course échouée, voir {output_dir / name})")
    return results


SUMMARY_COLUMNS = [
    ("generations", "générations", "{:d}"),
    ("candidats_par_s", "candidats/s", "{:.0f}"),
    ("evaluations", "évaluations", "{:d}"),
    ("rejets_memoire", "rejets mémoire", "{:d}"),
    ("meilleure_fitness", "meilleure fitness", "{:.4f}"),
    ("fitness_moyenne", "fitness moyenne", "{:.4f}"),
    ("atomes_moyen", "atomes moyens", "{:.1f}"),
    ("molecules_distinctes", "molécules distinctes", "{:d}"),
]


def summary_table(results: dict[str, dict]) -> str:
    """Tableau texte d'une ligne par course."""
    width = max(len(n) for n in results) if results else 10
    header = f"{'course':<{width}}  " + "  ".join(f"{label:>{max(len(label), 9)}}"
                                                   for _, label, _ in SUMMARY_COLUMNS)
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        cells = "  ".join(f"{fmt.format(r[key]):>{max(len(label), 9)}}"
                          for key, label, fmt in SUMMARY_COLUMNS)
        lines.append(f"{name:<{width}}  {cells}")
    return "\n".join(lines)


def plot_trajectories(results: dict[str, dict]):  # type: ignore[no-untyped-def]
    """Meilleure fitness et taille moyenne, en fonction du temps et des générations."""
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for name, r in results.items():
        t = r["trajectoire"]
        gens = [p["generation"] for p in t]
        times = [p["t"] for p in t]
        axes[0, 0].plot(times, [p["best"] for p in t], label=name)
        axes[0, 1].plot(gens, [p["best"] for p in t], label=name)
        axes[1, 0].plot(times, [p["mean_atoms"] for p in t], label=name)
        axes[1, 1].plot(gens, [p["mean_atoms"] for p in t], label=name)
    for ax, (x, y) in zip(axes.flat, [("temps (s)", "meilleure fitness"),
                                      ("génération", "meilleure fitness"),
                                      ("temps (s)", "atomes moyens"),
                                      ("génération", "atomes moyens")]):
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig
