# newmut : règles matricielles et nouvelles mutations

Ce dossier ajoute à RevoMol :
1. des **règles de validité lues sur la matrice** des ordres de liaison (R1 à R4), appliquées comme la valence,
   c'est-à-dire au moment de lister les mutations possibles, et jamais comme filtre sur les molécules produites ;
2. de **nouvelles mutations**, élémentaires (AppendAtomRule, SubstituteShift) et composées (greffe, annélation,
   inversion d'un chemin alterné, aromatisation).

Rien n'est modifié en dehors des dossiers `newmut` : avec toutes les nouvelles options à `False`, RevoMol se comporte
exactement comme avant (vérifié par `tests/newmut/test_non_regression.py`).

La spécification d'origine était écrite pour l'EvoMol de jules-leguy (masques de taille fixe). Dans RevoMol, chaque
type de mutation liste lui-même ses actions possibles (`list_actions`) : le « masque » de la spécification est donc
cette liste. La référence exécutable est l'atelier de mutations (`reference_core.js`, `reference_app.js`), traduit
fonction par fonction.

## Matrice et règles (`matrix_rules.py`)

Pour une molécule à n atomes lourds, M (n × n) a sur la diagonale le nombre d'électrons de valence de l'élément et
hors diagonale l'ordre de liaison (0 à 3). Les H sont implicites.

| Règle | Interdit |
|---|---|
| R1 | une ligne dont la somme dépasse la valence max (S : `SULFUR_MAX_VALENCE`) |
| R2 | deux carbones `s` (que des liaisons simples) liés, sauf si la liaison est dans un cycle et que chacun a un autre voisin sp² (C double ou triple, O, S, ou N conjugué) |
| R3 | un carbone avec deux doubles liaisons |
| R4a | deux petits cycles (≤ 8 atomes) qui partagent autre chose que rien, un atome ou une liaison (cycles pontés) |
| R4b | un grand cycle (> 8 atomes) qui ne peut pas se refermer à plat, ou qui contient un carbone `s` |

Une mutation est permise si son résultat ne crée **aucune nouvelle violation** : le nombre de violations de R1 ne
doit pas augmenter, ni celui de R2 + R3 + R4 (comptées ensemble, comme `ruleCount` du JS). Une molécule de départ
déjà hors règles reste donc modifiable.

`hybridization` (sp³ / sp² / sp, lue sur la matrice) est une information, pas une règle.

## Mutations

| Classe | Fichier | Rôle |
|---|---|---|
| `rule_filtered(Classe)` | `rule_filtered.py` | une mutation existante restreinte par les règles (même nom de classe : mêmes traces) |
| `AppendAtomRuleMG` | `append_atom_rule.py` | ajout d'un atome ; un C ajouté sur un carbone `s` l'est par une double liaison (≥ 2 H), sinon pas d'ajout |
| `SubstituteShiftMG` | `substitute_shift.py` | substitution ; si le nouvel élément accepte moins de liaisons (C= → O), les doubles liaisons sont déplacées le long d'un chemin alterné |
| `GraftFragmentMG` | `graft.py` | greffe d'un fragment du vocabulaire sur un atome |
| `AnnelateMG` | `annelate.py` | fusion d'un cycle sur une liaison (benzo, thiéno) |
| `FlipAlternatingPathMG` | `flip_path.py` | inversion 1 ↔ 2 du plus court chemin alterné 1, 2, …, 1 entre deux atomes (bascule quinoïde) |
| `AromatizeMG` | `aromatize.py` | structure de Kekulé d'un système de cycles, avec le moins de liaisons changées |

Nommer l'opération faite entre deux molécules (greffe, annélation, insertion, inversion, réécriture de Kekulé…) :
`matrix_diff.classify_mutation(avant, après)` ou `matrix_diff.classify_action(action)`, port de `mcs`, `diff` et
`classify` de `reference_core.js` (mêmes alignements que le JS, vérifiés avec node sur 9 paires). L'aromaticité vient de
RDKit, en ne comptant pas aromatique un cycle dont un atome porte une double liaison exocyclique (formes quinoïdes),
pour se rapprocher d'OpenChemLib. L'alignement est cherché dans les deux sens et le meilleur est gardé. Outil
d'analyse seulement : l'algorithme ne s'en sert pas.

Vocabulaire par défaut (`fragments.py`) : greffes vinyl, éthynyl, phényl, thién-2-yl, dicyanovinyl, diméthylamino,
méthoxy, cyano ; annélations benzo, thiéno. Un fragment dont un élément n'est pas dans les atomes acceptés n'est pas
utilisé (avec C, O, N seulement : ni thiényl ni thiéno).

## Utilisation

```python
from evomol import default_parameters as dp
from evomol import evaluation as evaluator
from evomol.newmut import (
    EvolutionaryAlgorithmNewMut,
    NewMutNeighborhoodStrategy,
    setup_newmut_action_space,
)
from evomol.representation import Molecule
from evomol.search.evolutionary_algorithm import ParametersEvolutionaryAlgorithm
from evomol.search.selection_strategy import BestSelection

dp.setup_default_parameters(accepted_atoms=["C", "O", "N"], max_heavy_atoms=38)
setup_newmut_action_space(
    matrix_rules=True,        # les mutations existantes respectent les règles
    append_atom_rule=True,    # AppendAtomRuleMG remplace AddAtomMG
    substitution_shift=True,  # SubstituteShiftMG remplace SubstituteAtomMG
    graft=True, annelate=True, flip_path=True, aromatize=True,
    # graft_fragments={...}, annelation_fragments={...}  # autre vocabulaire
)
algorithm = EvolutionaryAlgorithmNewMut(   # avertit si la population initiale est hors règles
    population=[Molecule("c1ccccc1")],
    parameters=ParametersEvolutionaryAlgorithm(size=100, max_generations=500),
    selection_strategy=BestSelection(),
    evaluations=[evaluator.QED],
    neighborhood_strategy=NewMutNeighborhoodStrategy(depth=2, lazy_rules=False),
)
algorithm.run()
```

`setup_newmut_action_space` accepte aussi tous les arguments de `dp.setup_default_action_space` (`insert_carbon`,
`bond_change_by_one`…), qu'il appelle en premier.

Pour essayer les mutations à la main et faire des marches aléatoires :
`notebooks/random_walk/newmut_walk.ipynb`.

## Coût des règles et option `lazy_rules`

Pour lister les mutations permises, chaque mutation candidate est appliquée sur une copie et ses règles sont
vérifiées (environ 0,25 ms par mutation sur une molécule de 35 atomes). Les listes sont gardées en mémoire par
molécule (`rule_filtered.CACHE_SIZE`, 10 000 molécules).

`NewMutNeighborhoodStrategy(lazy_rules=True)` (désactivé par défaut) essaie plutôt les mutations du type tiré une par
une, dans un ordre aléatoire, et garde la première permise : chaque mutation permise a la même probabilité, mais
on en vérifie en général une ou deux au lieu de toutes.

Mesure (une seule série, `scripts/newmut/bench_rules.py`, EA QED, 10 générations × 10 remplacements, population 20,
graine 0) :

| mode | ms par candidat | × par rapport à RevoMol sans règles |
|---|---|---|
| sans règles | 1,4 | 1 |
| règles, listes complètes, sans cache | 10,1 | 7,3 |
| règles, listes complètes, avec cache | 5,4 | 3,9 |
| règles, `lazy_rules=True` | 2,3 | 1,7 |

QED est très rapide à calculer : avec une évaluation plus lourde (Hückel, xTB), la part des règles diminue.

## Tests et scripts

```bash
/Applications/anaconda3/envs/revomol/bin/python -m pytest tests/newmut -q
/Applications/anaconda3/envs/revomol/bin/python scripts/newmut/bench_rules.py
/Applications/anaconda3/envs/revomol/bin/python scripts/newmut/run_all_options.py
```

## Limites connues

- R4 accepte les spiranes, le biphénylène, les hélicènes, le corannulène et le cyclooctatétraène (limites de la
  spécification, non corrigées).
- Quand plusieurs plus courts chemins ont la même longueur (cycles par liaison, chemins alternés), le choix dépend
  de l'ordre des atomes, qui n'est pas le même que dans le JS.
- Si S est accepté, les règles laissent S monter à la valence 4 ou 6 (comme les mutations existantes et le JS) :
  par exemple une greffe sur le S d'un thiophène. Réglable par `SULFUR_MAX_VALENCE`.
- `FlipAlternatingPathMG` : si le plus court chemin trouvé repasse par un atome, la paire est ignorée.
