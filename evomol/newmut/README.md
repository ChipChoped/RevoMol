# newmut : règles matricielles et nouvelles mutations

Ce dossier ajoute à RevoMol :
1. des **règles de validité lues sur la matrice** des ordres de liaison (R1 à R5), appliquées comme la valence,
   c'est-à-dire au moment de lister les mutations possibles, et jamais comme filtre sur les molécules produites ;
2. de **nouvelles mutations**, élémentaires (AddAtom, SubstituteShift) et composées (greffe, annélation,
   inversion d'un chemin alterné, aromatisation).

Rien n'est modifié en dehors des dossiers `newmut` : avec toutes les nouvelles options à `False` et
`evomol_add_atom=True`, RevoMol se comporte exactement comme avant (vérifié par `tests/newmut/test_non_regression.py`).
Sans `evomol_add_atom=True`, l'`AddAtomMG` de newmut remplace celui d'evomol (voir plus bas).

La spécification d'origine était écrite pour l'EvoMol de jules-leguy (masques de taille fixe). Dans RevoMol, chaque
type de mutation liste lui-même ses actions possibles (`list_actions`) : le « masque » de la spécification est donc
cette liste. La référence exécutable est l'atelier de mutations (`reference_core.js`, `reference_app.js`), traduit
fonction par fonction.

## Matrice et règles (`matrix_rules.py`)

Pour une molécule à n atomes lourds, M (n × n) a sur la diagonale le nombre d'électrons de valence de l'élément et
hors diagonale l'ordre de liaison (0 à 3). Les H sont implicites.

Deux versions des règles (`rules_version` de `setup_newmut_action_space`, `matrix_rules.RULES_VERSION`). **V1** est la
version par défaut depuis le 2026-10-16 (réglée dans `teste/parametrage_regles.ipynb`) ; **V0** (Regle_V0) est
l'ancienne, gardée pour reproduire les expériences d'avant.

V1 : C, N et O ont une étiquette `t` (une triple), `d` (une double) ou `s` (que des simples).

| Règle | Interdit (V1) |
|---|---|
| R1 | une ligne dont la somme dépasse la valence max (S : `SULFUR_MAX_VALENCE`) |
| R2a | deux carbones `s` liés |
| R2b | un C `s` lié à un N ou un O `s` qui n'a pas d'autre voisin qu'un C `s` (C`s`–OH, C`s`–NH₂, diméthylamine ; l'anisole et la N,N-diméthylaniline sont permis) |
| R2c | N–N, N–O, O–O entre deux atomes `s` dont aucun n'est relié à un atome `d` ou `t` |
| | exception de R2a, R2b, R2c : liaison dans un cycle, et chaque atome a un autre voisin `d` ou `t`, ou un O ou N `s` relié à un atome `d` ou `t` (EDOT permis, 1,4-dioxane refusé ; N ajouté le 2026-10-19) |
| R3 | un carbone avec deux doubles liaisons |
| R4 | deux petits cycles (≤ 8 atomes) qui partagent autre chose que rien, un atome ou une liaison (cycles pontés) |
| R4b | un grand cycle (> 8 atomes) qui ne peut pas se refermer à plat, ou qui contient un carbone `s`. **Désactivée par défaut** : option `r4b=True` de `setup_newmut_action_space` (`matrix_rules.R4B`) |
| R5 | un carbone `s` qui porte 3 atomes lourds ou plus (au moins 2 H : ni CH ramifié, ni carbone quaternaire) |

V0 (Regle_V0) : R1, R3, R4, R4b comme ci-dessus, et
- R2 : deux carbones `s` liés, sauf si la liaison est dans un cycle et que chacun a un autre voisin sp² (C double ou
  triple, O, S, ou N conjugué) ;
- R5 : un carbone `s` qui porte 4 atomes lourds (carbone quaternaire). Règle ajoutée, absente de l'atelier JS.

R4b est désactivée par défaut depuis le 2026-10-15 (`r4b=False`) : elle ne faisait pas mieux que le hasard sur les
grands cycles produits par les mutations (`teste/rapport_r4b.md`). Elle n'est plus comptée, ni dans la vérification
d'origine ni dans la vérification rapide, et le raccourci A (qui la suppose) n'est plus appliqué. `r4b=True` la
rétablit.

Options de R2 de V0 (absentes de l'atelier JS, désactivées par défaut, sans effet en V1) : `r2_hetero` étend R2 aux liaisons N–N, N–O et
O–O entre deux atomes sans liaison multiple, avec la même exception dans les cycles que pour les carbones ;
`r2_hetero_outside_rings` (avec `r2_hetero`) ne refuse ces liaisons qu'hors cycle.

Une mutation est permise si son résultat ne crée **aucune nouvelle violation** : le nombre de violations de R1 ne
doit pas augmenter, ni celui des autres règles (comptées ensemble, comme `ruleCount` du JS). Une molécule de départ
déjà hors règles reste donc modifiable.

`hybridization` (sp³ / sp² / sp, lue sur la matrice) est une information, pas une règle.

## Mutations

| Classe | Fichier | Rôle |
|---|---|---|
| `rule_filtered(Classe)` | `rule_filtered.py` | une mutation existante restreinte par les règles (même nom de classe : mêmes traces) |
| `SubstituteShiftMG` | `substitute_shift.py` | substitution ; si le nouvel élément accepte moins de liaisons (C= → O), les doubles liaisons sont déplacées le long d'un chemin alterné. Option `substitution_no_shift` : seulement les substitutions où le nouvel élément accepte les liaisons telles quelles (aucune double abaissée ni déplacée), comme SubstituteAtomMG ; si le nouvel élément accepte plus de liaisons (O s → N, O s → C, N s → C, O= → N…), une liaison de l'atome peut aussi monter vers un voisin qui a assez de H (C–O–C → C–N=C, C=C–C, C≡C…), une mutation par liaison et par montée : option `substitution_raise` (activée par défaut depuis le 2026-10-19) |
| `GraftFragmentMG` | `graft.py` | greffe d'un fragment du vocabulaire sur un atome |
| `AnnelateMG` | `annelate.py` | fusion d'un cycle sur une liaison (benzo, thiéno) |
| `FlipAlternatingPathMG` | `flip_path.py` | inversion 1 ↔ 2 du plus court chemin alterné 1, 2, …, 1 entre deux atomes (bascule quinoïde) |
| `AromatizeMG` | `aromatize.py` | structure de Kekulé d'un système de cycles, avec le moins de liaisons changées |
| `InsertCarbonDoubleMG` | `insert_carbon_double.py` | option `insert_carbon_keep_double` : un carbone inséré dans une C=C garde une double d'un côté (C=C–C et C–C=C) au lieu de deux simples |

`AddAtomMG` (`add_atom.py`) : l'ajout d'atome de newmut, ex-`AppendAtomRuleMG` (renommée le 2026-10-19), à la place
de l'`AddAtomMG` d'evomol par défaut (même nom de mutation) ; `evomol_add_atom=True` garde celui d'evomol. Sur chaque
atome non chargé qui a un H libre, chaque élément accepté par une simple, une double ou une triple, tant que l'atome a
assez de H et que l'élément accepte cet ordre (=CH2, ≡CH, =O, =NH, ≡N sur un C s comme sur un N s), sauf un C par une
simple sur un carbone `s` (règle sp3-sp3 de l'atelier JS) ; puis les règles de la matrice décident (toujours
appliquées, sans `matrix_rules`).

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
    # AddAtomMG de newmut à la place de celui d'evomol (evomol_add_atom=True : celui d'evomol)
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

## Coût des règles, vérification rapide et option `lazy_rules`

Pour lister les mutations permises, le résultat de chaque mutation candidate est vérifié. La vérification rapide
(`fast_rules.py`, active par défaut) donne exactement la même réponse que la vérification d'origine (copie RDKit du
résultat, toutes les règles recalculées sur toute la molécule), mais :
1. elle calcule le résultat directement sur la matrice, sans RDKit (sauf `AddGroupMG` et `RemoveGroupMG`) ;
2. elle ne recalcule que les règles autour du changement : R1, R3, R5 sur les lignes modifiées, R2 sur les liaisons
   à moins de deux atomes d'un changement (et celles qui entrent ou sortent d'un cycle), R4 seulement si les liaisons
   de cycle changent (sinon, seuls les grands cycles touchés sont revérifiés).

L'équivalence est testée sur toutes les mutations candidates de nombreuses molécules
(`tests/newmut/test_fast_rules.py` ; 56 852 mutations sur 195 molécules sans aucun désaccord, le 2026-10-13).
`fast_rules.FAST_RULES = False` revient à la vérification d'origine.

Raccourcis exacts (`fast_rules.PREFILTERS = True`, par défaut) : sur une molécule qui respecte les règles, les
mutations qui créent forcément une violation sont refusées sans être vérifiées. A : liaison simple créée avec un carbone
`s` qui ferme un cycle de plus de 8 atomes (R4b, seulement si R4b est activée) ; C : 4ᵉ atome lourd sur un carbone `s` qui en a déjà 3 (R5) ; D :
inversion de chemin dont une extrémité est un carbone qui a déjà une double liaison (R3) ; E : C=C hors cycle passée en
simple entre deux carbones sans autre double (R2). Même résultat (testé), vérification −16 %, voisinage complet −7 %
(120 molécules de 5 à 98 atomes, 2026-10-14). Les listes filtrées sont aussi gardées en mémoire
par molécule (`rule_filtered.CACHE_SIZE`, 10 000 molécules).

`NewMutNeighborhoodStrategy(lazy_rules=True)` (désactivé par défaut) essaie les mutations du type tiré une par une,
dans un ordre aléatoire, et garde la première permise : chaque mutation permise a la même probabilité, mais on en
vérifie en général une ou deux au lieu de toutes.

Mesures du 2026-10-13 (une seule série chacune, vérification rapide) :

EA QED (`scripts/newmut/bench_rules.py`, 10 générations × 10 remplacements, population 20, graine 0) :

| mode | ms par candidat | × par rapport à RevoMol sans règles |
|---|---|---|
| sans règles | 1,75 | 1 |
| règles, listes complètes, sans cache | 6,6 | 3,8 |
| règles, listes complètes, avec cache | 4,6 | 2,6 |
| règles, `lazy_rules=True` | 1,9 | 1,1 |

Marches de 200 pas et tirages (`scripts/newmut/compare_walk_time.py`, départ benzène, C, O, N, 4 marches par
condition) : anciennes mutations 0,12 ms par voisin, nouvelles mutations avec règles 0,51 ms par voisin (2,25 avant la
vérification rapide). Les marches « nouvelles » font des molécules plus grosses (34 atomes en moyenne contre 10) et
ont donc plus de voisins : 156 ms par pas contre 15. Dans l'algorithme évolutionnaire : 0,3 ms par mutation (anciennes),
11 ms (nouvelles, listes complètes), 2 ms (nouvelles, `lazy_rules`).

QED est très rapide à calculer : avec une évaluation plus lourde (Hückel, xTB), la part des règles diminue.

## Tests et scripts

```bash
/Applications/anaconda3/envs/revomol/bin/python -m pytest tests/newmut -q
/Applications/anaconda3/envs/revomol/bin/python scripts/newmut/bench_rules.py
/Applications/anaconda3/envs/revomol/bin/python scripts/newmut/run_all_options.py
/Applications/anaconda3/envs/revomol/bin/python scripts/newmut/compare_walk_time.py   # anciennes vs nouvelles mutations
```

## Limites connues

- R4 accepte les spiranes, le biphénylène, les hélicènes, le corannulène et le cyclooctatétraène (limites de la
  spécification, non corrigées).
- Quand plusieurs plus courts chemins ont la même longueur (cycles par liaison, chemins alternés), le choix dépend
  de l'ordre des atomes, qui n'est pas le même que dans le JS.
- Si S est accepté, les règles laissent S monter à la valence 4 ou 6 (comme les mutations existantes et le JS) :
  par exemple une greffe sur le S d'un thiophène. Réglable par `SULFUR_MAX_VALENCE`.
- `FlipAlternatingPathMG` : si le plus court chemin trouvé repasse par un atome, la paire est ignorée.
