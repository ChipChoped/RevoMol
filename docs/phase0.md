# Phase 0 — Fondations correctes (mesures solides avant d'optimiser)

> **HuckMol 2.0** : le contenu de `phase0/` a été intégré au code de base.
> - `config.py` → `evomol/default_parameters.py` : `setup_default_action_space(insert_carbon=…,
>   avoid_sp3_on_sp3=…, avoid_bond_breaking=…, avoid_bond_forming=…)`, `setup_huckel_filters(…)`
>   et `HUCKEL_ACCEPTED_ATOMS`. Les valeurs par défaut de la 2.0 sont celles d'EvoMol
>   (InsertCarbonMG présent, `avoid_sp3_on_sp3=False`) ; la config « pruned » ci-dessous
>   s'obtient avec `insert_carbon=False, avoid_sp3_on_sp3=True`.
> - `bfs.py` → `evomol/search/enumeration.py`.
> - `completeness_test.py`, `profile_run.py`, `profile_random_walk.py` → `scripts/`.
> - `prof/` → `output/prof/` ; notebooks → `notebooks/`.
>
> Les résultats ci-dessous datent d'avant le passage aux versions RevoMol-correlation de
> `RemoveAtomMG` et `ChangeBondMG`.


Objectif : **avoir des chiffres fiables avant de toucher au code d'evomol.**
Ce dossier ne contient QUE le code utile pour ça — pas le harnais de benchmark
tentaculaire du notebook.

Trois livrables :

1. **Corriger la config** — `config.py` est la source unique de vérité, corrigée
   pour coller à ce que les benchmarks du notebook avaient réellement trouvé.
2. **Test d'invariance de complétude** — `completeness_test.py` : le pruning
   fait-il perdre des molécules ? (prérequis qui légitime tout le reste)
3. **Profilage** — `profile_run.py` : où part vraiment le temps ?

Tout est **déterministe** (voisins triés + arrêt sur budget, pas sur le temps)
et se lance avec l'interpréteur du projet.

```bash
PY=/Applications/anaconda3/envs/revomol/bin/python   # Python + RDKit du projet

$PY scripts/Lea/completeness_test.py                 # invariance, 6 atomes, gate Huckel+Hetero
$PY scripts/Lea/profile_run.py                        # profilage 50k, config de production
```

---

## Fichiers

| Fichier | Rôle |
|---|---|
| `config.py` | Config canonique **corrigée** + constructeurs d'espace d'action `full` / `pruned`. Importé par les deux autres scripts. |
| `bfs.py` | Noyau BFS partagé, déterministe : `exhaustive_bfs` (épuisement) et `budget_bfs` (borné). Reproduit fidèlement le chemin coûteux actuel — on mesure l'existant. |
| `completeness_test.py` | Énumère à épuisement `full` vs `pruned`, compare les **ensembles** de valides. |
| `profile_run.py` | `cProfile` sur un run borné, synthèse hotspots + `.prof` réutilisable. |
| `prof/` | Profils binaires générés (`snakeviz prof/run_*.prof`). |

---

## 1. Ce qui était faux dans le notebook (et corrigé ici)

Le notebook `enumeration_huckel.ipynb` **annonçait** une config qui ne
correspondait pas à ses propres benchmarks (cellule 5 vs Tests 2/3/5) :

| Réglage | Notebook (cellule 5) | Ce que disent les tests | `config.py` |
|---|---|---|---|
| Ordre des filtres | Huckel → Hetero → ChEMBL | **Test 2** : Huckel → **ChEMBL** → Hetero (68.3 vs 41.5 v/s) | Huckel → ChEMBL → Hetero |
| `avoid_bond_breaking` | `True` | **Test 3** : « break ✓ » = `False` (98.7 v/s, le meilleur) | `False` |
| `avoid_sp3_on_sp3` | `True` | **Test 5** : `True` (100.4 vs 95.7 v/s) | `True` |
| `InsertCarbonMG` | retiré | **Test 3** : « InsertC ✗ » = retiré | retiré |

> ⚠️ Dans le Test 3, « break ✓ » = **bris de liaison autorisé**, donc
> `avoid_bond_breaking = False` (c'est aussi le défaut d'evomol). Le notebook le
> forçait à `True` à tort.

Conséquence importante : une fois corrigé, **`avoid_bond_breaking` n'est plus un
axe de pruning** (identique dans les deux configs). Les seules différences
`full` vs `pruned` sont donc : `InsertCarbonMG` et `avoid_sp3_on_sp3`.

---

## 2. Test d'invariance de complétude

**Question :** l'espace d'action pruné (rapide) atteint-il EXACTEMENT le même
ensemble de molécules valides que l'espace complet ? Sans cette garantie, « plus
rapide » pourrait juste vouloir dire « trouve moins de molécules ».

- `FULL`   = toutes les actions (InsertCarbon inclus), `avoid_sp3_on_sp3=False`
- `PRUNED` = InsertCarbon retiré, `avoid_sp3_on_sp3=True`
- Gate de validité : **Huckel + Hetero** (le bon référentiel : le pruning est
  justifié PAR Hückel). `--with-chembl-zinc` disponible pour info (plus lent,
  fragmente la connexité).

On énumère **à épuisement** puis on compare les ensembles (pas juste le compte).
Code retour `0` si identiques, `1` sinon.

```bash
$PY scripts/Lea/completeness_test.py                        # défaut : max_atoms=4 (~15s)
$PY scripts/Lea/completeness_test.py --max-atoms 5          # plusieurs minutes
$PY scripts/Lea/completeness_test.py --with-chembl-zinc     # gate production (lent)
```

> Défaut à **4** volontairement : avec le code actuel (non optimisé), `5` prend
> plusieurs minutes et `6-7` est impraticable. Débloquer 6-7 est justement l'un
> des objectifs de la Phase 1 — le rendre rapide, PUIS ré-exécuter ce test plus
> haut.

### Résultat

| max_atoms | valides FULL | valides PRUNED | perdues | verdict |
|---|---|---|---|---|
| 4 | 2 791 | 2 791 | 0 | ✅ ensembles identiques |
| 5 | 43 756 | 43 756 | 0 | ✅ ensembles identiques (86k vus, ~210 s/config) |

FULL et PRUNED atteignent **exactement le même ensemble** de molécules valides
(gate Huckel+Hetero), à `max_atoms=4` **comme à 5** (43 756 molécules) → le
pruning est légitime : il explore moins mais ne perd aucune molécule valide.

> Portée : « exhaustif » = épuisement de la composante connexe atteignable
> depuis le départ *en ne traversant que des molécules valides* (comme le run de
> production). La complétude absolue vis-à-vis d'une énumération chimique de
> référence (surge/MOLGEN) est une autre question, réservée à la Phase 3.

---

## 3. Profilage

`cProfile` sur un run borné par budget, avec la config de **production**
(pruned + filtres Huckel → ChEMBL/ZINC → Hetero).

```bash
$PY scripts/Lea/profile_run.py                      # 50k molécules, filtres complets
$PY scripts/Lea/profile_run.py --budget 20000       # plus court
$PY scripts/Lea/profile_run.py --no-chembl-zinc     # sans les filtres de réalisme
```

### Résultat (budget 50 000, filtres complets) — le point clé

Run : 2 651 valides / 47 386 invalides / 50 037 examinées en 48.7 s (avec
surcoût cProfile). Synthèse hotspots (temps propre) :

| Poste | tottime | % du temps |
|---|---|---|
| Itération atomes/liaisons RDKit (wrappers Python) | 12.67 s | **26.0 %** |
| `MolecularGraph.__init__/__copy__/canonical` (parse + écriture SMILES) | 12.50 s | **25.7 %** |
| `update_representation` (recalcul du graphe à chaque construction) | 6.63 s | **13.6 %** |
| **Filtres réunis** (Huckel 2.7 + Hetero 1.6 + ECFP 2.9 + GCF 2.6) | ~4.8 s | **~9.8 %** |
| Kekulize | 0.50 s | 1.0 % |
| Expansion voisins (enumeration/action) | 0.58 s | 1.2 % |
| autre / overhead Python | 9.77 s | 20.1 % |

**Le diagnostic est confirmé, chiffres à l'appui :** les filtres chimiques ne
coûtent que **~10 %**. Le temps part dans la **(re)construction d'objets RDKit**
— **1 259 222 `MolecularGraph` construits pour 50 037 molécules examinées, soit
~25×** — et l'itération Python sur les collections RDKit (40,8 M d'appels aux
wrappers). C'est exactement la cible de la Phase 1 : un seul objet RDKit par
molécule, dérivés (kekulisé, types d'atomes, fingerprint) mis en cache.

(Les mêmes ratios apparaissent dès un budget de 5 000 : mesure stable.)

> cProfile impute le temps C de `MolFromSmiles`/`MolToSmiles` (fonctions Boost,
> pas des builtins CPython) à leur appelant Python : il apparaît donc dans le
> poste `MolecularGraph …`, pas sur une ligne dédiée.

Profils binaires dans `prof/` : `snakeviz output/prof/run_full_50000.prof`.

---

## Environnement

- Interpréteur : `/Applications/anaconda3/bin/python3.12` (Python 3.12 + RDKit).
  Le `python3` système (3.9) **ne convient pas** (`list[int] | list[str]` en
  corps de classe).
- Les scripts ajoutent la racine du projet à `sys.path`, `chdir` vers la racine
  (les filtres ChEMBL/ZINC lisent `external_data/` en relatif) et coupent les
  warnings RDKit. Ils peuvent donc être lancés depuis n'importe quel répertoire.
