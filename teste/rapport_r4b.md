# La règle R4b classe-t-elle bien les grands cycles plans ? Test par la géométrie 3D

13 octobre 2026. Code : `teste/r4b_planeite.py` ; analyse et figures : `teste/r4b_planeite.ipynb` ; données :
`teste/resultats/`.

## En bref

- Sur les molécules de référence, R4b donne en général le bon verdict. La porphine, la phtalocyanine, le
  tridéshydro[12]annulène et les [14], [18] et [22]annulènes sont acceptés et plans ou presque. Les
  cycloparaphénylènes sont refusés et ne sont pas plans.
- Sur les grands cycles que produisent réellement les mutations, R4b ne sépare pas les cycles plans des autres.
  Presque tous ces cycles sont non plans en 3D, qu'ils soient acceptés ou refusés : aucun des 48 cycles acceptés
  n'est plan à l'énergie minimale. L'écart de la somme des virages à 360° ne prédit pas la planéité.
- On ne peut donc pas régler la tolérance de 40° avec ces données : aucune tolérance, de 0 à 360°, ne classe nettement
  mieux que le hasard. Le seuil de 14 atomes n'y change rien non plus.
- La cause : les grands cycles produits par les mutations sont surtout des polycycles pontés, créés quand
  ChangeBondMG ajoute une liaison à travers un système déjà fusionné. Le calcul des virages de R4b ne voit pas cette
  tension.

## Méthode

**Règle testée.** R4b (`evomol/newmut/matrix_rules.py`, `r4b_ring`) juge chaque grand cycle (plus de 8 atomes, plus
court cycle par liaison). Elle le refuse s'il contient un carbone s. Sinon, elle fait la somme des virages autour du
cycle et l'accepte si cette somme vaut 360° à ±40° près. Un H peut être tourné vers l'intérieur (−120°) si le cycle a
au moins 14 atomes. Le script recalcule le verdict cycle par cycle et vérifie qu'il est identique à celui de `r4b_ring`
sur les 383 cycles.

**Lot de molécules.**
- **19 molécules de référence :** annulènes [10] à [24], porphine, phtalocyanine, [5] à [8]cycloparaphénylènes (CPP),
  tridéshydro[12]annulène, cyclo[18]carbone, [2.2]paracyclophane, 18-couronne-6, cyclododécane.
- **347 résultats de mutations :**
  - on part des 120 molécules de `newmut_time`, avec les nouvelles mutations et les règles R1-R5 ;
  - on liste les 43 364 mutations candidates des molécules de départ d'au plus 61 atomes ;
  - 1 897 résultats différents ont un grand cycle, aucune autre violation que R4b, et au plus 60 atomes ;
  - parmi eux : 47 acceptés, 229 refusés « angle » et 1 621 refusés « carbone s » ;
  - on a gardé les 47 acceptés et tiré au hasard 150 refusés de chacune des deux autres classes ;
  - cela donne 364 grands cycles : 48 acceptés, 159 refusés « angle », 157 refusés « carbone s ».

**Géométries 3D.**
- Les isomères E/Z et R/S sont énumérés : au plus 64 par molécule, et 1 024 pour les références.
- Chaque isomère reçoit 4 conformères RDKit ETKDGv3, avec au moins 32 structures par molécule. Les structures sont
  ensuite optimisées avec le champ de force MMFF94.
- Il y a 64 structures par molécule en médiane.
- 7 molécules n'ont aucune structure 3D : le cyclo[18]carbone et 6 résultats de mutations.
- Le calcul a duré 92 minutes sur 8 cœurs.

**Mesures**, pour chaque grand cycle :
- `rms_plan` : écart quadratique moyen des atomes du cycle à leur plan moyen, en Å ;
- `angle_p` : angle moyen entre l'orbitale p de chaque atome du cycle et la normale de ce plan, en degrés (0° signifie
  des orbitales p parallèles, comme le suppose Hückel) ;
- `torsion` : écart moyen des dièdres du cycle à 0° ou 180°.

Chaque mesure est relevée à deux endroits :
- **à l'énergie minimale**, sur la structure la plus stable ;
- **au mieux**, sur la structure la plus plane, quelle que soit son énergie. C'est la question que pose la règle :
  « le cycle peut-il être plan ? »

## Résultats

### Références

| molécule | verdict R4b | `rms_plan` à E min (Å) | `angle_p` à E min (°) |
|---|---|---|---|
| porphine, phtalocyanine, tridéshydro[12]annulène | acceptée | 0,00 | 0 |
| [14], [18], [22]annulène | acceptée | 0,09 à 0,12 | 9 à 16 |
| [16], [20], [24]annulène | acceptée | 0,45 à 0,72 | 39 à 48 |
| [10]annulène | refusée angle | 0,09 | 9 |
| [12]annulène | refusée angle | 0,38 | 55 |
| [5] à [8]CPP | refusée angle | 0,60 à 0,75 | 68 à 78 |
| paracyclophane, 18-couronne-6, cyclododécane | refusée carbone s | 0,47 à 0,59 | 58 à 88 |
| cyclo[18]carbone | refusée angle | pas de structure 3D | — |

Il y a deux désaccords :
- **Les [16], [20] et [24]annulènes** (4n électrons π) sont acceptés mais non plans à l'énergie minimale. Leur
  meilleure structure reste à 0,22–0,29 Å du plan.
- **Le [10]annulène** est refusé alors que MMFF le donne aussi plat que le [18]annulène. MMFF n'est sans doute pas
  fiable pour ce cas.

### Cycles produits par les mutations

Médianes, et nombre de cycles plans à l'énergie minimale (`rms_plan` ≤ 0,25 Å) :

| verdict R4b | cycles | `rms_plan` à E min | `rms_plan` au mieux | `angle_p` à E min | plans (≤ 0,25 Å à E min) |
|---|---|---|---|---|---|
| acceptée | 48 | 0,47 Å | 0,31 Å | 57° | 0 sur 48 |
| refusée angle | 159 | 0,43 Å | 0,34 Å | 57° | 3 sur 154 |
| refusée carbone s | 157 | 0,45 Å | 0,31 Å | 55° | 0 sur 156 |

Avec le critère sur les orbitales p (`angle_p` ≤ 20° à l'énergie minimale), on compte 0 cycle plan sur 48 acceptés,
2 sur 154 refusés « angle » et 1 sur 155 refusés « carbone s ».

**Probabilité qu'un cycle accepté soit plus plan qu'un cycle refusé « angle »** (0,5 = hasard ; intervalle à 95 % par
rééchantillonnage) :
- `rms_plan` à E min : 0,45 [0,34 ; 0,55] ;
- `rms_plan` au mieux : 0,60 [0,50 ; 0,68] ;
- `angle_p` à E min : 0,56 [0,46 ; 0,65] ;
- `torsion` à E min : 0,63 [0,55 ; 0,71].

Les cycles acceptés sont donc au mieux à peine plus plans que les refusés. La corrélation de rang entre l'écart à 360°
et les mesures de planéité va de −0,08 à 0,24.

### Tolérance de 40°

On a fait varier la tolérance de 0 à 360°. Pour chaque critère « plan » qui a au moins 5 cycles plans et 5 non plans :
- l'aire sous la courbe ROC de l'écart à 360° va de 0,41 à 0,59 ;
- la précision équilibrée à 40° va de 0,44 à 0,56 ;
- la meilleure précision équilibrée ne dépasse pas 0,61, et elle est atteinte à des tolérances dispersées entre 0 et 315°, qui changent d’un
  critère à l’autre.

Aucune tolérance ne marche, donc 40° n'est ni justifié ni à remplacer par une autre valeur à partir de ces données.

### Seuil de 14 atomes

11 cycles de moins de 14 atomes sont refusés seulement à cause de ce seuil. Ils ne sont pas moins plans que les cycles
acceptés : 0,45 Å contre 0,47 Å en médiane à l'énergie minimale.

## Limites

- **Champ de force :** MMFF94 n'est pas un calcul quantique. Il ne traite pas l'aromaticité des annulènes, comme le
  montre le cas du [10]annulène. Une réoptimisation xTB d'un sous-lot servirait à vérifier.
- **Échantillonnage :** il est limité à 64 isomères × 4 conformères. ETKDG ne respecte pas toujours la configuration
  E/Z demandée dans un grand cycle, et le script note la configuration obtenue. La valeur « au mieux » peut donc
  surestimer l'écart au plan atteignable.
- **Molécules du lot :** ce sont les mutations de 120 molécules de départ seulement. Il n'y a que 47 molécules
  acceptées avec un grand cycle, et très peu de cycles plans : l'effet de la tolérance ne peut pas être mesuré sur
  cette population.
- **Écarts au plan prévu :**
  - le plongement 3D part de coordonnées aléatoires, sans imposer R/S, avec 20 essais au plus par conformère. Sinon,
    certains isomères bloquaient ETKDG plus de 9 minutes ;
  - l'énumération des isomères est refaite avec les H explicites quand RDKit 2024.03 plante, ce qui arrive sur les
    imines C=NH (46 molécules).
- **Reproductibilité :** le nombre de mutations candidates varie un peu d'un lancement à l'autre (42 941 contre 43 364).
  La cause n'a pas été cherchée. Le lot effectivement utilisé est enregistré dans `resultats/lot.csv`.

## Ce que ça suggère (à décider)

R4b fait ce pour quoi elle a été écrite : trier des macrocycles « de manuel » comme les annulènes, les porphyrines ou
les CPP. Mais ce n'est pas ce que les mutations produisent. Pour empêcher les polycycles pontés non plans, régler la
tolérance de 40° ne servira à rien. Il faudrait une autre règle. Deux pistes :
- interdire à un grand cycle de traverser un système de petits cycles fusionnés, comme R4a le fait pour les petits
  cycles ;
- faire une vérification 3D rapide des candidats qui ferment un grand cycle.
