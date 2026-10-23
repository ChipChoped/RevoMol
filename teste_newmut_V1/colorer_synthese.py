"""
Couleurs et clé de lecture du tableau de synthèse (teste/parametre/synthese_familles.xlsx), sans toucher aux
réglages déjà faits dans le fichier (largeurs, colonnes masquées, volet figé, filtres) : seules les couleurs des
cases et l'onglet « clé de lecture » sont ajoutés ou refaits.

À lancer avec le Python d'Anaconda de base (openpyxl, matplotlib) :
    /Applications/anaconda3/bin/python teste/colorer_synthese.py [fichier.xlsx]

- colonnes des règles (« refus Rx », « seule Rx ») : toute la colonne dans la couleur de la règle en pastel, plus
  foncée quand la case a un nombre (la règle refuse au moins une mutation) ;
- statistiques de certitude : rouge = conclusion sûre, bleu = pourrait être le hasard (trop peu de cas), dégradé
  entre les deux (échelle logarithmique) :
    « exception max 95 % », « oubli max 95 % » : la valeur elle-même ;
    « IC 95 % bas / haut », « couverture IC 95 % bas / haut » : la largeur de l'intervalle.
"""

import math
import os
import sys

from matplotlib import colormaps
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "parametre", "synthese_familles.xlsx")

# couleurs des règles : celles du notebook parametrage_regles (PALETTE, dans l'ordre de RULES)
RULE_COLOURS = {"R1": "e5484d", "R2a": "f08c00", "R2b": "8e4ec6", "R2c": "0090ff", "R3": "12a594", "R4": "e93d82",
                "R5": "a18072"}
SURE, CHANCE = 0.01, 50.0  # en % : ≤ 0,01 % → rouge (sûr), ≥ 50 % → bleu (le hasard)
BOUND_COLUMNS = ["exception max 95 % (%)", "oubli max 95 % (%)"]
INTERVALS = [("IC 95 % bas (%)", "IC 95 % haut (%)"),
             ("couverture IC 95 % bas (%)", "couverture IC 95 % haut (%)")]


def mix(hex_colour: str, part: float) -> str:
    """Couleur mélangée au blanc (part = proportion de la couleur gardée)."""
    rgb = [int(hex_colour[k:k + 2], 16) for k in (0, 2, 4)]
    return "".join(f"{round(255 - part * (255 - c)):02X}" for c in rgb)


def certainty_colour(value: float) -> str:
    """Rouge (valeur petite : sûr) → bleu (valeur grande : le hasard), échelle logarithmique, adoucie."""
    x = max(value, 1e-6)
    t = (math.log10(x) - math.log10(SURE)) / (math.log10(CHANCE) - math.log10(SURE))
    t = min(1.0, max(0.0, t))
    r, g, b, _ = colormaps["coolwarm"](1 - t)
    return mix("".join(f"{round(255 * c):02X}" for c in (r, g, b)), 0.6)


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


KEY = [
    ("Clé de lecture du tableau de synthèse", "titre"),
    ("", ""),
    ("Une ligne = une sous-famille de mutation vue par une source : « marches » (10 000 molécules des marches V1) ou "
     "« exhaustif » (toutes les molécules de 1 à 8 atomes). Les deux lignes d'une sous-famille se suivent.", ""),
    ("", ""),
    ("Lire une ligne dans cet ordre", "titre"),
    ("1. verdict : toujours autorisée (aucune mutation refusée), toujours interdite (toutes refusées) ou dépend "
     "(certaines oui, d'autres non selon la molécule).", ""),
    ("2. est-ce sûr ? « exception max 95 % » (pour un « toujours ») ou la largeur de l'intervalle « IC 95 % » (pour un "
     "« dépend ») : rouge = sûr, bleu = trop peu de cas pour conclure (pourrait être le hasard).", ""),
    ("3. qui refuse ? « refus Rx » = nombre de refus où la règle Rx est cassée (une mutation peut casser plusieurs "
     "règles) ; « seule Rx » = refus où Rx est la seule cassée : sans Rx, ces mutations passeraient.", ""),
    ("4. une règle suffit-elle seule ? « règle(s) qui suffisent seules » : règle cassée par 100 % des refus ; "
     "n'appliquer qu'elle donnerait le même résultat pour cette sous-famille. « oubli max 95 % » dit à quel point "
     "c'est sûr. « aucune » : il faut plusieurs règles ; voir « meilleure règle seule » et sa « couverture ».", ""),
    ("5. les deux sources sont-elles d'accord ? « statut » : accord (même verdict), désaccord, ou seulement exhaustif / "
     "seulement marches.", ""),
    ("", ""),
    ("Exemples", "titre"),
    ("addCs@Cs (ajouter un C s sur un C s) : toujours interdite dans les deux sources. Exception max 0,027 % (marches, "
     "11 065 cas) et 0,0008 % (exhaustif) : rouge, c'est sûr. Refus R2a = 100 % des refus → R2a suffit seule. R5 est "
     "aussi cassée dans 5 034 cas mais jamais seule : elle n'est pas nécessaire pour cette sous-famille.", ""),
    ("subOs→Nd+mont (O s remplacé par N, une liaison monte) : dépend, 65,7 % acceptées, IC 95 % de 64,8 à 66,6 % : "
     "intervalle étroit (rouge), la proportion est bien connue. Tous les refus cassent R3 → R3 suffit seule (oubli "
     "max 0,08 %).", ""),
    ("cutCs@Cs,Os (excision) : marches 80,95 % sur 21 cas, IC de 60,0 à 92,3 % : intervalle large (bleuté), peu sûr. "
     "Exhaustif : toujours autorisée (0 refus sur 2 027, exception max 0,15 %). Désaccord : les 4 refus des marches "
     "viennent de R4 (grands cycles absents des molécules de 8 atomes). « oubli max 52,7 % » (bleu) : avec 4 refus, "
     "on ne peut pas dire si R4 suffirait toujours.", ""),
    ("cutCt@Ct,Ct, marches : 2 candidates, toutes acceptées → « toujours autorisée », mais exception max 77,6 % : "
     "bleu, aucune conclusion possible avec 2 cas.", ""),
    ("", ""),
    ("Couleurs", "titre"),
    ("verdict et statut : vert = autorisée / accord, rouge = interdite / désaccord, jaune = dépend / une seule source.",
     ""),
    ("colonnes des règles : couleur de la règle (la même que dans le notebook) ; pâle = la règle ne refuse rien ici, "
     "plus foncé = elle refuse au moins une mutation.", ""),
    ("statistiques : rouge = sûr (≤ 0,01 %), bleu = le hasard (≥ 50 %), dégradé entre les deux.", "degrade"),
    ("", ""),
    ("Les formules sont dans l'onglet « méthode ».", ""),
]


def main(path: str) -> None:
    wb = load_workbook(path)
    ws = wb["synthèse"]
    header = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
    # colonnes des règles
    for rule, colour in RULE_COLOURS.items():
        for name in (f"refus {rule}", f"seule {rule}"):
            c = header.get(name)
            if c is None:
                continue
            ws.cell(1, c).fill = PatternFill("solid", fgColor=mix(colour, 0.35))
            for r in range(2, ws.max_row + 1):
                v = number(ws.cell(r, c).value)
                ws.cell(r, c).fill = PatternFill("solid", fgColor=mix(colour, 0.6 if v else 0.15))
    # statistiques de certitude
    for name in BOUND_COLUMNS:
        c = header.get(name)
        if c is None:
            continue
        for r in range(2, ws.max_row + 1):
            v = number(ws.cell(r, c).value)
            if v is not None and not math.isnan(v):
                ws.cell(r, c).fill = PatternFill("solid", fgColor=certainty_colour(v))
    for low_name, high_name in INTERVALS:
        lo, hi = header.get(low_name), header.get(high_name)
        if lo is None or hi is None:
            continue
        for r in range(2, ws.max_row + 1):
            a, b = number(ws.cell(r, lo).value), number(ws.cell(r, hi).value)
            if a is None or b is None or math.isnan(a) or math.isnan(b):
                continue
            fill = PatternFill("solid", fgColor=certainty_colour(b - a))
            ws.cell(r, lo).fill = fill
            ws.cell(r, hi).fill = fill
    # onglet « clé de lecture »
    if "clé de lecture" in wb.sheetnames:
        del wb["clé de lecture"]
    key = wb.create_sheet("clé de lecture", 1)
    key.column_dimensions["A"].width = 130
    for text, kind in KEY:
        key.append([text])
        cell = key.cell(key.max_row, 1)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if kind == "titre":
            cell.font = Font(bold=True, size=12 if "Clé" in text else 11)
    # dégradé de certitude sous la légende
    key.append([""])
    row = key.max_row + 1
    for k, value in enumerate([0.001, 0.01, 0.1, 1, 5, 20, 50]):
        cell = key.cell(row, 2 + k, f"{value} %")
        cell.fill = PatternFill("solid", fgColor=certainty_colour(value))
        key.column_dimensions[cell.column_letter].width = 9
    key.cell(row, 1, "exemple de dégradé (valeur ou largeur d'intervalle) :")
    wb.save(path)
    print("couleurs et clé de lecture ajoutées :", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
