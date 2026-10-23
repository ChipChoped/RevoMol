"""
Synthèse des familles de mutation V2 (teste_newmut_V2/Mutation_V2/parametre/{Famille}.csv, écrits par
mutations_V2.ipynb) dans un seul tableau, comme celle de V1 (teste_newmut_V1/synthese_familles.py, dont les calculs
sont repris tels quels) : une ligne par sous-famille, intervalles de confiance, règles qui suffisent seules comme
filtre, lien vers l'image de la sous-famille. Une seule source (les marches V2), pas d'énumération exhaustive.

Écrit dans teste_newmut_V2/Mutation_V2/parametre/ : synthese_familles_V2.csv, synthese_familles_V2_methode.md et
synthese_familles_V2.xlsx (couleurs et onglet « clé de lecture » de V1, et les réglages d'affichage du .xlsx de V1 :
largeurs, colonnes masquées, en-têtes inclinés, hauteur des lignes, volet figé, filtre, recopiés colonne par colonne).
Si le .xlsx existe déjà, synthese_familles_V2_nouveau.xlsx, pour ne pas écraser les réglages faits à la main.
    /Applications/anaconda3/envs/revomol/bin/python teste_newmut_V2/Mutation_V2/synthese_familles_V2.py [dossier]
dossier : celui des CSV et des images, relatif à teste_newmut_V2/Mutation_V2 (défaut : parametre) ; le nombre de
molécules testées est lu dans son test_info.json (10 000 s'il n'y en a pas : premier test, 2026-10-22).

Le .xlsx est écrit par le Python d'Anaconda de base (openpyxl, matplotlib), l'environnement revomol n'ayant pas de
module Excel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V1 = os.path.join(ROOT, "teste_newmut_V1")
sys.path.insert(0, V1)

# pylint: disable=wrong-import-position
import pandas as pd

import synthese_familles as sf  # teste_newmut_V1/synthese_familles.py

DIR = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "parametre")
WALKS = os.path.join(ROOT, "teste_newmut_V2", "Walk_V2", "resultats", "test_Regle_V2_Bd")
V1_XLSX = os.path.join(V1, "parametre", "synthese_familles.xlsx")
RULES = ["R1", "R2", "R3a", "R3b", "R4", "RB"]
# couleurs des règles : PALETTE de parametrage_regles.py dans l'ordre des règles (comme dans mutations_V2.ipynb)
PALETTE = ["e5484d", "f08c00", "8e4ec6", "0090ff", "12a594", "e93d82", "a18072"]
RULE_COLOURS = dict(zip(RULES, PALETTE))
# règles V1 dont la colonne prête ses réglages d'affichage (même rang dans le groupe des règles)
V1_RULES = ["R1", "R2a", "R2b", "R2c", "R3", "R4", "R5"]


def walks_description() -> str:
    with open(os.path.join(WALKS, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    bore = config.get("BORE_MAX_ORDRE")
    bore_text = {1: "B- seulement", 2: "B- et B=, pas de B#"}.get(bore, "pas de limite sur le bore")
    info = {"molécules": 10000, "graine": 0}
    if os.path.exists(os.path.join(DIR, "test_info.json")):
        with open(os.path.join(DIR, "test_info.json"), encoding="utf-8") as f:
            info.update(json.load(f))
    n = f"{info['molécules']:,}".replace(",", " ")
    return (f"marches V2 (test_Regle_V2_Bd : {config.get('N_STEPS')} pas, atomes {' '.join(config['ACCEPTED_ATOMS'])}, "
            f"{bore_text}), {n} molécules conformes tirées (graine {info['graine']})")


def has_exhaustive() -> bool:
    """Énumération exhaustive présente (DIR/exhaustif/, écrite par exhaustif_V2.py) ?"""
    return os.path.isdir(os.path.join(DIR, "exhaustif"))


def exhaustive_description() -> str:
    info: dict = {}
    path = os.path.join(DIR, "exhaustif", "test_info.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    n = f"{info['molécules']:,}".replace(",", " ") if "molécules" in info else "?"
    return (f"énumération V2 de 1 à 8 atomes (C N O B F, sans S ni P), {n} molécules conformes aux règles V2 + RB "
            "(B# écartés)")


def build() -> tuple[pd.DataFrame, dict]:
    """Le tableau de V1 (synthese_familles.build), avec les règles V2 ; marches V2 et, si elle est là,
    énumération exhaustive V2."""
    sf.DIR = DIR
    sf.RULES = RULES
    sf.JEUX = {"marches": walks_description(), "exhaustif": exhaustive_description() if has_exhaustive() else ""}
    return sf.build()


def example_lines(table: pd.DataFrame) -> list[str]:
    """Exemples de la clé de lecture, pris dans le tableau (aucun chiffre écrit à la main)."""
    def pct(x: float) -> str:
        return f"{x:.4g}".replace(".", ",")

    out = []
    never = table[(table.verdict == sf.NEVER) & (table["règle(s) qui suffisent seules"] != "aucune")]
    if not never.empty:
        r = never.sort_values("candidates", ascending=False).iloc[0]
        out.append(f"{r['sous-famille']} : toujours interdite, {r['candidates']} candidates ; exception max "
                   f"{pct(r['exception max 95 % (%)'])} % : rouge, c'est sûr. {r['règle(s) qui suffisent seules']} "
                   "est cassée par 100 % des refus : elle suffit seule pour cette sous-famille.")
    depends = table[table.verdict == sf.DEPENDS].copy()
    if not depends.empty:
        depends["largeur"] = depends["IC 95 % haut (%)"] - depends["IC 95 % bas (%)"]
        r = depends.sort_values("candidates", ascending=False).iloc[0]
        out.append(f"{r['sous-famille']} : dépend, {pct(r['% acceptées'])} % acceptées sur {r['candidates']} "
                   f"candidates, IC 95 % de {pct(r['IC 95 % bas (%)'])} à {pct(r['IC 95 % haut (%)'])} % : "
                   f"proportion bien connue. Règle(s) qui suffisent seules : {r['règle(s) qui suffisent seules']} ; "
                   f"meilleure règle seule {r['meilleure règle seule']} (couverture {pct(r['couverture (%)'])} %).")
    few = table[(table.verdict != sf.DEPENDS) & (table.candidates <= 3)]
    if not few.empty:
        r = few.iloc[0]
        out.append(f"{r['sous-famille']} : {r['candidates']} candidate(s) → « {r['verdict']} », mais exception max "
                   f"{pct(r['exception max 95 % (%)'])} % : bleu, aucune conclusion possible avec si peu de cas.")
    return out


def key_lines(table: pd.DataFrame) -> list[tuple[str, str]]:
    """Clé de lecture (onglet du .xlsx) : celle de V1, pour une seule source et les règles V2."""
    return [
        ("Clé de lecture du tableau de synthèse V2", "titre"),
        ("", ""),
        (("Une ligne = une sous-famille de mutation vue par une source : « marches » (" + walks_description()
          + ") ou « exhaustif » (" + exhaustive_description() + "). Les deux lignes d'une sous-famille se suivent."
          if has_exhaustive() else
          "Une ligne = une sous-famille de mutation vue dans les marches V2 : " + walks_description() + ". Pas "
          "d'énumération exhaustive : « statut » vaut toujours « seulement marches ».")
         + " Règles : V2 (R1 valence, R2 tout atome s a un voisin d ou t, R3a un C porte au plus une double, R3b un C "
         "s porte au moins 2 H, R4 ponts) et RB (ordre de liaison max sur un bore). Les mutations qui donneraient un "
         "B# ne sont pas proposées.", ""),
        ("", ""),
        ("Lire une ligne dans cet ordre", "titre"),
        ("1. verdict : toujours autorisée (aucune mutation refusée), toujours interdite (toutes refusées) ou dépend "
         "(certaines oui, d'autres non selon la molécule).", ""),
        ("2. est-ce sûr ? « exception max 95 % » (pour un « toujours ») ou la largeur de l'intervalle « IC 95 % » "
         "(pour un « dépend ») : rouge = sûr, bleu = trop peu de cas pour conclure (pourrait être le hasard).", ""),
        ("3. qui refuse ? « refus Rx » = nombre de refus où la règle Rx est cassée (une mutation peut casser "
         "plusieurs règles) ; « seule Rx » = refus où Rx est la seule cassée : sans Rx, ces mutations passeraient.", ""),
        ("4. une règle suffit-elle seule ? « règle(s) qui suffisent seules » : règle cassée par 100 % des refus ; "
         "n'appliquer qu'elle donnerait le même résultat pour cette sous-famille. « oubli max 95 % » dit à quel "
         "point c'est sûr. « aucune » : il faut plusieurs règles ; voir « meilleure règle seule » et sa "
         "« couverture ».", ""),
        *([("5. les deux sources sont-elles d'accord ? « statut » : accord (même verdict), désaccord, ou seulement "
             "exhaustif / seulement marches.", "")] if has_exhaustive() else []),
        ("", ""),
        ("Exemples (tirés du tableau)", "titre"),
        *[(text, "") for text in example_lines(table)],
        ("", ""),
        ("Couleurs", "titre"),
        ("verdict et statut : vert = autorisée / accord, rouge = interdite / désaccord, jaune = dépend / une seule "
         "source.", ""),
        ("colonnes des règles : couleur de la règle (la même que dans mutations_V2.ipynb) ; pâle = la règle ne "
         "refuse rien ici, plus foncé = elle refuse au moins une mutation.", ""),
        ("statistiques : rouge = sûr (≤ 0,01 %), bleu = le hasard (≥ 50 %), dégradé entre les deux.", "degrade"),
        ("", ""),
        ("Les formules sont dans l'onglet « méthode ».", ""),
    ]


def methode() -> str:
    """Méthode de V1, pour une seule source et les règles V2."""
    text = sf.METHODE
    start = text.index("# Méthode")
    body = text[start:]
    body = body.replace(
        "Fichiers : `synthese_familles.csv` et `synthese_familles.xlsx` (même tableau, liens cliquables dans le .xlsx),\n"
        "écrits par `teste/synthese_familles.py` à partir des CSV de `teste/parametre/` (marches) et\n"
        "`teste/parametre/exhaustif/` (énumération).",
        "Fichiers : `synthese_familles_V2.csv` et `synthese_familles_V2.xlsx` (même tableau, liens cliquables dans le\n"
        ".xlsx), écrits par `teste_newmut_V2/Mutation_V2/synthese_familles_V2.py` (calculs de\n"
        f"`teste_newmut_V1/synthese_familles.py`) à partir des CSV de `teste_newmut_V2/Mutation_V2/{os.path.basename(DIR)}/`\n"
        "(`mutations_V2.ipynb`).")
    sources_start = body.index("## Les deux sources")
    sources_end = body.index("## Intervalle de confiance")
    marches = ("- **marches** : " + walks_description() + " ; toutes les mutations candidates de la famille (ajout et "
               "substitution sans tri interne) sont jugées par les règles V2 de `parametrage_regles_V2.ipynb` et RB ; "
               "les mutations qui donneraient un B# ne sont pas proposées.\n")
    if has_exhaustive():
        sources = ("## Les deux sources\n" + marches + "- **exhaustif** : " + exhaustive_description() + ", "
                   "énumérées sur le serveur (`Dataset/Enumeration_rule_V2`) ; mêmes mutations et mêmes règles "
                   "(`exhaustif_V2.py`). Deux lignes par sous-famille, une par source ; une sous-famille absente d'une "
                   "source n'a qu'une ligne.\n\n")
        limits = ("- L'**exhaustif n'est pas un échantillon** : jusqu'à 8 atomes, ses chiffres sont exacts. Son "
                  "incertitude porte sur les molécules plus grandes, qu'il ne contient pas.\n")
        tail = body[body.index("## Statut et écart"):body.index("## Limites")]
    else:
        sources = ("## Source\n" + marches + "Une seule source : « statut » vaut toujours « seulement marches » et "
                   "« écart significatif » est vide.\n\n")
        limits, tail = "", ""
    body = body[:sources_start] + sources + body[sources_end:body.index("## Statut et écart")] + tail + (
        "## Limites\n" + limits +
        "- Les mutations d'une même molécule **ne sont pas indépendantes** : les intervalles sont un peu trop étroits.\n"
        "- Les **marches V2** sont un échantillon biaisé (5 types de mutation, départ benzène) : les contextes absents "
        "des marches n'y sont pas.\n"
        "- Les molécules du test sont conformes aux règles relues depuis leur SMILES ; R4 dépend de la numérotation "
        "des atomes, une partie des molécules des marches en sort (écartées du tirage).\n")
    return ("# Synthèse des familles de mutation V2 : méthode\n\n"
            "Règles : " + ", ".join(RULES) + " (V2 + RB). La clé de lecture est dans l'onglet « clé de lecture » du "
            ".xlsx.\n\n" + body)


EXCEL_SCRIPT = r'''
import json, sys
sys.path.insert(0, sys.argv[4])
import colorer_synthese as cs
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

xlsx_path, v1_path, settings = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
cs.RULE_COLOURS = settings["rule_colours"]
cs.KEY = [tuple(k) for k in settings["key"]]
cs.main(xlsx_path)

# réglages d'affichage du .xlsx de V1, colonne par colonne (par nom ; règles par rang dans leur groupe)
v1 = load_workbook(v1_path)["synthèse"]
v1_cols = {v1.cell(1, c).value: c for c in range(1, v1.max_column + 1)}

def v1_width(c):
    """Largeur et masquage d'une colonne V1 (une plage min-max partage ses réglages)."""
    letter = get_column_letter(c)
    for dim in v1.column_dimensions.values():
        if dim.min and dim.max and dim.min <= c <= dim.max:
            return dim.width, dim.hidden
    dim = v1.column_dimensions[letter]
    return dim.width, dim.hidden

wb = load_workbook(xlsx_path)
ws = wb["synthèse"]
header = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
alias = settings["alias"]
for name, c in header.items():
    source = v1_cols.get(alias.get(name, name))
    letter = get_column_letter(c)
    if source is None:
        continue
    width, hidden = v1_width(source)
    ws.column_dimensions[letter].width = width
    ws.column_dimensions[letter].hidden = hidden
    head = v1.cell(1, source)
    ws.cell(1, c).alignment = Alignment(horizontal=head.alignment.horizontal, vertical=head.alignment.vertical,
                                        wrap_text=head.alignment.wrap_text, text_rotation=head.alignment.textRotation)
for r, dim in v1.row_dimensions.items():
    if dim.height is not None and r <= ws.max_row:
        ws.row_dimensions[r].height = dim.height
# volet figé : sur la colonne de même nom qu'en V1
frozen = v1.freeze_panes
if frozen:
    col = "".join(ch for ch in frozen if ch.isalpha())
    row = "".join(ch for ch in frozen if ch.isdigit())
    from openpyxl.utils import column_index_from_string
    name = v1.cell(1, column_index_from_string(col)).value
    back = {v: k for k, v in alias.items()}
    target = header.get(back.get(name, name))
    ws.freeze_panes = f"{get_column_letter(target)}{row}" if target else frozen
ws.auto_filter.ref = ws.dimensions if v1.auto_filter.ref else None
ws.sheet_view.zoomScale = v1.sheet_view.zoomScale
wb.save(xlsx_path)
print("réglages d'affichage de V1 appliqués :", xlsx_path)
'''


def main() -> None:
    table, stats = build()
    csv_path = os.path.join(DIR, "synthese_familles_V2.csv")
    xlsx_path = os.path.join(DIR, "synthese_familles_V2.xlsx")
    if os.path.exists(xlsx_path):
        # ne pas écraser les réglages faits à la main dans le .xlsx existant
        xlsx_path = os.path.join(DIR, "synthese_familles_V2_nouveau.xlsx")
    md_path = os.path.join(DIR, "synthese_familles_V2_methode.md")
    table.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(methode())
    # tableau de base (script de V1), puis couleurs, clé de lecture et réglages d'affichage de V1
    result = subprocess.run([sf.PYTHON_EXCEL, "-c", sf.EXCEL_SCRIPT, csv_path, xlsx_path, md_path],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print("le .xlsx n'a pas pu être écrit :", result.stderr[-800:], file=sys.stderr)
    else:
        alias = {}
        for v2, v1 in zip(RULES, V1_RULES):
            alias[f"refus {v2}"] = f"refus {v1}"
            alias[f"seule {v2}"] = f"seule {v1}"
        settings = {"rule_colours": RULE_COLOURS, "key": key_lines(table), "alias": alias}
        styled = subprocess.run([sf.PYTHON_EXCEL, "-c", EXCEL_SCRIPT, xlsx_path, V1_XLSX, json.dumps(settings), V1],
                                capture_output=True, text=True, check=False)
        print(styled.stdout.strip() or styled.stderr[-800:])
    verdicts = table["verdict"].value_counts().to_dict()
    print(f"{len(table)} lignes ({stats['sous-familles']} sous-familles, familles : {', '.join(stats['familles'])})")
    print("verdicts :", verdicts, "· images manquantes :", stats["images manquantes"])
    print("écrit :", csv_path, xlsx_path if result.returncode == 0 else "(pas de .xlsx)", md_path, sep="\n  ")


if __name__ == "__main__":
    main()
