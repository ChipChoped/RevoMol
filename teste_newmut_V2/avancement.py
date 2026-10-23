"""Avancement des calculs V2 lancés à la suite (marches, tests de mutation, exhaustif, synthèse), rafraîchi toutes
les 60 s (Ctrl-C pour quitter) :
    /Applications/anaconda3/envs/revomol/bin/python teste_newmut_V2/avancement.py [secondes]
"""
import glob
import os
import re
import sys
import time
from datetime import datetime, timedelta

LOGS = "/private/tmp/claude-501/-Users-LeaClautour-Desktop-RevoMol/e3066078-5338-42a0-b32c-c239b84ddfcb/scratchpad/calculs"
MARCHES_LOG, SUITE_LOG = os.path.join(LOGS, "marches.log"), os.path.join(LOGS, "suite.log")
WALKS = "/Users/LeaClautour/Desktop/RevoMol/teste_newmut_V2/Walk_V2/resultats/test_Regle_V2_Bd"
N_WALKS = 20
FAMILLES = ["AddAtomMG", "ChangeBondMG", "CutAtomMG", "InsertCarbonMG", "RemoveAtomMG", "SubstituteShiftMG"]


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def duree(s):
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60:02d}" if s >= 3600 else f"{s // 60} min {s % 60:02d}"


def heure(s):
    return (datetime.now() + timedelta(seconds=s)).strftime("%a %d %H:%M")


def marches():
    log = read(MARCHES_LOG)
    if not log:
        return "pas commencées"
    start = os.path.getmtime(MARCHES_LOG) if "début" not in log else None
    m = re.search(r"début (\d\d:\d\d)", log)
    t0 = os.stat(MARCHES_LOG).st_birthtime if hasattr(os.stat(MARCHES_LOG), "st_birthtime") else start
    done = [f for f in glob.glob(os.path.join(WALKS, "walk*_meta.json")) if os.path.getmtime(f) > t0]
    ecoule = time.time() - t0
    if "ANALYSE FINIE" in log:
        return f"FINIES ({N_WALKS}/{N_WALKS}), analyse faite"
    if "Traceback" in log:
        return "ERREUR : voir " + MARCHES_LOG
    if "FINI en" in log:
        return f"{N_WALKS}/{N_WALKS} finies, analyse en cours"
    pas = re.findall(r"(\d+)%\|[^|]*\| (\d+)/(\d+) \[([\d:]+)<([\d:?]+)", log)   # barre de progression des pas
    texte = ""
    if pas:
        pct, fait, total, _, reste = pas[-1]
        texte = f"\n    pas {int(fait):,}/{int(total):,} ({pct} %), reste ≈ {reste}".replace(",", " ")
        if reste.count(":") >= 1 and "?" not in reste:
            parts = [int(x) for x in reste.split(":")]
            secondes = sum(v * 60 ** k for k, v in enumerate(reversed(parts)))
            texte += f" (fin ≈ {heure(secondes)})"
    return (f"{len(done)}/{N_WALKS} finies | depuis {m.group(1) if m else '?'} ({duree(ecoule)}){texte}")


def suite():
    log = read(SUITE_LOG)
    lignes = []
    if not log:
        return ["mutations, exhaustif, synthèse : en attente des marches"]
    # tests de mutation
    if "== mutations" in log:
        part = log.split("== mutations", 1)[1].split("== exhaustif", 1)[0]
        faites = re.findall(r"^(\w+) : (\d+) mutations, (\d+) s", part, re.M)
        test = re.search(r"molécules du test : (\d+)", part)
        etat = "FINIS" if "MUTATIONS FINIES" in part else "en cours"
        lignes.append(f"tests de mutation : {etat} | molécules {test.group(1) if test else 'tirage en cours'} "
                      f"| familles {len(faites)}/{len(FAMILLES)}")
        for fam, n, s in faites:
            lignes.append(f"    {fam:18s} {int(n):>10,} mutations  {duree(int(s))}".replace(",", " "))
    # exhaustif
    if "== exhaustif" in log:
        part = log.split("== exhaustif", 1)[1].split("== synthèse", 1)[0]
        conformes = re.search(r"(\d+) molécules conformes sur (\d+) \((\d+) s\)", part)
        lignes.append("exhaustif : " + (f"{int(conformes.group(1)):,} molécules conformes sur {int(conformes.group(2)):,}"
                                        .replace(",", " ") if conformes else "tri des molécules conformes en cours"))
        finies = re.findall(r"^(\w+) : (\d+) sous-familles, (\d+) mutations, .*?, (\d+) s", part, re.M)
        for fam, sf, n, s in finies:
            lignes.append(f"    {fam:18s} fini : {sf} sous-familles, {int(n):,} mutations, {duree(int(s))}".replace(",", " "))
        prog = re.findall(r"^  (\w+) : (\d+)/(\d+) molécules, (\d+) mutations, (\d+) s", part, re.M)
        if prog and prog[-1][0] not in {f for f, *_ in finies}:
            fam, fait, total, n, s = prog[-1]
            fait, total, s = int(fait), int(total), int(s)
            reste = s / fait * (total - fait) if fait else 0
            lignes.append(f"    {fam:18s} {fait:,}/{total:,} molécules ({100 * fait / total:.0f} %), {duree(s)} écoulées, "
                          f"reste ≈ {duree(reste)} (fin ≈ {heure(reste)})".replace(",", " "))
        lignes.append(f"    familles finies : {len(finies)}/{len(FAMILLES)}")
    if "== synthèse" in log:
        lignes.append("synthèse : " + ("faite" if "TOUT FINI" in log else "en cours"))
    if "TOUT FINI" in log:
        lignes.append("TOUT EST FINI : " + re.search(r"TOUT FINI (.*)", log).group(1))
    if re.search(r"Traceback|Error", log):
        lignes.append("ERREUR : voir " + SUITE_LOG)
    return lignes


def main():
    every = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    while True:
        os.system("clear")
        print(f"Toutes les {every:.0f} s   {datetime.now():%a %d %b %H:%M:%S}\n")
        print("marches V2 : " + marches())
        for line in suite():
            print(line)
        sys.stdout.flush()
        time.sleep(every)


if __name__ == "__main__":
    main()
