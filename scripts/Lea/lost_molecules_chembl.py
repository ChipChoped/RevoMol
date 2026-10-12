"""Compare l'énumération exhaustive C=C, 4 atomes, Hückel + ChEMBL/ZINC + Hetero,
avec les mutations actuelles (HuckMol-2.0) et anciennes (RevoMol-main), et
explique par quels chemins les molécules perdues étaient atteintes."""
import importlib.util
import os
import sys
from collections import deque, defaultdict

ROOT = os.path.expanduser("~/Desktop/HuckMol-2.0")
OLD = os.path.expanduser("~/Desktop/RevoMol-main/evomol/action/molecular_graph")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from rdkit import RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

from evomol import default_parameters as dp  # noqa: E402
import huckel_filters as hf  # noqa: E402
from evomol import evaluation as evaluator  # noqa: E402
from evomol.action import molecular_graph as mg  # noqa: E402
from evomol.representation import MolecularGraph, Molecule  # noqa: E402
from evomol.search.enumeration import canonical  # noqa: E402


def load_old(fname, modname):
    spec = importlib.util.spec_from_file_location(
        f"evomol.action.molecular_graph.{modname}", os.path.join(OLD, fname)
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "evomol.action.molecular_graph"
    spec.loader.exec_module(mod)
    return mod


OldRemove = load_old("remove_atom.py", "_old_remove_atom").RemoveAtomMG
OldChange = load_old("change_bond.py", "_old_change_bond").ChangeBondMG

MAX_ATOMS = 4


def setup(old: bool, pruned: bool, realism):
    dp.setup_default_parameters(accepted_atoms=hf.HUCKEL_ACCEPTED_ATOMS, max_heavy_atoms=MAX_ATOMS)
    dp.setup_default_action_space(insert_carbon=not pruned, avoid_sp3_on_sp3=pruned)
    if old:
        sp = MolecularGraph.action_space
        sp[sp.index(mg.RemoveAtomMG)] = OldRemove
        sp[sp.index(mg.ChangeBondMG)] = OldChange
    return hf.setup_huckel_filters(realism=realism)


def neighbors(mol):
    """(smiles voisin, action) pour chaque action possible."""
    mol.compute_possible_actions()
    return [
        (a.apply_smiles(), repr(a))
        for rep in mol.possible_actions.values()
        for lst in rep.values()
        for a in lst
    ]


def bfs(filters):
    start = canonical("C=C")
    seen, valid, invalid = {start}, set(), set()
    edges = defaultdict(list)  # enfant valide -> [(parent, action)]
    q = deque()
    if evaluator.is_valid_molecule(Molecule(start), filters):
        valid.add(start)
        q.append(start)
    while q:
        cur = q.popleft()
        for smi, act in neighbors(Molecule(cur)):
            if smi != cur:
                edges[smi].append((cur, act))
            if smi in seen:
                continue
            seen.add(smi)
            if Molecule(smi).get_representation(MolecularGraph).nb_atoms > MAX_ATOMS:
                continue
            if evaluator.is_valid_molecule(Molecule(smi), filters):
                valid.add(smi)
                q.append(smi)
            else:
                invalid.add(smi)
    return valid, invalid, seen, edges


if __name__ == "__main__":
    realism = "chembl_zinc"
    res = {}
    for old in (False, True):
        for pruned in (False, True):
            f = setup(old, pruned, realism)
            v, inv, seen, edges = bfs(f)
            res[(old, pruned)] = (v, inv, seen, edges)
            print(f"old={old} pruned={pruned}: {len(v)} valides / {len(seen)} vus")

    v_new, inv_new, seen_new, e_new = res[(False, False)]
    v_old, inv_old, seen_old, e_old = res[(True, False)]
    lost = sorted(v_old - v_new)
    print(f"\nPerdues : {len(lost)} ; gagnées : {sorted(v_new - v_old)}")
    # quelles actions de l'ancien code menaient aux perdues depuis une molécule valide
    for smi in lost:
        parents = [(p, a) for p, a in e_old[smi] if p in v_old]
        par_new = [(p, a) for p, a in e_new.get(smi, []) if p in v_new]
        flags = []
        # la molécule est-elle valide isolément (filtres) ?
        f = setup(False, False, realism)
        flags.append("valide" if evaluator.is_valid_molecule(Molecule(smi), f) else "INVALIDE")
        print(f"\n- {smi}  [{flags[0]}]  vue avec nouvelles mutations : {smi in seen_new}")
        for p, a in parents:
            tag = "parent perdu" if p not in v_new else "parent atteint"
            print(f"    ancien : {p} --{a}--> ({tag})")
        for p, a in par_new:
            print(f"    nouveau: {p} --{a}-->")
