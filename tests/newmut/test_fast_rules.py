"""
The fast check of the matrix rules (newmut.fast_rules) gives exactly the same
answer as the original one (RDKit copy of the result, all the rules
recomputed), for every candidate action of many molecules.
"""

import random

import pytest

from evomol import default_parameters as dp
from evomol.action import Action
from evomol.newmut import fast_rules
from evomol.newmut import rule_filtered as rf
from evomol.newmut.default_parameters import setup_newmut_action_space
from evomol.representation import MolecularGraph, Molecule

FIXED = [
    "",
    "C",
    "CC",
    "CC(C)(C)C",
    "C=CC=C",
    "C#CC=C",
    "c1ccccc1",
    "Cc1ccccc1",
    "CC1=CC=C(C)S1",
    "C1=CNC=CC1",
    "C1=CC=CC1",
    "C1CCCCC1",
    "C1CC2CCC1C2",
    "c1ccc2ccccc2c1",
    "c1cc2ccc3cccc4ccc(c1)c2c34",
    "C1=CC=CC=CC=CC=C1",
    "C1=CC=CC=CC=CC=CC=CC=CC=C1",
    "c1cc2ccc1CCc1ccc(cc1)CC2",
    "O=C1C=CC(=O)C=C1",
    "CC(=O)Nc1ccc(O)cc1",
    "N#Cc1ccc(C=O)s1",
    "C1COc2cscc2O1",
    "C1CCC2(CC1)CCCC2",
    "c1ccc(cc1)[N+](=O)[O-]",
    "CS(=O)(=O)c1ccccc1",
]


def random_molecules(n: int, length: int, seed: int) -> list[str]:
    """Molecules reached by chains of random mutations without rules (they
    break the rules in many ways)."""
    dp.setup_default_action_space(with_add_group=True, with_remove_group=True)
    rng = random.Random(seed)
    molecules = []
    for _ in range(n):
        molecule = Molecule(rng.choice(["c1ccccc1", "C=CC=O", "c1ccncc1", "CC1=CC=C(C)O1"]))
        for _ in range(length):
            action_class = rng.choice(MolecularGraph.action_space)
            actions = action_class.list_actions(molecule)
            if actions:
                molecule = rng.choice(actions).apply()
        molecules.append(str(molecule))
    return molecules


@pytest.fixture(scope="module")
def molecules() -> list[str]:
    """Fixed and random molecules."""
    dp.setup_default_parameters(accepted_atoms=["C", "O", "N", "S"], max_heavy_atoms=60)
    return FIXED + random_molecules(40, 25, seed=0) + valid_molecules(25, 120, seed=1)


def valid_molecules(n: int, length: int, seed: int) -> list[str]:
    """Molecules reached by chains of random mutations that respect the rules
    (they have no violation: the prefilters apply to them)."""
    from evomol.newmut import matrix_rules as mr  # pylint: disable=import-outside-toplevel
    from evomol.newmut.neighborhood_strategy import draw_allowed_action  # pylint: disable=import-outside-toplevel

    setup_newmut_action_space(
        matrix_rules=True, append_atom_rule=True, substitution_shift=True, graft=True,
        annelate=True, flip_path=True, aromatize=True, insert_carbon_keep_double=True,
    )
    rng = random.Random(seed)
    state = random.getstate()
    random.seed(seed)
    molecules = []
    try:
        for _ in range(n):
            molecule = Molecule(rng.choice(["c1ccccc1", "C=CC=O", "c1ccncc1", "C=CC=CC=C"]))
            for _ in range(rng.randint(5, length)):
                action = draw_allowed_action(rng.choice(MolecularGraph.action_space), molecule)
                if action is not None:
                    molecule = action.apply()
            if not mr.molecular_graph_violations(molecule.get_representation(MolecularGraph)):
                molecules.append(str(molecule))
    finally:
        random.setstate(state)
    return molecules


def all_rule_classes() -> list[type[Action]]:
    """Every action type, restricted by the rules."""
    setup_newmut_action_space(
        with_add_group=True,
        with_remove_group=True,
        matrix_rules=True,
        append_atom_rule=False,
        substitution_shift=False,
        graft=True,
        annelate=True,
        flip_path=True,
        aromatize=True,
    )
    space = list(MolecularGraph.action_space)
    setup_newmut_action_space(
        matrix_rules=True, append_atom_rule=True, substitution_shift=True, insert_carbon_keep_double=True
    )
    return space + [
        action for action in MolecularGraph.action_space
        if action.class_name() in ("AppendAtomRuleMG", "SubstituteShiftMG")
        or getattr(action, "inner_class", None).__name__ == "InsertCarbonDoubleMG"
    ]


def flags(action_class: type[Action], smiles: str, fast: bool) -> list[tuple[str, bool]]:
    """(action, allowed) for every candidate action, with the fast or the
    original check (no cache)."""
    fast_rules.FAST_RULES = fast
    fast_rules.clear_cache()
    rf.set_cache_size(0)
    try:
        molecule = Molecule(smiles)
        candidates = action_class.list_unchecked_actions(molecule)  # type: ignore[attr-defined]
        return [
            (repr(action), action_class.action_is_allowed(action, molecule))  # type: ignore[attr-defined]
            for action in candidates
        ]
    finally:
        fast_rules.FAST_RULES = True
        rf.set_cache_size(10_000)


def test_same_answer(molecules: list[str]) -> None:
    """Fast and original checks agree on every candidate action."""
    classes = all_rule_classes()
    nb_checked = 0
    mismatches = []
    for smiles in molecules:
        for action_class in classes:
            fast = flags(action_class, smiles, fast=True)
            slow = flags(action_class, smiles, fast=False)
            assert [a for a, _ in fast] == [a for a, _ in slow]
            nb_checked += len(fast)
            mismatches += [(a, f, s) for (a, f), (_, s) in zip(fast, slow) if f != s]
    assert not mismatches, mismatches[:10]
    assert nb_checked > 5000


@pytest.mark.parametrize("outside_rings", (False, True))
def test_same_answer_r2_hetero(molecules: list[str], outside_rings: bool) -> None:
    """Same agreement with the options R2_HETERO (and R2_HETERO_OUTSIDE_RINGS)."""
    from evomol.newmut import matrix_rules as mr  # pylint: disable=import-outside-toplevel

    classes = all_rule_classes()
    mr.R2_HETERO, mr.R2_HETERO_OUTSIDE_RINGS = True, outside_rings
    try:
        mismatches = []
        for smiles in molecules[::2]:
            for action_class in classes:
                fast = flags(action_class, smiles, fast=True)
                slow = flags(action_class, smiles, fast=False)
                assert [a for a, _ in fast] == [a for a, _ in slow]
                mismatches += [(a, f, s) for (a, f), (_, s) in zip(fast, slow) if f != s]
        assert not mismatches, mismatches[:10]
    finally:
        mr.R2_HETERO, mr.R2_HETERO_OUTSIDE_RINGS = False, False


def test_fast_is_used() -> None:
    """The fast path computes the effect of the usual actions on the matrix."""
    setup_newmut_action_space(matrix_rules=True)
    molecule = Molecule("Cc1ccccc1")
    parent = fast_rules.parent_rules(molecule.get_representation(MolecularGraph))
    for action_class in MolecularGraph.action_space:
        for action in action_class.list_actions(molecule):
            assert fast_rules.matrix_effect(action, parent.types, parent.matrix) is not None


def test_prefilters_are_refusals(molecules: list[str]) -> None:
    """Every action skipped by a prefilter is refused by the original check,
    and the prefilters do skip actions on the molecules within the rules."""
    nb_skipped = 0
    for action_class in all_rule_classes():
        for smiles in molecules:
            molecule = Molecule(smiles)
            fast_rules.clear_cache()
            parent = fast_rules.parent_rules(molecule.get_representation(MolecularGraph))
            if parent.charged:
                continue
            for action in action_class.list_unchecked_actions(molecule):  # type: ignore[attr-defined]
                if not parent.surely_refused(action):
                    continue
                nb_skipped += 1
                fast_rules.FAST_RULES = False
                try:
                    assert not action_class.action_is_allowed(action, molecule), (smiles, action)  # type: ignore[attr-defined]
                finally:
                    fast_rules.FAST_RULES = True
    assert nb_skipped > 100
