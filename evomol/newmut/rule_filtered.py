"""
Wrapper that applies the matrix rules to an existing action (newmut).

rule_filtered(ActionClass) returns a subclass of ActionClass whose
list_actions keeps only the actions whose result creates no new violation of
the rules R1 to R4 (see matrix_rules.result_is_allowed). The rules are thus
applied when the actions are listed, like the valence, and never as a filter
on the mutated molecules.

The wrapper keeps the class name of the wrapped action (class_name and
__name__), so that traces and counters stay the same. Its class attributes are
inherited, so the options set on the wrapped class (by_one_step,
avoid_sp3_on_sp3, ...) still apply.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Hashable

import numpy as np

from evomol.action import Action
from evomol.action import molecular_graph as mg
from evomol.action.molecular_graph.action_molecular_graph import ActionMolGraph
from evomol.logging import get_logger
from evomol.newmut.matrix_rules import matrices_allowed, to_matrix
from evomol.representation import MolecularGraph, Molecule

# Number of molecular graphs whose filtered actions are kept in memory (per
# process). 0 disables the cache.
CACHE_SIZE = 10_000

_cache: OrderedDict[Hashable, tuple[bool, ...]] = OrderedDict()
_wrapped_classes: dict[type[ActionMolGraph], type[ActionMolGraph]] = {}


def set_cache_size(size: int) -> None:
    """Set the size of the cache of filtered actions (0: no cache)."""
    global CACHE_SIZE  # pylint: disable=global-statement
    CACHE_SIZE = size
    clear_cache()


def clear_cache() -> None:
    """Empty the cache of filtered actions.

    To call when the options of an action change in the same process (done by
    setup_newmut_action_space).
    """
    _cache.clear()


def graph_key(mol_graph: MolecularGraph) -> tuple[Hashable, ...]:
    """Key of a molecular graph: atoms, bond orders (atom numbering included),
    formal charges and radicals."""
    types, matrix = to_matrix(mol_graph)
    atoms = [mol_graph.mol.GetAtomWithIdx(i) for i in range(len(types))]
    return (
        tuple(types),
        matrix.tobytes(),
        tuple(atom.GetFormalCharge() for atom in atoms),
        tuple(atom.GetNumRadicalElectrons() for atom in atoms),
    )


def result_graph(action: ActionMolGraph, mol_graph: MolecularGraph) -> MolecularGraph:
    """Molecular graph produced by the action (as in ActionMolGraph.apply_smiles).

    Raises:
        Exception: any error raised by the action or by RDKit.
    """
    new_mol_graph = mol_graph.raw_copy()
    action.apply_action(new_mol_graph)
    new_mol_graph.update_representation()
    return new_mol_graph


def action_is_allowed(
    action: ActionMolGraph,
    mol_graph: MolecularGraph,
    before: tuple[list[str], np.ndarray] | None = None,
) -> bool:
    """True if the result of the action creates no new violation.

    Args:
        action: action to check.
        mol_graph: molecular graph the action applies to.
        before: to_matrix(mol_graph), if already computed.
    """
    # pylint: disable=import-outside-toplevel
    from evomol.newmut import fast_rules

    if fast_rules.FAST_RULES:
        parent = fast_rules.parent_rules(mol_graph)
        # charged or radical atoms: RDKit may refuse a result that the rules
        # accept, so the original check is kept
        if not parent.charged:
            if fast_rules.PREFILTERS and parent.surely_refused(action):
                return False
            effect = fast_rules.matrix_effect(action, parent.types, parent.matrix)
            if effect is not None:
                return parent.allowed_after(*effect)
            try:
                after = to_matrix(result_graph(action, mol_graph))
            except Exception as exc:  # pylint: disable=broad-except
                get_logger().debug("Action %s refused (error: %r)", action, exc)
                return False
            if len(after[0]) >= parent.n:  # atoms only added (at the end)
                return parent.allowed_after(after[0], after[1], list(range(parent.n)))
            return matrices_allowed(parent.types, parent.matrix, after[0], after[1])

    if before is None:
        before = to_matrix(mol_graph)
    try:
        after = to_matrix(result_graph(action, mol_graph))
    except Exception as exc:  # pylint: disable=broad-except
        get_logger().debug("Action %s refused (error: %r)", action, exc)
        return False
    return matrices_allowed(before[0], before[1], after[0], after[1])


def filter_actions(
    action_class: type[RuleCheckedMixin], actions: list[Action], molecule: Molecule
) -> list[Action]:
    """Keep the actions whose result creates no new violation (cached)."""
    if not actions:
        return actions
    mol_graph = molecule.get_representation(MolecularGraph)

    flags: tuple[bool, ...] | None = None
    key: tuple[Hashable, ...] | None = None
    if CACHE_SIZE > 0:
        key = (action_class.class_name(), *graph_key(mol_graph))
        flags = _cache.get(key)
        if flags is not None and len(flags) == len(actions):
            _cache.move_to_end(key)
        else:
            flags = None

    if flags is None:
        before = to_matrix(mol_graph)
        flags = tuple(
            action_class.action_is_allowed(action, molecule, before)
            for action in actions
        )
        if key is not None:
            _cache[key] = flags
            if len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)

    return [action for action, flag in zip(actions, flags) if flag]


class RuleCheckedMixin:
    """Base of the actions that apply the matrix rules.

    A subclass gives its candidate actions (list_unchecked_actions);
    list_actions keeps those whose result creates no new violation
    (action_is_allowed), with a cache per molecular graph. The neighborhood
    strategy of newmut with lazy_rules=True uses list_unchecked_actions and
    action_is_allowed directly, to test the candidates one at a time.
    """

    # True for the action classes that already apply the matrix rules
    checks_rules: bool = True

    @classmethod
    def class_name(cls) -> str:
        """Return the class name of the action."""
        return cls.__name__

    @classmethod
    def list_unchecked_actions(cls, molecule: Molecule) -> list[Action]:
        """Candidate actions, before the matrix rules."""
        raise NotImplementedError

    @classmethod
    def action_is_allowed(
        cls,
        action: Action,
        molecule: Molecule,
        before: tuple[list[str], np.ndarray] | None = None,
    ) -> bool:
        """True if the result of the action creates no new violation."""
        assert isinstance(action, ActionMolGraph)
        return action_is_allowed(
            action, molecule.get_representation(MolecularGraph), before
        )

    @classmethod
    def list_actions(cls, molecule: Molecule) -> list[Action]:
        """Candidate actions whose result respects the rules."""
        return filter_actions(cls, cls.list_unchecked_actions(molecule), molecule)


class _RuleFilteredMixin(RuleCheckedMixin):
    """list_actions of the wrapped class, restricted by the matrix rules."""

    inner_class: type[ActionMolGraph]

    @classmethod
    def class_name(cls) -> str:
        """Same name as the wrapped action (same traces)."""
        return cls.inner_class.class_name()

    @classmethod
    def list_unchecked_actions(cls, molecule: Molecule) -> list[Action]:
        """Actions listed by the wrapped class."""
        return cls.inner_class.list_actions(molecule)


def rule_filtered(action_class: type[ActionMolGraph]) -> type[ActionMolGraph]:
    """Return action_class restricted by the matrix rules.

    The action classes that already apply the rules (checks_rules = True) are
    returned unchanged. The same wrapper is returned for the same class.
    """
    if getattr(action_class, "checks_rules", False):
        return action_class
    if action_class not in _wrapped_classes:
        wrapper = type(
            action_class.__name__,
            (_RuleFilteredMixin, action_class),
            {
                "inner_class": action_class,
                "__module__": __name__,
                "__qualname__": action_class.__name__,
                "__doc__": (
                    f"{action_class.__name__} restricted by the matrix rules "
                    "(see newmut.rule_filtered)."
                ),
            },
        )
        _wrapped_classes[action_class] = wrapper
    return _wrapped_classes[action_class]


# ChangeBondMG restricted by the rules ("ChBR" of the specification).
# ChangeBondMG never breaks a bridge bond, so the molecule stays connected.
ChangeBondRuleMG = rule_filtered(mg.ChangeBondMG)
