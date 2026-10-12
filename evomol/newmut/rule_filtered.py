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


def allowed_flags(
    actions: list[Action], mol_graph: MolecularGraph
) -> tuple[bool, ...]:
    """For each action, True if its result creates no new violation."""
    types_before, matrix_before = to_matrix(mol_graph)
    flags = []
    for action in actions:
        assert isinstance(action, ActionMolGraph)
        try:
            types_after, matrix_after = to_matrix(result_graph(action, mol_graph))
        except Exception as exc:  # pylint: disable=broad-except
            get_logger().debug("Action %s refused (error: %r)", action, exc)
            flags.append(False)
            continue
        flags.append(
            matrices_allowed(types_before, matrix_before, types_after, matrix_after)
        )
    return tuple(flags)


def filter_actions(
    name: str, actions: list[Action], molecule: Molecule
) -> list[Action]:
    """Keep the actions whose result creates no new violation (cached)."""
    if not actions:
        return actions
    mol_graph = molecule.get_representation(MolecularGraph)

    flags: tuple[bool, ...] | None = None
    key: tuple[Hashable, ...] | None = None
    if CACHE_SIZE > 0:
        key = (name, *graph_key(mol_graph))
        flags = _cache.get(key)
        if flags is not None and len(flags) == len(actions):
            _cache.move_to_end(key)
        else:
            flags = None

    if flags is None:
        flags = allowed_flags(actions, mol_graph)
        if key is not None:
            _cache[key] = flags
            if len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)

    return [action for action, flag in zip(actions, flags) if flag]


class _RuleFilteredMixin:
    """list_actions of the wrapped class, restricted by the matrix rules."""

    # True for the action classes that already apply the matrix rules
    checks_rules: bool = True
    inner_class: type[ActionMolGraph]

    @classmethod
    def class_name(cls) -> str:
        """Same name as the wrapped action (same traces)."""
        return cls.inner_class.class_name()

    @classmethod
    def list_actions(cls, molecule: Molecule) -> list[Action]:
        """Actions of the wrapped class whose result respects the rules."""
        actions: list[Action] = super().list_actions(molecule)  # type: ignore[misc]
        return filter_actions(cls.inner_class.class_name(), actions, molecule)


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
