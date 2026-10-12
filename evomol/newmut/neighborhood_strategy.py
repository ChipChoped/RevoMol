"""
Neighborhood strategy for the new mutations (newmut).

NewMutNeighborhoodStrategy is RandomNeighborhoodStrategy with one more
option, lazy_rules (off by default):

- lazy_rules=False: exactly RandomNeighborhoodStrategy. An action type that
  applies the matrix rules lists all its actions, checks each of them, and one
  is drawn among the allowed ones (cached per molecule).
- lazy_rules=True: once the action type is drawn, its candidate actions are
  tried in a random order, and the first one that respects the rules is kept.
  Each allowed action has the same probability as with lazy_rules=False, but
  usually only one or two actions are checked instead of all of them. Only used
  with preselect_actions=True and uniform_over_actions=False (the defaults);
  the other modes need the full lists and keep the behaviour of
  RandomNeighborhoodStrategy.
"""

from __future__ import annotations

import random

from typing_extensions import override

import evomol.action.action as action_module
from evomol.newmut.matrix_rules import to_matrix
from evomol.representation import MolecularGraph, Molecule
from evomol.search.neighborhood_strategy import RandomNeighborhoodStrategy


def draw_allowed_action(
    action_class: type[action_module.Action], molecule: Molecule
) -> action_module.Action | None:
    """Draw an action of action_class uniformly among the allowed ones.

    For an action type that applies the matrix rules (checks_rules), the
    candidates are tried in a random order and the first allowed one is
    returned. Otherwise, uniform draw in list_actions. None if there is no
    allowed action.
    """
    if not getattr(action_class, "checks_rules", False):
        actions = action_class.list_actions(molecule)
        return random.choice(actions) if actions else None

    pool = list(action_class.list_unchecked_actions(molecule))  # type: ignore[attr-defined]
    before = to_matrix(molecule.get_representation(MolecularGraph))
    while pool:
        # draw without replacement: the first allowed action of a uniformly
        # random order is uniform among the allowed actions
        k = random.randrange(len(pool))
        action = pool[k]
        pool[k] = pool[-1]
        pool.pop()
        if action_class.action_is_allowed(action, molecule, before):  # type: ignore[attr-defined]
            return action
    return None


class NewMutNeighborhoodStrategy(RandomNeighborhoodStrategy):
    """RandomNeighborhoodStrategy, with the lazy_rules option (see module)."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        depth: int = 1,
        nb_max_tries: int = 50,
        preselect_actions: bool = True,
        random_depth: bool = False,
        uniform_over_actions: bool = False,
        remember_invalid: bool = False,
        lazy_rules: bool = False,
    ):
        """Initialize the strategy.

        Args:
            depth, nb_max_tries, preselect_actions, random_depth,
            uniform_over_actions, remember_invalid: same as
                RandomNeighborhoodStrategy.
            lazy_rules: if True, the actions that apply the matrix rules are
                tried one at a time in a random order instead of being all
                checked (same draw, faster). False: behaviour of
                RandomNeighborhoodStrategy.
        """
        super().__init__(
            depth=depth,
            nb_max_tries=nb_max_tries,
            preselect_actions=preselect_actions,
            random_depth=random_depth,
            uniform_over_actions=uniform_over_actions,
            remember_invalid=remember_invalid,
        )
        self.lazy_rules: bool = lazy_rules

    @override
    def random_mutation(self, new_molecule: Molecule) -> Molecule:
        """Same as RandomNeighborhoodStrategy.random_mutation; with lazy_rules,
        the action of the drawn type is drawn by draw_allowed_action."""
        if (
            not self.lazy_rules
            or not self.preselect_actions
            or self.uniform_over_actions
        ):
            return super().random_mutation(new_molecule)

        nb_actions = random.randint(1, self.depth) if self.random_depth else self.depth

        for _ in range(nb_actions):
            representation_dict: dict[str, list[type[action_module.Action]]] = (
                new_molecule.list_available_actions_space()
            )

            representation_choice: str = random.choice(list(representation_dict.keys()))

            action_choice: type[action_module.Action] = random.choice(
                representation_dict[representation_choice]
            )

            action = draw_allowed_action(action_choice, new_molecule)

            if action is None:
                continue

            new_molecule = action.apply()

        return new_molecule
