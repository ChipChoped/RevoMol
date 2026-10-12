"""
New mutations (newmut): validity rules read on the bond-order matrix and new
elementary and composite actions on the molecular graph.

Nothing here changes the existing evomol behaviour: the new rules and actions
are only used when they are put in the action space (see
setup_newmut_action_space). Documentation: evomol/newmut/README.md.
"""

# flake8: noqa
from .annelate import AnnelateMG as AnnelateMG
from .append_atom_rule import AppendAtomRuleMG as AppendAtomRuleMG
from .aromatize import AromatizeMG as AromatizeMG
from .default_parameters import setup_newmut_action_space as setup_newmut_action_space
from .evolutionary_algorithm import EvolutionaryAlgorithmNewMut as EvolutionaryAlgorithmNewMut
from .evolutionary_algorithm import warn_rule_violations as warn_rule_violations
from .flip_path import FlipAlternatingPathMG as FlipAlternatingPathMG
from .fragments import ANNELATION_FRAGMENTS as ANNELATION_FRAGMENTS
from .fragments import GRAFT_FRAGMENTS as GRAFT_FRAGMENTS
from .graft import GraftFragmentMG as GraftFragmentMG
from .matrix_rules import is_valid as is_valid
from .matrix_rules import result_is_allowed as result_is_allowed
from .matrix_rules import rule_violations as rule_violations
from .matrix_rules import to_matrix as to_matrix
from .neighborhood_strategy import NewMutNeighborhoodStrategy as NewMutNeighborhoodStrategy
from .rule_filtered import ChangeBondRuleMG as ChangeBondRuleMG
from .substitute_shift import SubstituteShiftMG as SubstituteShiftMG
