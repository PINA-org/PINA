"""Module for the Problems."""

__all__ = [
    "ProblemInterface",
    "BaseProblem",
    "SpatialProblem",
    "TimeDependentProblem",
    "ParametricProblem",
    "InverseProblem",
]

from pina._src.problem.problem_interface import ProblemInterface
from pina._src.problem.base_problem import BaseProblem
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.problem.parametric_problem import ParametricProblem
from pina._src.problem.inverse_problem import InverseProblem

# Back-compatibility with version 0.2, to be removed soon
import warnings

_DEPRECATED_IMPORTS = {"AbstractProblem": "BaseProblem"}


def __getattr__(name):
    """
    Resolve the deprecated abstract-problem name.

    :param str name: The requested attribute name.
    :return: :class:`BaseProblem` for ``"AbstractProblem"``, emitting
        ``DeprecationWarning``. Unrecognized names return ``None``.
    :rtype: type | None
    """
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina.problem' is deprecated; use "
            f"pina.problem.{_DEPRECATED_IMPORTS[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return globals()[_DEPRECATED_IMPORTS[name]]
