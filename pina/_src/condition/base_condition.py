"""Module for the Base Condition class."""

from pina._src.condition.condition_interface import ConditionInterface
from pina._src.core.utils import check_consistency
from pina._src.problem.problem_interface import ProblemInterface


def _unwrap_single(value):
    """
    Unwrap a single-element list or tuple into its only element.

    This is used to expose a list of one graph as the graph itself.

    :param Any value: The value to unwrap.
    :return: The single element when ``value`` is a list or tuple of length one,
        otherwise ``value``.
    :rtype: Any
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]

    return value


class BaseCondition(ConditionInterface):
    """
    Base class for all conditions, implementing common functionality.

    Concrete conditions inherit either from :class:`TensorCondition` or
    :class:`GraphCondition` to obtain the storage and batching behaviour
    matching their data type, while defining their own :meth:`evaluate`.
    """

    @property
    def problem(self):
        """
        The problem associated with this condition.

        :raises AttributeError: If no problem has been assigned.
        :return: The problem associated with this condition.
        :rtype: pina.problem.ProblemInterface
        """
        return self._problem

    @problem.setter
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param pina.problem.ProblemInterface value: The problem to associate
            with this condition.
        :raises ValueError: If ``value`` fails validation against
            :class:`pina.problem.ProblemInterface`.
        """
        check_consistency(value, ProblemInterface)
        self._problem = value
