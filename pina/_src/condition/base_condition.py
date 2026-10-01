"""Module for the Base Condition class."""

from pina._src.condition.condition_interface import ConditionInterface
from pina._src.core.utils import check_consistency
from pina._src.problem.problem_interface import ProblemInterface


def _unwrap_single(value):
    """
    Unwrap a single-element list into its only element.

    This is used to expose a list of one graph as the graph itself.

    :param value: The value to unwrap.
    :return: The single element when ``value`` is a list of length one,
        otherwise ``value``.
    """
    if isinstance(value, list) and len(value) == 1:
        return value[0]

    return value


class BaseCondition(ConditionInterface):
    """
    Base class for all conditions, implementing common functionality.

    Concrete conditions inherit either from :class:`TensorCondition` or
    :class:`GraphCondition` to obtain the storage and batching behaviour
    matching their data type, while defining their own :meth:`evaluate`.
    """

    def __init__(self, **kwargs):
        """
        Initialization of the :class:`BaseCondition` class.

        :param dict kwargs: The keyword arguments representing the data to be
            stored in the condition.
        """
        super().__init__()
        self.data = self.store_data(**kwargs)

    @property
    def problem(self):
        """
        The problem associated with this condition.

        :return: The problem associated with this condition.
        :rtype: BaseProblem
        """
        return self._problem

    @problem.setter
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param BaseProblem value: The problem to associate with this condition.
        :raises ValueError: If the problem is not an instance of BaseProblem.
        """
        check_consistency(value, ProblemInterface)
        self._problem = value
