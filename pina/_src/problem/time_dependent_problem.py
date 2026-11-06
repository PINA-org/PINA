"""Module for the TimeDependentProblem class."""

from abc import abstractmethod
from pina._src.problem.base_problem import BaseProblem


class TimeDependentProblem(BaseProblem):
    """
    Base class for all time-dependent problems, extending the standard problem
    definition with time-dependent inputs.

    A time-dependent problem is defined over a temporal domain, where input
    variables represent the time at which the solution is evaluated.

    This class is not meant to be instantiated directly.

    :Example:

        >>> from pina.problem import TimeDependentProblem
        >>> from pina.domain import CartesianDomain
        >>> class MyTimeProblem(TimeDependentProblem):
        ...     output_variables = ["u"]
        ...     @property
        ...     def temporal_domain(self):
        ...         return CartesianDomain({"t": [0.0, 1.0]})
        ...     @property
        ...     def conditions(self): return {}
        >>> problem = MyTimeProblem()
        >>> problem.temporal_variables
        ['t']
    """

    @property
    @abstractmethod
    def temporal_domain(self):
        """
        The domain of temporal variables of the problem.

        :return: The geometry providing the temporal coordinate names.
        :rtype: pina.domain.DomainInterface
        """

    @property
    def temporal_variables(self):
        """
        The temporal variables of the problem.

        :return: The temporal variables of the problem.
        :rtype: list[str]
        """
        return self.temporal_domain.variables
