"""Module for the ParametricProblem class."""

from abc import abstractmethod
from pina._src.problem.base_problem import BaseProblem


class ParametricProblem(BaseProblem):
    """
    Base class for all parametric problems, extending the standard problem
    definition with parameter-dependent inputs.

    A parametric problem includes additional input variables, defined over a
    dedicated parameter domain, which represent external quantities
    (e.g., physical coefficients or control variables) that can vary across
    different evaluations and influence the solution.

    This class is not meant to be instantiated directly.

    :Example:

        >>> from pina.problem import ParametricProblem
        >>> from pina.domain import CartesianDomain
        >>> class MyParametricProblem(ParametricProblem):
        ...     output_variables = ["u"]
        ...     @property
        ...     def parameter_domain(self):
        ...         return CartesianDomain({"mu": [0.1, 10.0]})
        ...     @property
        ...     def conditions(self): return {}
        >>> problem = MyParametricProblem()
        >>> problem.parameters
        ['mu']
    """

    @property
    @abstractmethod
    def parameter_domain(self):
        """
        The domain of the parameters of the problem.

        :return: The geometry providing the names of parameter inputs, distinct
            from trainable unknowns in an inverse problem.
        :rtype: pina.domain.DomainInterface
        """

    @property
    def parameters(self):
        """
        The parameters of the problem.

        :return: The parameters of the problem.
        :rtype: list[str]
        """
        return self.parameter_domain.variables
