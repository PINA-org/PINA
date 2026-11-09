"""Module for the SpatialProblem class."""

from abc import abstractmethod
from pina._src.problem.base_problem import BaseProblem


class SpatialProblem(BaseProblem):
    """
    Base class for all spatial problems, extending the standard problem
    definition with spatial-dependent inputs.

    A spatial problem is defined over a spatial domain, where input variables
    represent the coordinates of the system (e.g., positions in one or more
    dimensions) on which the solution is evaluated.

    This class is not meant to be instantiated directly.

    :Example:

        >>> from pina.problem import SpatialProblem
        >>> from pina.domain import CartesianDomain
        >>> class MySpatialProblem(SpatialProblem):
        ...     output_variables = ["u"]
        ...     @property
        ...     def spatial_domain(self):
        ...         return CartesianDomain({"x": [0, 1], "y": [0, 1]})
        ...     @property
        ...     def conditions(self): return {}
        >>> problem = MySpatialProblem()
        >>> problem.spatial_variables
        ['x', 'y']
    """

    @property
    @abstractmethod
    def spatial_domain(self):
        """
        The domain of spatial variables of the problem.

        :return: The geometry providing the spatial coordinate names.
        :rtype: pina.domain.DomainInterface
        """

    @property
    def spatial_variables(self):
        """
        The spatial input variables of the problem.

        :return: The spatial input variables of the problem.
        :rtype: list[str]
        """
        return self.spatial_domain.variables
