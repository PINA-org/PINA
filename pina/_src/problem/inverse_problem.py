"""Module for the InverseProblem class."""

from abc import abstractmethod
import torch
from pina._src.problem.base_problem import BaseProblem


class InverseProblem(BaseProblem):
    """
    Base class for all inverse problems, extending the standard problem
    definition with unknown parameters to be determined through training.

    An inverse problem is defined by a set of unknown parameters that need to be
    estimated from observed data.

    This class is not meant to be instantiated directly.

    :Example:

        >>> import torch
        >>> from pina.problem import InverseProblem
        >>> from pina.domain import CartesianDomain
        >>> class MyInverseProblem(InverseProblem):
        ...     output_variables = ["u"]
        ...     @property
        ...     def unknown_parameter_domain(self):
        ...         return CartesianDomain({"k": [0.1, 5.0]})
        ...     @property
        ...     def conditions(self): return {}
        >>> problem = MyInverseProblem()
        >>> problem.unknown_variables
        ['k']
    """

    def __init__(self):
        """
        Initialization of the :class:`InverseProblem` class.

        :raises KeyError: If an unknown variable has no ranged bounds in
            ``unknown_parameter_domain._range``.

        .. note::

            Creates an independent one-element trainable parameter per unknown,
            uniformly initialized between its bounds. This constructor does
            not enforce bounds after initialization; solvers may clamp them.
        """
        super().__init__()

        # Set the unknown parameters as trainable parameters
        self.unknown_parameters = {}
        for var in self.unknown_variables:
            low, high = self.unknown_parameter_domain._range[var]
            tensor_var = low + (high - low) * torch.rand(1)
            self.unknown_parameters[var] = torch.nn.Parameter(tensor_var)

    @property
    @abstractmethod
    def unknown_parameter_domain(self):
        """
        The domain of the unknown parameters of the problem.

        :return: The primitive domain whose variables all have ranged bounds.
            Initialization reads its private ``_range`` mapping directly.
        :rtype: pina.domain.BaseDomain
        """

    @property
    def unknown_variables(self):
        """
        The unknown variables of the problem.

        :return: The unknown variables of the problem.
        :rtype: list[str]
        """
        return self.unknown_parameter_domain.variables

    @property
    def unknown_parameters(self):
        """
        The unknown parameters of the problem.

        :return: The stored mapping from unknown names to trainable one-element
            parameters. Mutations affect the problem directly.
        :rtype: dict[str, torch.nn.Parameter]
        """
        return self.__unknown_parameters

    @unknown_parameters.setter
    def unknown_parameters(self, value):
        """
        Set the unknown parameters of the problem.

        :param dict[str, torch.nn.Parameter] value: The new unknown-parameter
            mapping, stored by reference without validation or conversion.
        """
        self.__unknown_parameters = value
