"""Module for defining the fixed laplacian equation."""

import warnings
from pina._src.equation.equation import Equation
from pina._src.core.operator import laplacian
from pina._src.core.utils import check_consistency


class FixedLaplacian(Equation):
    """
    Equation to enforce a fixed laplacian for a specific condition.

    :Example:

        >>> from pina.equation import FixedLaplacian
        >>> eq = FixedLaplacian(value=0.0)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedLaplacian` class.

        :param value: The fixed value to be enforced to the laplacian.
        :type value: float | int
        :param components: The name of the output variables for which the fixed
            laplace condition is applied. It should be a subset of the output
            labels. If ``None``, all output variables are considered. Default is
            ``None``.
        :type components: str | list[str]
        :param d: The name of the input variables on which the laplacian is
            computed. It should be a subset of the input labels. If ``None``,
            all the input variables are considered. Default is ``None``.
        :type d: str | list[str]
        :raises ValueError: If ``value`` is neither a float nor an integer.
        :raises ValueError: If, when provided, ``components`` is neither a
            string nor a list of strings.
        :raises ValueError: If, when provided, ``d`` is neither a string nor a
            list of strings.
        """
        # Check consistency
        check_consistency(value, (float, int))
        if components is not None:
            check_consistency(components, str)
        if d is not None:
            check_consistency(d, str)

        def equation(input_, output_):
            """
            Definition of the equation to enforce a fixed laplacian.

            :param LabelTensor input_: The input points where the residual is
                computed.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :return: The residual values of the equation.
            :rtype: LabelTensor
            """
            return (
                laplacian(output_, input_, components=components, d=d) - value
            )

        super().__init__(equation)


# Back-compatibility with version 0.2, to be removed soon
class Laplace(FixedLaplacian):
    """
    Deprecated wrapper for :class:`FixedLaplacian` with ``value=0.0``.

    .. deprecated:: 0.3
        Use :class:`FixedLaplacian` with ``value=0.0`` instead.
    """

    def __init__(self, components=None, d=None):
        warnings.warn(
            "Laplace is deprecated, use FixedLaplacian with value=0.0 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(0.0, components=components, d=d)
