"""Module for defining the fixed gradient equation."""

from pina._src.equation.equation import Equation
from pina._src.core.operator import grad
from pina._src.core.utils import check_consistency


class FixedGradient(Equation):
    """
    Equation to enforce a fixed gradient for a specific condition.

    :Example:

        >>> from pina.equation import FixedGradient
        >>> eq = FixedGradient(value=0.0)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedGradient` class.

        :param float value: The fixed value to be enforced to the gradient.
        :param components: The name of the output variables for which the fixed
            gradient condition is applied. It should be a subset of the output
            labels. If ``None``, all output variables are considered. Default is
            ``None``.
        :type components: str | list[str]
        :param d: The name of the input variables on which the gradient is
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
            Definition of the equation to enforce a fixed gradient.

            :param LabelTensor input_: The input points where the residual is
                computed.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :return: The residual values of the equation.
            :rtype: LabelTensor
            """
            return grad(output_, input_, components=components, d=d) - value

        super().__init__(equation)
