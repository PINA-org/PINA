"""Module for defining the fixed value equation."""

from pina._src.equation.equation import Equation
from pina._src.core.utils import check_consistency


class FixedValue(Equation):
    """
    Equation to enforce a fixed value. Can be used to enforce Dirichlet Boundary
    conditions.

    :Example:

        >>> from pina.equation import FixedValue
        >>> eq = FixedValue(value=0.0)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, value, components=None):
        """
        Initialization of the :class:`FixedValue` class.

        :param value: The fixed value to be enforced.
        :type value: float | int
        :param components: The name of the output variables for which the fixed
            value condition is applied. It should be a subset of the output
            labels. If ``None``, all output variables are considered. Default is
            ``None``.
        :type components: str | list[str]
        :raises ValueError: If ``value`` is neither a float nor an integer.
        :raises ValueError: If, when provided, ``components`` is neither a
            string nor a list of strings.
        """
        # Check consistency
        check_consistency(value, (float, int))
        if components is not None:
            check_consistency(components, str)

        def equation(_, output_):
            """
            Definition of the equation to enforce a fixed value.

            :param LabelTensor input_: The input points where the residual is
                computed.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :return: The residual values of the equation.
            :rtype: LabelTensor
            """
            if components is None:
                return output_ - value
            return output_.extract(components) - value

        super().__init__(equation)
