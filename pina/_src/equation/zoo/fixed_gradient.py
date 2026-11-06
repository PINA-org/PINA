"""Module for defining the fixed gradient equation."""

from pina._src.equation.equation import Equation
from pina._src.core.operator import grad
from pina._src.core.utils import check_consistency


class FixedGradient(Equation):
    """
    Equation to enforce a fixed gradient for a specific condition.

    :Example:

        >>> from pina.equation.zoo import FixedGradient
        >>> eq = FixedGradient(value=0.0)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedGradient` class.

        :param value: The value subtracted from every selected derivative.
        :type value: float | int
        :param components: The name of the output variables for which the fixed
            gradient condition is applied. It should be a subset of the output
            labels. ``None`` or an empty list selects all output variables.
            Tuples and sets pass construction but fail nonempty operator
            selection at evaluation. Default is ``None``.
        :type components: str | list[str] | None
        :param d: The name of the input variables on which the gradient is
            computed. It should be a subset of the input labels. If ``None``,
            all the input variables are considered; an empty list does the same.
            Default is ``None``.
        :type d: str | list[str] | None
        :raises ValueError: If ``value`` is neither a float nor an integer.
        :raises ValueError: If component or coordinate validation rejects
            non-string entries.
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

            :param LabelTensor input_: The labeled input points of shape
                ``(n, n_inputs)``, with gradient tracking enabled.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :raises TypeError: If either argument is not a labeled tensor or
                differentiation returns an unused gradient.
            :raises RuntimeError: If selected labels are missing or autograd
                cannot differentiate the supplied graph.
            :raises ValueError: If derivative labels cannot be assigned.
            :return: Selected derivatives minus the fixed value, with shape
                ``(n, n_components * n_coordinates)`` in component-major order
                and labels assigned by :func:`pina.operator.grad`.
            :rtype: LabelTensor
            """
            return grad(output_, input_, components=components, d=d) - value

        super().__init__(equation)
