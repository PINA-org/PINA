"""Module for defining the fixed flux equation."""

from pina._src.equation.equation import Equation
from pina._src.core.operator import div
from pina._src.core.utils import check_consistency


class FixedFlux(Equation):
    """
    Equation to enforce a fixed flux, or divergence, for a specific condition.

    This constrains divergence, not a boundary-normal flux integral.

    :Example:

        >>> from pina.equation.zoo import FixedFlux
        >>> eq = FixedFlux(value=0.0)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedFlux` class.

        :param value: The fixed value to be enforced to the flux.
        :type value: float | int
        :param components: The name of the output variables for which the fixed
            flux condition is applied. It should be a subset of the output
            labels. ``None`` or an empty list selects all output variables.
            The selected count must match the coordinate count at evaluation.
            Default is ``None``.
        :type components: str | list[str] | None
        :param d: The name of the input variables on which the flux is computed.
            It should be a subset of the input labels. If ``None``, all the
            input variables are considered; an empty list does the same.
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
            Definition of the equation to enforce a fixed flux.

            :param LabelTensor input_: The labeled input points of shape
                ``(n, n_inputs)``, with gradient tracking enabled.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :raises TypeError: If either argument is not a labeled tensor or
                differentiation returns an unused gradient.
            :raises RuntimeError: If selected labels are missing or autograd
                cannot differentiate the supplied graph.
            :raises ValueError: If component and coordinate counts differ.
            :return: Paired derivative sum minus the fixed value, with shape
                ``(n, 1)`` and the label assigned by :func:`pina.operator.div`.
            :rtype: LabelTensor
            """
            return div(output_, input_, components=components, d=d) - value

        super().__init__(equation)
