"""Module for defining the fixed value equation."""

from pina._src.equation.equation import Equation
from pina._src.core.utils import check_consistency


class FixedValue(Equation):
    """
    Equation to enforce a fixed value. Can be used to enforce Dirichlet Boundary
    conditions.

    :Example:

        >>> from pina.equation.zoo import FixedValue
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
        :type components: str | list[str] | tuple[str, ...] | None
        :raises ValueError: If ``value`` is neither a float nor an integer.
        :raises ValueError: If, when provided, ``components`` is neither a
            string nor a list, tuple, or set of strings. Sets pass constructor
            validation but cannot be extracted when evaluating the residual.
        """
        # Check consistency
        check_consistency(value, (float, int))
        if components is not None:
            check_consistency(components, str)

        def equation(_, output_):
            """
            Definition of the equation to enforce a fixed value.

            :param LabelTensor _: The unused input points.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :raises ValueError: If selected output labels are unknown, repeated,
                or in an unsupported selection format.
            :raises IndexError: If the selected component sequence is empty.
            :return: Selected outputs minus the fixed value, with shape
                ``(n, n_selected)`` and the selected output labels retained.
            :rtype: LabelTensor
            """
            if components is None:
                return output_ - value
            return output_.extract(components) - value

        super().__init__(equation)
