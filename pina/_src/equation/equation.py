"""Module for the Equation."""

import inspect
from pina._src.equation.base_equation import BaseEquation


class Equation(BaseEquation):
    """
    Implementation of the Equation class, representing a single mathematical
    equation to be satisfied by the model outputs.

    It can be passed to a
    :class:`~pina.condition.DomainEquationCondition`
    or :class:`~pina.condition.InputEquationCondition`
    object to define the conditions under which the model is trained.

    :Example:

        >>> from pina.equation import Equation
        >>> eq = Equation(lambda x, u: u - x)
        >>> # Use within a DomainEquationCondition or InputEquationCondition:
        >>> # cond_domain = DomainEquationCondition(domain=domain, equation=eq)
        >>> # cond_input = InputEquationCondition(input=pts, equation=eq)
    """

    def __init__(self, equation):
        """
        Initialization of the :class:`Equation` class.

        :param Callable equation: The residual callable, accepting positional
            ``(input_, output_)`` or ``(input_, output_, params_)``. Dispatch
            uses the number of inspected signature parameters, not their names
            or kinds. The result is returned without conversion or validation.
        :raises ValueError: If the equation is not a callable function.
        :raises ValueError: If :func:`inspect.signature` cannot inspect it.
        :raises TypeError: If signature inspection is unsupported for it.
        """
        # Check consistency
        if not callable(equation):
            raise ValueError(f"Expected a callable function, got {equation}")

        # Compute the signature length
        sig = inspect.signature(equation)
        self.__len_sig = len(sig.parameters)
        self.__equation = equation

    def residual(self, input_, output_, params_=None):
        """
        Evaluate the equation residual at the given inputs.

        Stored tensor attributes are moved to the input device before dispatch;
        callable closures and parameter dictionaries are not moved here.

        :param LabelTensor input_: Labeled input points, normally of shape
            ``(n, n_inputs)``. Differential equations require gradient tracking.
        :param LabelTensor output_: Labeled predictions, normally of shape
            ``(n, n_outputs)``, connected to ``input_`` when differentiated.
        :param params_: The named unknown parameters for an
            :class:`~pina.problem.InverseProblem`. Passed unchanged to
            three-parameter callables, including when ``None``; ignored by
            two-parameter callables. Default is ``None``.
        :type params_: dict[str, torch.nn.Parameter] | None
        :raises RuntimeError: If the underlying equation signature is neither of
            length 2 for direct problems nor of length 3 for inverse problems.
        :raises TypeError: If the callable cannot accept positional dispatch.
        :return: The callable's unchanged result, normally a residual tensor of
            shape ``(n, n_residuals)``. Result type and labels are not enforced;
            exceptions raised by the callable propagate unchanged.
        :rtype: Any
        """
        # Move the equation to the input_ device
        self.to(input_.device)

        # Evaluate the equation for direct problems
        if self.__len_sig == 2:
            return self.__equation(input_, output_)

        # Evaluate the equation for inverse problems
        if self.__len_sig == 3:
            return self.__equation(input_, output_, params_)

        # Raise an error if the signature length is unexpected
        raise RuntimeError(
            f"Unexpected number of arguments in equation: {self.__len_sig}. "
            "Expected either 2 for direct problems, or 3  for inverse problems."
        )
