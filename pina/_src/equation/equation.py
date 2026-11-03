"""Module for the Equation."""

import inspect
from pina._src.equation.base_equation import BaseEquation


class Equation(BaseEquation):
    """
    Implementation of the Equation class, representing a single mathematical
    equation to be satisfied by the model outputs.

    It can be passed to a
    :class:`~pina.condition.domain_equation_condition.DomainEquationCondition`
    or :class:`~pina.condition.input_equation_condition.InputEquationCondition`
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

        :param Callable equation: A callable function used to compute the
            residual of a mathematical equation.
        :raises ValueError: If the equation is not a callable function.
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

        :param LabelTensor input_: The input points where the residual is
            computed.
        :param LabelTensor output_: The output tensor, potentially produced by a
            :class:`torch.nn.Module` instance.
        :param dict params_: An optional dictionary of unknown parameters, used
            in :class:`~pina.problem.inverse_problem.InverseProblem` settings.
            If the equation is not related to an inverse problem, this should be
            set to ``None``. Default is ``None``.
        :raises RuntimeError: If the underlying equation signature is neither of
            length 2 for direct problems nor of length 3 for inverse problems.
        :return: The residual values of the equation.
        :rtype: LabelTensor
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
