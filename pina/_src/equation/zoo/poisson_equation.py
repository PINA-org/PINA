"""Module for defining the Poisson equation."""

from typing import Callable
from pina._src.equation.equation import Equation
from pina._src.core.operator import laplacian
from pina._src.core.utils import check_consistency


class PoissonEquation(Equation):
    r"""
    Implementation of the Poisson equation, defined as follows:

    .. math::

            \Delta u - f = 0

    Here, :math:`f` is the forcing term.

    :Example:

        >>> from pina.equation import PoissonEquation
        >>> eq = PoissonEquation(forcing_term=lambda x: x**2)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, forcing_term):
        """
        Initialization of the :class:`PoissonEquation` class.

        :param Callable forcing_term: The forcing field function, taking as
            input the points on which evaluation is required.
        """
        check_consistency(forcing_term, (Callable))
        self.forcing_term = forcing_term

        def equation(input_, output_):
            """
            Implementation of the Poisson equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the Poisson equation.
            :rtype: LabelTensor
            """
            lap = laplacian(output_, input_)
            return lap - self.forcing_term(input_)

        super().__init__(equation)
