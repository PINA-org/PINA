"""Module for defining the Helmholtz equation."""

from typing import Callable
from pina._src.equation.equation import Equation
from pina._src.core.operator import laplacian
from pina._src.core.utils import check_consistency


class HelmholtzEquation(Equation):
    r"""
    Implementation of the Helmholtz equation, defined as follows:

    .. math::

            \Delta u + k u - f = 0

    Here, :math:`k` is the squared wavenumber, while :math:`f` is the
    forcing term.

    :Example:

        >>> from pina.equation import HelmholtzEquation
        >>> eq = HelmholtzEquation(k=1.0, forcing_term=lambda x: x**2)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, k, forcing_term):
        """
        Initialization of the :class:`HelmholtzEquation` class.

        :param k: The squared wavenumber.
        :type k: float | int
        :param Callable forcing_term: The forcing field function, taking as
            input the points on which evaluation is required.
        """
        check_consistency(k, (int, float))
        check_consistency(forcing_term, (Callable))
        self.k = k
        self.forcing_term = forcing_term

        def equation(input_, output_):
            """
            Implementation of the Helmholtz equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the Helmholtz equation.
            :rtype: LabelTensor
            """
            lap = laplacian(output_, input_)
            return lap + self.k * output_ - self.forcing_term(input_)

        super().__init__(equation)
