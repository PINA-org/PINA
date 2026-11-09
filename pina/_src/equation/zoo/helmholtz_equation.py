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

    The Laplacian uses every input coordinate and every output component.

    :Example:

        >>> from pina.equation.zoo import HelmholtzEquation
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
            input the full labeled points. Its numeric or tensor result must
            broadcast with the componentwise Laplacian, normally ``(n, m)``.
        :raises ValueError: If ``k`` is not numeric or ``forcing_term`` is not
            a callable instance. The sign of ``k`` is not restricted.
        """
        check_consistency(k, (int, float))
        check_consistency(forcing_term, (Callable))
        self.k = k
        self.forcing_term = forcing_term

        def equation(input_, output_):
            """
            Implementation of the Helmholtz equation.

            :param LabelTensor input_: Input coordinates of shape ``(n, d)``,
                with gradient tracking enabled.
            :param LabelTensor output_: Predictions of shape ``(n, m)``, with
                an autograd connection supporting second derivatives.
            :raises TypeError: If arguments are not labeled tensors or an
                unused gradient cannot be indexed.
            :raises RuntimeError: If differentiation or forcing broadcasting
                fails.
            :return: The Laplacian plus ``k * output_`` minus forcing, normally
                of shape ``(n, m)``. The stored ``k`` is not squared again.
            :rtype: LabelTensor
            """
            lap = laplacian(output_, input_)
            return lap + self.k * output_ - self.forcing_term(input_)

        super().__init__(equation)
