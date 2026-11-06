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

    The Laplacian uses every input coordinate and every output component;
    pass only spatial coordinates when no time derivative is intended.

    :Example:

        >>> from pina.equation.zoo import PoissonEquation
        >>> eq = PoissonEquation(forcing_term=lambda x: x**2)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, forcing_term):
        """
        Initialization of the :class:`PoissonEquation` class.

        :param Callable forcing_term: The forcing field function, taking as
            input the full labeled points. Its numeric or tensor result must
            broadcast with the componentwise Laplacian, normally ``(n, m)``.
        :raises ValueError: If ``forcing_term`` is not a callable instance.
        """
        check_consistency(forcing_term, (Callable))
        self.forcing_term = forcing_term

        def equation(input_, output_):
            """
            Implementation of the Poisson equation.

            :param LabelTensor input_: Input coordinates of shape ``(n, d)``,
                with gradient tracking enabled.
            :param LabelTensor output_: Predictions of shape ``(n, m)``, with
                an autograd connection supporting second derivatives.
            :raises TypeError: If arguments are not labeled tensors or an
                unused gradient cannot be indexed.
            :raises RuntimeError: If the graph cannot support differentiation
                or forcing values cannot broadcast with the Laplacian.
            :return: The Laplacian minus forcing, normally of shape ``(n, m)``.
            :rtype: LabelTensor
            """
            lap = laplacian(output_, input_)
            return lap - self.forcing_term(input_)

        super().__init__(equation)
