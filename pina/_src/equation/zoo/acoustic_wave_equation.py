"""Module for defining the acoustic wave equation."""

from pina._src.equation.equation import Equation
from pina._src.core.operator import laplacian
from pina._src.core.utils import check_consistency


class AcousticWaveEquation(Equation):
    r"""
    Implementation of the N-dimensional isotropic acoustic wave equation.
    The equation is defined as follows:

    .. math::

        \frac{\partial^2 u}{\partial t^2} - c^2 \Delta u = 0

    or alternatively:

    .. math::

        \Box u = 0

    Here, :math:`c` is the wave propagation speed, and :math:`\Box` is the
    d'Alembert operator.

    :Example:

        >>> from pina.equation.zoo import AcousticWaveEquation
        >>> eq = AcousticWaveEquation(c=1.0)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, c):
        """
        Initialization of the :class:`AcousticWaveEquation` class.

        :param c: The wave propagation speed.
        :type c: float | int
        :raises ValueError: If numeric type validation fails for ``c``.
        """
        check_consistency(c, (float, int))
        self.c = c

        def equation(input_, output_):
            """
            Implementation of the acoustic wave equation.

            :param LabelTensor input_: Points of shape ``(n, d + 1)`` with
                gradient tracking and time label ``"t"``. Every other label
                is treated as a spatial coordinate.
            :param LabelTensor output_: Predictions of shape ``(n, m)``, with
                an autograd connection supporting second derivatives.
            :raises ValueError: If the ``input_`` labels do not contain the time
                variable ``"t"``.
            :raises TypeError: If arguments are not labeled tensors or an
                unused gradient cannot be indexed.
            :raises RuntimeError: If the graph cannot support differentiation.
            :return: Time second derivatives minus ``c ** 2`` times spatial
                Laplacians, with shape ``(n, m)``.
            :rtype: LabelTensor
            """
            # Ensure time is passed as input
            if "t" not in input_.labels:
                raise ValueError(
                    "The ``input_`` labels must contain the time 't' variable."
                )

            # Compute the time second derivative and the spatial laplacian
            u_tt = laplacian(output_, input_, d=["t"])
            u_xx = laplacian(
                output_, input_, d=[di for di in input_.labels if di != "t"]
            )

            return u_tt - self.c**2 * u_xx

        super().__init__(equation)
