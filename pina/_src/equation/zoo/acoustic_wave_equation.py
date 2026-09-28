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

        >>> from pina.equation import AcousticWaveEquation
        >>> eq = AcousticWaveEquation(c=1.0)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, c):
        """
        Initialization of the :class:`AcousticWaveEquation` class.

        :param c: The wave propagation speed.
        :type c: float | int
        """
        check_consistency(c, (float, int))
        self.c = c

        def equation(input_, output_):
            """
            Implementation of the acoustic wave equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the acoustic wave equation.
            :rtype: LabelTensor
            :raises ValueError: If the ``input_`` labels do not contain the time
                variable 't'.
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
