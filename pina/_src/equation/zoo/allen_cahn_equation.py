"""Module for defining the Allen-Cahn equation."""

from pina._src.equation.equation import Equation
from pina._src.core.operator import grad, laplacian
from pina._src.core.utils import check_consistency


class AllenCahnEquation(Equation):
    r"""
    Implementation of the N-dimensional Allen-Cahn equation, defined as follows:

    .. math::

        \frac{\partial u}{\partial t} - \alpha \Delta u + \beta(u^3 - u) = 0

    Here, :math:`\alpha` and :math:`\beta` are parameters of the equation.

    :Example:

        >>> from pina.equation import AllenCahnEquation
        >>> eq = AllenCahnEquation(alpha=1.0, beta=1.0)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, alpha, beta):
        """
        Initialization of the :class:`AllenCahnEquation` class.

        :param alpha: The diffusion coefficient.
        :type alpha: float | int
        :param beta: The reaction coefficient.
        :type beta: float | int
        """
        check_consistency(alpha, (float, int))
        check_consistency(beta, (float, int))
        self.alpha = alpha
        self.beta = beta

        def equation(input_, output_):
            """
            Implementation of the Allen-Cahn equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the Allen-Cahn equation.
            :rtype: LabelTensor
            :raises ValueError: If the ``input_`` labels do not contain the time
                variable 't'.
            """
            # Ensure time is passed as input
            if "t" not in input_.labels:
                raise ValueError(
                    "The ``input_`` labels must contain the time 't' variable."
                )

            # Compute the time derivative and the spatial laplacian
            u_t = grad(output_, input_, d=["t"])
            u_xx = laplacian(
                output_, input_, d=[di for di in input_.labels if di != "t"]
            )

            return u_t - self.alpha * u_xx + self.beta * (output_**3 - output_)

        super().__init__(equation)
