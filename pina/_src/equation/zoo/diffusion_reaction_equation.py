"""Module for defining the Diffusion-Reaction equation."""

from typing import Callable
from pina._src.equation.equation import Equation
from pina._src.core.operator import grad, laplacian
from pina._src.core.utils import check_consistency


class DiffusionReactionEquation(Equation):
    r"""
    Implementation of the N-dimensional Diffusion-Reaction equation,
    defined as follows:

    .. math::

        \frac{\partial u}{\partial t} - \alpha \Delta u - f = 0

    Here, :math:`\alpha` is a parameter of the equation, while :math:`f` is the
    reaction term.

    :Example:

        >>> from pina.equation import DiffusionReactionEquation
        >>> eq = DiffusionReactionEquation(alpha=1.0, forcing_term=lambda x: x**2)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, alpha, forcing_term):
        """
        Initialization of the :class:`DiffusionReactionEquation` class.

        :param alpha: The diffusion coefficient.
        :type alpha: float | int
        :param Callable forcing_term: The forcing field function, taking as
            input the points on which evaluation is required.
        """
        check_consistency(alpha, (float, int))
        check_consistency(forcing_term, (Callable))
        self.alpha = alpha
        self.forcing_term = forcing_term

        def equation(input_, output_):
            """
            Implementation of the Diffusion-Reaction equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the Diffusion-Reaction equation.
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

            return u_t - self.alpha * u_xx - self.forcing_term(input_)

        super().__init__(equation)
