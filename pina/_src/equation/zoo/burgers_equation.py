"""Module for defining the Burgers equation."""

from pina._src.core.operator import laplacian, grad
from pina._src.core.utils import check_consistency
from pina._src.equation.equation import Equation
import torch


class BurgersEquation(Equation):
    r"""
    Implementation of the N-dimensional Burgers' equation, defined as follows:

    .. math::

        \frac{\partial u}{\partial t} + u \cdot \nabla u = \nu \Delta u

    Here, :math:`\nu` is the viscosity coefficient.

    :Example:

        >>> from pina.equation import BurgersEquation
        >>> eq = BurgersEquation(nu=0.01)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, nu):
        """
        Initialization of the :class:`BurgersEquation` class.

        :param nu: The viscosity coefficient.
        :type nu: float | int
        :raises ValueError: If ``nu`` is not a float or an int.
        :raises ValueError: If ``nu`` is negative.
        """
        # Check consistency
        check_consistency(nu, (float, int))
        if nu < 0:
            raise ValueError(
                "The viscosity ``nu`` must be a non-negative float or int."
            )

        # Store viscosity coefficient
        self.nu = nu

        def equation(input_, output_):
            """
            Implementation of the Burgers' equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :raises ValueError: If the number of output components does not
                match the number of spatial dimensions.
            :raises ValueError: If the ``input_`` labels do not contain the time
                variable 't'.
            :return: The residual of the Burgers' equation.
            :rtype: LabelTensor
            """
            # Store labels
            spatial_d = [di for di in input_.labels if di != "t"]

            # Ensure consistency between output and spatial dimensions
            if len(output_.labels) != len(spatial_d):
                raise ValueError(
                    f"The number of output components must match the number of "
                    f"spatial dimensions. Got {len(output_.labels)} and "
                    f"{len(spatial_d)}."
                )

            # Ensure time is passed as input
            if "t" not in input_.labels:
                raise ValueError(
                    "The ``input_`` labels must contain the time 't' variable."
                )

            # Compute the differential terms
            u_t = grad(output_, input_, d=["t"])
            u_x = grad(output_, input_, d=spatial_d)
            u_xx = laplacian(output_, input_, d=spatial_d)

            # Compute the convective term componentwise
            convection = torch.zeros_like(output_)
            for i, c in enumerate(output_.labels):
                convection[:, i] = sum(
                    output_[output_.labels[j]] * u_x[f"d{c}d{spatial_d[j]}"]
                    for j in range(len(spatial_d))
                ).reshape(-1)

            return u_t + convection - self.nu * u_xx

        super().__init__(equation)
