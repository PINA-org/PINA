"""Module for defining the advection equation."""

import torch
from pina._src.equation.equation import Equation
from pina._src.core.operator import grad
from pina._src.core.utils import check_consistency


class AdvectionEquation(Equation):
    r"""
    Implementation of the N-dimensional advection equation with constant
    velocity parameter. The equation is defined as follows:

    .. math::

        \frac{\partial u}{\partial t} + c \cdot \nabla u = 0

    Here, :math:`c` is the advection velocity parameter.

    :Example:

        >>> from pina.equation import AdvectionEquation
        >>> eq = AdvectionEquation(c=1.0)
        >>> # Use within a DomainEquationCondition:
        >>> # condition = DomainEquationCondition(domain=domain, equation=eq)
    """

    def __init__(self, c):
        """
        Initialization of the :class:`AdvectionEquation` class.

        :param c: The advection velocity. If a scalar is provided, the same
            velocity is applied to all spatial dimensions. If a list is
            provided, it must contain one value per spatial dimension.
        :type c: float | int | List[float] | List[int]
        :raises ValueError: If ``c`` is an empty list.
        """
        # Check consistency
        check_consistency(c, (float, int))
        if isinstance(c, list):
            if len(c) < 1:
                raise ValueError("'c' cannot be an empty list.")
        else:
            c = [c]

        # Store advection velocity parameter
        self.c = torch.tensor(c).unsqueeze(0)

        def equation(input_, output_):
            """
            Implementation of the advection equation.

            :param LabelTensor input_: The input data of the problem.
            :param LabelTensor output_: The output data of the problem.
            :return: The residual of the advection equation.
            :rtype: LabelTensor
            :raises ValueError: If the ``input_`` labels do not contain the time
                variable 't'.
            :raises ValueError: If ``c`` is a list and its length is not
                consistent with the number of spatial dimensions.
            """
            # Store labels
            input_lbl = input_.labels
            spatial_d = [di for di in input_lbl if di != "t"]

            # Ensure time is passed as input
            if "t" not in input_lbl:
                raise ValueError(
                    "The ``input_`` labels must contain the time 't' variable."
                )

            # Ensure consistency of c length
            if self.c.shape[-1] != len(input_lbl) - 1 and self.c.shape[-1] > 1:
                raise ValueError(
                    "If 'c' is passed as a list, its length must be equal to "
                    "the number of spatial dimensions."
                )

            # Repeat c to ensure consistent shape for advection
            c = self.c.repeat(output_.shape[0], 1)
            if c.shape[1] != (len(input_lbl) - 1):
                c = c.repeat(1, len(input_lbl) - 1)

            # Add a dimension to c for the following operations
            c = c.unsqueeze(-1)

            # Compute the time derivative and the spatial gradient
            time_der = grad(output_, input_, components=None, d="t")
            grads = grad(output_=output_, input_=input_, d=spatial_d)

            # Reshape and transpose
            tmp = grads.reshape(*output_.shape, len(spatial_d))
            tmp = tmp.transpose(-1, -2)

            # Compute advection term
            adv = (tmp * c).sum(dim=tmp.tensor.ndim - 2)

            return time_der + adv

        super().__init__(equation)
