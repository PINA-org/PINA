"""Formulation of the Burgers' problem."""

import torch
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.domain.cartesian_domain import CartesianDomain
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.core.utils import check_consistency
from pina._src.equation.equation import Equation
from pina._src.equation.zoo.fixed_value import FixedValue
from pina._src.equation.zoo.burgers_equation import BurgersEquation


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the Burgers' problem.

    :param LabelTensor input_: The input data of the problem.
    :param LabelTensor output_: The output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    return output_ + torch.sin(torch.pi * input_["x"])


class BurgersProblem(TimeDependentProblem, SpatialProblem):
    r"""
    Implementation of the one-dimensional Burgers' problem on the space-time
    domain :math:`\Omega\times T = [-1, 1] \times [0, 1]`.

    The problem is governed by the Burgers' equation

    .. math::

        \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} =
        \nu \frac{\partial^2 u}{\partial x^2},

    where :math:`u = u(x, t)` is the solution field and :math:`\nu \geq 0`
    is the viscosity coefficient. For :math:`\nu = 0`, the equation reduces
    to the inviscid Burgers' equation.

    Homogeneous Dirichlet boundary conditions are imposed at the spatial
    boundaries:

    .. math::
        u(-1, t) = u(1, t) = 0, \qquad t \in [0, 1].

    The initial condition is prescribed as

    .. math::
        u(x, 0) = -\sin(\pi x), \qquad x \in [-1, 1].


    .. seealso::

        **Original reference**: Raissi M., Perdikaris P., Karniadakis G. E.
        (2017).
        *Physics Informed Deep Learning (Part I): Data-driven Solutions of
        Nonlinear Partial Differential Equations*.
        DOI: `10.48550 <https://doi.org/10.48550/arXiv.1711.10561>`_.

    :Example:

        >>> problem = BurgersProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
        "boundary": spatial_domain.partial().update(temporal_domain),
    }

    conditions = {
        "boundary": DomainEquationCondition(
            domain="boundary", equation=FixedValue(0.0)
        ),
        "t0": DomainEquationCondition(
            domain="t0", equation=Equation(initial_condition)
        ),
    }

    def __init__(self, nu=0):
        """
        Initialization of the :class:`BurgersProblem` class.

        :param nu: The viscosity coefficient.
        :type nu: float | int
        :raises ValueError: If ``nu`` is not a float or an int.
        :raises ValueError: If ``nu`` is negative.
        """
        super().__init__()

        # Check consistency
        check_consistency(nu, (float, int))
        if nu < 0:
            raise ValueError(
                "The viscosity ``nu`` must be a non-negative float or int."
            )

        self.conditions["D"] = DomainEquationCondition(
            domain="D", equation=BurgersEquation(nu)
        )
