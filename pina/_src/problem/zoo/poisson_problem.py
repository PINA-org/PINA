"""Formulation of the Poisson problem in a square domain."""

import torch

from pina._src.equation.zoo.fixed_value import FixedValue
from pina._src.domain.cartesian_domain import CartesianDomain
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.equation.zoo.poisson_equation import PoissonEquation


def forcing_term(input_):
    """
    Implementation of the forcing term of the Poisson problem.

    :param LabelTensor input_: The points where the forcing term is evaluated.
    :return: The forcing term of the Poisson problem.
    :rtype: LabelTensor
    """
    return (
        torch.sin(input_.extract(["x"]) * torch.pi)
        * torch.sin(input_.extract(["y"]) * torch.pi)
        * (2 * torch.pi**2)
    )


class Poisson2DSquareProblem(SpatialProblem):
    r"""
    Implementation of the two-dimensional Poisson problem on the square domain
    :math:`\Omega = [0, 1] \times [0, 1]`.

    The problem is governed by the Poisson equation

    .. math::

        \Delta u = f(x, y),

    where :math:`u = u(x, y)` is the solution field and :math:`f(x, y)` is the
    forcing term.

    Homogeneous Dirichlet boundary conditions are imposed on the boundary of the
    domain:

    .. math::

        u(x, y) = 0, \qquad (x, y) \in \partial \Omega.

    The forcing term is given by

    .. math::

        f(x, y)
        =
        2\pi^2 \sin(\pi x)\sin(\pi y).

    The analytical solution is given by

    .. math::

        u(x, y)
        =
        -\sin(\pi x)\sin(\pi y).

    :Example:

        >>> problem = Poisson2DSquareProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

    domains = {
        "D": spatial_domain,
        "boundary": spatial_domain.partial(),
    }

    conditions = {
        "boundary": DomainEquationCondition(
            domain="boundary", equation=FixedValue(0.0)
        ),
        "D": DomainEquationCondition(
            domain="D", equation=PoissonEquation(forcing_term=forcing_term)
        ),
    }

    def solution(self, pts):
        """
        Implementation of the analytical solution of the Poisson problem.

        :param LabelTensor pts: The points where the solution is evaluated.
        :return: The analytical solution of the Poisson problem.
        :rtype: LabelTensor
        """
        sol = -(
            torch.sin(pts.extract(["x"]) * torch.pi)
            * torch.sin(pts.extract(["y"]) * torch.pi)
        )
        sol.labels = self.output_variables
        return sol
