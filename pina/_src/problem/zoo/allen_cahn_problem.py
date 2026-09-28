"""Formulation of the Allen Cahn problem."""

import torch
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.equation.equation import Equation
from pina._src.equation.zoo.allen_cahn_equation import AllenCahnEquation
from pina._src.core.utils import check_consistency
from pina._src.domain.cartesian_domain import CartesianDomain


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the Allen Cahn problem.

    :param LabelTensor input_: The input data of the problem.
    :param LabelTensor output_: The output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    x = input_.extract("x")
    u_0 = x**2 * torch.cos(torch.pi * x)
    return output_ - u_0


class AllenCahnProblem(TimeDependentProblem, SpatialProblem):
    r"""
    Implementation of the one-dimensional Allen-Cahn problem on the space-time
    domain :math:`\Omega\times T = [-1, 1] \times [0, 1]`.

    The problem is governed by the Allen-Cahn equation

    .. math::

        \frac{\partial u}{\partial t}
        -
        \alpha \frac{\partial^2 u}{\partial x^2}
        +
        \beta \left(u^3 - u\right)
        =
        0,

    where :math:`u = u(x, t)` is the solution field, :math:`\alpha` is the
    diffusion coefficient, and :math:`\beta` is the reaction coefficient.

    Periodic boundary conditions are imposed at the spatial boundaries:

    .. math::

        u(-1, t) = u(1, t), \qquad t \in [0, 1].

    The initial condition is prescribed as

    .. math::

        u(x, 0) = x^2 \cos(\pi x), \qquad x \in [-1, 1].

    .. seealso::

        **Original reference**: Sokratis J. Anagnostopoulos, Juan D. Toscano,
        Nikolaos Stergiopulos, and George E. Karniadakis.
        *Residual-based attention and connection to information
        bottleneck theory in PINNs*.
        Computer Methods in Applied Mechanics and Engineering 421 (2024): 116805
        DOI: `10.1016/
        j.cma.2024.116805 <https://doi.org/10.1016/j.cma.2024.116805>`_.

    :Example:

        >>> problem = AllenCahnProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
    }

    conditions = {
        "t0": DomainEquationCondition(domain="t0", equation=Equation(initial_condition)),
    }

    def __init__(self, alpha=1e-4, beta=5):
        """
        Initialization of the :class:`AllenCahnProblem`.

        :param alpha: The diffusion coefficient. Default is ``1e-4``.
        :type alpha: float | int
        :param beta: The reaction coefficient. Default is ``5.0``.
        :type beta: float | int
        """
        super().__init__()
        check_consistency(alpha, (float, int))
        check_consistency(beta, (float, int))
        self.alpha = alpha
        self.beta = beta

        self.conditions["D"] = DomainEquationCondition(
            domain="D",
            equation=AllenCahnEquation(alpha=self.alpha, beta=self.beta),
        )
