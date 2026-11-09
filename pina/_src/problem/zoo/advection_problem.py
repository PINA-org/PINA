"""Formulation of the advection problem."""

import torch
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.domain.cartesian_domain import CartesianDomain
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.equation.zoo.advection_equation import AdvectionEquation
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.core.utils import check_consistency
from pina._src.equation.equation import Equation


def initial_condition(input_, output_):
    """
    Implementation of the initial condition.

    :param LabelTensor input_: Points of shape ``(n, d)`` including ``"x"``.
    :param LabelTensor output_: Predictions of shape ``(n, 1)``.
    :raises ValueError: If the spatial label is missing.
    :return: Predictions minus ``sin(x)``, of shape ``(n, 1)`` for scalar
        predictions.
    :rtype: LabelTensor
    """
    return output_ - torch.sin(input_.extract("x"))


class AdvectionProblem(SpatialProblem, TimeDependentProblem):
    r"""
    Implementation of the one-dimensional advection problem on the space-time
    domain :math:`\Omega\times T = [0, 2\pi] \times [0, 1]`.

    The problem is governed by the linear advection equation

    .. math::

        \frac{\partial u}{\partial t}
        +
        c \frac{\partial u}{\partial x}
        =
        0,

    where :math:`u = u(x, t)` is the solution field and :math:`c` is the
    advection velocity.

    The analytical profile is periodic at the spatial boundaries:

    .. math::

        u(0, t) = u(2\pi, t), \qquad t \in [0, 1].

    No periodic boundary condition is installed by this class; only the PDE
    and initial condition are added. Enforce periodicity separately if needed.

    The initial condition is prescribed as

    .. math::

        u(x, 0) = \sin(x), \qquad x \in [0, 2\pi].

    The analytical solution is given by

    .. math::

        u(x, t) = \sin(x - ct).

    .. seealso::

        **Original reference**: Wang, Sifan, et al. *An expert's guide to
        training physics-informed neural networks*.
        arXiv preprint arXiv:2308.08468 (2023).
        DOI: `arXiv:2308.08468  <https://arxiv.org/abs/2308.08468>`_.

    :Example:

        >>> from pina.problem.zoo import AdvectionProblem
        >>> problem = AdvectionProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 2 * torch.pi]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
    }

    conditions = {
        "t0": DomainEquationCondition(
            domain="t0", equation=Equation(initial_condition)
        ),
    }

    def __init__(self, c=1.0):
        """
        Initialization of the :class:`AdvectionProblem` class.

        :param c: The advection velocity parameter. Default is ``1.0``.
        :type c: float | int
        :raises ValueError: If numeric validation rejects ``c``.
        """
        super().__init__()
        check_consistency(c, (float, int))
        self.c = c

        self.conditions["D"] = DomainEquationCondition(
            domain="D", equation=AdvectionEquation(self.c)
        )

    def solution(self, pts):
        """
        Implementation of the analytical solution of the advection problem.

        :param LabelTensor pts: Points of shape ``(n, 2)`` labeled ``"x"`` and
            ``"t"``, in either column order.
        :raises ValueError: If a required coordinate label is missing.
        :return: The sine-wave profile of shape ``(n, 1)``, labeled ``"u"``.
        :rtype: LabelTensor
        """
        sol = torch.sin(pts.extract("x") - self.c * pts.extract("t"))
        sol.labels = self.output_variables
        return sol
