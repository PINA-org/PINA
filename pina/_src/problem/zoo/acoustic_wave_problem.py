"""Formulation of the acoustic wave problem."""

import torch
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.domain.cartesian_domain import CartesianDomain
from pina._src.equation.system_equation import SystemEquation
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.core.utils import check_consistency
from pina._src.equation.equation import Equation
from pina._src.equation.zoo.fixed_value import FixedValue
from pina._src.equation.zoo.fixed_gradient import FixedGradient
from pina._src.equation.zoo.acoustic_wave_equation import AcousticWaveEquation


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the acoustic wave problem.

    :param LabelTensor input_: The input data of the problem.
    :param LabelTensor output_: The output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    arg = torch.pi * input_["x"]
    return output_ - torch.sin(arg) - 0.5 * torch.sin(4 * arg)


class AcousticWaveProblem(TimeDependentProblem, SpatialProblem):
    r"""
    Implementation of the one-dimensional acoustic wave problem on the
    space-time domain :math:`\Omega\times T = [0, 1] \times [0, 1]`.

    The problem is governed by the acoustic wave equation

    .. math::

        \frac{\partial^2 u}{\partial t^2}
        =
        c^2 \frac{\partial^2 u}{\partial x^2},

    where :math:`u = u(x, t)` is the solution field and :math:`c > 0` is the
    wave propagation speed.

    Homogeneous Dirichlet boundary conditions are imposed at the spatial
    boundaries:

    .. math::

        u(0, t) = u(1, t) = 0, \qquad t \in [0, 1].

    The initial displacement is prescribed as

    .. math::

        u(x, 0) = \sin(\pi x) + \frac{1}{2}\sin(4\pi x),
        \qquad x \in [0, 1],

    together with zero initial velocity:

    .. math::

        \frac{\partial u}{\partial t}(x, 0) = 0,
        \qquad x \in [0, 1].

    The analytical solution is given by

    .. math::

        u(x, t)
        =
        \sin(\pi x)\cos(c\pi t)
        +
        \frac{1}{2}\sin(4\pi x)\cos(4c\pi t).

    .. seealso::

        **Original reference**: Wang, Sifan, Xinling Yu, and
        Paris Perdikaris. *When and why PINNs fail to train:
        A neural tangent kernel perspective*. Journal of
        Computational Physics 449 (2022): 110768.
        DOI: `10.1016 <https://doi.org/10.1016/j.jcp.2021.110768>`_.

    :Example:

        >>> problem = AcousticWaveProblem(c=2.0)
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
        "boundary": spatial_domain.partial().update(temporal_domain),
    }

    conditions = {
        "boundary": DomainEquationCondition(domain="boundary", equation=FixedValue(0.0)),
        "t0": DomainEquationCondition(
            domain="t0",
            equation=SystemEquation(
                [Equation(initial_condition), FixedGradient(0.0, d="t")]
            ),
        ),
    }

    def __init__(self, c=2.0):
        """
        Initialization of the :class:`AcousticWaveProblem` class.

        :param c: The wave propagation speed. Default is ``2.0``.
        :type c: float | int
        """
        super().__init__()
        check_consistency(c, (float, int))
        self.c = c

        self.conditions["D"] = DomainEquationCondition(
            domain="D", equation=AcousticWaveEquation(self.c)
        )

    def solution(self, pts):
        """
        Implementation of the analytical solution of the acoustic wave problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the acoustic wave problem.
        :rtype: LabelTensor
        """
        arg_x = torch.pi * pts["x"]
        arg_t = self.c * torch.pi * pts["t"]
        term1 = torch.sin(arg_x) * torch.cos(arg_t)
        term2 = 0.5 * torch.sin(4 * arg_x) * torch.cos(4 * arg_t)

        sol = term1 + term2
        sol.labels = self.output_variables
        return sol
