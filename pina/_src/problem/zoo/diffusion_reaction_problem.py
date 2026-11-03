"""Formulation of the diffusion-reaction problem."""

import torch
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.equation.equation import Equation
from pina._src.equation.zoo.fixed_value import FixedValue
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.core.utils import check_consistency
from pina._src.domain.cartesian_domain import CartesianDomain
from pina._src.equation.zoo.diffusion_reaction_equation import (
    DiffusionReactionEquation,
)


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the diffusion-reaction problem.

    :param LabelTensor input_: The input data of the problem.
    :param LabelTensor output_: The output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    x = input_.extract("x")
    u_0 = (
        torch.sin(x)
        + (1 / 2) * torch.sin(2 * x)
        + (1 / 3) * torch.sin(3 * x)
        + (1 / 4) * torch.sin(4 * x)
        + (1 / 8) * torch.sin(8 * x)
    )
    return output_ - u_0


class DiffusionReactionProblem(TimeDependentProblem, SpatialProblem):
    r"""
    Implementation of the one-dimensional diffusion-reaction problem on the
    space-time domain :math:`\Omega\times T = [-\pi, \pi] \times [0, 1]`.

    The problem is governed by the forced diffusion-reaction equation

    .. math::

        \frac{\partial u}{\partial t}
        -
        \alpha \frac{\partial^2 u}{\partial x^2}
        =
        f(x, t),

    where :math:`u = u(x, t)` is the solution field, :math:`\alpha` is the
    diffusion coefficient, and :math:`f(x, t)` is a forcing term.

    Homogeneous Dirichlet boundary conditions are imposed at the spatial
    boundaries:

    .. math::

        u(-\pi, t) = u(\pi, t) = 0, \qquad t \in [0, 1].

    The initial condition is prescribed as

    .. math::

        u(x, 0)
        =
        \sin(x)
        +
        \frac{1}{2}\sin(2x)
        +
        \frac{1}{3}\sin(3x)
        +
        \frac{1}{4}\sin(4x)
        +
        \frac{1}{8}\sin(8x).

    The analytical solution is given by

    .. math::

        u(x, t)
        =
        e^{-t}
        \left(
        \sin(x)
        +
        \frac{1}{2}\sin(2x)
        +
        \frac{1}{3}\sin(3x)
        +
        \frac{1}{4}\sin(4x)
        +
        \frac{1}{8}\sin(8x)
        \right).

    .. seealso::

        **Original reference**: Si, Chenhao, et al. *Complex Physics-Informed
        Neural Network.* arXiv preprint arXiv:2502.04917 (2025).
        DOI: `arXiv:2502.04917 <https://arxiv.org/abs/2502.04917>`_.

    :Example:

        >>> problem = DiffusionReactionProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-torch.pi, torch.pi]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "boundary": spatial_domain.partial().update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
    }

    conditions = {
        "boundary": DomainEquationCondition(
            domain="boundary", equation=FixedValue(0.0)
        ),
        "t0": DomainEquationCondition(
            domain="t0", equation=Equation(initial_condition)
        ),
    }

    def __init__(self, alpha=1e-4):
        """
        Initialization of the :class:`DiffusionReactionProblem`.

        :param alpha: The diffusion coefficient. Default is ``1e-4``.
        :type alpha: float | int
        """
        super().__init__()
        check_consistency(alpha, (float, int))
        self.alpha = alpha

        def forcing_term(input_):
            """
            Implementation of the forcing term.
            """
            # Extract spatial and temporal variables
            spatial_d = [di for di in input_.labels if di != "t"]
            x = input_.extract(spatial_d)
            t = input_.extract("t")

            return torch.exp(-t) * (
                (self.alpha - 1) * torch.sin(x)
                + ((4 * self.alpha - 1) / 2) * torch.sin(2 * x)
                + ((9 * self.alpha - 1) / 3) * torch.sin(3 * x)
                + ((16 * self.alpha - 1) / 4) * torch.sin(4 * x)
                + ((64 * self.alpha - 1) / 8) * torch.sin(8 * x)
            )

        self.conditions["D"] = DomainEquationCondition(
            domain="D",
            equation=DiffusionReactionEquation(self.alpha, forcing_term),
        )

    def solution(self, pts):
        """
        Implementation of the analytical solution of the diffusion-reaction
        problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the diffusion-reaction problem.
        :rtype: LabelTensor
        """
        t = pts.extract("t")
        x = pts.extract("x")
        sol = torch.exp(-t) * (
            torch.sin(x)
            + (1 / 2) * torch.sin(2 * x)
            + (1 / 3) * torch.sin(3 * x)
            + (1 / 4) * torch.sin(4 * x)
            + (1 / 8) * torch.sin(8 * x)
        )
        sol.labels = self.output_variables
        return sol
