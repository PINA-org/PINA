"""Formulation of the Helmholtz problem."""

import torch
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.equation.zoo.fixed_value import FixedValue
from pina._src.equation.zoo.helmholtz_equation import HelmholtzEquation
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.core.utils import check_consistency
from pina._src.domain.cartesian_domain import CartesianDomain


class HelmholtzProblem(SpatialProblem):
    r"""
    Implementation of the two-dimensional Helmholtz problem on the square domain
    :math:`\Omega = [-1, 1] \times [-1, 1]`.

    The problem is governed by the forced Helmholtz equation

    .. math::

        \Delta u + k u = f(x, y),

    where :math:`u = u(x, y)` is the solution field, :math:`k` is the squared
    wavenumber, and :math:`f(x, y)` is a forcing term.

    Homogeneous Dirichlet boundary conditions are imposed on the boundary of
    the domain:

    .. math::

        u(x, y) = 0, \qquad (x, y) \in \partial \Omega.

    The analytical solution is given by

    .. math::

        u(x, y)
        =
        \sin(\alpha_x \pi x)
        \sin(\alpha_y \pi y),

    with forcing term

    .. math::

        f(x, y)
        =
        \left[
        k - (\alpha_x^2 + \alpha_y^2)\pi^2
        \right]
        \sin(\alpha_x \pi x)
        \sin(\alpha_y \pi y).

    .. seealso::

        **Original reference**: Si, Chenhao, et al. *Complex Physics-Informed
        Neural Network.* arXiv preprint arXiv:2502.04917 (2025).
        DOI: `arXiv:2502.04917 <https://arxiv.org/abs/2502.04917>`_.

    :Example:

        >>> from pina.problem.zoo import HelmholtzProblem
        >>> problem = HelmholtzProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1], "y": [-1, 1]})

    domains = {
        "D": spatial_domain,
        "boundary": spatial_domain.partial(),
    }

    conditions = {
        "boundary": DomainEquationCondition(
            domain="boundary", equation=FixedValue(0.0)
        ),
    }

    def __init__(self, k=1.0, alpha_x=1, alpha_y=4):
        """
        Initialization of the :class:`HelmholtzProblem` class.

        :param k: The squared wavenumber. Default is ``1.0``.
        :type k: float | int
        :param int alpha_x: The frequency in the x-direction. Default is ``1``.
        :param int alpha_y: The frequency in the y-direction. Default is ``4``.
        :raises ValueError: If numeric validation rejects ``k`` or either
            frequency is not an integer.
        """
        super().__init__()
        check_consistency(k, (int, float))
        check_consistency(alpha_x, int)
        check_consistency(alpha_y, int)
        self.k = k
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        def forcing_term(input_):
            """
            Implementation of the forcing term.

            :param LabelTensor input_: Points of shape ``(n, 2)`` including
                ``"x"`` and ``"y"`` coordinate labels.
            :raises ValueError: If a required coordinate label is missing.
            :return: The forcing matching the sine-product solution, with shape
                ``(n, 1)``.
            :rtype: LabelTensor
            """
            x, y, pi = input_["x"], input_["y"], torch.pi
            factor = (self.alpha_x**2 + self.alpha_y**2) * pi**2
            return (
                (self.k - factor)
                * torch.sin(self.alpha_x * pi * x)
                * torch.sin(self.alpha_y * pi * y)
            )

        self.conditions["D"] = DomainEquationCondition(
            domain="D",
            equation=HelmholtzEquation(self.k, forcing_term),
        )

    def solution(self, pts):
        """
        Implementation of the analytical solution of the Helmholtz problem.

        :param LabelTensor pts: Points of shape ``(n, 2)`` labeled ``"x"`` and
            ``"y"``, in either column order.
        :raises ValueError: If a required coordinate label is missing.
        :return: The sine-product solution of shape ``(n, 1)``, labeled ``"u"``.
        :rtype: LabelTensor
        """
        x, y, pi = pts["x"], pts["y"], torch.pi
        sol = torch.sin(self.alpha_x * pi * x) * torch.sin(
            self.alpha_y * pi * y
        )
        sol.labels = self.output_variables
        return sol
