"""Module for the System of Equation."""

from typing import Callable
import torch
from pina._src.equation.base_equation import BaseEquation
from pina._src.core.utils import check_consistency
from pina._src.equation.equation import Equation


class SystemEquation(BaseEquation):
    """
    Implementation of the SystemEquation class, representing a system of
    mathematical equation to be satisfied by the model outputs. It is useful for
    multi-component outputs or coupled problems, where multiple constraints must
    be evaluated together.

    It can be passed to a
    :class:`~pina.condition.domain_equation_condition.DomainEquationCondition`
    object to define the conditions under which the model is trained.

    Each equation in the system must be either an instance of
    :class:`~pina.equation.equation.Equation`, or a callable function.

    Residuals are computed independently for each equation and then aggregated
    using an optional reduction (e.g., ``mean``, ``sum``). The final result is
    returned as a single :class:`~pina.LabelTensor`.

    :Example:

    >>> pts = LabelTensor(torch.rand(10, 2), labels=["x", "y"])
    >>> pts.requires_grad = True
    >>> output_ = torch.pow(pts, 2)
    >>> output_.labels = ["u", "v"]
    >>> system_equation = SystemEquation(
    ...     [
    ...         FixedValue(value=1.0, components=["u"]),
    ...         FixedGradient(value=0.0, components=["v"], d=["y"]),
    ...     ],
    ...     reduction="mean",
    ... )
    >>> residual = system_equation.residual(pts, output_)
    """

    def __init__(self, list_equation, reduction=None):
        """
        Initialization of the :class:`SystemEquation` class.

        :param list_equation: The list of equations used for the computation of
            the residuals. Each element of the list can be either a callable
            function or a :class:`~pina.equation.equation.Equation` instance.
        :type list_equation: list[Callable] | list[Equation]
        :param reduction: The method used to combine the residuals from each
            equation. Available options are: ``None``, ``"mean"``, ``"sum"``, or
            a custom callable. If ``None``, no reduction is applied. If
            ``"mean"``, the residuals are averaged. If ``"sum"``, the residuals
            are summed. If a callable is provided, it is used as a custom
            reduction (no validation is performed).
        :raises ValueError: If the list of equations is not a list.
        :raises ValueError: If any element of the list of equations is not a
            callable function or a :class:`~pina.equation.equation.Equation`
            instance.
        :raises ValueError: If an invalid reduction method is used.
        """
        # Check consistency
        check_consistency([list_equation], list)
        check_consistency(list_equation, (Callable, Equation))

        # Convert all callable functions to Equation instances, if necessary
        self.equations = [
            equation if isinstance(equation, Equation) else Equation(equation)
            for equation in list_equation
        ]

        # Validate and set the reduction method
        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif (reduction is None) or callable(reduction):
            self.reduction = reduction
        else:
            raise ValueError(
                "Invalid reduction method. Available options include: None, "
                "'mean', 'sum', or a custom callable."
            )

    def residual(self, input_, output_, params_=None):
        """
        Evaluate each equation residual from the system of equations at the
        given inputs and aggregate it according to the specified ``reduction``.

        :param LabelTensor input_: The input points where the residual is
            computed.
        :param LabelTensor output_: The output tensor, potentially produced by a
            :class:`torch.nn.Module` instance.
        :param dict params_: An optional dictionary of unknown parameters, used
            in :class:`~pina.problem.inverse_problem.InverseProblem` settings.
            If the equation is not related to an inverse problem, this should be
            set to ``None``. Default is ``None``.
        :return: The aggregated residuals of the system of equations.
        :rtype: LabelTensor
        """
        # Move the equation to the input_ device
        self.to(input_.device)

        # Compute the residual for each equation
        residual = torch.cat(
            [
                equation.residual(input_, output_, params_)
                for equation in self.equations
            ],
            dim=-1,
        )

        # Skip reduction if not specified
        if self.reduction is None:
            return residual

        return self.reduction(residual, dim=-1)
