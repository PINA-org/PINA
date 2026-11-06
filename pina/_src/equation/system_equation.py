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
    :class:`~pina.condition.DomainEquationCondition`
    or :class:`~pina.condition.InputEquationCondition`
    object to define the conditions under which the model is trained.

    Each equation in the system must be either an instance of
    :class:`~pina.equation.Equation`, or a callable function.

    Residuals are computed independently for each equation and then aggregated
    using an optional reduction (e.g., ``mean``, ``sum``). The final result is
    returned as a single tensor, with no new residual labels assigned.

    :Example:

        >>> import torch
        >>> from pina import LabelTensor
        >>> from pina.equation import SystemEquation
        >>> from pina.equation.zoo import FixedValue, FixedGradient
        >>> pts = LabelTensor(torch.ones(3, 2), ["x", "y"]).requires_grad_()
        >>> output_ = LabelTensor(pts ** 2, ["u", "v"])
        >>> system_equation = SystemEquation(
        ...     [FixedValue(1., ["u"]), FixedGradient(0., ["v"], ["y"])],
        ...     reduction="mean",
        ... )
        >>> residual = system_equation.residual(pts, output_)
        >>> tuple(residual.shape)
        (3,)
        >>> torch.allclose(residual.tensor, torch.ones(3))
        True
    """

    def __init__(self, list_equation, reduction=None):
        """
        Initialization of the :class:`SystemEquation` class.

        :param list[Callable | Equation] list_equation: The equations used to
            compute residuals. Each element can be either a callable
            function or a :class:`~pina.equation.Equation` instance, including
            its subclasses. Callables are wrapped in ``Equation``; arbitrary
            non-callable ``BaseEquation`` instances are not accepted.
            An empty list is accepted but cannot produce a residual.
        :param reduction: The method used to combine the residuals from each
            equation. Available options are: ``None``, ``"mean"``, ``"sum"``, or
            a custom callable. If ``None``, no reduction is applied. If
            ``"mean"``, the residuals are averaged. If ``"sum"``, the residuals
            are summed. If a callable is provided, it is used as a custom
            reduction called as ``reduction(residuals, dim=-1)``. Only
            callability is checked for custom reductions. Default is ``None``.
        :type reduction: str | Callable | None
        :raises ValueError: If the list of equations is not a list.
        :raises ValueError: If any element of the list of equations is not a
            callable function or a :class:`~pina.equation.Equation`
            instance.
        :raises ValueError: If an invalid reduction method is used.
        :raises ValueError: If a constituent callable cannot be inspected.
        :raises TypeError: If signature inspection is unsupported for a
            constituent callable.
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

        :param LabelTensor input_: Labeled input points, normally of shape
            ``(n, n_inputs)``. Differential equations require gradient tracking.
        :param LabelTensor output_: Labeled predictions, normally of shape
            ``(n, n_outputs)``, connected to ``input_`` when differentiated.
        :param params_: The named unknown parameters for an
            :class:`~pina.problem.InverseProblem`, forwarded to every equation.
            ``None`` is forwarded unchanged when supplied. Default is ``None``.
        :type params_: dict[str, torch.nn.Parameter] | None
        :raises ValueError: If the system is empty.
        :raises RuntimeError: If constituent tensor shapes cannot be
            concatenated along their last dimension.
        :raises TypeError: If constituents do not return tensors or a custom
            reduction cannot accept the ``dim`` keyword.
        :return: Residuals concatenated in equation order, normally with shape
            ``(n, sum(n_residuals))`` when unreduced. Built-in reductions remove
            the last axis, giving shape ``(n,)``. Custom reductions determine
            their own result type and shape; no new labels are assigned.
            Constituent and reduction exceptions propagate unchanged.
        :rtype: Any
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
