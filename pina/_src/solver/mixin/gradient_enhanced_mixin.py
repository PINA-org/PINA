"""Module for the gradient-enhanced mixin class."""

import torch
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.core.utils import check_consistency
from pina._src.core.operator import grad


class GradientEnhancedMixin:
    """
    Mixin that augments residual losses with a gradient-based regularization
    term.

    The additional penalty is the norm of the residual gradient with respect
    to the spatial input variables. It is only applied to the conditions whose
    names are listed in ``regularized_conditions``.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver` and using
    :class:`~pina.LabelTensor` as input.
    """

    def _init_gradient_enhanced_components(
        self, regularization_weight=1.0, regularized_conditions=None
    ):
        """
        Initialize the gradient-enhancement parameters.

        :param regularization_weight: The weight of the gradient regularization
            term. Default is ``1.0``.
        :type regularization_weight: float | int
        :param regularized_conditions: The names of the conditions that should
            receive gradient regularization. If ``None``, all conditions are
            regularized. Default is ``None``.
        :type regularized_conditions: str | list[str] | tuple[str, ...] | None
        :raises ValueError: If ``regularization_weight`` is not a float or int.
        :raises ValueError: If ``regularized_conditions`` is not a string or a
            list or tuple of strings.
        :raises ValueError: If ``problem`` is not an instance of
            :class:`~pina._src.problem.spatial_problem.SpatialProblem`.
        :raises ValueError: If the solver's input data are not instances of
            labeled tensors, as indicated by ``use_lt=False``. Actual data
            tensors are checked later by the differential operator.
        :raises ValueError: If any of the specified ``regularized_conditions``
            are not present in the ``problem``'s conditions.
        """
        # Use all conditions if regularized_conditions is None
        if regularized_conditions is None:
            regularized_conditions = list(self.problem.conditions.keys())

        # Check consistency
        check_consistency(regularization_weight, (float, int))
        check_consistency(regularized_conditions, str)

        # Map conditions to list if a single condition is provided
        if not isinstance(regularized_conditions, (list, tuple)):
            regularized_conditions = [regularized_conditions]

        # Assert the problem is instance of SpatialProblem
        if not isinstance(self.problem, SpatialProblem):
            raise ValueError(
                "Gradient-enhanced regularization requires the problem to be "
                f"an instance of SpatialProblem. Got {type(self.problem)}."
            )

        # Ensure that the solver is using LabelTensors as input
        if not self.use_lt:
            raise ValueError(
                "Gradient-enhanced regularization requires the solver to use "
                f"LabelTensors as input. Got use_lt={self.use_lt}."
            )

        # Ensure that all regularized conditions are present in the problem
        problem_conditions = set(self.problem.conditions.keys())
        for condition in regularized_conditions:
            if condition not in problem_conditions:
                raise ValueError(
                    f"Condition '{condition}' is not present in the problem."
                )

        # Initialize the gradient-enhancement parameters
        self.regularization_weight = regularization_weight
        self.regularized_conditions = regularized_conditions

    def _prepare_condition_data(self, data):
        """
        Enable gradient tracking on an input tensor in place when needed.

        :param dict[str, Any] data: The original condition payload, optionally
            containing an ``"input"`` tensor. Applies to every condition, not
            only selected regularization conditions.
        :raises RuntimeError: If the input dtype cannot require gradients.
        :return: The same dictionary, with its input tensor possibly changed.
        :rtype: dict[str, Any]
        """
        # If data does not require grad, force requires_grad to True
        if "input" in data and not data["input"].requires_grad:
            data["input"].requires_grad_(True)

        return data

    def _regularize_condition_loss(
        self,
        condition_tensor_loss,
        condition_name,
        data,
        batch_idx,
    ):
        """
        Add residual-gradient loss for selected conditions.

        Replaces residual labels with ``"res_i"`` names before differentiation
        in the problem's spatial coordinates, then applies the same loss module
        against zero to the gradient tensor.

        :param condition_tensor_loss: The original tensor loss for the
            condition.
        :type condition_tensor_loss: torch.Tensor | LabelTensor
        :param condition_name: The condition name; ``None`` is not selected.
        :type condition_name: str | None
        :param dict[str, Any] data: The condition payload with labeled input.
        :param int batch_idx: The batch index, unused by this hook.
        :raises KeyError: If selected data lacks ``"input"``.
        :raises RuntimeError: If differentiation or addition of differently
            shaped base/gradient losses fails.
        :raises TypeError: If a residual/input is not labeled or an unused
            autograd gradient cannot be indexed.
        :return: The original loss plus ``regularization_weight`` times gradient
            loss, or the original tensor for an unselected condition. Loss
            shapes must be compatible for broadcasting.
        :rtype: torch.Tensor | LabelTensor
        """
        # Regularize the loss with the gradient penalty if needed
        if condition_name in self.regularized_conditions:

            # Apply labels to the residual tensor for gradient computation
            self.residual_tensor.labels = [
                f"res_{i}" for i in range(self.residual_tensor.shape[1])
            ]

            # Compute the gradient of the residual with respect to spatial input
            residual_gradient = grad(
                output_=self.residual_tensor,
                input_=data["input"],
                d=self.problem.spatial_variables,
            )

            # Compute the norm of the residual gradient
            residual_gradient_norm = self._loss_fn(
                residual_gradient, torch.zeros_like(residual_gradient)
            )

            # Compute the gradient penalty term
            penalty = self.regularization_weight * residual_gradient_norm

            # Add the gradient penalty to the original condition tensor loss
            condition_tensor_loss = condition_tensor_loss + penalty

        return condition_tensor_loss
