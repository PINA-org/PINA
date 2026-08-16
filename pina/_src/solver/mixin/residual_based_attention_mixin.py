"""Module for the residual-based attention mixin class."""

import torch
from pina._src.core.utils import check_consistency
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class ResidualBasedAttentionMixin:
    """
    Mixin that augments the residual loss with an attention mechanism based on
    the residual values.

    The attention weights are computed as a function of the residuals, and they
    are used to weight the contribution of each condition to the overall loss.
    This allows the solver to focus more on conditions with larger residuals,
    potentially improving convergence and accuracy.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def _init_residual_attention_components(
        self, eta=0.001, gamma=0.999, regularized_conditions=None
    ):
        """
        Initialize the residual-based attention parameters.

        :param eta: The learning rate for the residual-based attention weights
            update. Default is ``0.001``.
        :type eta: float | int
        :param float gamma: The decay factor for the residual-based attention
            mechanism. Default is ``0.999``.
        :param regularized_conditions: The names of the conditions that should
            receive attention regularization. If ``None``, all conditions are
            regularized. Default is ``None``.
        :type regularized_conditions: str | list[str]
        :raises ValueError: If ``eta`` is not a positive float or int.
        :raises ValueError: If ``gamma`` is not a float in the range (0, 1).
        :raises ValueError: If ``regularized_conditions`` is not a string or a
            list of strings.
        :raises ValueError: If any of the specified ``regularized_conditions``
            are not present in the ``problem``'s conditions.
        """
        # Use all conditions if regularized_conditions is None
        if regularized_conditions is None:
            regularized_conditions = list(self.problem.conditions.keys())

        # Check consistency
        check_consistency(eta, (float, int))
        check_consistency(gamma, float)
        check_consistency(regularized_conditions, str)

        # Assert gamma is in range (0, 1)
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in range (0, 1)")

        # Assert eta is positive
        if eta <= 0:
            raise ValueError("eta must be positive")

        # Map conditions to list if a single condition is provided
        if not isinstance(regularized_conditions, (list, tuple)):
            regularized_conditions = [regularized_conditions]

        # Ensure that all regularized conditions are present in the problem
        problem_conditions = set(self.problem.conditions.keys())
        for condition in regularized_conditions:
            if condition not in problem_conditions:
                raise ValueError(
                    f"Condition '{condition}' is not present in the problem."
                )

        # Initialize residual-based attention parameters
        self.regularized_conditions = regularized_conditions
        self.gamma = gamma
        self.eta = eta
        self.weight_buffers = {}

        # Iterate over all conditions to initialize the attention weights
        for cond in self.regularized_conditions:

            # Get the condition object
            condition = self.problem.conditions[cond]

            # Determine the number of points in the condition
            if isinstance(condition, DomainEquationCondition):
                n_pts = self.problem._discretised_domains[cond].shape[0]
            else:
                n_pts = condition.data.input.shape[0]

            # Register the attention weights as a buffer in the module
            self.register_buffer(f"weight_{cond}", torch.zeros((n_pts, 1)))
            self.weight_buffers[cond] = f"weight_{cond}"

    def _weight_condition_loss(
        self,
        condition_tensor_loss,
        condition_name,
        data,
        batch_idx,
    ):
        """
        Apply residual-based attention weighting to the condition loss.

        :param condition_tensor_loss: The tensor loss for the condition.
        :type condition_tensor_loss: torch.Tensor | LabelTensor
        :param str condition_name: The name of the condition.
        :param dict data: The data corresponding to the condition.
        :param int batch_idx: The index of the current batch.
        :return: The weighted tensor loss for the condition.
        :rtype: torch.Tensor | LabelTensor
        """
        # Apply residual-based attention mechanism if needed
        if condition_name in self.regularized_conditions:

            # Compute the normalized residuals norm for the current condition
            res_abs = torch.linalg.vector_norm(
                self.residual_tensor, ord=2, dim=1, keepdim=True
            )
            res_norm = res_abs / (res_abs.max() + 1e-12)

            # Get the correct indices to retrieve the weights for the batch
            len_residuals = self.residual_tensor.shape[0]

            # Get the weights buffer for the current condition
            weights = getattr(self, self.weight_buffers[condition_name])

            # Get the total number of points, together with the start / end idx
            total_points = weights.shape[0]
            start = (batch_idx * len_residuals) % total_points
            end = start + len_residuals

            # Retrieve the weights for the current batch using modular indexing
            idx = torch.arange(start, end, device=weights.device)
            idx = idx % total_points

            # Update weights
            with torch.no_grad():
                weights[idx] = self.gamma * weights[idx] + self.eta * res_norm

            # Weight the condition tensor loss with attention weights
            condition_tensor_loss = condition_tensor_loss * weights[idx]

        return condition_tensor_loss
