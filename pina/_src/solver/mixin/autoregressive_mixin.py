"""Module for the autoregressive mixin class."""

import torch

from pina._src.core.utils import check_consistency


class AutoregressiveMixin:
    """
    Mixin that enables the autoregressive rollout loss logic by maintaining a
    running average of step losses and computing adaptive weights for each step
    based on the cumulative loss. This allows the solver to focus more on steps
    that are currently underperforming, which can help improve training
    stability and convergence.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def _init_autoregressive_components(
        self, eps, reset_weights_at_epoch_start
    ):
        """
        Initialize the components related to the autoregressive rollout loss.

        :param eps: The hyperparameter controlling the influence of the
            cumulative loss on the adaptive weights. Higher values of eps will
            lead to more aggressive weighting of steps with higher cumulative
            loss.
        :type eps: float | int
        :param bool reset_weights_at_epoch_start: Whether to reset the running
            average and step count at the start of each epoch. If ``True``, the
            adaptive weights will be recalibrated at the beginning of each epoch
            based on the new training dynamics.
        :raises ValueError: If ``eps`` is not a float or int.
        :raises ValueError: If ``reset_weights_at_epoch_start`` is not a bool.
        """
        # Check consistency
        check_consistency(eps, (float, int))
        check_consistency(reset_weights_at_epoch_start, bool)

        # Initialize the components for autoregressive rollout loss
        self.reset_weights_at_epoch_start = reset_weights_at_epoch_start
        self.eps = eps
        self._running_avg = {}
        self._step_count = {}

    def _loss_from_residual(self, condition_name=None):
        """
        Compute the tensor loss from the residual tensor.

        :param str condition_name: The name of the condition.
        :return: The tensor loss computed from the residual tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        # Compute the step losses from the residual tensor
        step_loss = self._loss_fn(
            self.residual_tensor, torch.zeros_like(self.residual_tensor)
        )

        # Retrieve the temporal adaptive weights for the current step losses
        with torch.no_grad():
            weights = self._get_weights(condition_name or "default", step_loss)

        return step_loss * weights

    def _get_weights(self, condition_name, step_loss):
        """
        Get temporal adaptive weights for each step based on the running average
        of step losses.

        :param str condition_name: The name of the condition.
        :param step_loss: The tensor of step losses for the current condition.
        :type step_loss: torch.Tensor | LabelTensor
        :return: The tensor of adaptive weights for each step.
        :rtype: torch.Tensor | LabelTensor
        """
        # Use the condition name for tracking the running average and step count
        key = condition_name or "default"
        reduce_dims = tuple(range(1, step_loss.dim()))
        step_loss = step_loss.detach().mean(dim=reduce_dims, keepdim=True)

        # Update the running average and step count for the current condition
        if key not in self._running_avg:
            self._running_avg[key] = step_loss.detach().clone()
            self._step_count[key] = 1
        else:
            self._step_count[key] += 1
            value = step_loss.detach() - self._running_avg[key]
            self._running_avg[key] += value / self._step_count[key]

        return self._compute_adaptive_weights(self._running_avg[key])

    def _compute_adaptive_weights(self, step_loss):
        """
        Compute the adaptive weights for each step based on the cumulative loss.

        :param step_loss: The tensor of step losses for the current condition.
        :type step_loss: torch.Tensor | LabelTensor
        :return: The tensor of adaptive weights for each step.
        :rtype: torch.Tensor | LabelTensor
        """
        cumulative_loss = -self.eps * torch.cumsum(step_loss, dim=0)
        return torch.exp(cumulative_loss)

    def on_train_epoch_start(self):
        """
        Clear the running average and step count at the start of each epoch if
        ``reset_weights_at_epoch_start`` is ``True``.
        """
        if self.reset_weights_at_epoch_start:
            self._running_avg.clear()
            self._step_count.clear()
