"""Module for the autoregressive mixin class."""

import torch

from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency


def _concat_state_control(state, control):
    """
    Concatenate state and control along the feature dimension.

    :param state: The current state.
    :type state: torch.Tensor | LabelTensor
    :param control: The control, or ``None``.
    :type control: torch.Tensor | LabelTensor | None
    :return: The concatenated input. Labels are preserved when both inputs are
        :class:`~pina.label_tensor.LabelTensor` instances.
    :rtype: torch.Tensor | LabelTensor
    """
    if control is None:
        return state

    model_input = torch.cat([state, control], dim=-1)

    # Restore labels when both inputs are LabelTensor instances
    if isinstance(state, LabelTensor) and isinstance(control, LabelTensor):
        model_input = model_input.as_subclass(LabelTensor)
        model_input.labels = state.labels + control.labels

    return model_input


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
        self, eps, reset_weights_at_epoch_start, kwargs
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
        :param dict kwargs: Additional keyword arguments for the autoregressive
            rollout.
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
        self._kwargs = kwargs or {}

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

    def rollout_step(self, state, control=None, **kwargs):
        """
        Perform a single autoregressive rollout step.

        This method defines how the current state and control are combined and
        passed to the model. Override it to customize the rollout convention,
        for example to update a graph attribute before the forward pass.

        :param state: The current state.
        :type state: torch.Tensor | LabelTensor
        :param control: The control at the current step. Defaults to ``None``.
        :type control: torch.Tensor | LabelTensor | None
        :param dict kwargs: Additional keyword arguments. Recognized keys are
            ``graph`` and ``key`` for graph-based rollouts. When ``graph`` is
            provided, the concatenated state and control are stored on the
            graph under ``key`` before the forward pass.
        :return: The next state.
        :rtype: torch.Tensor | LabelTensor
        """
        graph = kwargs.get("graph")
        key = kwargs.get("key", "x")

        # Graph rollout: update the graph attribute before the forward pass
        if graph is not None:
            setattr(graph, key, _concat_state_control(state, control))
            return self.forward(graph) + state

        # Tensor rollout: concatenate state and control
        return self.forward(_concat_state_control(state, control))

    def rollout(self, initial_state, n_steps, control=None, **kwargs):
        """
        Run an autoregressive rollout for ``n_steps`` steps.

        The rollout repeatedly calls :meth:`rollout_step`, feeding the
        predicted state back as the next input. The initial state is included
        in the returned trajectory.

        :param initial_state: The initial state from which to start the
            rollout. The initial state must be of shape
            ``[trajectories, 1, *features]``.
        :type initial_state: torch.Tensor | LabelTensor
        :param int n_steps: The number of autoregressive steps to predict.
        :param control: Optional control inputs of shape
            ``[trajectories, n_steps, *control_features]``. When provided, the
            control at each step is concatenated with the current state before
            the forward pass. Default is ``None``.
        :type control: torch.Tensor | LabelTensor | None
        :param dict kwargs: Additional keyword arguments forwarded to
            :meth:`rollout_step`.
        :raises ValueError: If the provided initial_state tensor has less than 3
            dimensions.
        :raises ValueError: If ``control`` is provided and has fewer than
            ``n_steps`` time steps.
        :return: The predicted trajectory, including the initial state. It has
            shape ``[trajectories, n_steps + 1, *features]``, where the first
            step corresponds to the initial state.
        :rtype: torch.Tensor | LabelTensor
        """
        # Set model to evaluation mode for prediction
        self.eval()

        # Raise error if the initial_state does not have at least 3 dimensions
        if initial_state.dim() < 3:
            raise ValueError(
                "The provided initial_state tensor must have at least 3"
                "dimensions: [trajectories, 1, *features]."
                f" Got shape {initial_state.shape}."
            )

        # Check that enough control steps are available
        if control is not None and control.shape[1] < n_steps:
            raise ValueError(
                "The control tensor must provide at least `n_steps` time "
                f"steps. Got {control.shape[1]} for {n_steps} steps."
            )

        # Initialize the list of predictions with the initial state
        predictions = [initial_state]

        # Preserve the graph attribute so the rollout does not mutate it
        graph = kwargs.get("graph")
        key = kwargs.get("key", "x")
        original = getattr(graph, key) if graph is not None else None

        try:
            # Disable gradient computation for autoregressive prediction
            with torch.no_grad():

                # Unroll the autoregressive prediction for n_steps
                for step in range(n_steps):

                    # Select the control at the current step
                    current_control = (
                        control[:, step : step + 1]
                        if control is not None
                        else None
                    )

                    # Advance the rollout by one step
                    predictions.append(
                        self.rollout_step(
                            predictions[-1], current_control, **kwargs
                        )
                    )
        finally:
            if graph is not None:
                setattr(graph, key, original)

        return torch.cat(predictions, dim=1)

    def predict(self, initial_state, n_steps, control=None, **kwargs):
        """
        Generate predictions by recursively calling the model's forward.

        This is an alias for :meth:`rollout`.

        :param initial_state: The initial state from which to start prediction.
            The initial state must be of shape ``[trajectories, 1, *features]``.
        :type initial_state: torch.Tensor | LabelTensor
        :param int n_steps: The number of autoregressive steps to predict.
        :param control: Optional control inputs of shape
            ``[trajectories, n_steps, *control_features]``. When provided, the
            control at each step is concatenated with the current state before
            the forward pass. Default is ``None``.
        :type control: torch.Tensor | LabelTensor | None
        :param dict kwargs: Additional keyword arguments forwarded to
            :meth:`rollout_step`.
        :return: The predicted trajectory, including the initial state. It has
            shape ``[trajectories, n_steps + 1, *features]``, where the first
            step corresponds to the initial state.
        :rtype: torch.Tensor | LabelTensor
        """
        return self.rollout(initial_state, n_steps, control=control, **kwargs)
