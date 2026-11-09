"""Module for the autoregressive mixin class."""

import torch
from pina._src.core.utils import check_consistency


class AutoregressiveMixin:
    """
    Mixin that enables the autoregressive rollout loss logic by maintaining a
    running average of step losses and computing adaptive weights for each step
    based on the cumulative loss. Positive ``eps`` suppresses steps whose
    cumulative running loss is larger, including the current step.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def _init_autoregressive_components(
        self, eps, reset_weights_at_epoch_start, kwargs
    ):
        """
        Initialize the components related to the autoregressive rollout loss.

        :param eps: The hyperparameter controlling the influence of the
            cumulative loss on adaptive weights. Positive values suppress
            larger cumulative losses. No sign constraint is enforced.
        :type eps: float | int
        :param bool reset_weights_at_epoch_start: Whether to reset the running
            average and step count at the start of each epoch. If ``True``, the
            adaptive weights will be recalibrated at the beginning of each epoch
            based on the new training dynamics.
        :param kwargs: Options forwarded to rollout preprocessing/postprocessing
            hooks by time-series conditions. Base hooks ignore all options.
            ``None`` stores an empty mapping.
        :type kwargs: dict[str, Any] | None
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

        :param condition_name: The statistics key. ``None`` or an empty name
            uses ``"default"``. Default is ``None``.
        :type condition_name: str | None
        :return: Residual-versus-zero losses multiplied by detached temporal
            weights. Running statistics update even outside training.
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

        :param str condition_name: The statistics key; an empty name selects
            ``"default"``.
        :param step_loss: Losses with rollout steps on the first axis. All later
            axes are averaged before updating the per-condition running mean.
        :type step_loss: torch.Tensor | LabelTensor
        :raises RuntimeError: If stored and current step-loss shapes cannot
            broadcast during the running-average update.
        :return: Weights broadcastable to the loss tensor, normally of shape
            ``(n_steps, 1, ..., 1)``. Stored means are detached; each call
            increments the condition's count, including ensemble member calls.
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

        :param step_loss: Running average losses, with rollout steps on axis
            zero.
        :type step_loss: torch.Tensor | LabelTensor
        :return: ``exp(-eps * cumsum(step_loss, dim=0))`` of the same shape,
            including the current step. This helper does not detach its input.
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

    def preprocess_step(self, current_state, **kwargs):
        """
        Preprocess the current state before each step.

        :param current_state: The current state tensor.
        :type current_state: torch.Tensor | LabelTensor
        :param dict[str, Any] kwargs: Options for subclass hooks, ignored here.
        :return: The same state tensor, unchanged.
        :rtype: torch.Tensor | LabelTensor
        """
        return current_state

    def postprocess_step(self, predicted_state, **kwargs):
        """
        Return the predicted state without modification.

        Ensemble outputs are not averaged here. Override this hook when rollout
        feedback must combine the leading model axis into a single state.

        :param predicted_state: The predicted state tensor.
        :type predicted_state: torch.Tensor | LabelTensor
        :param dict[str, Any] kwargs: Options for subclass hooks, ignored here.
        :return: The same prediction tensor, unchanged.
        :rtype: torch.Tensor | LabelTensor
        """
        return predicted_state

    def predict(self, initial_state, n_steps, **kwargs):
        """
        Generate predictions by recursively calling the model's forward.

        :param initial_state: The initial state from which to start prediction.
            The initial state must be of shape ``[trajectories, 1, *features]``.
        :type initial_state: torch.Tensor | LabelTensor
        :param int n_steps: The number of autoregressive updates. Zero or
            negative values perform no updates and return the initial data.
        :param dict[str, Any] kwargs: Options passed to preprocessing and
            postprocessing on every prediction step. Not merged with the
            constructor's stored rollout options; base hooks ignore all keys.
        :raises ValueError: If the provided initial_state tensor has less than 3
            dimensions.
        :raises TypeError: If ``n_steps`` cannot be used by :func:`range`.
        :raises RuntimeError: If predicted states cannot be concatenated with
            the initial state on axis one, such as uncombined ensemble outputs.
        :return: The predicted trajectory, including the initial state. It has
            shape ``[trajectories, n_steps + 1, *features]``, where the first
            step corresponds to the initial state.
        :rtype: torch.Tensor | LabelTensor

        .. note::

            Leaves the solver in evaluation mode, including after invalid-input
            errors. Does not restore its prior mode. Forward updates run with
            gradients disabled; the initial state's second-axis length is not
            explicitly checked to be one.
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

        # Initialize the list of predictions with the initial state
        predictions = [initial_state]

        # Disable gradient computation for autoregressive prediction
        with torch.no_grad():

            # Unroll the autoregressive prediction for n_steps
            for _ in range(n_steps):

                # Preprocess the current state before the forward pass
                current_state = self.preprocess_step(predictions[-1], **kwargs)
                output = self.forward(current_state)

                # Postprocess the predicted state after the forward pass
                next_state = self.postprocess_step(output, **kwargs)
                predictions.append(next_state)

        return torch.cat(predictions, dim=1)
