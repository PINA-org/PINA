"""Module for the Time-Series Condition class."""

import torch

from pina._src.condition.base_condition import _unwrap_single
from pina._src.condition.tensor_condition import TensorCondition
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency, check_positive_integer


def _check_time_series_params(data, n_windows, unroll_length, randomize):
    """
    Check the temporal data and the parameters defining its windows.

    :param data: The temporal data, of shape
        ``[trajectories, time_steps, *features]``.
    :type data: torch.Tensor | LabelTensor
    :param int n_windows: The strictly positive number of windows to extract.
    :param int unroll_length: The number of time steps in each window.
    :param bool randomize: If ``True``, randomly permute the valid starting
        indices before selecting the windows.
    :raises AssertionError: If ``unroll_length`` or ``n_windows`` is not a
        positive integer.
    :raises ValueError: If ``randomize`` is not a boolean value.
    :raises ValueError: If ``data`` has fewer than three dimensions.
    :raises ValueError: If ``unroll_length`` is lower than 2.
    """
    # Check consistency
    check_consistency(randomize, bool)
    check_positive_integer(n_windows, strict=True)
    check_positive_integer(unroll_length, strict=True)

    # Check the data shape
    if data.dim() < 3:
        raise ValueError(
            "The provided data tensor must have at least 3 dimensions: "
            f"[trajectories, time, *features]. Got shape {data.shape}."
        )

    # Check the unroll length
    if unroll_length < 2:
        raise ValueError(
            f"unroll_length must be strictly greater than 1 to create "
            f"temporal windows. Got unroll_length={unroll_length}."
        )


def _unroll_windows(data, n_windows, unroll_length, randomize):
    """
    Build temporal windows from time-series data.

    Given data with shape ``[trajectories, time_steps, *features]``, this
    function returns a tensor of temporal windows with shape
    ``[trajectories, windows, unroll_length, *features]``.

    :param data: The temporal data tensor to be unrolled.
    :type data: torch.Tensor | LabelTensor
    :param int n_windows: The number of windows to extract per trajectory.
    :param int unroll_length: The number of time steps in each window.
    :param bool randomize: If ``True``, starting indices are randomly
        permuted before selecting ``n_windows``. The same starting indices are
        used for all trajectories.
    :raises ValueError: If ``unroll_length`` is greater than the number of
        time steps in the data.
    :raises ValueError: If fewer than ``n_windows`` valid starting indices are
        available.
    :return: The unrolled windows with shape
        ``[trajectories, n_windows, unroll_length, *features]``. LabelTensor
        labels are preserved.
    :rtype: torch.Tensor | LabelTensor
    """
    # Store the number of time steps in the data
    time_steps = data.shape[1]

    # Compute the last valid starting index for unroll windows
    last_idx = time_steps - unroll_length

    # Raise error if unroll_length is greater than time_steps
    if last_idx < 0:
        raise ValueError(
            f"Cannot create unroll windows: unroll_length {unroll_length} "
            f"exceeds the available number of time steps {time_steps}."
        )

    # Extract starting indices
    start_indices = torch.arange(last_idx + 1)

    # Randomly permute starting indices if randomize is True
    if randomize:
        start_indices = start_indices[torch.randperm(len(start_indices))]

    # Raise error if n_windows is greater than the number of valid windows
    if len(start_indices) < n_windows:
        raise ValueError(
            f"Cannot create {n_windows} unroll windows with the selected "
            f"unroll_length {unroll_length} from data with {time_steps} "
            f"time steps. Only {len(start_indices)} valid windows are "
            "available."
        )

    # Limit the number of windows to n_windows
    start_indices = start_indices[:n_windows]

    # Create unroll windows by slicing the input data at the starting idx
    windows = [data[:, s : s + unroll_length] for s in start_indices]

    # Stack the windows along a new dimension
    unrolled_data = torch.stack(windows, dim=1)

    # Preserve labels if the input data is a LabelTensor
    if isinstance(data, LabelTensor):
        unrolled_data = unrolled_data.as_subclass(LabelTensor)
        unrolled_data.labels = data.labels

    return unrolled_data


class TimeSeriesCondition(TensorCondition):
    """
    The :class:`TimeSeriesCondition` class represents an autoregressive time
    series condition defined by temporal ``input`` data. The input is expected
    to have shape ``[trajectories, time_steps, *features]``, where the second
    dimension corresponds to the temporal evolution of each trajectory.

    The constructor and each :meth:`store_data` call extract overlapping
    temporal windows from the original trajectories. The parameter
    ``unroll_length`` defines the number of consecutive time steps in each
    temporal window, while
    ``n_windows`` controls how many temporal windows are created from the
    available trajectories. Exactly ``n_windows`` are selected per trajectory;
    requesting more than the number of valid starting indices raises an error.
    When ``randomize`` is ``False``, the earliest valid windows are selected.

    Internally, the unrolled data is stored as a tensor of shape
    ``[trajectories, n_windows, unroll_length, *features]``.

    Supported data types include :class:`~pina.label_tensor.LabelTensor` and
    :class:`torch.Tensor`.

    :Example:

        >>> from pina import LabelTensor
        >>> from pina.condition import TimeSeriesCondition
        >>> import torch
        >>> data = LabelTensor(torch.rand(5, 10, 2), labels=["u", "v"])
        >>> condition = TimeSeriesCondition(
        ...     input=data, unroll_length=5, n_windows=3
        ... )
    """

    # Available fields and input data types
    __fields__ = ["input", "unroll_length", "n_windows", "randomize"]
    _avail_input_cls = (torch.Tensor, LabelTensor)

    def __init__(self, input, n_windows, unroll_length, randomize=False):
        """
        Initialization of the :class:`TimeSeriesCondition` class.

        :param input: The temporal input data with shape
            ``[trajectories, time_steps, *features]``. The original data is
            retained for subsequent window generation.
        :type input: torch.Tensor | LabelTensor
        :param int n_windows: The strictly positive number of windows to extract
            per trajectory. At least this many valid starting indices must
            exist.
        :param int unroll_length: The number of time steps in each window. Must
            be at least ``2`` and no greater than the available time steps.
        :param bool randomize: If ``True``, randomly permute the valid starting
            indices before selecting the windows. Default is ``False``.
        :raises ValueError: If ``input`` is not of type
            :class:`torch.Tensor` or :class:`~pina.label_tensor.LabelTensor`.
        :raises AssertionError: If ``unroll_length`` is not a positive integer.
        :raises AssertionError: If ``n_windows`` is not a positive integer.
        :raises ValueError: If ``randomize`` is not a boolean value.
        :raises ValueError: If ``input`` has fewer than three dimensions.
        :raises ValueError: If ``unroll_length`` is lower than 2.
        :raises ValueError: If ``unroll_length`` exceeds the available time
            steps, or fewer than ``n_windows`` valid windows are available.
        """
        check_consistency(input, self._avail_input_cls)
        _check_time_series_params(
            data=input,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )
        self._raw_input = input
        self._n_windows = n_windows
        self._unroll_length = unroll_length
        self._randomize = randomize
        self._input = _unroll_windows(
            data=input,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )

    def store_data(self, batch_size=None, **kwargs):
        """
        Store the unrolled time-series input data in batches.

        The method extracts the time-series input data and creates the temporal
        windows based on the specified ``unroll_length`` and ``n_windows``.
        Parameters passed through ``kwargs`` override the values provided to
        the constructor.

        New window settings replace the stored settings for subsequent calls.
        With randomization enabled, windows are selected again on each call.
        Existing batches are replaced.

        :param batch_size: The number of trajectories per batch. ``None`` or
            ``0`` stores all trajectories in one batch. A negative value
            produces no batches. Default is ``None``.
        :type batch_size: int | None
        :param dict[str, int | bool] kwargs: Optional unrolling settings:

            * ``n_windows`` (int): The strictly positive number of windows per
              trajectory. Default is the currently stored value.
            * ``unroll_length`` (int): The number of steps per window, at least
              ``2``. Default is the currently stored value.
            * ``randomize`` (bool): Whether to permute valid starting indices
              before selecting windows. Default is the currently stored value.

            Other keyword arguments are ignored.
        :raises AssertionError: If ``n_windows`` or ``unroll_length`` is not a
            positive integer.
        :raises ValueError: If ``randomize`` is not a boolean, the original data
            has fewer than three dimensions, ``unroll_length`` is less than two
            or exceeds the available steps, or too few valid windows exist.
        :raises TypeError: If a truthy ``batch_size`` is not an integer accepted
            by :func:`range`.
        :return: The stored batch dictionaries, also assigned to ``self.data``.
            Each ``"input"`` has shape
            ``[batch_trajectories, n_windows, unroll_length, *features]``; the
            other fields contain the current temporal settings.
        :rtype: list[dict[str, Any]]
        """
        n_windows = kwargs.get("n_windows", self._n_windows)
        unroll_length = kwargs.get("unroll_length", self._unroll_length)
        randomize = kwargs.get("randomize", self._randomize)

        _check_time_series_params(
            data=self._raw_input,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )

        self._n_windows = n_windows
        self._unroll_length = unroll_length
        self._randomize = randomize
        self._input = _unroll_windows(
            data=self._raw_input,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )
        return super().store_data(batch_size=batch_size)

    def _get_series(self, batch):
        """
        Return the unrolled time-series data carried by the given batch.

        :param dict[str, Any] batch: The batch containing time-series data under
            ``"input"``.
        :raises KeyError: If the batch lacks ``"input"``.
        :return: The unrolled data, of shape
            ``[trajectories, n_windows, unroll_length, *features]``.
        :rtype: torch.Tensor | LabelTensor
        """
        return batch["input"]

    def evaluate(self, batch, solver):
        """
        Evaluate the residual of the condition on the given batch using the
        solver.

        This method computes the per-step residuals through autoregressive
        unrolling. A forward pass of the solver's model is performed at each
        time step, and the per-step residuals (predicted - target) are
        returned as a stacked tensor.

        The returned tensor preserves all per-step residual values without
        reduction or loss aggregation.

        A single copy of ``solver._kwargs`` is passed to the solver's
        ``preprocess_step`` and ``postprocess_step`` hooks throughout the
        rollout. Predictions, rather than stored targets, initialize later
        steps, and the computation remains attached to the autograd graph.

        :param dict[str, Any] batch: The batch containing ``"input"`` with shape
            ``[trajectories, windows, unroll_length, *features]``.
        :param pina.solver.BaseSolver solver: The autoregressive solver
            providing ``preprocess_step``, ``forward``, ``postprocess_step``,
            and the ``_kwargs`` settings used by the rollout.
        :raises KeyError: If the batch lacks ``"input"``.
        :raises ValueError: If the unrolled data has less than 4 dimensions.
        :raises RuntimeError: If prediction and target shapes are incompatible
            for subtraction, or no prediction steps are available to stack.
        :return: The stacked per-step residuals as a plain tensor with shape
            ``[unroll_length - 1, trajectories, windows, *features]``.
        :rtype: torch.Tensor
        """
        # Extract the unrolled time-series data from the batch
        series = self._get_series(batch)

        # Raise error if the unrolled data does not have at least 4 dimensions
        if series.dim() < 4:
            raise ValueError(
                "The provided input tensor must have at least 4 dimensions:"
                " [trajectories, windows, time_steps, *features]."
                f" Got shape {series.shape}."
            )

        # Copy the kwargs to avoid modifying the original settings
        kwargs = solver._kwargs.copy()

        # Extract the initial state and initialize the step-wise residuals list
        current_state = series[:, :, 0]
        residuals = []

        # Iterate over the time steps
        for step in range(1, series.shape[2]):

            # Pre-process, forward, and post-process the current state
            processed_input = solver.preprocess_step(current_state, **kwargs)
            output = solver.forward(processed_input)
            predicted_state = solver.postprocess_step(output, **kwargs)

            # Retrieve the target and compute the step-wise residual
            target_state = series[:, :, step]
            step_residual = predicted_state - target_state
            residuals.append(step_residual)

            # Update the current state for the next iteration
            current_state = predicted_state

        # Stack the step-wise residuals
        return torch.stack(residuals).as_subclass(torch.Tensor)

    @property
    def input(self):
        """
        The input data associated with the condition.

        :return: The unrolled input data with shape
            ``[trajectories, n_windows, unroll_length, *features]``.
        :rtype: torch.Tensor | LabelTensor
        """
        return _unwrap_single(self._input)
