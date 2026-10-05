"""Module for the Time-Series Condition class."""

import torch

from pina._src.condition.base_condition import BaseCondition, _unwrap_single
from pina._src.condition.tensor_condition import TensorCondition
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency, check_positive_integer


def _check_time_series_params(data, n_windows, unroll_length, randomize):
    """
    Check the temporal data and the parameters defining its windows.

    :param data: The temporal data, of shape
        ``[trajectories, time_steps, *features]``.
    :type data: torch.Tensor | LabelTensor
    :param int n_windows: The maximum number of temporal windows to extract.
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
    :param int n_windows: The maximum number of temporal windows to extract.
    :param int unroll_length: The number of time steps in each window.
    :param bool randomize: If ``True``, starting indices are randomly
        permuted before applying ``n_windows``. Default is ``False``.
    :raises ValueError: If ``unroll_length`` is greater than the number of
        time steps in the data.
    :return: A tensor of unrolled windows.
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

    During training, the condition automatically extracts overlapping temporal
    windows from the trajectories. The parameter ``unroll_length`` defines the
    number of consecutive time steps contained in each temporal window, while
    ``n_windows`` controls how many temporal windows are created from the
    available trajectories.

    Internally, the unrolled data is stored as a tensor of shape
    ``[trajectories, n_windows, unroll_length, *features]``.

    Supported data types include :class:`~pina.label_tensor.LabelTensor` and
    :class:`torch.Tensor`.

    :Example:

    >>> from pina import Condition, LabelTensor
    >>> import torch

    >>> data = LabelTensor(torch.rand(5, 10, 2), labels=["u", "v"])
    >>> condition = Condition(input=data, unroll_length=5, n_windows=3)
    """

    # Available fields and input data types
    __fields__ = ["input", "unroll_length", "n_windows", "randomize"]
    _avail_input_cls = (torch.Tensor, LabelTensor)

    def __new__(cls, input, n_windows, unroll_length, randomize=False):
        """
        Validate the input data and time-series parameters.

        :param input: The temporal input data.
        :type input: torch.Tensor | LabelTensor
        :param int n_windows: The maximum number of temporal windows to extract.
        :param int unroll_length: The number of time steps in each window.
        :param bool randomize: If ``True``, randomly permute the valid starting
            indices before selecting the windows. Default is ``False``.
        :raises ValueError: If ``input`` is not of type :class:`torch.Tensor` or
            :class:`~pina.label_tensor.LabelTensor`.
        :raises AssertionError: If ``unroll_length`` is not a positive integer.
        :raises AssertionError: If ``n_windows`` is not a positive integer.
        :raises ValueError: If ``randomize`` is not a boolean value.
        :raises ValueError: If ``input`` has fewer than three dimensions.
        :raises ValueError: If ``unroll_length`` is lower than 2.
        :return: A new :class:`TimeSeriesCondition` instance.
        :rtype: TimeSeriesCondition
        """
        # Check consistency
        check_consistency(input, cls._avail_input_cls)
        _check_time_series_params(
            data=input,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )

        return BaseCondition.__new__(cls)

    def store_data(self, **kwargs):
        """
        Store the unrolled time-series input data.

        The method extracts the time-series input data and creates the temporal
        windows based on the specified ``unroll_length`` and ``n_windows``.

        :param dict kwargs: The keyword arguments containing the data to be
            stored.
        :return: A namespace-like structure containing the stored data.
        :rtype: SimpleNamespace
        """
        # Extract unrolling parameters from kwargs
        unroll_length = kwargs.get("unroll_length")
        n_windows = kwargs.get("n_windows")
        randomize = kwargs.get("randomize", False)
        data = kwargs.get("input")

        # Create unrolled windows from the input data
        unrolled_data = _unroll_windows(
            data=data,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )

        return TensorCondition.store_data(self, input=unrolled_data)

    def _get_series(self, batch):
        """
        Return the unrolled time-series data carried by the given batch.

        :param dict batch: The batch to extract the time-series data from.
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

        :param dict batch: The batch containing the data required by the
            condition evaluation.
        :param BaseSolver solver: The solver used to perform the forward pass
            and compute the residual. The solver provides access to the model
            and its parameters, which may be necessary for evaluating the
            condition residual.
        :raises ValueError: If the unrolled data has less than 4 dimensions.
        :return: The stacked per-step residual tensor of shape
            ``[time_steps - 1, trajectories, windows, *features]``.
        :rtype: torch.Tensor | LabelTensor
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

        :return: The unrolled input data.
        :rtype: torch.Tensor | LabelTensor
        """
        return _unwrap_single(self.data.input)
