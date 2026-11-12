"""Module for the Time-Series Condition class."""

from typing import ClassVar

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
    :param int n_windows: The maximum number of temporal windows to extract.
    :param int unroll_length: The number of time steps in each window.
    :param bool randomize: If ``True``, randomly permute the valid starting
        indices before selecting the windows.
    :raises AssertionError: If ``unroll_length`` or ``n_windows`` is not a
        positive integer.
    :raises ValueError: If ``randomize`` is not a boolean value.
    :raises ValueError: If ``data`` has fewer than three dimensions.
    :raises ValueError: If ``unroll_length`` is lower than 2.
    :raises ValueError: If ``unroll_length`` exceeds the number of time steps.
    :raises ValueError: If fewer than ``n_windows`` valid windows are
        available.
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
    time_steps = data.shape[1]
    if unroll_length < 2:
        raise ValueError(
            f"unroll_length must be strictly greater than 1 to create "
            f"temporal windows. Got unroll_length={unroll_length}."
        )

    # Check that the requested unroll length can be built from the data
    if unroll_length > time_steps:
        raise ValueError(
            f"Cannot create unroll windows: unroll_length {unroll_length} "
            f"exceeds the available number of time steps {time_steps}."
        )

    # Check that enough valid windows are available
    n_valid_windows = time_steps - unroll_length + 1
    if n_windows > n_valid_windows:
        raise ValueError(
            f"Cannot create {n_windows} unroll windows with the selected "
            f"unroll_length {unroll_length} from data with {time_steps} "
            f"time steps. Only {n_valid_windows} valid windows are "
            "available."
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


def _resolve_feature_selectors(data, state, control):
    """
    Resolve state and control selectors into feature column indices.

    Selectors can be lists of integer column indices or, when ``data`` is a
    :class:`LabelTensor`, lists of feature labels. When ``control`` is
    provided without ``state``, all columns not selected as control are used
    as state. When both are omitted, every feature column is treated as
    state.

    :param data: The temporal input data of shape
        ``[trajectories, time_steps, *features]``.
    :type data: torch.Tensor | LabelTensor
    :param state: The columns predicting the state. It can be a list of
        integer indices, a list of labels, or ``None``.
    :type state: list[int] | list[str] | None
    :param control: The columns acting as control inputs. It can be a list
        of integer indices, a list of labels, or ``None``.
    :type control: list[int] | list[str] | None
    :raises TypeError: If a selector is not a list of integers or labels.
    :raises ValueError: If a selector references an out-of-range column or a
        missing label, contains duplicates, leaves the state empty, or
        overlaps with the control.
    :return: The resolved state and control column indices.
    :rtype: tuple[list[int], list[int]]
    """
    n_features = data.shape[-1]

    def _resolve(selector, name):
        if selector is None:
            return None

        # Allow a single column to be passed directly
        if isinstance(selector, (int, str)):
            selector = [selector]
        elif not isinstance(selector, (list, tuple, range)):
            raise TypeError(
                f"'{name}' must be None, an integer, a string, or a list "
                f"of integers/labels. Got {type(selector).__name__}."
            )
        selector = list(selector)

        # Empty selectors are allowed for control, checked by the caller
        if len(selector) == 0:
            return []

        # Resolve integer column indices
        if all(isinstance(item, int) for item in selector):
            indices = list(selector)
            for index in indices:
                if index < 0 or index >= n_features:
                    raise ValueError(
                        f"'{name}' contains the column index {index}, which "
                        f"is out of range for data with {n_features} "
                        "features."
                    )

        # Resolve label selectors for LabelTensor data
        elif all(isinstance(item, str) for item in selector):
            if not isinstance(data, LabelTensor):
                raise ValueError(
                    f"'{name}' was provided as labels, but the input data is "
                    "not a LabelTensor."
                )
            labels = data.labels
            if labels is None:
                raise ValueError(
                    f"'{name}' was provided as labels, but the input "
                    "LabelTensor does not define feature labels."
                )
            missing = [label for label in selector if label not in labels]
            if missing:
                raise ValueError(
                    f"'{name}' contains labels {missing} that are not "
                    f"present in the input labels {list(labels)}."
                )
            indices = [labels.index(label) for label in selector]

        else:
            raise ValueError(
                f"'{name}' must contain either only integers or only labels, "
                f"not a mixture. Got {selector}."
            )

        if len(indices) != len(set(indices)):
            raise ValueError(
                f"'{name}' contains duplicate columns: {selector}."
            )

        return indices

    state_indices = _resolve(state, "state")
    control_indices = _resolve(control, "control")

    # By default, all columns not used as control are predicted as state
    if state_indices is None:
        if control_indices is None:
            state_indices = list(range(n_features))
        else:
            state_indices = [
                index
                for index in range(n_features)
                if index not in control_indices
            ]

    # No control columns by default
    if control_indices is None:
        control_indices = []

    # At least one state column is required
    if len(state_indices) == 0:
        raise ValueError(
            "At least one feature column must be selected as state."
        )

    # State and control columns must be disjoint
    overlapping = set(state_indices) & set(control_indices)
    if overlapping:
        raise ValueError(
            "State and control columns must be disjoint. Overlapping "
            f"columns: {sorted(overlapping)}."
        )

    return state_indices, control_indices


def _select_features(tensor, indices):
    """
    Select feature columns along the last dimension of a tensor.

    :param tensor: The tensor to select columns from.
    :type tensor: torch.Tensor | LabelTensor
    :param list[int] indices: The feature column indices to select.
    :return: The selected tensor, or ``None`` if ``indices`` is empty.
    :rtype: torch.Tensor | LabelTensor | None
    """
    if not indices:
        return None

    selected = tensor[..., indices]

    # LabelTensor indexing does not always update labels for advanced
    # indexing, so explicitly restore the labels of the selected columns
    if isinstance(selected, LabelTensor) and tensor.labels is not None:
        selected.labels = [tensor.labels[index] for index in indices]

    return selected


def _concat_features(
    state, control, state_labels=None, control_labels=None
):
    """
    Concatenate state and control columns along the feature dimension.

    :param state: The selected state tensor.
    :type state: torch.Tensor | LabelTensor
    :param control: The selected control tensor, or ``None``.
    :type control: torch.Tensor | LabelTensor | None
    :param state_labels: The labels of the state columns, if available.
        Default is ``None``.
    :type state_labels: list[str] | None
    :param control_labels: The labels of the control columns, if available.
        Default is ``None``.
    :type control_labels: list[str] | None
    :return: The concatenated model input.
    :rtype: torch.Tensor | LabelTensor
    """
    if control is None:
        return state

    # Concatenate the underlying tensors. Working on the tensor part avoids
    # relying on the label bookkeeping of mixed LabelTensor/Tensor inputs.
    model_input = torch.cat([state, control], dim=-1)

    # Restore labels when both state and control labels are known
    if state_labels is not None and control_labels is not None:
        model_input = model_input.as_subclass(LabelTensor)
        model_input.labels = list(state_labels) + list(control_labels)

    return model_input


class TimeSeriesCondition(TensorCondition):
    r"""
    The :class:`TimeSeriesCondition` class represents an autoregressive time
    series condition defined by temporal ``input`` data. The input is expected
    to have shape ``[trajectories, time_steps, *features]``, where the second
    dimension corresponds to the temporal evolution of each trajectory.

    The input data is stored as provided to the constructor. During
    :meth:`store_data`, it is unrolled into temporal windows of shape
    ``[trajectories, n_windows, unroll_length, *features]`` before being
    split into batches. The temporal window parameters ``unroll_length``,
    ``n_windows``, and ``randomize`` are arguments of :meth:`store_data` and
    are not stored on the condition.

    Feature columns can be partitioned into ``state`` and ``control`` columns
    using integer indices or, for :class:`~pina.label_tensor.LabelTensor`
    data, feature labels. The model predicts the next state from the current
    state and the control at the same time step:

    .. math::

        \hat{\mathbf{y}}_{k+1} = \mathcal{M}(\mathbf{y}_k, \mathbf{a}_k),

    where :math:`\mathbf{y}_k` and :math:`\mathbf{a}_k` are obtained by
    selecting the ``state`` and ``control`` columns from the input. Before
    the forward pass, the selected state and control columns are concatenated
    along the feature dimension, i.e. the model receives
    ``[y_k, a_k]``. Only the state is compared with the target and fed back
    into the next step; the control is never predicted.

    When ``state`` and ``control`` are both omitted, every feature column is
    treated as state and the condition reduces to the standard autoregressive
    update :math:`\hat{\mathbf{y}}_{k+1} = \mathcal{M}(\mathbf{y}_k)`.

    Supported data types include :class:`~pina.label_tensor.LabelTensor` and
    :class:`torch.Tensor`.

    :Example:

    >>> from pina import LabelTensor
    >>> from pina.condition import TimeSeriesCondition
    >>> import torch

    >>> data = LabelTensor(torch.rand(5, 10, 3), labels=["u", "v", "a"])
    >>> condition = TimeSeriesCondition(
    ...     input=data,
    ...     state=["u", "v"],
    ...     control=["a"],
    ... )
    >>> condition.store_data(unroll_length=5, n_windows=3)
    """

    # Available fields and input data types
    __fields__: ClassVar[list[str]] = ["input", "state", "control"]
    _avail_input_cls = (torch.Tensor, LabelTensor)

    def __init__(self, input, state=None, control=None):
        """
        Validate the input data and the feature selectors.

        The temporal window parameters are provided to :meth:`store_data`,
        not to the constructor.

        :param input: The temporal input data. It must contain all the
            feature columns used as state and control.
        :type input: torch.Tensor | LabelTensor
        :param state: The columns used as state. It can be a list of integer
            column indices, a list of labels for :class:`LabelTensor` input,
            or ``None``. If ``None`` and ``control`` is provided, all columns
            not selected as control are used as state. If both are ``None``,
            all columns are treated as state. Default is ``None``.
        :type state: list[int] | list[str] | None
        :param control: The columns used as control inputs. It can be a list
            of integer column indices, a list of labels for
            :class:`LabelTensor` input, or ``None``. Control columns are
            concatenated with the state before the forward pass and are never
            predicted. Default is ``None``.
        :type control: list[int] | list[str] | None
        :raises ValueError: If ``input`` is not of type
            :class:`torch.Tensor` or :class:`~pina.label_tensor.LabelTensor`.
        :raises ValueError: If ``input`` has fewer than three dimensions.
        :raises ValueError: If ``state`` or ``control`` selectors are invalid.
        """
        check_consistency(input, self._avail_input_cls)
        if input.dim() < 3:
            raise ValueError(
                "The provided data tensor must have at least 3 dimensions: "
                f"[trajectories, time, *features]. Got shape {input.shape}."
            )
        self._input = input
        self._state = state
        self._control = control
        self._state_indices, self._control_indices = (
            _resolve_feature_selectors(input, state, control)
        )

    def store_data(
        self,
        batch_size=None,
        n_windows=None,
        unroll_length=None,
        randomize=False,
        state=None,
        control=None,
        **kwargs,
    ):
        """
        Unroll the time-series input data and store it in batches.

        The temporal windows are created when this method is called, so
        ``n_windows``, ``unroll_length``, and ``randomize`` are not stored on
        the condition. If ``unroll_length`` is ``None``, the full trajectory
        is used. If ``n_windows`` is ``None``, all valid windows for the
        selected ``unroll_length`` are used.

        :param int batch_size: Number of trajectories per batch. If ``None``,
            all trajectories are stored as a single batch.
        :param int n_windows: The maximum number of temporal windows to
            extract. Default is ``None``, which uses all valid windows.
        :param int unroll_length: The number of time steps in each temporal
            window. Default is ``None``, which uses the full trajectory.
        :param bool randomize: If ``True``, randomly permute the valid
            starting indices before selecting the windows. Default is
            ``False``.
        :param state: Optional override of the state column selector. If
            ``None``, the selector provided to the constructor is used.
        :param control: Optional override of the control column selector. If
            ``None``, the selector provided to the constructor is used.
        :param dict kwargs: Additional keyword arguments are ignored.
        :return: The stored batches.
        :rtype: list[dict]
        :raises AssertionError: If ``n_windows`` or ``unroll_length`` is not
            a positive integer.
        :raises ValueError: If ``randomize`` is not a boolean value,
            ``unroll_length`` is lower than 2 or exceeds the available time
            steps, or fewer than ``n_windows`` valid windows are available.
        """
        time_steps = self._input.shape[1]
        unroll_length = (
            time_steps if unroll_length is None else unroll_length
        )
        n_windows = (
            max(time_steps - unroll_length + 1, 1)
            if n_windows is None
            else n_windows
        )
        state = self._state if state is None else state
        control = self._control if control is None else control

        _check_time_series_params(
            data=self._input,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )
        self._state = state
        self._control = control
        self._state_indices, self._control_indices = (
            _resolve_feature_selectors(self._input, state, control)
        )

        # Unroll the raw input only while creating the batches. The base
        # TensorCondition.store_data slices _input along the trajectory
        # dimension, so it is temporarily replaced by the unrolled tensor
        # and restored afterwards.
        raw_input = self._input
        self._input = _unroll_windows(
            data=raw_input,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )
        try:
            return super().store_data(batch_size=batch_size)
        finally:
            self._input = raw_input

    def evaluate(self, batch, solver):
        r"""
        Evaluate the residual of the condition on the given batch using the
        solver.

        This method computes the per-step residuals through autoregressive
        unrolling. At each time step, the state and control columns are
        selected from the unrolled data and concatenated along the feature
        dimension before the forward pass. The model prediction is compared
        with the next state columns, while the control columns are only used
        as inputs and are never predicted.

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
            ``[time_steps - 1, trajectories, windows, *state_features]``.
        :rtype: torch.Tensor | LabelTensor
        """
        # Extract the unrolled time-series data from the batch
        series = batch["input"]

        # Raise error if the unrolled data does not have at least 4 dimensions
        if series.dim() < 4:
            raise ValueError(
                "The provided input tensor must have at least 4 dimensions:"
                " [trajectories, windows, time_steps, *features]."
                f" Got shape {series.shape}."
            )

        # Copy the kwargs to avoid modifying the original settings
        kwargs = solver._kwargs.copy()

        # Resolve labels for the selected columns, if available
        if isinstance(series, LabelTensor) and series.labels is not None:
            state_labels = [series.labels[i] for i in self._state_indices]
            control_labels = [series.labels[i] for i in self._control_indices]
        else:
            state_labels = None
            control_labels = None

        # Extract the initial state and initialize the step-wise residuals list
        current_state = _select_features(
            series[:, :, 0], self._state_indices
        )
        residuals = []

        # Iterate over the time steps
        for step in range(1, series.shape[2]):

            # Select the control at the current time step. The control does
            # not need to be predicted, so it is only used as model input.
            current_control = _select_features(
                series[:, :, step - 1], self._control_indices
            )
            model_input = _concat_features(
                current_state,
                current_control,
                state_labels=state_labels,
                control_labels=control_labels,
            )

            # Pre-process, forward, and post-process the model input
            processed_input = solver.preprocess_step(model_input, **kwargs)
            output = solver.forward(processed_input)
            predicted_state = solver.postprocess_step(output, **kwargs)

            # Retrieve the target state and compute the step-wise residual
            target_state = _select_features(
                series[:, :, step], self._state_indices
            )
            step_residual = predicted_state - target_state
            residuals.append(step_residual)

            # Update the current state for the next iteration. Only the
            # predicted state is fed back; the control is read from the data.
            current_state = predicted_state

        # Stack the step-wise residuals
        return torch.stack(residuals).as_subclass(torch.Tensor)

    @property
    def input(self):
        """
        The input data associated with the condition.

        :return: The original input data, before temporal unrolling and
            batching.
        :rtype: torch.Tensor | LabelTensor
        """
        return _unwrap_single(self._input)

    @property
    def state(self):
        """
        The state column selector provided to the condition.

        :return: The state selector.
        :rtype: list[int] | list[str] | None
        """
        return self._state

    @property
    def control(self):
        """
        The control column selector provided to the condition.

        :return: The control selector.
        :rtype: list[int] | list[str] | None
        """
        return self._control
