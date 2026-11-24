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


def _select_features(tensor, selector):
    """
    Select feature columns along the last dimension of a tensor.

    Columns can be selected by feature labels or by column indices.

    :param tensor: The tensor to select columns from.
    :type tensor: torch.Tensor | LabelTensor
    :param selector: The columns to select, given as labels or indices. A
        single column is accepted as well as a list of columns.
    :type selector: int | str | list[int] | list[str] | None
    :raises AttributeError: If ``selector`` contains labels and ``tensor``
        is not a :class:`~pina.label_tensor.LabelTensor`.
    :raises ValueError: If ``selector`` contains labels that ``tensor`` does
        not define.
    :return: The selected columns, or ``None`` if ``selector`` is empty.
    :rtype: torch.Tensor | LabelTensor | None
    """
    if not selector:
        return None
    if isinstance(selector, (int, str)):
        selector = [selector]

    # Labels are extracted by the LabelTensor, indices by plain slicing
    if isinstance(selector[0], str):
        return tensor.extract(selector)

    selected = tensor[..., selector]

    # LabelTensor indexing does not always update labels, so explicitly
    # restore the labels of the selected columns
    if isinstance(selected, LabelTensor) and tensor.labels is not None:
        selected.labels = [tensor.labels[index] for index in selector]

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
    :meth:`store_mini_batches`, it is unrolled into temporal windows of shape
    ``[trajectories, n_windows, unroll_length, *features]`` before being
    split into batches. The temporal window parameters ``unroll_length``,
    ``n_windows``, and ``randomize`` are arguments of :meth:`store_mini_batches` and
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
    >>> condition.store_mini_batches(unroll_length=5, n_windows=3)
    """

    # Available fields and input data types
    __fields__: ClassVar[list[str]] = ["input", "state", "control"]
    _avail_input_cls = (torch.Tensor, LabelTensor)

    def __init__(
        self, input, state=None, control=None, residual_addition=False
    ):
        """
        Validate the input data and store the feature selectors.

        The selectors are stored as provided and resolved when the columns are
        selected. The temporal window parameters are provided to
        :meth:`store_mini_batches`, not to the constructor.

        :param input: The temporal input data. It must contain all the
            feature columns used as state and control.
        :type input: torch.Tensor | LabelTensor
        :param state: The columns used as state. It can be a list of integer
            column indices, a list of labels for :class:`LabelTensor` input,
            or ``None``. If ``None`` and ``control`` is provided, all columns
            not selected as control are used as state. If both are ``None``,
            all columns are treated as state. Default is ``None``.
        :type state: int | str | list[int] | list[str] | None
        :param control: The columns used as control inputs. It can be a list
            of integer column indices, a list of labels for
            :class:`LabelTensor` input, or ``None``. Control columns are
            concatenated with the state before the forward pass and are never
            predicted. Default is ``None``.
        :type control: int | str | list[int] | list[str] | None
        :param bool residual_addition: If ``True``, the previous state is added
            to the model output at each step, so that the model predicts the
            increment of the state instead of the state itself. If ``False``,
            the model output is used as the next state. Default is ``False``.
        :raises ValueError: If ``input`` is not of type
            :class:`torch.Tensor` or :class:`~pina.label_tensor.LabelTensor`.
        :raises ValueError: If ``input`` has fewer than three dimensions.
        :raises ValueError: If ``residual_addition`` is not a boolean.
        """
        check_consistency(input, self._avail_input_cls)
        check_consistency(residual_addition, bool)
        if input.dim() < 3:
            raise ValueError(
                "The provided data tensor must have at least 3 dimensions: "
                f"[trajectories, time, *features]. Got shape {input.shape}."
            )
        self._input = input
        self._state = state
        self._control = control
        self._residual_addition = residual_addition

        # By default the state is made of every column not used as control
        if isinstance(self._control, (int, str)):
            self._control = [self._control]
        if self._state is None:
            columns = input.labels if isinstance(input, LabelTensor) else None
            columns = columns or list(range(input.shape[-1]))
            self._state = [
                column
                for column in columns
                if column not in (self._control or [])
            ]

    def store_mini_batches(
        self,
        batch_size=None,
        n_windows=None,
        unroll_length=None,
        randomize=False,
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
        _check_time_series_params(
            data=self._input,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )

        # Unroll the raw input only while creating the batches. The base
        # TensorCondition.store_mini_batches slices _input along the trajectory
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
            return super().store_mini_batches(batch_size=batch_size)
        finally:
            self._input = raw_input

    def _rollout(
        self,
        solver,
        initial_state,
        control,
        n_steps,
        state_labels=None,
        control_labels=None,
    ):
        """
        Unroll the autoregressive model starting from ``initial_state``.

        At each step the control is concatenated with the current state along
        the feature dimension and passed to the solver. Only the prediction is
        fed back as the state of the next step, so the control is always read
        from ``control``. When ``residual_addition`` is ``True``, the previous
        state is added to the prediction before feeding it back.

        :param BaseSolver solver: The solver used to perform the forward pass.
        :param initial_state: The state at the first time step, with shape
            ``[records, *state_features]``.
        :type initial_state: torch.Tensor | LabelTensor
        :param control: The control inputs with shape
            ``[records, n_steps, *control_features]``. If ``None``, no control
            is provided. Control-free conditions store ``None``.
        :type control: torch.Tensor | LabelTensor | None
        :param int n_steps: The number of autoregressive steps to unroll.
        :param state_labels: The labels of the state features. If ``None``,
            the labels of ``initial_state`` are used when available. Default
            is ``None``.
        :type state_labels: list[str] | None
        :param control_labels: The labels of the control features. If
            ``None``, the labels of ``control`` are used when available.
            Default is ``None``.
        :type control_labels: list[str] | None
        :return: The predicted trajectory, including the initial state, with
            shape ``[records, n_steps + 1, *state_features]``.
        :rtype: torch.Tensor | LabelTensor
        """
        # Read the labels of the state and control columns, if available
        if state_labels is None and hasattr(initial_state, "_labels"):
            state_labels = initial_state.labels
        if control_labels is None and hasattr(control, "_labels"):
            control_labels = control.labels

        # Restore the labels of the initial state, which are dropped when the
        # data is flattened or reshaped by the caller
        if state_labels and isinstance(initial_state, LabelTensor):
            initial_state = initial_state.as_subclass(LabelTensor)
            initial_state.labels = list(state_labels)

        # Initialize the trajectory with the initial state
        trajectory = [initial_state]
        current_state = initial_state

        # Unroll the autoregressive prediction
        for step in range(n_steps):

            # The control is an input only and is never predicted
            current_control = None if control is None else control[:, step]
            model_input = _concat_features(
                current_state,
                current_control,
                state_labels=state_labels,
                control_labels=control_labels,
            )

            # Forward the model input and update the current state
            predicted_state = solver.forward(model_input)
            if self._residual_addition:
                predicted_state = predicted_state + current_state

            trajectory.append(predicted_state)
            current_state = predicted_state

        # Stack the trajectory along a new temporal dimension
        return torch.stack(trajectory, dim=1)

    def evaluate(self, batch, solver):
        r"""
        Evaluate the residual of the condition on the given batch using the
        solver.

        This method computes the per-step residuals through autoregressive
        unrolling, as implemented by :meth:`_rollout`. At each time step, the
        state and control columns are selected from the unrolled data and
        concatenated along the feature dimension before the forward pass. The
        model prediction is compared with the next state columns, while the
        control columns are only used as inputs and are never predicted.

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

        # Extract the initial states, the controls, and the target states. The
        # temporal dimension of the unrolled data is the third one.
        n_steps = series.shape[2] - 1
        records = series.shape[:2]
        initial_state = _select_features(series[:, :, 0], self._state)
        control = _select_features(series[:, :, :-1], self._control)
        targets = _select_features(series[:, :, 1:], self._state)

        # Read the labels before flattening, which drops them
        state_labels = (
            initial_state.labels
            if isinstance(initial_state, LabelTensor)
            else None
        )
        control_labels = (
            control.labels if isinstance(control, LabelTensor) else None
        )

        # Unroll the trajectories, treating trajectories and windows as records
        trajectory = self._rollout(
            solver,
            initial_state.flatten(0, 1),
            None if control is None else control.flatten(0, 1),
            n_steps,
            state_labels=state_labels,
            control_labels=control_labels,
        )
        trajectory = trajectory.reshape(
            *records, n_steps + 1, *initial_state.shape[2:]
        )

        # Compare the predictions with the target states
        residuals = trajectory[:, :, 1:] - targets
        return residuals.movedim(2, 0).as_subclass(torch.Tensor)

    def test(self, solver):
        """
        Predict the full trajectory of the stored input data.

        The initial state and the control are read from the stored ``input``
        column by column, and the rollout is computed by :meth:`_rollout`
        without gradient tracking. This is meant to be used after training, to
        compare the prediction of the trained solver with the stored data.

        :param BaseSolver solver: The solver providing the trained model.
        :return: The predicted trajectory, including the initial state, with
            shape ``[trajectories, time_steps, *state_features]``.
        :rtype: torch.Tensor | LabelTensor
        """
        # Read the initial state and the control from the stored input
        initial_state = _select_features(self._input[:, 0], self._state)
        control = _select_features(self._input[:, :-1], self._control)

        # Unroll the trajectory without tracking the gradient
        with torch.no_grad():
            return self._rollout(
                solver, initial_state, control, self._input.shape[1] - 1
            )

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
