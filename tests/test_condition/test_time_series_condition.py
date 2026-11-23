import pytest
import torch

from pina import LabelTensor
from pina._src.core.utils import labelize_forward
from pina.condition import TimeSeriesCondition

# Number of samples and time steps for testing
n_samples = 5
time_steps = 10


# Helper function to create tensor data
def _create_tensor_data(use_lt=False):

    # Input tensor
    input_tensor = torch.rand((n_samples, time_steps, 2))

    # If LabelTensor is used, create tensor data with labels
    if use_lt:
        return LabelTensor(input_tensor, labels=["u", "v"])

    return input_tensor


# Helper function to create tensor data with state and control columns
def _create_control_tensor_data(use_lt=False):

    # Input tensor with two state columns and one control column
    input_tensor = torch.rand((n_samples, time_steps, 3))

    # If LabelTensor is used, create tensor data with labels
    if use_lt:
        return LabelTensor(input_tensor, labels=["u", "v", "a"])

    return input_tensor


# Helper function to check tensor types
def _assert_tensor_type(t, use_lt):
    if use_lt:
        assert isinstance(t, LabelTensor)
    else:
        assert isinstance(t, torch.Tensor) and not isinstance(t, LabelTensor)


# Helper function to compute expected unroll windows
def _expected_unroll(data, n_windows, unroll_length, randomize):

    # Compute valid starting indices
    last_idx = data.shape[1] - unroll_length
    start_indices = torch.arange(last_idx + 1)

    # Randomize indices if required
    if randomize:
        start_indices = start_indices[torch.randperm(len(start_indices))]

    # Limit the number of windows
    if n_windows is not None and n_windows < len(start_indices):
        start_indices = start_indices[:n_windows]

    # Build expected windows
    windows = [data[:, s : s + unroll_length] for s in start_indices]

    return torch.stack(windows, dim=1)


# Define a dummy solver for testing
class DummySolver:

    def __init__(self, use_lt, input_vars):
        if use_lt:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=input_vars,
                output_variables=input_vars,
            )

        self._params = None
        self._kwargs = {}
        self.aggregation_strategy = torch.mean

    def forward(self, samples):
        return samples

    def _get_weights(self, condition_name, step_losses):
        return 1.0


# Define a dummy solver that consumes state and control columns
class ControlledDummySolver:

    def __init__(self, n_state):
        self.n_state = n_state
        self._kwargs = {}
        self.aggregation_strategy = torch.mean
        self.last_input = None

    def forward(self, samples):
        self.last_input = samples
        tensor = (
            samples.as_subclass(torch.Tensor)
            if isinstance(samples, LabelTensor)
            else samples
        )
        return tensor[..., : self.n_state] + tensor[..., self.n_state :]

    def _get_weights(self, condition_name, step_losses):
        return 1.0


# Verify the constructor stores raw data and validates its shape
@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor(use_lt):

    # Define the condition
    input_tensor = _create_tensor_data(use_lt)
    condition = TimeSeriesCondition(input=input_tensor)

    # Assert correct types
    assert isinstance(condition, TimeSeriesCondition)
    _assert_tensor_type(condition.input, use_lt)

    # The constructor stores the raw input; unrolling happens in store_mini_batches
    expected_shape = torch.Size([n_samples, time_steps, 2])
    assert condition.input.shape == expected_shape
    assert torch.allclose(condition.input, input_tensor)

    # Assert labels if LabelTensor is used
    if use_lt:
        assert condition.input.labels == ["u", "v"]

    # Should fail if the input tensor has less than 3 dimensions
    with pytest.raises(ValueError):
        TimeSeriesCondition(input=torch.rand(n_samples, 2))

    # Should fail if the input is not a tensor
    with pytest.raises(ValueError):
        TimeSeriesCondition(input=[[0.0, 1.0]])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_store_mini_batches_unrolls_input(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    input_tensor = _create_tensor_data(use_lt)
    condition = TimeSeriesCondition(input=input_tensor)

    # The constructor keeps the raw input, without temporal unrolling
    raw_shape = torch.Size([n_samples, time_steps, 2])
    assert condition.input.shape == raw_shape

    # store_mini_batches builds the temporal windows in the batches
    condition.store_mini_batches(
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    batch = condition.materialize(0)
    expected_shape = torch.Size([n_samples, n_windows, unroll_length, 2])
    assert batch["input"].shape == expected_shape

    # Assert numerical parity when windows are not randomized
    if not randomize:
        expected_tensor = _expected_unroll(
            input_tensor, n_windows, unroll_length, randomize
        )
        assert torch.allclose(batch["input"], expected_tensor)

    # The public input property still exposes the raw data
    assert condition.input.shape == raw_shape

    # Temporal window parameters are not stored on the condition or batches
    for name in ("n_windows", "unroll_length", "randomize"):
        assert not hasattr(condition, f"_{name}")
        assert name not in batch


@pytest.mark.parametrize("use_lt", [True, False])
def test_store_mini_batches_defaults(use_lt):

    # Without temporal arguments, use the full trajectory in one window
    input_tensor = _create_tensor_data(use_lt)
    condition = TimeSeriesCondition(input=input_tensor)
    condition.store_mini_batches()

    batch = condition.materialize(0)
    expected_shape = torch.Size([n_samples, 1, time_steps, 2])
    assert batch["input"].shape == expected_shape
    assert torch.allclose(batch["input"][:, 0], input_tensor)


@pytest.mark.parametrize("use_lt", [True, False])
def test_store_mini_batches_validation(use_lt):

    # Invalid temporal window parameters are rejected by store_mini_batches
    input_tensor = _create_tensor_data(use_lt)
    condition = TimeSeriesCondition(input=input_tensor)

    # Should fail if unroll_length is not a positive integer
    with pytest.raises(AssertionError):
        condition.store_mini_batches(unroll_length=0)

    # Should fail if n_windows is not a positive integer
    with pytest.raises(AssertionError):
        condition.store_mini_batches(n_windows=0)

    # Should fail if randomize is not a boolean value
    with pytest.raises(ValueError):
        condition.store_mini_batches(randomize="not_a_boolean")

    # Should fail if unroll_length is not greater than 1
    with pytest.raises(ValueError):
        condition.store_mini_batches(unroll_length=1)

    # Should fail if unroll_length is greater than the number of time steps
    with pytest.raises(ValueError):
        condition.store_mini_batches(unroll_length=time_steps + 1)

    # Should fail if n_windows is greater than the number of valid windows
    with pytest.raises(ValueError):
        condition.store_mini_batches(n_windows=10, unroll_length=3)


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_get_item(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    input_tensor = _create_tensor_data(use_lt)
    condition = TimeSeriesCondition(input=input_tensor)

    # Store and extract item using __getitem__
    index = 0
    condition.store_mini_batches(
        batch_size=1,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    item = condition[index]

    # Assert correct types
    assert isinstance(item, dict)
    assert "input" in item
    _assert_tensor_type(item["input"], use_lt)

    # Assert correct shapes
    expected_shape = torch.Size([1, n_windows, unroll_length, 2])
    assert item["input"].shape == expected_shape

    # Assert numerical parity
    if not randomize:
        expected_tensor = _expected_unroll(
            input_tensor, n_windows, unroll_length, randomize
        )
        assert torch.allclose(item["input"], expected_tensor[index : index + 1])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_materialize(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    input_tensor = _create_tensor_data(use_lt)
    condition = TimeSeriesCondition(input=input_tensor)

    # Materialize the first batch of size two
    condition.store_mini_batches(
        batch_size=2,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    batch = condition.materialize(0)

    # Check that the batch is a dictionary holding the input data
    assert isinstance(batch, dict)
    assert "input" in batch

    # Assert that the batch input is correct
    expected_shape = torch.Size([2, n_windows, unroll_length, 2])
    assert batch["input"].shape == expected_shape

    # Create input values
    if not randomize:
        expected_tensor = _expected_unroll(
            input_tensor, n_windows, unroll_length, randomize
        )
        assert torch.allclose(batch["input"], expected_tensor[:2])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_evaluate(use_lt, n_windows, unroll_length, randomize):

    # Define the input tensor
    input_tensor = _create_tensor_data(use_lt)
    input_vars = input_tensor.labels if use_lt else None

    # Define the condition and the solver
    condition = TimeSeriesCondition(input=input_tensor)
    solver = DummySolver(use_lt, input_vars)

    # Store and extract the batch
    condition.store_mini_batches(
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    batch = condition.materialize(0)

    # Evaluate the condition and compute the expected residuals
    residuals = condition.evaluate(batch, solver)

    # Compute expected autoregressive step residuals
    step_residuals = []
    current_state = batch["input"][:, :, 0]

    for step in range(1, batch["input"].shape[2]):
        predicted_state = current_state
        target_state = batch["input"][:, :, step]

        step_residual = predicted_state - target_state
        step_residuals.append(step_residual)

        current_state = predicted_state

    expected = torch.stack(step_residuals).as_subclass(torch.Tensor)

    # Assert that the evaluated residuals are correct
    assert torch.allclose(residuals, expected)


# Verify selector resolution for state and control columns
@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor_control_indices(use_lt):

    # Define the condition with integer selectors
    input_tensor = _create_control_tensor_data(use_lt)
    condition = TimeSeriesCondition(
        input=input_tensor,
        state=[0, 1],
        control=[2],
    )

    # Assert that the selectors are stored and resolved correctly
    assert condition.state == [0, 1]
    assert condition.control == [2]

    # The constructor keeps the raw input shape
    expected_shape = torch.Size([n_samples, time_steps, 3])
    assert condition.input.shape == expected_shape


def test_constructor_control_labels():

    # Define the condition with label selectors
    input_tensor = _create_control_tensor_data(use_lt=True)
    condition = TimeSeriesCondition(
        input=input_tensor,
        state=["u", "v"],
        control=["a"],
    )

    # Assert that the labels are stored as provided
    assert condition.state == ["u", "v"]
    assert condition.control == ["a"]


def test_control_only_selector():

    # When only control is provided, the state is its complement
    input_tensor = _create_control_tensor_data(use_lt=True)
    condition = TimeSeriesCondition(
        input=input_tensor,
        control=["a"],
    )

    assert condition.state == ["u", "v"]
    assert condition.control == ["a"]


def test_no_selectors_all_state():

    # Without selectors every feature is treated as state
    input_tensor = _create_control_tensor_data(use_lt=False)
    condition = TimeSeriesCondition(input=input_tensor)

    assert condition.state == [0, 1, 2]
    assert condition.control is None


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_evaluate_with_control(
    use_lt, n_windows, unroll_length, randomize
):

    # Define the condition and the solver
    input_tensor = _create_control_tensor_data(use_lt)
    condition = TimeSeriesCondition(
        input=input_tensor,
        state=[0, 1],
        control=[2],
    )
    solver = ControlledDummySolver(n_state=2)

    # Store and extract the batch
    condition.store_mini_batches(
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    batch = condition.materialize(0)

    # Evaluate the condition
    residuals = condition.evaluate(batch, solver)

    # The model must receive state and control concatenated
    assert solver.last_input.shape[-1] == 3
    if use_lt:
        assert solver.last_input.labels == ["u", "v", "a"]

    # Compute the expected autoregressive step residuals. Only the state
    # is predicted and fed back; the control is always read from the data.
    series = batch["input"]
    tensor = (
        series.as_subclass(torch.Tensor)
        if isinstance(series, LabelTensor)
        else series
    )
    current_state = tensor[:, :, 0, :2]
    step_residuals = []
    for step in range(1, tensor.shape[2]):
        control_state = tensor[:, :, step - 1, 2:]
        predicted_state = current_state + control_state
        target_state = tensor[:, :, step, :2]
        step_residuals.append(predicted_state - target_state)
        current_state = predicted_state

    expected = torch.stack(step_residuals)
    assert residuals.shape[-1] == 2
    assert torch.allclose(residuals, expected)


# Verify that residual_addition is stored and validated by the constructor
@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor_residual_addition(use_lt):

    # Define the condition with the residual addition enabled
    condition = TimeSeriesCondition(
        input=_create_tensor_data(use_lt), residual_addition=True
    )
    assert condition._residual_addition is True

    # Residual addition is disabled by default
    condition = TimeSeriesCondition(input=_create_tensor_data(use_lt))
    assert condition._residual_addition is False

    # Should fail if residual_addition is not a boolean
    with pytest.raises(ValueError):
        TimeSeriesCondition(
            input=_create_tensor_data(use_lt), residual_addition="yes"
        )


# Verify that residual_addition adds the previous state to the prediction
@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("randomize", [True, False])
def test_evaluate_residual_addition(use_lt, randomize):

    # Define the condition and the solver
    input_tensor = _create_control_tensor_data(use_lt)
    condition = TimeSeriesCondition(
        input=input_tensor,
        state=[0, 1],
        control=[2],
        residual_addition=True,
    )
    solver = ControlledDummySolver(n_state=2)

    # Store and extract the batch
    condition.store_mini_batches(
        n_windows=4, unroll_length=3, randomize=randomize
    )
    batch = condition.materialize(0)

    # Evaluate the condition
    residuals = condition.evaluate(batch, solver)

    # Compute the expected residuals. The model output is the state plus the
    # control, and the previous state is added to the prediction.
    series = batch["input"]
    tensor = (
        series.as_subclass(torch.Tensor)
        if isinstance(series, LabelTensor)
        else series
    )
    current_state = tensor[:, :, 0, :2]
    step_residuals = []
    for step in range(1, tensor.shape[2]):
        control_state = tensor[:, :, step - 1, 2:]
        predicted_state = current_state + control_state + current_state
        target_state = tensor[:, :, step, :2]
        step_residuals.append(predicted_state - target_state)
        current_state = predicted_state

    expected = torch.stack(step_residuals)
    assert torch.allclose(residuals, expected)


# Verify the trajectory returned by the test method
@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("residual_addition", [True, False])
def test_test(use_lt, residual_addition):

    # Define the condition and the solver
    input_tensor = _create_control_tensor_data(use_lt)
    condition = TimeSeriesCondition(
        input=input_tensor,
        state=[0, 1],
        control=[2],
        residual_addition=residual_addition,
    )
    solver = ControlledDummySolver(n_state=2)

    # Predict the trajectory of the stored input data
    trajectory = condition.test(solver)

    # The trajectory keeps the trajectories and the time steps, and only the
    # state columns are predicted
    assert isinstance(trajectory, LabelTensor if use_lt else torch.Tensor)
    assert trajectory.shape == (n_samples, time_steps, 2)
    assert not trajectory.requires_grad

    # Compute the expected trajectory
    tensor = (
        input_tensor.as_subclass(torch.Tensor)
        if isinstance(input_tensor, LabelTensor)
        else input_tensor
    )
    states = [tensor[:, 0, :2]]
    for step in range(1, time_steps):
        predicted_state = states[-1] + tensor[:, step - 1, 2:]
        if residual_addition:
            predicted_state = predicted_state + states[-1]
        states.append(predicted_state)
    expected = torch.stack(states, dim=1)

    assert torch.allclose(
        trajectory.as_subclass(torch.Tensor), expected
    )
