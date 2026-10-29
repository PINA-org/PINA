import pytest
import torch

from pina import Condition, LabelTensor
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

    def preprocess_step(self, current_state, **kwargs):
        return current_state

    def postprocess_step(self, predicted_state, **kwargs):
        return predicted_state

    def _get_weights(self, condition_name, step_losses):
        return 1.0


# Verify the unroll semantics of the condition w.r.t. the stored input
@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_constructor(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    input_tensor = _create_tensor_data(use_lt)
    condition = Condition(
        input=input_tensor,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )

    # Assert correct types
    assert isinstance(condition, TimeSeriesCondition)
    _assert_tensor_type(condition.input, use_lt)

    # Assert correct shape
    expected_shape = torch.Size([n_samples, n_windows, unroll_length, 2])
    assert condition.input.shape == expected_shape

    # Assert numerical parity
    if not randomize:
        expected_tensor = _expected_unroll(
            input_tensor, n_windows, unroll_length, randomize
        )
        assert torch.allclose(condition.input, expected_tensor)

    # Assert labels if LabelTensor is used
    if use_lt:
        assert condition.input.labels == ["u", "v"]

    # Should fail if unroll_length is not a positive integer
    with pytest.raises(AssertionError):
        Condition(
            input=input_tensor,
            n_windows=n_windows,
            unroll_length=0,
            randomize=randomize,
        )

    # Should fail if n_windows is not a positive integer
    with pytest.raises(AssertionError):
        Condition(
            input=input_tensor,
            n_windows=0,
            unroll_length=unroll_length,
            randomize=randomize,
        )

    # Should fail if randomize is not a boolean value
    with pytest.raises(ValueError):
        Condition(
            input=input_tensor,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize="not_a_boolean",
        )

    # Should fail if the input tensor has less than 3 dimensions
    with pytest.raises(ValueError):
        Condition(
            input=torch.rand(n_samples, 2),
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )

    # Should fail if unroll_length is not greater than 1
    with pytest.raises(ValueError):
        Condition(
            input=input_tensor,
            n_windows=n_windows,
            unroll_length=1,
            randomize=randomize,
        )

    # Should fail if unroll_length is greater than the number of time steps
    with pytest.raises(ValueError):
        Condition(
            input=input_tensor,
            n_windows=n_windows,
            unroll_length=time_steps + 1,
            randomize=randomize,
        )

    # Should fail if n_windows is greater than the number of valid windows
    with pytest.raises(ValueError):
        Condition(
            input=input_tensor,
            n_windows=10,
            unroll_length=unroll_length,
            randomize=randomize,
        )


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_get_item(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    input_tensor = _create_tensor_data(use_lt)
    condition = Condition(
        input=input_tensor,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )

    # Extract item using __getitem__
    index = 0
    item = condition.materialize([index])

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
    condition = Condition(
        input=input_tensor,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )

    # Materialize the batch for the given ids
    idx = [0, 2]
    batch = condition.materialize(idx)

    # Check that the batch is a dictionary holding the input data
    assert isinstance(batch, dict)
    assert "input" in batch

    # Assert that the batch input is correct
    expected_shape = torch.Size([len(idx), n_windows, unroll_length, 2])
    assert batch["input"].shape == expected_shape

    # Create input values
    if not randomize:
        expected_tensor = _expected_unroll(
            input_tensor, n_windows, unroll_length, randomize
        )
        assert torch.allclose(batch["input"], expected_tensor[idx])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_evaluate(use_lt, n_windows, unroll_length, randomize):

    # Define the input tensor
    input_tensor = _create_tensor_data(use_lt)
    input_vars = input_tensor.labels if use_lt else None

    # Define the condition and the solver
    condition = Condition(
        input=input_tensor,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    solver = DummySolver(use_lt, input_vars)
    loss_fn = torch.nn.MSELoss(reduction="none")

    # Extract the batch
    batch = {"input": condition.input}

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
