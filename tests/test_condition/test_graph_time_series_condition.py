import pytest
import torch
from torch_geometric.data import Data

from pina import Condition, LabelTensor
from pina.condition import GraphTimeSeriesCondition, TimeSeriesCondition
from pina.graph import RadiusGraph

# Number of samples and time steps for testing
n_samples = 5
n_nodes = 20
time_steps = 10


# Helper function to check tensor types
def _assert_tensor_type(t, use_lt):
    if use_lt:
        assert isinstance(t.x, LabelTensor)
    else:
        assert isinstance(t.x, torch.Tensor) and not isinstance(
            t.x, LabelTensor
        )


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


# Helper function to create graph data
def _create_graph_data(use_lt):

    # If LabelTensor is used, create graph data with LabelTensors
    if use_lt:
        x = LabelTensor(torch.rand(n_nodes, time_steps, 2), ["u", "v"])
        pos = LabelTensor(torch.rand(n_nodes, 2), ["x", "y"])

    # Standard torch.Tensor without labels
    else:
        x = torch.rand(n_nodes, time_steps, 2)
        pos = torch.rand(n_nodes, 2)

    # Create a list of Graphs
    graph = RadiusGraph(
        pos=pos,
        radius=0.1,
        x=x,
    )

    return graph


# Define a dummy solver for testing
class DummySolver:

    def __init__(self, use_lt, input_vars):
        self._params = None
        self._kwargs = {}
        self.aggregation_strategy = torch.mean

    def forward(self, samples):
        # The whole graph batch goes in, the current state comes out
        return samples.x[:, :, 0]

    def preprocess_step(self, current_state, **kwargs):
        return current_state

    def postprocess_step(self, predicted_state, **kwargs):
        return predicted_state

    def _get_weights(self, condition_name, step_losses):
        return 1.0


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_constructor(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    graph = _create_graph_data(use_lt=use_lt)
    original_timeseries = (
        graph.x.clone()
    )  # Store original time series for later comparison
    condition = Condition(
        input=graph,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
        key="x",
    )

    # Assert correct types
    assert isinstance(condition, GraphTimeSeriesCondition)

    # The condition must also be a time-series condition, so that solvers
    # accepting TimeSeriesCondition (e.g. the autoregressive ones) take it
    assert isinstance(condition, TimeSeriesCondition)

    # Assert numerical parity
    if not randomize:
        expected_tensor = _expected_unroll(
            original_timeseries, n_windows, unroll_length, randomize
        )
        assert torch.allclose(condition.input.x, expected_tensor)

    # Assert labels if LabelTensor is used
    if use_lt:
        assert condition.input["x"].labels == ["u", "v"]

    # Should fail if unroll_length is not a positive integer
    with pytest.raises(AssertionError):
        GraphTimeSeriesCondition(
            input=graph,
            n_windows=n_windows,
            unroll_length=0,
            randomize=randomize,
        )

    # Should fail if n_windows is not a positive integer
    with pytest.raises(AssertionError):
        GraphTimeSeriesCondition(
            input=graph,
            n_windows=0,
            unroll_length=unroll_length,
            randomize=randomize,
        )

    # Should fail if randomize is not a boolean value
    with pytest.raises(ValueError):
        Condition(
            input=graph,
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
            input=graph,
            n_windows=n_windows,
            unroll_length=1,
            randomize=randomize,
        )

    # Should fail if unroll_length is greater than the number of time steps
    with pytest.raises(ValueError):
        Condition(
            input=graph,
            n_windows=n_windows,
            unroll_length=time_steps + 1,
            randomize=randomize,
        )

    # Should fail if n_windows is greater than the number of valid windows
    with pytest.raises(ValueError):
        Condition(
            input=graph,
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
    graph = _create_graph_data(use_lt=use_lt)
    condition = GraphTimeSeriesCondition(
        input=graph,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )

    # Extract item using materialize
    index = 0
    item = condition.materialize([index])

    # Assert correct types
    assert isinstance(item, dict)
    _assert_tensor_type(item["input"], use_lt)

    # Assert correct shapes
    expected_shape = torch.Size([n_nodes, n_windows, unroll_length, 2])
    assert item["input"].x.shape == expected_shape


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_materialize(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    graph = _create_graph_data(use_lt=use_lt)
    condition = GraphTimeSeriesCondition(
        input=graph,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )

    # Materialize the batch for the given ids
    idx = [0]
    batch = condition.materialize(idx)

    # Check that the batch is a dictionary holding the input data
    assert isinstance(batch, dict)
    assert "input" in batch
    assert batch["input"].num_graphs == len(idx)


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_evaluate(use_lt, n_windows, unroll_length, randomize):

    # Define the input tensor
    graph = _create_graph_data(use_lt=use_lt)
    input_vars = graph.x.labels if use_lt else None

    # Define the condition and the solver
    condition = GraphTimeSeriesCondition(
        input=graph,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    solver = DummySolver(use_lt, input_vars)

    # Extract the batch
    batch = {"input": condition.input}

    # Evaluate the condition and compute the expected residuals
    residuals = condition.evaluate(batch, solver)

    # Compute expected autoregressive step residuals
    step_residuals = []
    print(batch["input"].x.shape)
    current_state = batch["input"].x[:, :, 0, :]

    for step in range(1, batch["input"].x.shape[2]):
        predicted_state = current_state
        target_state = batch["input"].x[:, :, step, :]

        step_residual = predicted_state - target_state
        step_residuals.append(step_residual)

        current_state = predicted_state

    expected = torch.stack(step_residuals).as_subclass(torch.Tensor)

    # Assert that the evaluated residuals are correct
    assert torch.allclose(residuals, expected)


def test_evaluate_forward_receives_graph():

    graph = _create_graph_data(use_lt=False)
    condition = GraphTimeSeriesCondition(
        input=graph, n_windows=4, unroll_length=3
    )

    received = []

    class CaptureSolver:
        _kwargs = {}

        def preprocess_step(self, state, **kwargs):
            return state

        def postprocess_step(self, state, **kwargs):
            return state

        def forward(self, samples):
            received.append(samples)
            return samples.x[:, :, 0]

    batch = {"input": condition.input}
    residuals = condition.evaluate(batch, CaptureSolver())

    inp = received[0]
    assert isinstance(inp, Data)
    assert inp.x.shape == (n_nodes, 4, 3, 2)
    assert residuals.shape == (2, n_nodes, 4, 2)
