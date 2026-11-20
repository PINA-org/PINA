from typing import ClassVar

import pytest
import torch
from torch_geometric.data import Data

from pina import LabelTensor
from pina.condition import GraphTimeSeriesCondition, TimeSeriesCondition
from pina.graph import RadiusGraph

n_nodes = 20
time_steps = 10


def _create_graph_data(use_lt, time_steps=time_steps):
    if time_steps is None:
        shape = (n_nodes, 2)
    else:
        shape = (n_nodes, time_steps, 2)

    x = torch.rand(shape)
    pos = torch.rand(n_nodes, 2)
    if use_lt:
        x = LabelTensor(x, ["u", "v"])
        pos = LabelTensor(pos, ["x", "y"])

    return RadiusGraph(pos=pos, radius=0.1, x=x)


class DummySolver:
    def __init__(self, use_lt):
        self._params = None
        self._kwargs = {}
        self.aggregation_strategy = torch.mean

    def forward(self, samples):
        return samples.x

    def _get_weights(self, condition_name, step_losses):
        return 1.0


@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor(use_lt):
    graph = _create_graph_data(use_lt=use_lt)
    condition = GraphTimeSeriesCondition(graph, key="x")

    assert isinstance(condition, GraphTimeSeriesCondition)
    assert isinstance(condition, TimeSeriesCondition)
    assert condition.graph is graph
    assert condition.key == "x"
    assert not hasattr(condition, "data")


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_store_data(use_lt, n_windows, unroll_length, randomize):
    graph = _create_graph_data(use_lt=use_lt)
    condition = GraphTimeSeriesCondition(graph, key="x")
    condition.store_data(
        n_subgraphs=2,
        num_hops=1,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )

    assert len(condition) == 2
    for subgraph in condition.data:
        assert isinstance(subgraph, Data)
        assert subgraph.x.shape[0] == n_windows
        assert subgraph.x.shape[1] == unroll_length
        assert subgraph.x.shape[2] == subgraph.pos.shape[0]
        assert subgraph.x.shape[3] == 2

        if use_lt:
            assert isinstance(subgraph.x, LabelTensor)
            assert subgraph.x.labels == ["u", "v"]


@pytest.mark.parametrize("use_lt", [True, False])
def test_store_data_validation(use_lt):
    graph = _create_graph_data(use_lt=use_lt)
    condition = GraphTimeSeriesCondition(graph, key="x")

    with pytest.raises(AssertionError):
        condition.store_data(n_windows=4, unroll_length=0)
    with pytest.raises(AssertionError):
        condition.store_data(n_windows=0, unroll_length=3)
    with pytest.raises(ValueError):
        condition.store_data(
            n_windows=4, unroll_length=3, randomize="not_a_boolean"
        )
    with pytest.raises(ValueError):
        condition.store_data(n_windows=4, unroll_length=1)
    with pytest.raises(ValueError):
        condition.store_data(n_windows=4, unroll_length=time_steps + 1)
    with pytest.raises(ValueError):
        condition.store_data(n_windows=10, unroll_length=3)


def test_store_data_requires_temporal_input():
    graph = _create_graph_data(use_lt=False, time_steps=None)
    condition = GraphTimeSeriesCondition(graph, key="x")

    with pytest.raises(ValueError):
        condition.store_data(n_windows=2, unroll_length=2)


@pytest.mark.parametrize("use_lt", [True, False])
def test_get_item(use_lt):
    graph = _create_graph_data(use_lt=use_lt)
    condition = GraphTimeSeriesCondition(graph, key="x")
    condition.store_data(
        n_subgraphs=2, num_hops=1, n_windows=4, unroll_length=3
    )

    item = condition[0]

    assert isinstance(item, Data)
    assert item.x.shape[0] == 4
    assert item.x.shape[1] == 3
    assert item.x.shape[2] == item.pos.shape[0]
    assert item.x.shape[3] == 2


@pytest.mark.parametrize("use_lt", [True, False])
def test_materialize(use_lt):
    graph = _create_graph_data(use_lt=use_lt)
    condition = GraphTimeSeriesCondition(graph, key="x")
    condition.store_data(
        n_subgraphs=2, num_hops=1, n_windows=4, unroll_length=3
    )

    batch = condition.materialize(1)

    assert isinstance(batch, Data)
    assert batch is condition.data[1]
    assert batch.x.shape[0] == 4
    assert batch.x.shape[1] == 3
    assert batch.x.shape[2] == batch.pos.shape[0]
    assert batch.x.shape[3] == 2


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_evaluate(use_lt, n_windows, unroll_length, randomize):
    graph = _create_graph_data(use_lt=use_lt)
    condition = GraphTimeSeriesCondition(graph, key="x")
    condition.store_data(
        n_subgraphs=1,
        num_hops=1,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    batch = condition.materialize(0)
    series = batch.x.clone()
    solver = DummySolver(use_lt)

    residuals = condition.evaluate(batch, solver)

    step_residuals = []
    current_state = series[:, 0, :, :]
    for step in range(1, series.shape[1]):
        predicted_state = current_state
        target_state = series[:, step, :, :]
        step_residuals.append(predicted_state - target_state)
        current_state = predicted_state

    expected = torch.stack(step_residuals).as_subclass(torch.Tensor)
    assert torch.allclose(residuals, expected)


def test_evaluate_forward_receives_graph():
    graph = _create_graph_data(use_lt=False)
    condition = GraphTimeSeriesCondition(graph, key="x")
    condition.store_data(
        n_subgraphs=1, num_hops=1, n_windows=4, unroll_length=3
    )
    batch = condition.materialize(0)

    received = []
    shapes = []

    class CaptureSolver:
        _kwargs: ClassVar[dict] = {}

        def forward(self, samples):
            received.append(samples)
            shapes.append(samples.x.shape)
            return samples.x

    residuals = condition.evaluate(batch, CaptureSolver())

    assert len(received) == 2
    assert all(isinstance(inp, Data) for inp in received)
    assert shapes[0] == (4, batch.pos.shape[0], 2)
    assert residuals.shape == (2, 4, batch.pos.shape[0], 2)


def _create_control_graph_data(use_lt):
    x = torch.rand(n_nodes, time_steps, 3)
    pos = torch.rand(n_nodes, 2)
    if use_lt:
        x = LabelTensor(x, ["u", "v", "a"])
        pos = LabelTensor(pos, ["x", "y"])
    return RadiusGraph(pos=pos, radius=0.1, x=x)


class ControlledGraphSolver:
    def __init__(self, n_state):
        self.n_state = n_state
        self._kwargs = {}
        self.aggregation_strategy = torch.mean
        self.last_input = None

    def forward(self, samples):
        self.last_input = samples.x
        x = (
            samples.x.as_subclass(torch.Tensor)
            if isinstance(samples.x, LabelTensor)
            else samples.x
        )
        return x[..., : self.n_state] + x[..., self.n_state :]

    def _get_weights(self, condition_name, step_losses):
        return 1.0


@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor_control_selectors(use_lt):
    graph = _create_control_graph_data(use_lt)
    condition = GraphTimeSeriesCondition(
        graph, key="x", state=[0, 1], control=[2]
    )

    assert condition.state == [0, 1]
    assert condition.control == [2]


def test_constructor_control_labels():
    graph = _create_control_graph_data(use_lt=True)
    condition = GraphTimeSeriesCondition(
        graph, key="x", state=["u", "v"], control=["a"]
    )

    assert condition.state == ["u", "v"]
    assert condition.control == ["a"]


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_evaluate_with_control(use_lt, n_windows, unroll_length, randomize):
    graph = _create_control_graph_data(use_lt)
    condition = GraphTimeSeriesCondition(
        graph, key="x", state=[0, 1], control=[2]
    )
    condition.store_data(
        n_subgraphs=1,
        num_hops=1,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    batch = condition.materialize(0)
    series = batch.x.clone()
    solver = ControlledGraphSolver(n_state=2)

    residuals = condition.evaluate(batch, solver)

    # The model receives state and control concatenated
    assert solver.last_input.shape[-1] == 3
    if use_lt:
        assert solver.last_input.labels == ["u", "v", "a"]

    # The batch is restored after evaluation
    assert torch.allclose(batch.x, series)

    # Compute expected residuals. Only the state is predicted and fed back.
    tensor = (
        series.as_subclass(torch.Tensor)
        if isinstance(series, LabelTensor)
        else series
    )
    current_state = tensor[:, 0, :, :2]
    step_residuals = []
    for step in range(1, tensor.shape[1]):
        control_state = tensor[:, step - 1, :, 2:]
        predicted_state = current_state + control_state
        target_state = tensor[:, step, :, :2]
        step_residuals.append(predicted_state - target_state)
        current_state = predicted_state

    expected = torch.stack(step_residuals)
    assert residuals.shape[-1] == 2
    assert torch.allclose(residuals, expected)


@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor_residual_addition(use_lt):
    graph = _create_control_graph_data(use_lt)

    # Residual addition is disabled by default
    condition = GraphTimeSeriesCondition(
        graph, key="x", state=[0, 1], control=[2]
    )
    assert condition._residual_addition is False

    # Residual addition can be enabled
    condition = GraphTimeSeriesCondition(
        graph, key="x", state=[0, 1], control=[2], residual_addition=True
    )
    assert condition._residual_addition is True

    # Should fail if residual_addition is not a boolean
    with pytest.raises(ValueError):
        GraphTimeSeriesCondition(
            graph, key="x", residual_addition="yes"
        )


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
def test_evaluate_residual_addition(use_lt, n_windows, unroll_length):
    graph = _create_control_graph_data(use_lt)
    condition = GraphTimeSeriesCondition(
        graph,
        key="x",
        state=[0, 1],
        control=[2],
        residual_addition=True,
    )
    condition.store_data(
        n_subgraphs=1,
        num_hops=1,
        n_windows=n_windows,
        unroll_length=unroll_length,
    )
    batch = condition.materialize(0)
    series = batch.x.clone()
    solver = ControlledGraphSolver(n_state=2)

    residuals = condition.evaluate(batch, solver)

    # The previous state is added to the model output at each step
    tensor = (
        series.as_subclass(torch.Tensor)
        if isinstance(series, LabelTensor)
        else series
    )
    current_state = tensor[:, 0, :, :2]
    step_residuals = []
    for step in range(1, tensor.shape[1]):
        control_state = tensor[:, step - 1, :, 2:]
        predicted_state = current_state + control_state + current_state
        target_state = tensor[:, step, :, :2]
        step_residuals.append(predicted_state - target_state)
        current_state = predicted_state

    expected = torch.stack(step_residuals)
    assert torch.allclose(residuals, expected)
