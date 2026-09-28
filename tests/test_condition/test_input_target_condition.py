import pytest
import torch

from pina import Condition, LabelTensor
from pina._src.core.graph import LabelBatch
from pina._src.core.utils import labelize_forward
from pina.condition import InputTargetCondition
from pina.graph import RadiusGraph

# Number of graphs and tensor samples for testing
n_samples = 10
n_graphs = 10
n_nodes = 20


# Helper function to create tensor data
def _create_tensor_data(use_lt=False):

    # If LabelTensor is used, create tensor data with labels
    if use_lt:
        input_tensor = LabelTensor(torch.rand((n_samples, 3)), ["x", "y", "z"])
        target_tensor = LabelTensor(torch.rand((n_samples, 2)), ["a", "b"])
        return input_tensor, target_tensor

    # Standard torch.Tensor without labels
    input_tensor = torch.rand((n_samples, 3))
    target_tensor = torch.rand((n_samples, 2))

    return input_tensor, target_tensor


# Helper function to create graph data
def _create_graph_data(is_input, use_lt):

    # If LabelTensor is used, create graph data with LabelTensors
    if use_lt:
        x = LabelTensor(torch.rand(n_graphs, n_nodes, 2), ["u", "v"])
        pos = LabelTensor(torch.rand(n_graphs, n_nodes, 2), ["x", "y"])
        tensor = LabelTensor(torch.rand(n_graphs, n_nodes, 2), ["f", "g"])

    # Standard torch.Tensor without labels
    else:
        x = torch.rand(n_graphs, n_nodes, 2)
        pos = torch.rand(n_graphs, n_nodes, 2)
        tensor = torch.rand(n_graphs, n_nodes, 2)

    # Create a list of Graphs
    graph = [
        RadiusGraph(
            pos=pos[i],
            radius=0.1,
            x=x[i] if is_input else None,
            y=x[i] if not is_input else None,
        )
        for i in range(len(x))
    ]

    return graph, tensor


# Helper function to check tensor types
def _assert_tensor_type(t, use_lt):
    if use_lt:
        assert isinstance(t, LabelTensor)
    else:
        assert isinstance(t, torch.Tensor) and not isinstance(t, LabelTensor)


# Helper function to check input graph
def _assert_graph_type(graph_list, use_lt, is_input):

    assert isinstance(graph_list, list)
    for graph in graph_list:
        value = graph.x if is_input else graph.y
        _assert_tensor_type(value, use_lt)


# Define a dummy solver for testing
class DummySolver:

    def __init__(self, use_lt, input_vars, output_vars):
        if use_lt is True:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=input_vars,
                output_variables=output_vars,
            )

    def forward(self, pts):

        # Tensor case
        if isinstance(pts, torch.Tensor):
            return torch.cat(
                [pts.mean(dim=-1, keepdim=True), pts.sum(dim=-1, keepdim=True)],
                dim=-1,
            )

        # Graph case
        else:
            return torch.cat(
                [
                    pts.x.mean(dim=-1, keepdim=True),
                    pts.x.sum(dim=-1, keepdim=True),
                ],
                dim=-1,
            ).reshape(n_graphs, n_nodes, -1)


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize(
    "case", [["tensor", "tensor"], ["tensor", "graph"], ["graph", "tensor"]]
)
def test_constructor(use_lt, case):

    # Tensor - tensor
    if case == ["tensor", "tensor"]:

        # Define the condition
        input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
        condition = Condition(input=input_tensor, target=target_tensor)

        # Assert correct types
        assert isinstance(condition, InputTargetCondition)
        _assert_tensor_type(condition.input, use_lt)
        _assert_tensor_type(condition.target, use_lt)

        # Assert numerical parity
        assert torch.allclose(condition.input, input_tensor)
        assert torch.allclose(condition.target, target_tensor)

        # Assert labels if LabelTensor is used
        if use_lt:
            assert condition.input.labels == ["x", "y", "z"]
            assert condition.target.labels == ["a", "b"]

    # Tensor - graph
    elif case == ["tensor", "graph"]:

        # Define the condition
        target_graph, input_tensor = _create_graph_data(
            is_input=False, use_lt=use_lt
        )
        condition = Condition(input=input_tensor, target=target_graph)

        # Assert correct types
        assert isinstance(condition, InputTargetCondition)
        _assert_tensor_type(condition.input, use_lt)
        _assert_graph_type(condition.target, use_lt, is_input=False)

        # Assert numerical parity
        assert torch.allclose(condition.input, input_tensor)
        for i, graph in enumerate(target_graph):
            assert torch.allclose(condition.target[i].y, graph.y)

        # Assert labels if LabelTensor is used
        if use_lt:
            assert condition.input.labels == ["f", "g"]
            for i in range(len(target_graph)):
                assert condition.target[i].y.labels == ["u", "v"]
                assert condition.target[i].pos.labels == ["x", "y"]

    # Graph - tensor
    elif case == ["graph", "tensor"]:

        # Define the condition
        input_graph, target_tensor = _create_graph_data(
            is_input=True, use_lt=use_lt
        )
        condition = Condition(input=input_graph, target=target_tensor)

        # Assert correct types
        assert isinstance(condition, InputTargetCondition)
        _assert_graph_type(condition.input, use_lt, is_input=True)
        _assert_tensor_type(condition.target, use_lt)

        # Assert numerical parity
        assert torch.allclose(condition.target, target_tensor)
        for i, graph in enumerate(input_graph):
            assert torch.allclose(condition.input[i].x, graph.x)

        # Assert labels if LabelTensor is used
        if use_lt:
            assert condition.target.labels == ["f", "g"]
            for i in range(len(input_graph)):
                assert condition.input[i].x.labels == ["u", "v"]
                assert condition.input[i].pos.labels == ["x", "y"]

    # Prepare for invalid input tests
    input_ = input_tensor if case[0] == "tensor" else input_graph
    target_ = target_tensor if case[1] == "tensor" else target_graph

    # Should fail if the input is neither a tensor nor a graph
    with pytest.raises(ValueError):
        Condition(input="invalid_input", target=target_)

    # Should fail if the target is neither a tensor nor a graph
    with pytest.raises(ValueError):
        Condition(input=input_, target="invalid_target")

    # Should fail if the input is a list of tensors
    if case[0] == "tensor":
        with pytest.raises(ValueError):
            Condition(input=[input_], target=target_)

    # Should fail if the target is a list of tensors
    if case[1] == "tensor":
        with pytest.raises(ValueError):
            Condition(input=input_, target=[target_])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize(
    "case", [["tensor", "tensor"], ["tensor", "graph"], ["graph", "tensor"]]
)
def test_get_item(use_lt, case):

    # Tensor - tensor
    if case == ["tensor", "tensor"]:

        # Define the condition
        input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
        condition = Condition(input=input_tensor, target=target_tensor)

        # Extract item using __getitem__
        index = 0
        item = condition[index]

        # Assert correct types
        assert isinstance(item, dict)
        _assert_tensor_type(item["input"], use_lt)
        _assert_tensor_type(item["target"], use_lt)

        # Assert numerical parity
        assert torch.allclose(item["input"][0], input_tensor[index])
        assert torch.allclose(item["target"][0], target_tensor[index])

    # Tensor - graph
    elif case == ["tensor", "graph"]:

        # Define the condition
        target_graph, input_tensor = _create_graph_data(
            is_input=False, use_lt=use_lt
        )
        condition = Condition(input=input_tensor, target=target_graph)

        # Extract item using __getitem__
        index = 0
        item = condition[index]

        # Assert correct types
        assert isinstance(item, dict)
        _assert_tensor_type(item["input"], use_lt)
        assert isinstance(item["target"], LabelBatch)
        assert item["target"].num_graphs == 1

        # Assert numerical parity
        assert torch.allclose(item["input"], input_tensor[index])
        assert torch.allclose(item["target"].y, target_graph[index].y)

    # Graph - tensor
    elif case == ["graph", "tensor"]:

        # Define the condition
        input_graph, target_tensor = _create_graph_data(
            is_input=True, use_lt=use_lt
        )
        condition = Condition(input=input_graph, target=target_tensor)

        # Extract item using __getitem__
        index = 0
        item = condition[index]

        # Assert correct types
        assert isinstance(item, dict)
        assert isinstance(item["input"], LabelBatch)
        assert item["input"].num_graphs == 1
        _assert_tensor_type(item["target"], use_lt)

        # Assert numerical parity
        assert torch.allclose(item["target"], target_tensor[index])
        assert torch.allclose(item["input"].x, input_graph[index].x)


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize(
    "case", [["tensor", "tensor"], ["tensor", "graph"], ["graph", "tensor"]]
)
def test_materialize(use_lt, case):

    # Tensor - tensor
    if case == ["tensor", "tensor"]:

        # Define the condition
        input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
        condition = Condition(input=input_tensor, target=target_tensor)

        # Materialize the batch for the given ids
        idx = [0, 2]
        batch = condition.materialize(idx)

        # Check that the batch is a dictionary holding the data
        assert isinstance(batch, dict)
        assert "input" in batch
        assert "target" in batch

        # Create expected input and target batches
        expected_input = torch.stack([input_tensor[i] for i in idx])
        expected_target = torch.stack([target_tensor[i] for i in idx])

        # Assert that the batch input and target are correct
        assert torch.allclose(batch["input"], expected_input)
        assert torch.allclose(batch["target"], expected_target)
        assert batch["input"].shape == expected_input.shape
        assert batch["target"].shape == expected_target.shape

    # Tensor - graph
    elif case == ["tensor", "graph"]:

        # Define the condition
        target_graph, input_tensor = _create_graph_data(
            is_input=False, use_lt=use_lt
        )
        condition = Condition(input=input_tensor, target=target_graph)

        # Materialize the batch for the given ids
        idx = [0, 2]
        batch = condition.materialize(idx)

        # Check that the batch is a dictionary holding the data
        assert isinstance(batch, dict)
        assert "input" in batch
        assert "target" in batch

        # Create expected input and target batches
        expected_input = torch.cat([input_tensor[i] for i in idx])

        # Assert that the batch input and target are correct
        assert torch.allclose(batch["input"], expected_input)
        assert batch["input"].shape == expected_input.shape
        assert batch["target"].num_graphs == len(idx)
        assert torch.allclose(
            batch["target"].y,
            torch.cat([g.y for g in [target_graph[i] for i in idx]]),
        )

    # Graph - tensor
    elif case == ["graph", "tensor"]:

        # Define the condition
        input_graph, target_tensor = _create_graph_data(
            is_input=True, use_lt=use_lt
        )
        condition = Condition(input=input_graph, target=target_tensor)

        # Materialize the batch for the given ids
        idx = [0, 2]
        batch = condition.materialize(idx)

        # Check that the batch is a dictionary holding the data
        assert isinstance(batch, dict)
        assert "input" in batch
        assert "target" in batch

        # Create expected input and target batches
        expected_target = torch.cat([target_tensor[i] for i in idx])

        # Assert that the batch input and target are correct
        assert batch["input"].num_graphs == len(idx)
        assert torch.allclose(
            batch["input"].x,
            torch.cat([g.x for g in [input_graph[i] for i in idx]]),
        )
        assert torch.allclose(batch["target"], expected_target)
        assert batch["target"].shape == expected_target.shape


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize(
    "case", [["tensor", "tensor"], ["tensor", "graph"], ["graph", "tensor"]]
)
def test_evaluate(case, use_lt):

    # Tensor - tensor
    if case == ["tensor", "tensor"]:

        # Define the input and the target
        input_, target_ = _create_tensor_data(use_lt=use_lt)
        input_vars = input_.labels if use_lt else None
        output_vars = target_.labels if use_lt else None

        # Define the condition and the solver
        condition = Condition(input=input_, target=target_)
        solver = DummySolver(use_lt, input_vars, output_vars)
        loss_fn = torch.nn.MSELoss(reduction="none")

        # Extract the batch
        batch = {"input": condition.input, "target": condition.target}

    # Tensor - graph
    elif case == ["tensor", "graph"]:

        # Define the input and the target
        target_, input_ = _create_graph_data(is_input=False, use_lt=use_lt)
        input_vars = input_.labels if use_lt else None
        output_vars = target_[0].y.labels if use_lt else None

        # Define the condition and the solver
        condition = Condition(input=input_, target=target_)
        solver = DummySolver(use_lt, input_vars, output_vars)
        loss_fn = torch.nn.MSELoss(reduction="none")

        # Extract the batch
        batch = {
            "input": condition.input,
            "target": LabelBatch.from_data_list(condition.target).y.reshape(
                n_graphs, n_nodes, -1
            ),
        }

    # Graph - tensor
    elif case == ["graph", "tensor"]:

        # Define the input and the target
        input_, target_ = _create_graph_data(is_input=True, use_lt=use_lt)
        input_vars = input_[0].x.labels if use_lt else None
        output_vars = target_.labels if use_lt else None

        # Define the condition and the solver
        condition = Condition(input=input_, target=target_)
        solver = DummySolver(use_lt, input_vars, output_vars)
        loss_fn = torch.nn.MSELoss(reduction="none")

        # Extract the batch
        batch = {
            "input": LabelBatch.from_data_list(condition.input),
            "target": condition.target,
        }

    # Evaluate the condition and compute the expected residual
    residual = condition.evaluate(batch, solver)
    expected = solver.forward(batch["input"]) - batch["target"]

    # Assert that the evaluated residual is correct
    assert torch.allclose(residual, expected)
