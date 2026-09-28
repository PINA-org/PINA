import pytest
import torch

from pina import Condition, LabelTensor
from pina._src.core.graph import LabelBatch
from pina._src.core.utils import labelize_forward
from pina.condition import DataCondition
from pina.graph import RadiusGraph

# Number of graphs and tensor samples for testing
n_samples = 10
n_graphs = 10
n_nodes = 20


# Helper function to create tensor data
def _create_tensor_data(use_lt, conditional_variables):

    # If LabelTensor is used, create tensors with labels
    if use_lt:
        input_tensor = LabelTensor(torch.rand((n_samples, 3)), ["x", "y", "z"])
        cond_vars = LabelTensor(torch.rand((n_samples, 1)), ["a"])
        cond_vars = cond_vars if conditional_variables else None

        return input_tensor, cond_vars

    # Standard torch.Tensor without labels
    input_tensor = torch.rand((n_samples, 3))
    cond_vars = torch.rand((n_samples, 1))
    cond_vars = cond_vars if conditional_variables else None

    return input_tensor, cond_vars


# Helper function to create graph data
def _create_graph_data(use_lt, conditional_variables):

    # If LabelTensor is used, create graph data with LabelTensors
    if use_lt:
        x = LabelTensor(torch.rand(n_graphs, n_nodes, 2), ["u", "v"])
        pos = LabelTensor(torch.rand(n_graphs, n_nodes, 2), ["x", "y"])
        cond_vars = LabelTensor(torch.rand(n_graphs, n_nodes, 1), ["f"])

    # Standard torch.Tensor without labels
    else:
        x = torch.rand(n_graphs, n_nodes, 2)
        pos = torch.rand(n_graphs, n_nodes, 2)
        cond_vars = torch.rand(n_graphs, n_nodes, 1)

    # Create a list of Graphs
    graph = [
        RadiusGraph(pos=pos[i], radius=0.1, x=x[i], cond_vars=cond_vars[i])
        for i in range(len(x))
    ]

    # Create conditional variables if needed
    cond_vars = cond_vars if conditional_variables else None

    return graph, cond_vars


# Helper function to check tensor types
def _assert_tensor_type(t, use_lt):
    if use_lt:
        assert isinstance(t, LabelTensor)
    else:
        assert isinstance(t, torch.Tensor) and not isinstance(t, LabelTensor)


# Helper function to check input graph
def _assert_graph_type(graph_list, use_lt):
    assert isinstance(graph_list, list)
    for graph in graph_list:
        _assert_tensor_type(graph.x, use_lt)


# Define a dummy solver for testing
class DummySolver:

    def __init__(self, use_lt, input_vars, cond_vars):
        if use_lt:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=input_vars,
                output_variables="z",
            )

        self.cond_vars = cond_vars
        self._params = None

    def forward(self, pts):

        # Tensor case
        if isinstance(pts, torch.Tensor):
            factor = self.cond_vars if self.cond_vars is not None else 1.0
            return pts.mean(dim=-1, keepdim=True) * factor

        # Graph case
        else:
            factor = pts.cond_vars if pts.cond_vars is not None else 1.0
            output_ = pts.x.mean(dim=-1, keepdim=True) * factor
            return output_.reshape(n_graphs, n_nodes, 1)


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("conditional_variables", [False, True])
@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_constructor(case, use_lt, conditional_variables):

    # Tensor input case
    if case == "tensor":

        # Define the condition
        input_tensor, cond_vars = _create_tensor_data(
            use_lt, conditional_variables
        )
        condition = Condition(
            input=input_tensor, conditional_variables=cond_vars
        )

        # Assert correct types
        assert isinstance(condition, DataCondition)
        _assert_tensor_type(condition.input, use_lt)
        if cond_vars is not None:
            _assert_tensor_type(condition.conditional_variables, use_lt)

        # Assert numerical parity
        assert torch.allclose(condition.input, input_tensor)
        if cond_vars is not None:
            assert torch.allclose(condition.conditional_variables, cond_vars)

        # Assert labels if LabelTensor is used
        if use_lt:
            assert condition.input.labels == ["x", "y", "z"]
            if cond_vars is not None:
                assert condition.conditional_variables.labels == ["a"]

    # Graph input case
    elif case == "graph":

        # Define the condition
        input_graph, cond_vars = _create_graph_data(
            use_lt, conditional_variables
        )
        condition = Condition(
            input=input_graph, conditional_variables=cond_vars
        )

        # Assert correct types
        assert isinstance(condition, DataCondition)
        _assert_graph_type(condition.input, use_lt)
        if cond_vars is not None:
            _assert_tensor_type(condition.conditional_variables, use_lt)

        # Assert numerical parity for graph inputs
        for i in range(len(input_graph)):
            assert torch.allclose(condition.input[i].x, input_graph[i].x)
            assert torch.allclose(condition.input[i].pos, input_graph[i].pos)

        # Assert numerical parity for conditional variables
        if cond_vars is not None:
            assert torch.allclose(condition.conditional_variables, cond_vars)

        # Assert labels if LabelTensor is used
        if use_lt:
            for graph in condition.input:
                assert graph.x.labels == ["u", "v"]
                assert graph.pos.labels == ["x", "y"]
            if cond_vars is not None:
                assert condition.conditional_variables.labels == ["f"]

    # Prepare for invalid input tests
    input_ = input_tensor if case == "tensor" else input_graph

    # Should fail if the input is neither a tensor nor a graph
    with pytest.raises(ValueError):
        Condition(input="invalid_input", conditional_variables=cond_vars)

    # Should fail if the conditional_variables is neither a tensor nor a graph
    with pytest.raises(ValueError):
        Condition(input=input_, conditional_variables="invalid_cond_vars")

    # Should fail if the input is a list of tensors
    if case == "tensor":
        with pytest.raises(ValueError):
            Condition(input=[input_], conditional_variables=cond_vars)

    # Should fail if the conditional_variables is a list of tensors
    if case == "tensor":
        with pytest.raises(ValueError):
            Condition(input=input_, conditional_variables=[cond_vars])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("conditional_variables", [False, True])
@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_get_item(case, use_lt, conditional_variables):

    # Tensor input case
    if case == "tensor":

        # Define the condition
        input_tensor, cond_vars = _create_tensor_data(
            use_lt, conditional_variables
        )
        condition = Condition(
            input=input_tensor, conditional_variables=cond_vars
        )

        # Extract item using materialize
        index = 0
        item = condition.materialize([index])

        # Assert correct types
        assert isinstance(item, dict)
        _assert_tensor_type(item["input"], use_lt)
        if cond_vars is not None:
            _assert_tensor_type(item["conditional_variables"], use_lt)

        # Assert numerical parity
        assert torch.allclose(item["input"][0], input_tensor[index])
        if cond_vars is not None:
            assert torch.allclose(
                item["conditional_variables"][0], cond_vars[index]
            )

    # Graph input case
    elif case == "graph":

        # Define the condition
        input_graph, cond_vars = _create_graph_data(
            use_lt, conditional_variables
        )
        condition = Condition(
            input=input_graph, conditional_variables=cond_vars
        )

        # Extract item using materialize
        index = 0
        item = condition.materialize([index])

        # Assert correct types
        assert isinstance(item, dict)
        assert isinstance(item["input"], LabelBatch)
        assert item["input"].num_graphs == 1
        if cond_vars is not None:
            _assert_tensor_type(item["conditional_variables"], use_lt)

        # Assert numerical parity
        assert torch.allclose(item["input"].x, input_graph[index].x)
        assert torch.allclose(item["input"].pos, input_graph[index].pos)
        if cond_vars is not None:
            assert torch.allclose(
                item["conditional_variables"], cond_vars[index]
            )


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("conditional_variables", [False, True])
@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_materialize(case, use_lt, conditional_variables):

    # Tensor case
    if case == "tensor":
        input_, cond_vars = _create_tensor_data(use_lt, conditional_variables)

    # Graph case
    elif case == "graph":
        input_, cond_vars = _create_graph_data(use_lt, conditional_variables)

    # Define the condition
    condition = Condition(input=input_, conditional_variables=cond_vars)

    # Materialize the batch for the given ids
    idx = [0, 2]
    batch = condition.materialize(idx)

    # Check that the batch is a dictionary holding the data
    assert isinstance(batch, dict)
    assert "input" in batch
    if cond_vars is not None:
        assert "conditional_variables" in batch

    # Retrieve tensor class for expected batch creation
    cls = LabelTensor if use_lt else torch

    # Validate batch contents for tensor case
    if case == "tensor":

        # Create expected input batch
        expected_input = cls.stack([input_[i] for i in idx])
        if cond_vars is not None:
            exp_cond = cls.stack([cond_vars[i] for i in idx])

        # Assert that the batch input is correct
        assert torch.allclose(batch["input"], expected_input)
        assert batch["input"].shape == expected_input.shape
        if cond_vars is not None:
            assert torch.allclose(batch["conditional_variables"], exp_cond)
            assert batch["conditional_variables"].shape == exp_cond.shape

    # Validate batch contents for graph case
    elif case == "graph":

        if cond_vars is not None:
            exp_cond = cls.cat([cond_vars[i] for i in idx])

        # Assert that the batch input is correct
        for pos, i in enumerate(idx):
            assert torch.allclose(batch["input"][pos].x, input_[i].x)
        assert batch["input"].num_graphs == len(idx)
        if cond_vars is not None:
            assert torch.allclose(batch["conditional_variables"], exp_cond)
            assert batch["conditional_variables"].shape == exp_cond.shape


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("conditional_variables", [False, True])
@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_evaluate(case, use_lt, conditional_variables):

    # Tensor case
    if case == "tensor":

        # Define the input and the target
        input_, cond_vars = _create_tensor_data(use_lt, conditional_variables)
        input_vars = input_.labels if use_lt else None

        # Define the condition and the solver
        condition = Condition(input=input_, conditional_variables=cond_vars)
        solver = DummySolver(use_lt, input_vars, cond_vars)
        loss_fn = torch.nn.MSELoss(reduction="none")

        # Extract the batch
        batch = {
            "input": condition.input,
            "conditional_variables": condition.conditional_variables,
        }

    # Graph case
    elif case == "graph":

        # Define the input and the target
        input_, cond_vars = _create_graph_data(use_lt, conditional_variables)
        input_vars = input_[0].x.labels if use_lt else None

        # Define the condition and the solver
        condition = Condition(input=input_, conditional_variables=cond_vars)
        solver = DummySolver(use_lt, input_vars, cond_vars)
        loss_fn = torch.nn.MSELoss(reduction="none")

        # Extract the batch
        batch = {
            "input": LabelBatch.from_data_list(condition.input),
            "conditional_variables": condition.conditional_variables,
        }

    # Evaluate the condition and compute the expected value
    loss = condition.evaluate(batch, solver)
    expected = solver.forward(batch["input"])

    # Assert that the evaluated loss is correct
    assert torch.allclose(loss, expected)
