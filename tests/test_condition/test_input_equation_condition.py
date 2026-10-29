import pytest
import torch

from pina import Condition, LabelTensor
from pina._src.core.graph import LabelBatch
from pina._src.core.utils import labelize_forward
from pina.condition import InputEquationCondition
from pina.equation.zoo import FixedValue
from pina.graph import Graph, RadiusGraph

# Number of graphs and tensor samples for testing
n_samples = 10
n_graphs = 10
n_nodes = 20

# Generate input data for testing - tensor case
input_tensor = LabelTensor(torch.rand((n_samples, 2)), ["x", "y"])

# Generate input and equation data for testing - graph case
input_graph_list = [
    RadiusGraph(
        x=LabelTensor(torch.rand(n_nodes, 2), labels=["u", "v"]),
        pos=LabelTensor(torch.rand(n_nodes, 2), labels=["x", "y"]),
        radius=0.1,
        edge_attr=True,
    )
    for _ in range(n_graphs)
]

# Initialize the testing equation
test_equation = FixedValue(0.0)


# Define a dummy solver for testing
class DummySolver:

    def __init__(self, input_vars, output_vars):

        self.forward = labelize_forward(
            forward=self.forward,
            input_variables=input_vars,
            output_variables=output_vars,
        )

        self._params = None

    def forward(self, samples):
        return samples


@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_constructor(case):

    # Tensor case
    if case == "tensor":
        input_, equation = input_tensor, test_equation

    # Graph case
    elif case == "graph":
        input_, equation = input_graph_list, test_equation

    # Define the condition
    condition = Condition(input=input_, equation=equation)

    # Assert correct types
    assert isinstance(condition, InputEquationCondition)

    # Assert that the equation is stored correctly
    assert condition.equation is equation

    # Assert correct input type
    if case == "tensor":
        assert isinstance(condition.input, LabelTensor)

    elif case == "graph":
        assert isinstance(condition.input, list)
        for graph in condition.input:
            assert isinstance(graph, Graph)

    # Should fail if input is not an instance of LabelTensor or Graph
    with pytest.raises(ValueError):
        Condition(input=torch.rand(n_samples, 2), equation=equation)

    # Should fail if equation is not an instance of BaseEquation
    with pytest.raises(ValueError):
        Condition(input=input_, equation="not_an_equation")

    # Should fail if input is a list with wrong elements
    with pytest.raises(ValueError):
        Condition(
            input=[LabelTensor(torch.rand(n_samples, 2), ["x", "y"])],
            equation=equation,
        )


@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_get_item(case):

    # Tensor case
    if case == "tensor":
        input_, equation = input_tensor, test_equation

    # Graph case
    elif case == "graph":
        input_, equation = input_graph_list, test_equation

    # Define the condition
    condition = Condition(input=input_, equation=equation)

    # Extract item using materialize
    index = 0
    item = condition.materialize([index])

    # Assert correct types and numerical parity
    assert isinstance(item, dict)
    if case == "tensor":
        assert "input" in item
        assert isinstance(item["input"], LabelTensor)
        assert torch.allclose(item["input"][0], input_[index])

    elif case == "graph":
        assert "input" in item
        assert isinstance(item["input"], LabelBatch)
        assert item["input"].num_graphs == 1
        assert torch.allclose(item["input"].x, input_[index].x)


@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_materialize(case):

    # Tensor case
    if case == "tensor":
        input_, equation = input_tensor, test_equation

    # Graph case
    elif case == "graph":
        input_, equation = input_graph_list, test_equation

    # Define the condition
    condition = Condition(input=input_, equation=equation)

    # Materialize the batch for the given ids
    idx = [0, 2]
    batch = condition.materialize(idx)

    # Check that the batch is a dictionary holding the input data
    assert isinstance(batch, dict)
    assert "input" in batch

    # Validate batch contents for tensor case
    if case == "tensor":

        # Create expected input batch
        expected_input = LabelTensor.stack([input_[i] for i in idx])

        # Assert that the batch input is correct
        assert torch.allclose(batch["input"], expected_input)
        assert batch["input"].shape == expected_input.shape

    # Validate batch contents for graph case
    elif case == "graph":

        # Create expected input batch
        expected_input = [input_[i] for i in idx]

        # Assert that the batch input is correct
        for i, graph in enumerate(expected_input):
            assert torch.allclose(batch["input"][i].x, graph.x)
        assert batch["input"].num_graphs == len(idx)


@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_evaluate(case):

    # Tensor case
    if case == "tensor":

        # Define the input and the target
        input_, equation = input_tensor, test_equation
        input_vars = input_.labels
        output_vars = ["z", "t"]

        # Define the condition and the solver
        condition = Condition(input=input_, equation=equation)
        solver = DummySolver(input_vars=input_vars, output_vars=output_vars)
        loss_fn = torch.nn.MSELoss(reduction="none")

        # Extract the batch
        batch = {"input": condition.input}

    # Graph case
    elif case == "graph":

        # Define the input and the target
        input_, equation = input_graph_list, test_equation
        input_vars = input_[0].x.labels
        output_vars = ["z", "t"]

        # Define the condition and the solver
        condition = Condition(input=input_, equation=equation)
        solver = DummySolver(input_vars=input_vars, output_vars=output_vars)
        loss_fn = torch.nn.MSELoss(reduction="none")

        # Extract the batch
        batch = {"input": LabelBatch.from_data_list(condition.input).x}

    # Evaluate the condition and compute the expected residuals
    residual = condition.evaluate(batch, solver)
    expected = solver.forward(batch["input"])

    # Assert that the evaluated residual is correct
    assert torch.allclose(residual, expected)
