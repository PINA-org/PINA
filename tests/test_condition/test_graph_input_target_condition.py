import pytest
import torch

from pina import LabelTensor
from pina._src.core.utils import labelize_forward
from pina.condition import GraphCondition, GraphInputTargetCondition
from pina.graph import Graph

n_nodes = 10
input_variables = ["a", "b"]
output_variables = ["u"]


def _create_graph(use_lt=False):
    x = torch.rand((n_nodes, len(input_variables)))
    y = torch.rand((n_nodes, len(output_variables)))
    pos = torch.rand((n_nodes, 2))

    if use_lt:
        x = LabelTensor(x, input_variables)
        y = LabelTensor(y, output_variables)
        pos = LabelTensor(pos, ["x", "y"])

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        ]
    )
    return Graph(x=x, y=y, pos=pos, edge_index=edge_index)


class DummyGraphSolver:
    def __init__(self, use_lt):
        if use_lt:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=input_variables,
                output_variables=output_variables,
            )

    def forward(self, graph):
        return graph.x.mean(dim=-1, keepdim=True)


@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor(use_lt):
    graph = _create_graph(use_lt=use_lt)
    condition = GraphInputTargetCondition(graph=graph)

    assert isinstance(condition, GraphInputTargetCondition)
    assert isinstance(condition, GraphCondition)
    assert condition.graph is graph
    assert condition.key == "x"
    assert condition.output_key == "y"


@pytest.mark.parametrize("use_lt", [True, False])
def test_store_and_materialize(use_lt):
    graph = _create_graph(use_lt=use_lt)
    condition = GraphInputTargetCondition(graph=graph)
    condition.store_data(n_subgraphs=3, num_hops=1)

    assert len(condition) == 3

    batch = condition.materialize(0)

    assert isinstance(batch, Graph)
    assert batch.x.shape[0] == batch.y.shape[0]


@pytest.mark.parametrize("use_lt", [True, False])
def test_evaluate(use_lt):
    graph = _create_graph(use_lt=use_lt)
    condition = GraphInputTargetCondition(graph=graph)
    condition.store_data(n_subgraphs=1, num_hops=1)

    batch = condition.materialize(0)
    solver = DummyGraphSolver(use_lt=use_lt)

    residual = condition.evaluate(batch, solver)
    expected = solver.forward(batch) - batch.y

    assert residual.shape == expected.shape
    torch.testing.assert_close(residual, expected)
