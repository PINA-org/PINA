import pytest
import torch

from pina import LabelTensor
from pina._src.condition.graph_condition import GraphCondition
from pina._src.condition.graph_input_equation_condition import GraphInputEquationCondition
from pina.graph import Graph
from pina.equation.zoo import FixedValue


def _create_graph(use_lt=False):
    x = torch.rand(6, 2)
    if use_lt:
        x = LabelTensor(x, ["x", "y"])
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    return Graph(x=x, edge_index=edge_index, pos=torch.rand(6, 2))


class DummyGraphSolver:
    _params = None

    def forward(self, graph):
        return graph.x.sum(dim=-1, keepdim=True)


@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor_and_batching(use_lt):
    graph = _create_graph(use_lt=use_lt)
    equation = FixedValue(0.0)
    condition = GraphInputEquationCondition(graph=graph, equation=equation)

    assert isinstance(condition, GraphCondition)
    assert condition.graph is graph
    assert condition.key == "x"
    assert condition.equation is equation

    condition.store_data(n_subgraphs=2, num_hops=1)
    assert len(condition) == 2
    assert isinstance(condition.materialize(0), Graph)


def test_invalid_equation():
    with pytest.raises(ValueError):
        GraphInputEquationCondition(
            graph=_create_graph(), equation="not_an_equation"
        )


@pytest.mark.parametrize("use_lt", [True, False])
def test_evaluate(use_lt):
    condition = GraphInputEquationCondition(
        graph=_create_graph(use_lt=use_lt), equation=FixedValue(0.0)
    )
    condition.store_data(n_subgraphs=1, num_hops=1)
    batch = condition.materialize(0)

    residual = condition.evaluate(batch, DummyGraphSolver())
    expected = batch.x.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(residual, expected)
    assert batch.x.requires_grad
