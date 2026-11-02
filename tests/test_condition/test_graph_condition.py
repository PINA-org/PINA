import torch

from pina import LabelTensor
from pina.condition import GraphCondition
from pina.graph import KNNGraph

n_nodes = 10


class DummyGraphCondition(GraphCondition):
    def evaluate(self, batch, solver):
        return batch[self.key]


def _create_graph(labels=False):
    x = torch.rand(n_nodes, 2)
    if labels:
        x = LabelTensor(x, ["u", "v"])
    return KNNGraph(
        x=x,
        pos=torch.rand(n_nodes, 3),
        neighbours=3,
        edge_attr=True,
    )


def test_constructor():
    graph = _create_graph()
    condition = DummyGraphCondition(graph, key="x")

    assert condition.graph is graph
    assert condition.key == "x"


def test_store_data():
    graph = _create_graph()
    condition = DummyGraphCondition(graph)
    condition.store_data(n_subgraphs=4, num_hops=1)

    assert len(condition) == 4
    assert len(condition.data) == 4
    for subgraph in condition.data:
        assert subgraph.num_nodes <= graph.num_nodes
        assert subgraph.edge_index.max() < subgraph.num_nodes


def test_create_subgraph():
    graph = _create_graph()
    condition = DummyGraphCondition(graph)

    subgraph = condition.create_subgraph(0, num_hops=1)

    assert subgraph.num_nodes <= graph.num_nodes
    assert subgraph.edge_index.max() < subgraph.num_nodes


def test_materialize():
    graph = _create_graph()
    condition = DummyGraphCondition(graph)
    condition.store_data(n_subgraphs=3, num_hops=1)

    assert condition.materialize(0) is condition.data[0]
    assert condition[1] is condition.data[1]
    assert condition.materialize(0, device="cpu").num_nodes == (
        condition.data[0].num_nodes
    )


def test_store_data_preserves_labels():
    graph = _create_graph(labels=True)
    condition = DummyGraphCondition(graph)
    condition.store_data(n_subgraphs=2, num_hops=1)

    for subgraph in condition.data:
        assert isinstance(subgraph.x, LabelTensor)
        assert subgraph.x.labels == ["u", "v"]
