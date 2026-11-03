import pytest
import torch
from pina import LabelTensor
from pina.graph import RadiusGraph, KNNGraph, Graph


def build_edge_attr(pos, edge_index):
    return torch.cat([pos[edge_index[0]], pos[edge_index[1]]], dim=-1)


@pytest.mark.parametrize(
    "x, pos",
    [
        (torch.rand(10, 2), torch.rand(10, 3)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
        ),
    ],
)
def test_build_graph(x, pos):
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
        dtype=torch.int64,
    )
    graph = Graph(x=x, pos=pos, edge_index=edge_index)
    assert hasattr(graph, "x")
    assert hasattr(graph, "pos")
    assert hasattr(graph, "edge_index")
    assert torch.isclose(graph.x, x).all()
    if isinstance(x, LabelTensor):
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == x.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert torch.isclose(graph.pos, pos).all()
    if isinstance(pos, LabelTensor):
        assert isinstance(graph.pos, LabelTensor)
        assert graph.pos.labels == pos.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
        dtype=torch.int64,
    )
    graph = Graph(x=x, edge_index=edge_index)
    assert hasattr(graph, "x")
    assert hasattr(graph, "pos")
    assert hasattr(graph, "edge_index")
    assert torch.isclose(graph.x, x).all()
    if isinstance(x, LabelTensor):
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == x.labels
    else:
        assert isinstance(graph.x, torch.Tensor)


@pytest.mark.parametrize(
    "x, pos",
    [
        (torch.rand(10, 2), torch.rand(10, 3)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
        ),
    ],
)
@pytest.mark.parametrize("loop", [True, False])
def test_build_radius_graph(x, pos, loop):
    graph = RadiusGraph(x=x, pos=pos, radius=0.5, loop=loop)
    assert hasattr(graph, "x")
    assert hasattr(graph, "pos")
    assert hasattr(graph, "edge_index")
    assert torch.isclose(graph.x, x).all()
    if isinstance(x, LabelTensor):
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == x.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert torch.isclose(graph.pos, pos).all()
    if isinstance(pos, LabelTensor):
        assert isinstance(graph.pos, LabelTensor)
        assert graph.pos.labels == pos.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    if not loop:
        assert (
            len(
                torch.nonzero(
                    graph.edge_index[0] == graph.edge_index[1], as_tuple=True
                )[0]
            )
            == 0
        )  # Detect self loops


@pytest.mark.parametrize(
    "x, pos",
    [
        (torch.rand(10, 2), torch.rand(10, 3)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
        ),
    ],
)
def test_build_radius_graph_edge_attr(x, pos):
    graph = RadiusGraph(x=x, pos=pos, radius=0.5, edge_attr=True)
    assert hasattr(graph, "x")
    assert hasattr(graph, "pos")
    assert hasattr(graph, "edge_index")
    assert torch.isclose(graph.x, x).all()
    if isinstance(x, LabelTensor):
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == x.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert torch.isclose(graph.pos, pos).all()
    if isinstance(pos, LabelTensor):
        assert isinstance(graph.pos, LabelTensor)
        assert graph.pos.labels == pos.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert hasattr(graph, "edge_attr")
    assert isinstance(graph.edge_attr, torch.Tensor)
    assert graph.edge_attr.shape[-1] == 3
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]


@pytest.mark.parametrize(
    "x, pos",
    [
        (torch.rand(10, 2), torch.rand(10, 3)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
        ),
    ],
)
def test_build_radius_graph_custom_edge_attr(x, pos):
    graph = RadiusGraph(
        x=x,
        pos=pos,
        radius=0.5,
        edge_attr=True,
        custom_edge_func=build_edge_attr,
    )
    assert hasattr(graph, "x")
    assert hasattr(graph, "pos")
    assert hasattr(graph, "edge_index")
    assert torch.isclose(graph.x, x).all()
    if isinstance(x, LabelTensor):
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == x.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert torch.isclose(graph.pos, pos).all()
    if isinstance(pos, LabelTensor):
        assert isinstance(graph.pos, LabelTensor)
        assert graph.pos.labels == pos.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert hasattr(graph, "edge_attr")
    assert isinstance(graph.edge_attr, torch.Tensor)
    assert graph.edge_attr.shape[-1] == 6
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]


@pytest.mark.parametrize(
    "x, pos",
    [
        (torch.rand(10, 2), torch.rand(10, 3)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
        ),
    ],
)
@pytest.mark.parametrize("loop", [True, False])
def test_build_knn_graph(x, pos, loop):
    graph = KNNGraph(x=x, pos=pos, neighbours=2, loop=loop)
    assert hasattr(graph, "x")
    assert hasattr(graph, "pos")
    assert hasattr(graph, "edge_index")
    assert torch.isclose(graph.x, x).all()
    if isinstance(x, LabelTensor):
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == x.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert torch.isclose(graph.pos, pos).all()
    if isinstance(pos, LabelTensor):
        assert isinstance(graph.pos, LabelTensor)
        assert graph.pos.labels == pos.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert graph.edge_attr is None
    self_loops = len(
        torch.nonzero(
            graph.edge_index[0] == graph.edge_index[1], as_tuple=True
        )[0]
    )
    if loop:
        assert self_loops != 0
    else:
        assert self_loops == 0


@pytest.mark.parametrize(
    "x, pos",
    [
        (torch.rand(10, 2), torch.rand(10, 3)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
        ),
    ],
)
def test_build_knn_graph_edge_attr(x, pos):
    graph = KNNGraph(x=x, pos=pos, neighbours=2, edge_attr=True)
    assert hasattr(graph, "x")
    assert hasattr(graph, "pos")
    assert hasattr(graph, "edge_index")
    assert torch.isclose(graph.x, x).all()
    if isinstance(x, LabelTensor):
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == x.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert torch.isclose(graph.pos, pos).all()
    if isinstance(pos, LabelTensor):
        assert isinstance(graph.pos, LabelTensor)
        assert graph.pos.labels == pos.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert isinstance(graph.edge_attr, torch.Tensor)
    assert graph.edge_attr.shape[-1] == 3
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]


@pytest.mark.parametrize(
    "x, pos",
    [
        (torch.rand(10, 2), torch.rand(10, 3)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
        ),
    ],
)
def test_build_knn_graph_custom_edge_attr(x, pos):
    graph = KNNGraph(
        x=x,
        pos=pos,
        neighbours=2,
        edge_attr=True,
        custom_edge_func=build_edge_attr,
    )
    assert hasattr(graph, "x")
    assert hasattr(graph, "pos")
    assert hasattr(graph, "edge_index")
    assert torch.isclose(graph.x, x).all()
    if isinstance(x, LabelTensor):
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == x.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert torch.isclose(graph.pos, pos).all()
    if isinstance(pos, LabelTensor):
        assert isinstance(graph.pos, LabelTensor)
        assert graph.pos.labels == pos.labels
    else:
        assert isinstance(graph.pos, torch.Tensor)
    assert isinstance(graph.edge_attr, torch.Tensor)
    assert graph.edge_attr.shape[-1] == 6
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]


@pytest.mark.parametrize(
    "x, pos, y",
    [
        (torch.rand(10, 2), torch.rand(10, 3), torch.rand(10, 4)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
            LabelTensor(torch.rand(10, 4), ["a", "b", "c", "d"]),
        ),
    ],
)
def test_additional_params(x, pos, y):
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
        dtype=torch.int64,
    )
    graph = Graph(x=x, pos=pos, edge_index=edge_index, y=y)
    assert hasattr(graph, "y")
    assert torch.isclose(graph.y, y).all()
    if isinstance(y, LabelTensor):
        assert isinstance(graph.y, LabelTensor)
        assert graph.y.labels == y.labels
    else:
        assert isinstance(graph.y, torch.Tensor)
    assert torch.isclose(graph.y, y).all()
    if isinstance(y, LabelTensor):
        assert isinstance(graph.y, LabelTensor)
        assert graph.y.labels == y.labels
    else:
        assert isinstance(graph.y, torch.Tensor)


@pytest.mark.parametrize(
    "x, pos, y",
    [
        (torch.rand(10, 2), torch.rand(10, 3), torch.rand(10, 4)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
            LabelTensor(torch.rand(10, 4), ["a", "b", "c", "d"]),
        ),
    ],
)
def test_additional_params_radius_graph(x, pos, y):
    graph = RadiusGraph(x=x, pos=pos, radius=0.5, y=y)
    assert hasattr(graph, "y")
    assert torch.isclose(graph.y, y).all()
    if isinstance(y, LabelTensor):
        assert isinstance(graph.y, LabelTensor)
        assert graph.y.labels == y.labels
    else:
        assert isinstance(graph.y, torch.Tensor)
    assert torch.isclose(graph.y, y).all()
    if isinstance(y, LabelTensor):
        assert isinstance(graph.y, LabelTensor)
        assert graph.y.labels == y.labels
    else:
        assert isinstance(graph.y, torch.Tensor)


@pytest.mark.parametrize(
    "x, pos, y",
    [
        (torch.rand(10, 2), torch.rand(10, 3), torch.rand(10, 4)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
            LabelTensor(torch.rand(10, 4), ["a", "b", "c", "d"]),
        ),
    ],
)
def test_additional_params_knn_graph(x, pos, y):
    graph = KNNGraph(x=x, pos=pos, neighbours=3, y=y)
    assert hasattr(graph, "y")
    assert torch.isclose(graph.y, y).all()
    if isinstance(y, LabelTensor):
        assert isinstance(graph.y, LabelTensor)
        assert graph.y.labels == y.labels
    else:
        assert isinstance(graph.y, torch.Tensor)
    assert torch.isclose(graph.y, y).all()
    if isinstance(y, LabelTensor):
        assert isinstance(graph.y, LabelTensor)
        assert graph.y.labels == y.labels
    else:
        assert isinstance(graph.y, torch.Tensor)


@pytest.mark.parametrize(
    "x, pos, y",
    [
        (torch.rand(10, 2), torch.rand(10, 3), torch.rand(10, 4)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
            LabelTensor(torch.rand(10, 4), ["a", "b", "c", "d"]),
        ),
    ],
)
def test_create_subgraph(x, pos, y):
    graph = KNNGraph(
        x=x,
        pos=pos,
        y=y,
        neighbours=3,
        edge_attr=True,
    )
    subgraph = graph.create_subgraph(0, num_hops=1)

    assert isinstance(subgraph, Graph)
    assert subgraph.num_nodes <= graph.num_nodes
    assert subgraph.edge_index.min() >= 0
    assert subgraph.edge_index.max() < subgraph.num_nodes

    if isinstance(x, LabelTensor):
        assert isinstance(subgraph.x, LabelTensor)
        assert subgraph.x.labels == x.labels
        assert isinstance(subgraph.pos, LabelTensor)
        assert subgraph.pos.labels == pos.labels
        assert isinstance(subgraph.y, LabelTensor)
        assert subgraph.y.labels == y.labels


def test_create_subgraph_num_hops():
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.int64,
    )
    graph = Graph(
        x=torch.rand(4, 2),
        pos=torch.rand(4, 2),
        edge_index=edge_index,
    )

    assert graph.create_subgraph(0, num_hops=1).num_nodes == 2
    assert graph.create_subgraph(0, num_hops=2).num_nodes == 3


def test_create_subgraph_preserves_edge_attr_labels():
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.int64,
    )
    edge_attr = LabelTensor(torch.rand(edge_index.shape[1], 1), ["w"])
    graph = Graph(
        x=torch.rand(4, 2),
        pos=torch.rand(4, 2),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    subgraph = graph.create_subgraph(0, num_hops=1)

    assert isinstance(subgraph.edge_attr, LabelTensor)
    assert subgraph.edge_attr.labels == edge_attr.labels
    assert subgraph.edge_attr.shape[0] == subgraph.edge_index.shape[1]
