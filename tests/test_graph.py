import pytest
import torch
from pina import LabelTensor
from pina.graph import RadiusGraph, KNNGraph, Graph
from torch_geometric.data import Data
from torch_geometric.sampler import BaseSampler, SamplerOutput


# Deterministic sampler returning the induced subgraph of the seed nodes
class InducedSampler(BaseSampler):
    def __init__(self, data):
        self.data = data

    def sample_from_nodes(self, index, **kwargs):
        seeds = index.node
        edge_index = self.data.edge_index
        mask = torch.isin(edge_index[0], seeds) & torch.isin(
            edge_index[1], seeds
        )
        mapping = torch.full((self.data.num_nodes,), -1, dtype=torch.long)
        mapping[seeds] = torch.arange(seeds.numel())
        edge = mask.nonzero(as_tuple=False).view(-1)
        return SamplerOutput(
            node=seeds,
            row=mapping[edge_index[0][mask]],
            col=mapping[edge_index[1][mask]],
            edge=edge,
        )


# Sampler returning an invalid object instead of a SamplerOutput
class InvalidOutputSampler(BaseSampler):
    def sample_from_nodes(self, index, **kwargs):
        return "not a sampler output"


# Sampler not providing the sampled edge ids
class NoEdgeIdSampler(BaseSampler):
    def sample_from_nodes(self, index, **kwargs):
        seeds = index.node
        return SamplerOutput(
            node=seeds,
            row=torch.arange(seeds.numel()),
            col=torch.arange(seeds.numel()),
            edge=None,
        )


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
    "x, pos",
    [
        (torch.rand(10, 2), torch.rand(10, 3)),
        (
            LabelTensor(torch.rand(10, 2), ["u", "v"]),
            LabelTensor(torch.rand(10, 3), ["x", "y", "z"]),
        ),
    ],
)
def test_create_subgraph(x, pos):
    graph = KNNGraph(
        x=x,
        pos=pos,
        neighbours=3,
        edge_attr=True,
        scale=torch.tensor(0.5),
    )
    subgraphs = graph.create_subgraph(InducedSampler(graph), batch_size=4)

    # The seed nodes are split into chunks of batch_size nodes
    assert len(subgraphs) == 3
    covered = torch.cat([sub.seed_n_id for sub in subgraphs])
    assert torch.equal(covered, torch.arange(10))

    for sub in subgraphs:
        seeds = sub.seed_n_id
        assert isinstance(sub, Graph)
        assert sub.num_nodes == seeds.numel()

        # Node-level attributes are sliced with the sampled nodes
        assert torch.allclose(sub.x, graph.x[seeds])
        assert torch.allclose(sub.pos, graph.pos[seeds])
        if isinstance(x, LabelTensor):
            assert isinstance(sub.x, LabelTensor)
            assert sub.x.labels == x.labels
            assert isinstance(sub.pos, LabelTensor)
            assert sub.pos.labels == pos.labels

        # Edge-level attributes are sliced with the sampled edges
        assert torch.allclose(sub.edge_attr, graph.edge_attr[sub.e_id])
        assert sub.edge_attr.shape[0] == sub.edge_index.shape[1]

        # Graph-level attributes are copied as they are
        assert torch.allclose(sub.scale, graph.scale)

        # Provenance attributes
        assert torch.equal(sub.n_id, seeds)
        assert sub.seed_n_id.numel() <= 4

        # The local edge index refers to the subgraph nodes
        assert sub.edge_index.min() >= 0
        assert sub.edge_index.max() < sub.num_nodes
        original_edges = set(
            zip(
                graph.edge_index[0].tolist(),
                graph.edge_index[1].tolist(),
            )
        )
        for row, col in zip(
            seeds[sub.edge_index[0]].tolist(),
            seeds[sub.edge_index[1]].tolist(),
        ):
            assert (row, col) in original_edges


def test_create_subgraph_seed_nodes():
    graph = KNNGraph(x=torch.rand(10, 2), pos=torch.rand(10, 3), neighbours=3)
    subgraphs = graph.create_subgraph(
        InducedSampler(graph), batch_size=2, seed_nodes=[8, 9, 1]
    )
    assert len(subgraphs) == 2
    assert subgraphs[0].seed_n_id.tolist() == [8, 9]
    assert subgraphs[1].seed_n_id.tolist() == [1]
    assert torch.allclose(subgraphs[0].x, graph.x[torch.tensor([8, 9])])


def test_create_subgraph_no_edge_attr():
    # Without edge-level attributes the sampler does not need edge ids
    graph = KNNGraph(x=torch.rand(10, 2), pos=torch.rand(10, 3), neighbours=3)
    subgraphs = graph.create_subgraph(NoEdgeIdSampler(), seed_nodes=[0, 1])
    assert len(subgraphs) == 2
    assert "e_id" not in subgraphs[0]


def test_create_subgraph_invalid_sampler():
    graph = KNNGraph(x=torch.rand(10, 2), pos=torch.rand(10, 3), neighbours=3)
    with pytest.raises(TypeError, match="sample_from_nodes"):
        graph.create_subgraph(object())
    with pytest.raises(TypeError, match="sample_from_nodes"):
        graph.create_subgraph(InducedSampler)
    with pytest.raises(TypeError, match="SamplerOutput"):
        graph.create_subgraph(InvalidOutputSampler())


def test_create_subgraph_invalid_batch_size():
    graph = KNNGraph(x=torch.rand(10, 2), pos=torch.rand(10, 3), neighbours=3)
    with pytest.raises(ValueError, match="batch_size"):
        graph.create_subgraph(InducedSampler(graph), batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        graph.create_subgraph(InducedSampler(graph), batch_size=1.5)


def test_create_subgraph_missing_edge_ids():
    graph = KNNGraph(
        x=torch.rand(10, 2),
        pos=torch.rand(10, 3),
        neighbours=3,
        edge_attr=True,
    )
    with pytest.raises(ValueError, match="edge-level"):
        graph.create_subgraph(NoEdgeIdSampler(), seed_nodes=[0, 1, 2])
