import pytest
import torch
from torch_geometric.sampler import BaseSampler, SamplerOutput

from pina._src.core.graph import LabelBatch
from pina.condition import InputEquationCondition, InputTargetCondition
from pina.data import DataModule
from pina.equation.zoo import FixedValue
from pina.graph import Graph, KNNGraph
from pina.problem.zoo import SupervisedProblem

# Number of nodes of the graph used for testing
n_nodes = 10


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


def _create_graph():
    return KNNGraph(
        x=torch.rand(n_nodes, 2),
        pos=torch.rand(n_nodes, 3),
        neighbours=3,
        edge_attr=True,
    )


def test_store_data_single_graph():
    graph = _create_graph()
    condition = InputTargetCondition(input=graph, target=torch.rand(1, 5))

    assert len(condition) == 1
    batch = condition[0]
    assert isinstance(batch["input"], LabelBatch)
    assert batch["input"].num_graphs == 1
    assert batch["target"].shape == (5,)
    assert torch.allclose(batch["target"], condition.target[0])


def test_store_data_graph_list():
    graphs = [_create_graph() for _ in range(4)]
    condition = InputTargetCondition(input=graphs, target=torch.rand(4, 5))

    assert len(condition) == 4
    batch = condition.materialize([0, 3])
    assert isinstance(batch["input"], LabelBatch)
    assert batch["input"].num_graphs == 2
    assert batch["target"].shape == (10,)


def test_store_data_with_sampler():
    graph = _create_graph()
    condition = InputTargetCondition(input=graph, target=torch.rand(1, 5))

    condition.store_data(
        input=graph,
        target=torch.rand(3, 5),
        sampler=InducedSampler(graph),
        batch_size=4,
    )

    assert len(condition) == 3
    assert all(isinstance(sub, Graph) for sub in condition.data.input)
    assert condition.data.target.shape == (3, 5)

    batch = condition.materialize([0, 2])
    assert isinstance(batch["input"], LabelBatch)
    assert batch["input"].num_graphs == 2
    assert batch["target"].shape == (10,)

    item = condition[1]
    assert item["input"].num_graphs == 1
    assert item["target"].shape == (5,)
    assert torch.allclose(item["target"], condition.data.target[1])


def test_store_data_with_sampler_default_batch_size():
    graph = _create_graph()
    condition = InputTargetCondition(input=graph, target=torch.rand(1, 5))

    condition.store_data(
        input=graph,
        target=torch.rand(n_nodes, 5),
        sampler=InducedSampler(graph),
    )

    # Without batch_size every node originates a subgraph
    assert len(condition) == n_nodes


def test_store_data_with_sampler_target_mismatch():
    graph = _create_graph()
    condition = InputTargetCondition(input=graph, target=torch.rand(1, 5))

    with pytest.raises(ValueError, match="does not match"):
        condition.store_data(
            input=graph,
            target=torch.rand(2, 5),
            sampler=InducedSampler(graph),
            batch_size=4,
        )


def test_store_data_sampler_requires_single_graph():
    graphs = [_create_graph() for _ in range(2)]
    condition = InputTargetCondition(input=graphs, target=torch.rand(2, 5))

    with pytest.raises(TypeError, match="single pina Graph"):
        condition.store_data(
            input=graphs,
            target=torch.rand(2, 5),
            sampler=InducedSampler(graphs[0]),
            batch_size=4,
        )


def test_store_data_sampling_params_without_sampler():
    graph = _create_graph()
    condition = InputTargetCondition(input=graph, target=torch.rand(1, 5))

    with pytest.raises(ValueError, match="only available"):
        condition.store_data(input=graph, target=torch.rand(1, 5), batch_size=4)
    with pytest.raises(ValueError, match="only available"):
        condition.store_data(
            input=graph, target=torch.rand(1, 5), seed_nodes=[0, 1]
        )


def test_store_data_with_sampler_keeps_equation():
    graph = _create_graph()
    equation = FixedValue(0.0)
    condition = InputEquationCondition(input=graph, equation=equation)

    condition.store_data(
        input=graph,
        equation=equation,
        sampler=InducedSampler(graph),
        batch_size=4,
    )

    assert len(condition) == 3
    assert condition.equation is equation
    batch = condition.materialize([0])
    assert batch["input"].num_graphs == 1


def test_data_module_with_sampled_condition():
    graph = _create_graph()
    problem = SupervisedProblem(input_=graph, output_=torch.rand(1, 5))
    condition = problem.conditions["data"]

    # Sampling with batch_size=2 splits the 10 nodes into 5 subgraphs
    condition.store_data(
        input=graph,
        target=torch.rand(5, 5),
        sampler=InducedSampler(graph),
        batch_size=2,
    )
    assert len(condition) == 5

    dm = DataModule(
        problem=problem,
        train_size=0.6,
        val_size=0.4,
        test_size=0.0,
        batch_size=2,
        batching_mode="proportional",
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    assert len(dm.train_datasets["data"]) == 3
    assert len(dm.val_datasets["data"]) == 2

    # The dataloader yields ids that materialize into subgraph batches
    id_batch = next(iter(dm.train_dataloader()))
    ((name, batch),) = dm.transfer_batch_to_device(
        id_batch, torch.device("cpu"), 0
    )
    assert name == "data"
    assert isinstance(batch["input"], LabelBatch)
    assert batch["input"].num_graphs == len(id_batch["data"])
    assert batch["target"].shape == (len(id_batch["data"]) * 5,)
