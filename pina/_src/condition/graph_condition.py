"""Module for the Graph Data Condition class."""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from pina._src.condition.base_condition import BaseCondition
from pina._src.core.graph import Graph
from pina._src.core.label_tensor import LabelTensor


def _is_graph(value):
    """Whether ``value`` is graph data or a collection of graph data."""
    if isinstance(value, (Graph, Data)):
        return True
    return isinstance(value, (list, tuple))


class GraphCondition(BaseCondition):
    """Base condition for graph data stored as a list of graph batches."""

    def __init__(self, graph, key="x"):
        super().__init__()
        self.graph = graph
        self.key = key

    def store_data(self, **kwargs):
        """Create and store one subgraph per sampled starting node."""
        n_subgraphs = kwargs.get("n_subgraphs", 10)
        num_hops = kwargs.get("num_hops", 5)

        start_nodes = torch.randint(0, self.graph.num_nodes, (n_subgraphs,))
        self.data = [
            self.create_subgraph(start_node, num_hops=num_hops)
            for start_node in start_nodes
        ]
        return self.data

    def create_subgraph(self, start_node, num_hops=5):
        """Create the ``num_hops``-hop subgraph centered on ``start_node``."""
        subset = k_hop_subgraph(
            node_idx=int(start_node),
            num_hops=num_hops,
            edge_index=self.graph.edge_index,
            relabel_nodes=True,
            num_nodes=self.graph.num_nodes,
        )[0]
        subgraph = self.graph.subgraph(subset)

        data = getattr(self.graph, self.key)
        if isinstance(data, LabelTensor):
            setattr(
                subgraph,
                self.key,
                LabelTensor(getattr(subgraph, self.key), data.labels),
            )
        return subgraph

    def __len__(self):
        return len(self.data)

    def materialize(self, id, device=None):
        """Return the graph batch with the given id."""
        batch = self.data[int(id)]
        return batch.to(device) if device is not None else batch

    def __getitem__(self, id):
        return self.materialize(id)
