"""Module for the Graph Data Condition class."""

from types import SimpleNamespace
import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from pina._src.condition.base_condition import BaseCondition
from pina._src.condition.tensor_condition import (
    _move_to_device,
    _normalize_ids,
)
from pina._src.core.graph import Graph, LabelBatch
from pina._src.core.label_tensor import LabelTensor


def _is_graph(value):
    """
    Whether the given value holds graph data.

    A value is considered graph data when it is a :class:`Graph`, a
    :class:`~torch_geometric.data.Data`, or a list / tuple of such types.

    :param value: The value to check.
    :return: Whether ``value`` holds graph data.
    :rtype: bool
    """
    if isinstance(value, (Graph, Data)):
        return True
    return isinstance(value, (list, tuple))


class GraphCondition(BaseCondition):
    """
    Abstract condition for graph-based data.

    It handles storage, slicing, and batching of :class:`Graph`,
    :class:`~torch_geometric.data.Data`, or lists / tuples of these types,
    together with any per-sample tensor field (e.g. a ``target``).
    Conditions built on graph data should inherit from this class.
    """

    def store_data(self, **kwargs):
        """
        Store the graph data and the associated per-graph tensors.

        When ``sampler`` is provided, the graph data is split into subgraphs
        with :meth:`~pina.graph.Graph.create_subgraph` and the subgraphs are
        stored as the samples of the condition, otherwise the graph data is
        stored as it is.

        Tensor fields are attached to each stored graph, so that batching
        concatenates them node-wise alongside the graph structures. The field
        is kept available as an attribute of the stored data as well.

        :param dict kwargs: The keyword arguments containing the graph data
            and the associated tensor fields. The optional sampling
            parameters ``sampler``, ``batch_size`` and ``seed_nodes`` are
            consumed here and are not stored.
        :raises TypeError: If ``sampler`` is provided and the graph data is
            not a single :class:`~pina.graph.Graph`.
        :raises ValueError: If ``batch_size`` or ``seed_nodes`` are provided
            without ``sampler``, or if the number of graphs does not match
            the number of samples of a tensor field.
        :return: The stored data.
        :rtype: SimpleNamespace
        """
        sampler = kwargs.pop("sampler", None)
        batch_size = kwargs.pop("batch_size", None)
        seed_nodes = kwargs.pop("seed_nodes", None)

        if sampler is not None:
            graph_key = next(
                key
                for key, value in kwargs.items()
                if isinstance(value, (Graph, Data, list, tuple))
            )
            graph = kwargs[graph_key]
            if not isinstance(graph, Graph):
                raise TypeError(
                    "When 'sampler' is provided the graph data must be a "
                    f"single pina Graph, got '{type(graph).__name__}'."
                )
            kwargs[graph_key] = graph.create_subgraph(
                sampler,
                batch_size=1 if batch_size is None else batch_size,
                seed_nodes=seed_nodes,
            )
        elif batch_size is not None or seed_nodes is not None:
            raise ValueError(
                "'batch_size' and 'seed_nodes' are only available when "
                "'sampler' is provided."
            )

        # Identify the unique graph field
        self.graph_key = next(
            key
            for key, value in kwargs.items()
            if isinstance(value, (Graph, Data, list, tuple))
        )

        # Normalize the graph field to a list of graphs
        graphs = kwargs[self.graph_key]
        if not isinstance(graphs, (list, tuple)):
            graphs = [graphs]
        self._graphs = list(graphs)

        # Keep the remaining tensor fields separate
        self._tensor_keys = [
            key
            for key, value in kwargs.items()
            if key != self.graph_key
            and isinstance(value, (torch.Tensor, LabelTensor))
        ]

        # Verify the consistency between graphs and tensor fields
        for key in self._tensor_keys:
            if len(self._graphs) != kwargs[key].shape[0]:
                raise ValueError(
                    f"Number of graphs ({len(self._graphs)}) does not match "
                    f"the number of samples for key '{key}' "
                    f"({kwargs[key].shape[0]})."
                )

            # Attach the sample of the tensor field to the corresponding graph
            for i, graph in enumerate(self._graphs):
                setattr(graph, key, kwargs[key][i])

        stored = {self.graph_key: self._graphs}
        stored.update({key: kwargs[key] for key in self._tensor_keys})

        # Keep any remaining non-tensor field (e.g. the equation)
        stored.update(
            {
                key: value
                for key, value in kwargs.items()
                if key != self.graph_key and key not in self._tensor_keys
            }
        )
        return SimpleNamespace(**stored)

    def __len__(self):
        """
        Return the number of samples in the condition.

        :return: The number of samples.
        :rtype: int
        """
        return len(self._graphs)

    def _select(self, idx):
        """
        Select the data points at the given indices.

        :param idx: The indices of the data points to select.
        :type idx: int | slice | list[int] | torch.Tensor
        :return: A mapping between data field names and selected values.
        :rtype: dict
        """
        idx = _normalize_ids(idx)
        graphs = [self._graphs[i] for i in idx]
        return {self.graph_key: graphs}

    def _collate(self, raw):
        """
        Collate the selected graphs into a batched graph.

        Tensor fields, attached to each graph, are concatenated node-wise
        during batching and exposed as batched-level attributes.

        :param dict raw: The selected raw data.
        :return: The batch.
        :rtype: dict
        """
        graphs = raw[self.graph_key]
        batching_fn = (
            LabelBatch.from_data_list
            if isinstance(graphs[0], Graph)
            else Batch.from_data_list
        )
        batched = batching_fn(graphs)
        batch = {self.graph_key: batched}

        # Extract the tensor fields back as batched attributes
        for key in self._tensor_keys:
            batch[key] = getattr(batched, key)
            delattr(batched, key)

        return batch

    def materialize(self, ids, device=None, batch_fn=None):
        """
        Build a batch from the given data point ids.

        :param ids: The ids of the data points to batch.
        :type ids: int | list[int] | torch.Tensor
        :param device: The target device for the batch. Default is ``None``.
        :param batch_fn: Optional callable overriding the batch construction.
            It receives the selected raw data and the ids and returns the
            batch. Default is ``None``.
        :return: The materialized batch.
        :rtype: dict
        """
        raw = self._select(ids)
        if batch_fn is None:
            batch = self._collate(raw)
        else:
            batch = batch_fn(raw, ids)

        return _move_to_device(batch, device)

    def __getitem__(self, idx):
        """
        Materialize the data points at the specified index (or indices).

        :param idx: The index of the data point to retrieve.
        :type idx: int | slice | list[int] | torch.Tensor
        :return: The materialized batch.
        :rtype: dict
        """
        return self.materialize(idx)
