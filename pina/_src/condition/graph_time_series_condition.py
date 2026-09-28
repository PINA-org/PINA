"""Module for the Graph Time-Series Condition class."""

import torch

from pina import logger
from pina._src.condition.graph_condition import GraphCondition
from pina._src.condition.time_series_condition import (
    TimeSeriesCondition,
    _check_time_series_params,
    _unroll_windows,
)
from pina._src.core.graph import Graph
from pina._src.core.label_tensor import LabelTensor
from torch_geometric.data import Data


class GraphTimeSeriesCondition(GraphCondition, TimeSeriesCondition):
    """
    Autoregressive time-series condition on graph data.

    Each stored batch is one subgraph carrying unrolled temporal windows with
    shape ``[n_windows, unroll_length, n_nodes, *features]``.
    """

    _avail_input_cls = (Data, Graph)

    def __init__(self, graph, key="x"):
        GraphCondition.__init__(self, graph, key=key)

    def store_data(self, **kwargs):
        """Store subgraphs carrying unrolled temporal windows."""
        unroll_length = kwargs.get("unroll_length", 10)
        n_windows = kwargs.get("n_windows", 4)
        randomize = kwargs.get("randomize", False)

        _check_time_series_params(
            data=getattr(self.graph, self.key),
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )

        subgraphs = GraphCondition.store_data(self, **kwargs)
        for subgraph in subgraphs:
            data = getattr(subgraph, self.key)
            labels = getattr(data, "labels", None)
            series = torch.permute(
                _unroll_windows(
                    data=data,
                    n_windows=n_windows,
                    unroll_length=unroll_length,
                    randomize=randomize,
                ),
                (1, 2, 0, 3),
            )
            if labels is not None:
                series = LabelTensor(series, labels)
            setattr(subgraph, self.key, series)

        logger.debug("Stored %d graph time-series batches.", len(self.data))
        return self.data

    def evaluate(self, batch, solver):
        """Compute autoregressive residuals over the unrolled graph series."""
        series = batch[self.key]
        if series.dim() < 4:
            raise ValueError(
                "The provided input tensor must have at least 4 dimensions:"
                " [windows, time_steps, nodes, *features]."
                f" Got shape {series.shape}."
            )

        original = getattr(batch, self.key).clone()
        setattr(batch, self.key, series[:, 0, :, :])

        residuals = []
        for step in range(1, series.shape[1]):
            prediction = solver.forward(batch)
            residuals.append(prediction - series[:, step, :, :])
            setattr(batch, self.key, prediction)

        setattr(batch, self.key, original)
        return torch.stack(residuals).as_subclass(torch.Tensor)
