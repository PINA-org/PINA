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
    shape ``[n_windows, unroll_length, n_nodes, features]``. Temporal windows
    are generated during :meth:`store_data`, not during construction.
    """

    _avail_input_cls = (Data, Graph)

    def __init__(self, graph, key="x"):
        """
        Initialization of the :class:`GraphTimeSeriesCondition` class.

        :param Graph graph: The graph from which subgraphs are sampled.
            Its feature attribute must contain temporal data with shape
            ``[n_nodes, time_steps, features]``.
        :param str key: The name of the graph attribute containing the
            temporal features. Default is ``"x"``.
        """
        GraphCondition.__init__(self, graph, key=key)

    def store_data(self, **kwargs):
        """
        Sample subgraphs and store their temporal windows.

        Starting nodes are sampled uniformly with replacement. Each
        subgraph's feature attribute is replaced with windows of shape
        ``[n_windows, unroll_length, n_nodes, features]``. LabelTensor
        features retain their labels. Existing stored batches are replaced.

        :param dict[str, Any] kwargs: Sampling and temporal window settings:

            * ``n_subgraphs`` (int): Number of subgraphs to generate.
              Default is ``10``.
            * ``num_hops`` (int): Number of hops around each starting node.
              Default is ``5``.
            * ``unroll_length`` (int): Number of time steps in each window.
              Must be at least ``2``. Default is ``10``.
            * ``n_windows`` (int): Number of windows to extract per
              subgraph. Default is ``4``.
            * ``randomize`` (bool): Whether to randomly permute valid
              window starting indices before selecting windows. Default
              is ``False``.

            Other keyword arguments, including ``batch_size``, are ignored.
        :raises AttributeError: If the graph does not contain the feature
            attribute specified by ``key``.
        :raises AssertionError: If ``n_windows`` or ``unroll_length`` is
            not a positive integer.
        :raises ValueError: If ``randomize`` is not a boolean, the temporal
            data has fewer than three dimensions, ``unroll_length`` is
            less than two or exceeds the available time steps, or fewer
            than ``n_windows`` valid windows are available.
        :raises RuntimeError: If the temporal windows do not have the
            four dimensions required by the permutation.
        :return: The subgraphs containing temporal windows, also stored
            in ``self.data``.
        :rtype: list[Graph]
        """
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
        """
        Compute autoregressive residuals for a graph's temporal windows.

        The first state of each window initializes the rollout. At each
        subsequent step, the model prediction is compared with the
        corresponding stored state and becomes the input for the next step.

        The batch's feature attribute is temporarily replaced during the
        rollout and restored from a clone after successful evaluation.
        Residuals are stacked along a new leading time-step dimension,
        without loss aggregation.

        :param batch: The graph batch containing temporal windows under
            the attribute specified by ``key``, with shape
            ``[n_windows, unroll_length, n_nodes, features]``.
        :type batch: pina.graph.Graph | torch_geometric.data.Data
        :param pina.solver.BaseSolver solver: The solver providing the forward
            pass. Predictions must be compatible with the stored states.
        :raises KeyError: If the batch lacks the feature attribute
            specified by ``key``.
        :raises ValueError: If the temporal feature tensor has fewer than
            four dimensions.
        :raises RuntimeError: If prediction and target shapes are
            incompatible for subtraction, or no prediction steps are
            available to stack.
        :return: The prediction-minus-target residuals as a plain tensor with
            shape ``[unroll_length - 1, n_windows, n_nodes, features]``.
        :rtype: torch.Tensor

        .. warning::

            The feature attribute is not restored if a forward pass or
            subtraction raises during the rollout.
        """
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
