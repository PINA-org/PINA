"""Module for the Graph Time-Series Condition class."""

import torch

from pina._src.condition.base_condition import BaseCondition
from pina._src.condition.graph_condition import GraphCondition
from pina._src.condition.time_series_condition import (
    TimeSeriesCondition,
    _check_time_series_params,
    _unroll_windows,
)
from pina._src.core.graph import Graph
from pina._src.core.utils import check_consistency
from torch_geometric.data import Data
from pina._src.core.label_tensor import LabelTensor


class GraphTimeSeriesCondition(GraphCondition, TimeSeriesCondition):
    """
    The :class:`GraphTimeSeriesCondition` class represents an autoregressive
    time series condition defined by temporal ``input`` data. The input is
    expected to have shape ``[trajectories, time_steps, *features]``, where the
    second dimension corresponds to the temporal evolution of each trajectory.

    During training, the condition automatically extracts overlapping temporal
    windows from the trajectories. The parameter ``unroll_length`` defines the
    number of consecutive time steps contained in each temporal window, while
    ``n_windows`` controls how many temporal windows are created from the
    available trajectories.

    Internally, the unrolled data is stored as a tensor of shape
    ``[trajectories, n_windows, unroll_length, *features]``, attached to the
    graph attribute given by ``key``.

    The temporal logic (window extraction and autoregressive residual) is
    inherited from :class:`~pina.condition.TimeSeriesCondition`, while storage
    and batching of the graphs are inherited from
    :class:`~pina.condition.GraphCondition`.

    Supported data types include :class:`~pina.graph.Graph` and
    :class:`~torch_geometric.data.Data`.

    :Example:

    >>> from pina.graph import Graph
    >>> import torch
    >>> from torch_geometric.data import Data

    >>> data = Graph(torch.rand(5, 10, 2), edge_index=torch.randint(0, 5, (2, 20)))
    >>> condition = Condition(input=data, unroll_length=5, n_windows=3)
    """

    # Available fields and input data types
    __fields__ = ["input", "unroll_length", "n_windows", "key", "randomize"]
    _avail_input_cls = (Data, Graph)

    # Name of the graph attribute holding the temporal data
    _key = "x"

    # def __new__(cls, input, n_windows, unroll_length, key="x", randomize=False):
    #     """
    #     Check the graph input and the time-series parameters.

    #     :param input: The graph holding the temporal data.
    #     :type input: Graph | Data
    #     :param int n_windows: The maximum number of temporal windows to extract.
    #     :param int unroll_length: The number of time steps in each window.
    #     :param str key: The name of the graph attribute holding the temporal
    #         data. Default is ``"x"``.
    #     :param bool randomize: If ``True``, randomly permute the valid starting
    #         indices before selecting the windows. Default is ``False``.
    #     :raises ValueError: If ``input`` is not of type :class:`~pina.graph.Graph`
    #         or :class:`~torch_geometric.data.Data`.
    #     :raises ValueError: If ``key`` is not a string value.
    #     :raises ValueError: If ``input`` has no attribute named ``key``.
    #     :raises ValueError: If the temporal data is not a valid time series, or
    #         if the windowing parameters are not valid.
    #     :return: A new :class:`GraphTimeSeriesCondition` instance.
    #     :rtype: GraphTimeSeriesCondition
    #     """
    #     # Check consistency
    #     check_consistency(input, cls._avail_input_cls)
    #     check_consistency(key, str)
    #     if not hasattr(input, key):
    #         raise ValueError(
    #             f"The provided graph does not have the specified key '{key}'."
    #         )

    #     # Check the time-series parameters on the temporal data of the graph
    #     _check_time_series_params(
    #         data=getattr(input, key),
    #         n_windows=n_windows,
    #         unroll_length=unroll_length,
    #         randomize=randomize,
    #     )

    #     return BaseCondition.__new__(cls)

    def __init__(self, graph, output_feature, conditional_feature=None):
        self.graph = graph
        self.output_feature = output_feature
        self.conditional_feature = conditional_feature


    def store_data(self, **kwargs):
        """
        Store the graph data, with its temporal windows already extracted.

        The temporal windows replace the data held by the ``key`` attribute of
        the graph, and the resulting graph is then stored by
        :class:`~pina.condition.GraphCondition`.

        :param dict kwargs: The keyword arguments containing the graph data.
        :return: A namespace-like structure containing the stored data.
        :rtype: SimpleNamespace
        """
        print("store_data called with kwargs:", kwargs)
        # Extract unrolling parameters from kwargs
        unroll_length = kwargs.pop("unroll_length", 10)
        n_windows = kwargs.pop("n_windows", 4)
        randomize = kwargs.pop("randomize", False)

        # Keep the key available for later retrieval of the temporal data
        key = self.output_feature
        graph = self.graph

        # Create unrolled windows and attach them to the graph
        unrolled_series = _unroll_windows(
            data=graph.x,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )
        old_labels = graph.x.labels

        unrolled_series = torch.permute(unrolled_series, (1, 2, 0, 3))
        unrolled_series = LabelTensor(unrolled_series, old_labels)
        setattr(graph, key, unrolled_series)

        from torch_geometric.utils import k_hop_subgraph

        n_batches = 10
        start_nodes = torch.randint(0, graph.num_nodes, (n_batches,))
        subgraphs = []
        for start_node in start_nodes:
            subset = k_hop_subgraph(
                node_idx=start_node.item(),
                num_hops=5,
                edge_index=graph.edge_index,
                relabel_nodes=True,
                num_nodes=graph.num_nodes,
            )[0]  # only the subset of nodes is needed
            subgraph = graph.subgraph(subset)
            subgraphs.append(subgraph)


        self.data = subgraphs
        # subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        #     node_idx=10,
        #     num_hops=5,
        #     edge_index=graph.edge_index,
        #     relabel_nodes=True,
        #     num_nodes=graph.num_nodes,
        # )

        # subgraph = graph.subgraph(subset)
        # print("subgraph:", subgraph)
        # fff

        # self.data = GraphCondition.store_data(self, input=graph)

    def __len__(self):
        return len(self.data)
    def _get_series(self, batch):
        """
        Return the unrolled time-series data carried by the given batch.

        :param dict batch: The batch to extract the time-series data from.
        :return: The unrolled data, of shape
            ``[nodes, n_windows, unroll_length, *features]``.
        :rtype: torch.Tensor | LabelTensor
        """
        return getattr(batch["input"], self.output_feature)

    def evaluate(self, batch, solver):
        """
        Evaluate the residual of the condition on the given batch using the
        solver.

        This method computes the per-step residuals through autoregressive
        unrolling. At each time step the whole graph batch (carrying the
        unrolled series on the attribute given by ``key``) is passed to the
        solver, which must return the predicted state as a tensor of shape
        ``[trajectories, n_windows, *features]``. The per-step residuals
        (predicted - target) are returned as a stacked tensor.

        :param dict batch: The batch containing the data required by the
            condition evaluation.
        :param BaseSolver solver: The solver used to perform the forward pass
            and compute the residual.
        :raises ValueError: If the unrolled data has less than 4 dimensions.
        :return: The stacked per-step residual tensor of shape
            ``[time_steps - 1, trajectories, windows, *features]``.
        :rtype: torch.Tensor | LabelTensor
        """
        # Extract the unrolled time-series data carried by the graph
        series = self._get_series(batch)

        # Raise error if the unrolled data does not have at least 4 dimensions
        if series.dim() < 4:
            raise ValueError(
                "The provided input tensor must have at least 4 dimensions:"
                " [windows, time_steps, nodes, *features]."
                f" Got shape {series.shape}."
            )

        # Copy the kwargs to avoid modifying the original settings
        kwargs = solver._kwargs.copy()

        # Extract the graph batch and initialize the step-wise residuals list
        graph = batch["input"]
        residuals = []

        setattr(graph, "series", series)
        setattr(graph, 'x', series[:, 0, :, :])

        # Iterate over the time steps
        for step in range(1, series.shape[1]):

            # Pre-process, forward, and post-process the graph batch
            #processed_input = solver.preprocess_step(graph, **kwargs)
            predicted_state = solver.forward(graph)
            #predicted_state = solver.postprocess_step(output, **kwargs)
            setattr(graph, 'x', predicted_state)

            # Retrieve the target and compute the step-wise residual
            target_state = series[:, step, :, :]
            step_residual = predicted_state - target_state
            residuals.append(step_residual)

        # Stack the step-wise residuals
        return torch.stack(residuals).as_subclass(torch.Tensor)
