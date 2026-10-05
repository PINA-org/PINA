"""Module for the Graph Time-Series Condition class."""

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

    def __new__(cls, input, n_windows, unroll_length, key="x", randomize=False):
        """
        Check the graph input and the time-series parameters.

        :param input: The graph holding the temporal data.
        :type input: Graph | Data
        :param int n_windows: The maximum number of temporal windows to extract.
        :param int unroll_length: The number of time steps in each window.
        :param str key: The name of the graph attribute holding the temporal
            data. Default is ``"x"``.
        :param bool randomize: If ``True``, randomly permute the valid starting
            indices before selecting the windows. Default is ``False``.
        :raises ValueError: If ``input`` is not of type :class:`~pina.graph.Graph`
            or :class:`~torch_geometric.data.Data`.
        :raises ValueError: If ``key`` is not a string value.
        :raises ValueError: If ``input`` has no attribute named ``key``.
        :raises ValueError: If the temporal data is not a valid time series, or
            if the windowing parameters are not valid.
        :return: A new :class:`GraphTimeSeriesCondition` instance.
        :rtype: GraphTimeSeriesCondition
        """
        # Check consistency
        check_consistency(input, cls._avail_input_cls)
        check_consistency(key, str)
        if not hasattr(input, key):
            raise ValueError(
                f"The provided graph does not have the specified key '{key}'."
            )

        # Check the time-series parameters on the temporal data of the graph
        _check_time_series_params(
            data=getattr(input, key),
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )

        return BaseCondition.__new__(cls)

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
        # Extract unrolling parameters from kwargs
        unroll_length = kwargs.get("unroll_length")
        n_windows = kwargs.get("n_windows")
        randomize = kwargs.get("randomize", False)
        key = kwargs.get("key", self._key)
        graph = kwargs.get("input")

        # Keep the key available for later retrieval of the temporal data
        self._key = key

        # Create unrolled windows and attach them to the graph
        setattr(
            graph,
            key,
            _unroll_windows(
                data=getattr(graph, key),
                n_windows=n_windows,
                unroll_length=unroll_length,
                randomize=randomize,
            ),
        )

        return GraphCondition.store_data(self, input=graph)

    def _get_series(self, batch):
        """
        Return the unrolled time-series data carried by the given batch.

        :param dict batch: The batch to extract the time-series data from.
        :return: The unrolled data, of shape
            ``[nodes, n_windows, unroll_length, *features]``.
        :rtype: torch.Tensor | LabelTensor
        """
        return getattr(batch["input"], self._key)
