"""Module for the Graph Time-Series Condition class."""

import torch
from torch_geometric.data import Data

from pina import logger
from pina._src.condition.graph_condition import GraphCondition
from pina._src.condition.time_series_condition import (
    TimeSeriesCondition,
    _check_time_series_params,
    _concat_features,
    _resolve_feature_selectors,
    _select_features,
    _unroll_windows,
)
from pina._src.core.graph import Graph
from pina._src.core.label_tensor import LabelTensor


class GraphTimeSeriesCondition(GraphCondition, TimeSeriesCondition):
    """
    Autoregressive time-series condition on graph data.

    Each stored batch is one subgraph carrying unrolled temporal windows with
    shape ``[n_windows, unroll_length, n_nodes, *features]``.

    Feature columns can be partitioned into ``state`` and ``control`` columns
    using integer indices or, for
    :class:`~pina.label_tensor.LabelTensor` graph features, feature labels.
    At each autoregressive step, the selected state and control columns are
    concatenated along the feature dimension and passed to the solver. Only
    the state is compared with the target and fed back into the next step.
    """

    _avail_input_cls = (Data, Graph)

    def __init__(self, graph, key="x", state=None, control=None):
        """
        Initialization of the :class:`GraphTimeSeriesCondition` class.

        :param Graph graph: The graph from which subgraphs are sampled.
            Its feature attribute must contain temporal data with shape
            ``[n_nodes, time_steps, features]``.
        :param str key: The name of the graph attribute containing the
            temporal features. Default is ``"x"``.
        :param state: The columns used as state. It can be a list of integer
            feature indices, a list of labels for
            :class:`~pina.label_tensor.LabelTensor` graph features, or
            ``None``. If ``None`` and ``control`` is provided, all columns
            not selected as control are used as state. If both are ``None``,
            all feature columns are treated as state. Default is ``None``.
        :type state: list[int] | list[str] | None
        :param control: The columns used as control inputs. It can be a list
            of integer feature indices, a list of labels for
            :class:`~pina.label_tensor.LabelTensor` graph features, or
            ``None``. Control columns are concatenated with the state before
            the forward pass and are never predicted. Default is ``None``.
        :type control: list[int] | list[str] | None
        :raises ValueError: If ``state`` or ``control`` selectors are invalid.
        """
        GraphCondition.__init__(self, graph, key=key)
        features = getattr(graph, key)
        self._state = state
        self._control = control
        self._state_indices, self._control_indices = (
            _resolve_feature_selectors(features, state, control)
        )

    def store_data(self, **kwargs):
        """
        Sample subgraphs and store their temporal windows.

        Starting nodes are sampled uniformly with replacement. Each
        subgraph's feature attribute is replaced with windows of shape
        ``[n_windows, unroll_length, n_nodes, features]``. LabelTensor
        features retain their labels. Existing stored batches are replaced.

        :param dict kwargs: Sampling and temporal window settings:

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

            Other keyword arguments are ignored.
        :return: The subgraphs containing temporal windows, also stored
            in ``self.data``.
        :rtype: list[Graph]
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
        subsequent step, the selected state and control columns are
        concatenated along the feature dimension and passed to the solver.
        The model prediction is compared with the next state columns and
        becomes the state input for the next step, while the control is
        always read from the stored data.

        The batch's feature attribute is temporarily replaced during the
        rollout and restored after evaluation. Residuals are stacked along a
        new leading time-step dimension, without loss aggregation.

        :param batch: The graph batch containing temporal windows under
            the attribute specified by ``key``, with shape
            ``[n_windows, unroll_length, n_nodes, features]``.
        :type batch: Graph | Data
        :param BaseSolver solver: The solver providing the model forward
            pass. Predictions must be compatible with the stored states.
        :return: The prediction-minus-target residuals with shape
            ``[unroll_length - 1, n_windows, n_nodes, *state_features]``.
        :rtype: torch.Tensor
        :raises KeyError: If the batch lacks the feature attribute
            specified by ``key``.
        :raises ValueError: If the temporal feature tensor has fewer than
            four dimensions.
        :raises RuntimeError: If prediction and target shapes are
            incompatible for subtraction, or no prediction steps are
            available to stack.
        """
        original = getattr(batch, self.key).clone()
        series = batch[self.key]
        if series.dim() < 4:
            raise ValueError(
                "The provided input tensor must have at least 4 dimensions:"
                " [windows, time_steps, nodes, *features]."
                f" Got shape {series.shape}."
            )

        # Resolve labels for the selected columns, if available
        if isinstance(series, LabelTensor) and series.labels is not None:
            state_labels = [series.labels[i] for i in self._state_indices]
            control_labels = [series.labels[i] for i in self._control_indices]
        else:
            state_labels = None
            control_labels = None

        current_state = _select_features(
            series[:, 0], self._state_indices
        )
        residuals = []

        try:
            for step in range(1, series.shape[1]):

                # The control is an input only and is never predicted
                current_control = _select_features(
                    series[:, step - 1], self._control_indices
                )

                dirichlet_nodes = torch.isclose(_select_features(
                    series[:, step - 1], [-1]
                ), torch.tensor(1.0)).squeeze()

                # print("Dirichlet nodes:", dirichlet_nodes.shape)
                # print(torch.sum(dirichlet_nodes).tensor)
                
                model_input = _concat_features(
                    current_state,
                    current_control,
                    state_labels=state_labels,
                    control_labels=control_labels,
                )
                setattr(batch, self.key, model_input)

                # Forward the graph and compare only the state columns
                prediction = solver.forward(batch) + current_state
                prediction[dirichlet_nodes] = 0.0
                target_state = _select_features(
                    series[:, step], self._state_indices
                )
                residuals.append(prediction - target_state)

                # Only the predicted state is fed back
                current_state = prediction
        finally:
            setattr(batch, self.key, original)

        return torch.stack(residuals).as_subclass(torch.Tensor)
