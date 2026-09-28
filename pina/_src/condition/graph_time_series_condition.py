"""Module for the Graph Time-Series Condition class."""

import torch

from pina._src.condition.base_condition import _unwrap_single
from pina._src.condition.graph_condition import GraphCondition
from pina._src.condition.time_series_condition import _unroll_windows
from pina._src.core.graph import Graph
from pina._src.core.utils import check_consistency, check_positive_integer
from torch_geometric.data import Data


class GraphTimeSeriesCondition(GraphCondition):
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
    ``[trajectories, n_windows, unroll_length, *features]``.

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

    def __new__(cls, input, n_windows, unroll_length, key="x", randomize=False):
        # Check consistency
        check_consistency(input, cls._avail_input_cls)
        check_consistency(randomize, bool)
        check_consistency(key, str)
        check_positive_integer(n_windows, strict=True)
        check_positive_integer(unroll_length, strict=True)

        return super().__new__(cls)

    def store_data(self, **kwargs):
        """
        Store the unrolled time-series input data.

        The method extracts the time-series input data and creates the temporal
        windows based on the specified ``unroll_length`` and ``n_windows``.

        :param dict kwargs: The keyword arguments containing the data to be
            stored.
        :return: A namespace-like structure containing the stored data.
        :rtype: SimpleNamespace
        """
        # Extract unrolling parameters from kwargs
        unroll_length = kwargs.get("unroll_length")
        n_windows = kwargs.get("n_windows")
        randomize = kwargs.get("randomize", False)
        key = kwargs.get("key", "x")
        graph = kwargs.get("input")

        # Create unrolled windows from the input data
        if not hasattr(graph, key):
            raise ValueError(
                f"The provided graph does not have the specified key '{key}'."
            )

        unrolled_data = _unroll_windows(
            data=graph.__getattribute__(key),
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )
        graph.__setattr__(key, unrolled_data)

        return super().store_data(input=graph)

    def evaluate(self, batch, solver):
        """
        Evaluate the residual of the condition on the given batch using the
        solver.

        This method computes the per-step residuals through autoregressive
        unrolling. A forward pass of the solver's model is performed at each
        time step, and the per-step residuals (predicted - target) are
        returned as a stacked tensor.

        The returned tensor preserves all per-step residual values without
        reduction or loss aggregation.

        :param dict batch: The batch containing the data required by the
            condition evaluation.
        :param SolverInterface solver: The solver used to perform the forward
            pass and compute the residual. The solver provides access to the
            model and its parameters, which may be necessary for evaluating the
            condition residual.
        :raises ValueError: If the input tensor in the batch has less than 4
            dimensions.
        :return: The stacked per-step residual tensor of shape
            ``[time_steps - 1, trajectories, windows, *features]``.
        :rtype: torch.Tensor | LabelTensor
        """
        # Raise error if input tensor does not have at least 4 dimensions
        if batch["input"].x.dim() < 4:
            raise ValueError(
                "The provided input tensor must have at least 4 dimensions:"
                " [trajectories, windows, time_steps, *features]."
                f" Got shape {batch['input'].shape}."
            )

        # Copy the kwargs to avoid modifying the original settings
        kwargs = solver._kwargs.copy()

        # Extract the initial state and initialize the step-wise residuals list
        current_state = batch["input"].x[:, :, 0, :]
        residuals = []

        # Iterate over the time steps
        for step in range(1, batch["input"].x.shape[2]):

            # Pre-process, forward, and post-process the current state
            processed_input = solver.preprocess_step(current_state, **kwargs)
            output = solver.forward(processed_input)
            predicted_state = solver.postprocess_step(output, **kwargs)

            # Retrieve the target and compute the step-wise residual
            target_state = batch["input"].x[:, :, step, :]
            step_residual = predicted_state - target_state
            residuals.append(step_residual)

            # Update the current state for the next iteration
            current_state = predicted_state

        # Stack the step-wise residuals
        return torch.stack(residuals).as_subclass(torch.Tensor)

    @property
    def input(self):
        """
        The input data associated with the condition.

        :return: The (unrolled) graph data.
        :rtype: Graph | Data
        """
        return _unwrap_single(self.data.input)
