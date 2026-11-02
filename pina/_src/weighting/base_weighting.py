"""Module for the Base Weighting class."""

from typing import final, Callable
import torch
from pina._src.weighting.weighting_interface import WeightingInterface
from pina._src.core.utils import check_positive_integer, check_consistency


class BaseWeighting(WeightingInterface):
    """
    Base class for all weighting schemas, implementing common functionality.

    A weighting schema defines how scalar loss terms coming from different
    conditions are aggregated into a single scalar loss.

    All weighting schemas must inherit from this class and implement the methods
    defined in :class:`~pina.weighting.weighting_interface.WeightingInterface`.

    This class is not meant to be instantiated directly.
    """

    # Supported aggregation methods
    _AGGREGATE_METHODS = {"sum": torch.sum, "mean": torch.mean}

    def __init__(self, update_every_n_epochs=1, aggregator="sum"):
        """
        Initialization of the :class:`BaseWeighting` class.

        :param int update_every_n_epochs: The number of training epochs between
            weight updates. If set to 1, the weights are updated at every epoch.
            This parameter is ignored by static weighting schemes.
            Default is ``1``.
        :param aggregator: The aggregation method. Available options include:
            ``"sum"`` which sums the weighted losses, ``"mean"`` which averages
            the weighted losses, or a custom callable that takes an iterable of
            weighted losses and returns a single scalar. Default is ``"sum"``.
        :type aggregator: str | Callable
        :raises ValueError: If ``update_every_n_epochs`` is not a positive
            integer.
        :raises ValueError: If ``aggregator`` is invalid.
        """
        # Check consistency
        check_positive_integer(value=update_every_n_epochs, strict=True)
        check_consistency(aggregator, (str, Callable))

        # Validate aggregator
        if isinstance(aggregator, str):
            if aggregator not in self._AGGREGATE_METHODS:
                raise ValueError(
                    f"Invalid aggregator '{aggregator}'. Available options: "
                    f"{list(self._AGGREGATE_METHODS.keys())}. Got {aggregator}."
                )
            aggregator = self._AGGREGATE_METHODS[aggregator]

        # Initialization
        self.update_every_n_epochs = update_every_n_epochs
        self.aggregator_fn = aggregator
        self._solver = None
        self._saved_weights = {}

    @final
    def aggregate(self, losses):
        """
        Aggregate a collection of loss terms into a single scalar.

        This method applies the current weighting scheme to the provided losses
        and returns the aggregated result. Implementations may internally update
        the weights (e.g., based on training state) before performing the
        aggregation.

        :param dict losses: The mapping from loss names to loss tensors.
        :return: The aggregated loss value.
        :rtype: torch.Tensor
        """
        # Update weights when required. Merge the updated weights because a
        # batch may contain only a subset of the problem conditions.
        if self.solver.trainer.current_epoch % self.update_every_n_epochs == 0:
            self._saved_weights.update(self.update_weights(losses))

        # Apply weights to the corresponding losses
        weighted_losses = torch.stack(
            [
                (self._saved_weights[condition] * loss).reshape(-1)
                for condition, loss in losses.items()
            ]
        )

        return self.aggregator_fn(weighted_losses)

    def last_saved_weights(self):
        """
        Get the most recently computed weights.

        :return: The mapping from loss names to their corresponding weights.
        :rtype: dict
        """
        return self._saved_weights

    @property
    def solver(self):
        """
        Solver associated with this weighting strategy.

        Provides access to the solver instance that uses this weighting scheme,
        enabling strategies that depend on training state or model information.

        :return: The solver instance.
        :rtype: :class:`~pina.solver.base_solver.BaseSolver`
        """
        return self._solver
