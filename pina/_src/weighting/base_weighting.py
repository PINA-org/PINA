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
    defined in :class:`~pina.weighting.WeightingInterface`.

    This class is not meant to be instantiated directly.
    """

    # Supported aggregation methods
    _AGGREGATE_METHODS = {"sum": torch.sum, "mean": torch.mean}

    def __init__(self, update_every_n_epochs=1, aggregator="sum"):
        """
        Initialization of the :class:`BaseWeighting` class.

        :param int update_every_n_epochs: The number of training epochs between
            eligible epochs. Updates occur on every aggregation call in epochs
            divisible by this value, including epoch zero, not once per epoch.
            Default is ``1``.
        :param aggregator: The aggregation method. Available options include:
            ``"sum"`` which sums the weighted losses, ``"mean"`` which averages
            the weighted losses, or a custom callable receiving the stacked
            tensor of flattened weighted losses, normally ``(n_conditions, 1)``.
            Built-in methods reduce all entries. A custom callable should
            return a scalar tensor; its output is not validated.
            Default is ``"sum"``.
        :type aggregator: str | Callable
        :raises AssertionError: If ``update_every_n_epochs`` is not a positive
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

        In eligible epochs, call ``update_weights`` and merge its result into
        the saved mapping before aggregation. Each call in that epoch can
        update weights again. Previously saved keys survive partial-condition
        batches. The weighting must be attached to a solver with a trainer.

        :param dict[str, torch.Tensor] losses: The nonempty mapping from
            condition names to losses, normally scalar or one-element tensors.
            Flattened weighted losses must have equal lengths for stacking.
        :raises AttributeError: If no solver is attached.
        :raises RuntimeError: If the solver has no trainer, losses cannot be
            stacked, or a strategy's gradient computation fails.
        :raises ValueError: If a strategy has no usable gradients for a loss.
        :raises KeyError: If a current loss has no saved weight and this call
            does not initialize one, or a strategy lacks its configuration.
        :return: The scalar tensor from a built-in aggregator, or the unchanged
            output of a custom aggregator.
        :rtype: Any
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

        :return: The stored mapping from condition names to weights. Initially
            empty; mutating it affects future aggregation directly.
        :rtype: dict[str, int | float | torch.Tensor]
        """
        return self._saved_weights

    @property
    def solver(self):
        """
        Solver associated with this weighting strategy.

        Provides access to the solver instance that uses this weighting scheme,
        enabling strategies that depend on training state or model information.

        :return: The associated solver, or ``None`` before solver attachment.
        :rtype: pina.solver.BaseSolver | None
        """
        return self._solver
