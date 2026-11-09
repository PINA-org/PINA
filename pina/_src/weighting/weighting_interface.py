"""Module for the Weighting Interface."""

from abc import ABCMeta, abstractmethod


class WeightingInterface(metaclass=ABCMeta):
    """
    Abstract interface for all weighting schemas.
    """

    @abstractmethod
    def aggregate(self, losses):
        """
        Aggregate a collection of loss terms into a single scalar.

        This method applies the current weighting scheme to the provided losses
        and returns the aggregated result. Implementations may internally update
        the weights (e.g., based on training state) before performing the
        aggregation.

        :param dict[str, torch.Tensor] losses: The mapping from condition names
            to loss tensors, normally scalar or one-element tensors.
        :return: The aggregated loss value.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def update_weights(self, losses):
        """
        Update the weights based on the current losses.

        Compute weights for the supplied condition subset. Aggregation is
        responsible for merging returned weights into the saved mapping;
        calling this method directly need not save them.

        :param dict[str, torch.Tensor] losses: The mapping from condition names
            to loss tensors, normally scalar or one-element tensors.
        :return: The updated weights.
        :rtype: dict[str, int | float | torch.Tensor]
        """

    @abstractmethod
    def last_saved_weights(self):
        """
        Get the most recently computed weights.

        :return: The mapping from loss names to their corresponding weights.
        :rtype: dict[str, int | float | torch.Tensor]
        """

    @property
    @abstractmethod
    def solver(self):
        """
        Solver associated with this weighting strategy.

        Provides access to the solver instance that uses this weighting scheme,
        enabling strategies that depend on training state or model information.

        :return: The associated solver, or ``None`` before solver attachment.
        :rtype: pina.solver.BaseSolver | None
        """
