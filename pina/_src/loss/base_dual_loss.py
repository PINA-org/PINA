"""Module for the BaseDualLoss class."""

import torch
from pina._src.loss.dual_loss_interface import DualLossInterface


class BaseDualLoss(DualLossInterface):
    """
    Base class for all losses requiring both an input and a target tensor,
    implementing common functionality.

    All specific loss types should inherit from this class and implement its
    abstract methods.

    This class is not meant to be instantiated directly.
    """

    # Define available reduction methods
    _REDUCTION_METHOD = {
        "sum": lambda x: torch.sum(x, keepdim=True, dim=-1),
        "mean": lambda x: torch.mean(x, keepdim=True, dim=-1),
        "none": lambda x: x,
    }

    def __init__(self, reduction="mean"):
        """
        Initialization of the :class:`BaseDualLoss` class.

        :param str reduction: The reduction method to aggregate pointwise loss
            values. Available options include: ``"none"`` for unreduced loss,
            ``"mean"`` for the average along the last axis, and ``"sum"`` for
            the sum along that axis. Mean and sum retain the reduced dimension.
            No other axes are reduced. Default is ``"mean"``.
        :raises ValueError: If the specified reduction method is not among the
            available options.
        """
        # Check that the reduction method is available
        if reduction not in self._REDUCTION_METHOD:
            raise ValueError(
                f"Invalid reduction method. Available options: "
                f"{list(self._REDUCTION_METHOD.keys())}. Got {reduction}."
            )

        # Initialization
        super().__init__(reduction=reduction, size_average=None, reduce=None)

    def _reduction(self, loss):
        """
        Apply the configured reduction operation to pointwise loss values.

        :param torch.Tensor loss: The tensor of pointwise losses, with the
            aggregation axis last.
        :return: The same tensor for ``"none"``, otherwise its last-axis mean
            or sum with that dimension retained at size one. For a vector of
            shape ``(n,)``, mean and sum return shape ``(1,)``.
        :rtype: torch.Tensor
        """
        return self._REDUCTION_METHOD[self.reduction](loss)
