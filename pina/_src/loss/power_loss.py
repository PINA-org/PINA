"""Module for the Power Loss class."""

import torch
from pina._src.loss.base_dual_loss import BaseDualLoss
from pina._src.core.utils import check_consistency, check_positive_integer


class PowerLoss(BaseDualLoss):
    r"""
    Implementation of the Power loss, measuring the pointwise averaged
    :math:`p`-power error between an input tensor :math:`x` and a target tensor
    :math:`y`.

    Given a batch of size :math:`N` and feature dimension :math:`D`, the
    unreduced loss (``reduction="none"``) is defined as:

    .. math::
        L = \{l_1, \dots, l_N\}^\top, \quad
        l_n = \frac{1}{D} \sum_{i=1}^{D} \left| x_n^i - y_n^i \right|^p

    If ``relative=True``, each term is normalized by the averaged
    :math:`p`-power magnitude of the input tensor :math:`x`:

    .. math::
        l_n = \frac{\frac{1}{D} \sum_{i=1}^{D} |x_n^i - y_n^i|^p}
                {\frac{1}{D} \sum_{i=1}^{D} |x_n^i|^p}

    If ``reduction`` is set to ``"mean"`` or ``"sum"``, the vector :math:`L`
    is aggregated accordingly:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{``mean''} \\
            \operatorname{sum}(L),  & \text{if reduction} = \text{``sum''}
        \end{cases}

    where :math:`N` is the batch size. The feature average removes the last
    input axis; mean and sum then reduce only the last remaining axis and keep
    its dimension. Higher leading dimensions are not globally reduced.

    Relative normalization uses the input, not the target, without an epsilon
    or zero-magnitude guard. Zero denominators can produce infinity or NaN.

    :Example:

        >>> import torch
        >>> from pina.loss import PowerLoss
        >>> loss = PowerLoss(p=2, reduction="mean")
        >>> input = torch.tensor([[1., 3.], [2., 2.]])
        >>> target = torch.zeros_like(input)
        >>> loss(input, target)
        tensor([4.5000])
    """

    def __init__(self, p=2, reduction="mean", relative=False):
        """
        Initialization of the :class:`PowerLoss` class.

        :param int p: The positive integer exponent applied to absolute
            differences before feature averaging. No root is taken.
            Default is ``2``.
        :param str reduction: The reduction method to aggregate pointwise loss
            values. Available options include: ``"none"`` for unreduced loss,
            ``"mean"`` for the average of the loss values, and ``"sum"`` for
            their total sum, along the last axis of the per-point loss tensor.
            Mean and sum retain that dimension. Default is ``"mean"``.
        :param bool relative: Whether to divide by the averaged input magnitude
            raised to ``p``, without guarding against zero denominators.
            Default is ``False``.
        :raises ValueError: If ``relative`` is not a boolean.
        :raises AssertionError: If ``p`` is not a positive integer.
        :raises ValueError: If ``reduction`` is not ``"none"``, ``"mean"``, or
            ``"sum"``.
        """
        super().__init__(reduction=reduction)

        # Check consistency
        check_consistency(relative, bool)
        check_positive_integer(p, strict=True)

        # Initialize attributes
        self.p = p
        self.relative = relative

    def forward(self, input, target):
        """
        Forward method of the loss function.

        :param torch.Tensor input: Predictions of shape ``(..., d)``, with
            features on the last axis. Absolute powered differences must have
            a dtype supported by :func:`torch.mean`.
        :param torch.Tensor target: Targets broadcastable with ``input``.
            For relative loss, averaged input magnitudes must also broadcast
            with the per-point errors.
        :raises RuntimeError: If subtraction, feature averaging, or relative
            normalization fails for the supplied shapes or dtypes.
        :return: Averaged absolute ``p``-power errors with the feature axis
            removed for ``"none"``. Mean and sum reduce the last remaining
            axis and retain it at size one. Inputs of shape ``(n, d)`` yield
            ``(n,)`` unreduced or ``(1,)`` reduced. Relative mode divides by
            the averaged ``p``-power input magnitudes.
        :rtype: torch.Tensor
        """
        # Compute the standard loss
        loss = torch.abs((input - target)).pow(self.p).mean(-1)

        # Compute the input norm for relative error
        if self.relative:
            loss = loss / torch.abs(input).pow(self.p).mean(-1)

        return self._reduction(loss)
