"""Module for the Averaging Neural Operator Block class."""

import torch
from torch import nn
from pina._src.core.utils import check_consistency


class AVNOBlock(nn.Module):
    r"""
    The inner block of the Averaging Neural Operator.

    The operator layer performs an affine transformation where the convolution
    is approximated with the arithmetic mean over axis one. Given the input
    :math:`v(x)\in\mathbb{R}^{\rm{emb}}` the layer computes the operator update
    :math:`K(v)` as:

    .. math::
        K(v) = \sigma\left(Wv(x) + b + \frac{1}{|\mathcal{A}|}\int v(y)dy\right)

    where:

    *   :math:`\mathbb{R}^{\rm{emb}}` is the embedding (hidden) size
        corresponding to the ``hidden_size`` object
    *   :math:`\sigma` is a non-linear activation, corresponding to the
        ``func`` object
    *   :math:`W\in\mathbb{R}^{\rm{emb}\times\rm{emb}}` is a tunable matrix.
    *   :math:`b\in\mathbb{R}^{\rm{emb}}` is a tunable bias.

    .. seealso::

        **Original reference**: Lanthaler S., Li, Z., Stuart, A. (2020).
        *The Nonlocal Neural Operator: Universal Approximation*.
        DOI: `arXiv preprint arXiv:2304.13221.
        <https://arxiv.org/abs/2304.13221>`_

    :Example:

        >>> import torch
        >>> from pina.model.block import AVNOBlock
        >>> block = AVNOBlock(hidden_size=20)
        >>> x = torch.randn(10, 5, 20)
        >>> out = block(x)
    """

    def __init__(self, hidden_size=100, func=nn.GELU):
        """
        Initialization of the :class:`AVNOBlock` class.

        :param int hidden_size: The size of the hidden layer.
            Default is ``100``.
        :param type[torch.nn.Module] func: The no-argument activation class.
            Default is :class:`torch.nn.GELU`.
        :raises ValueError: If hidden width or activation types are invalid.
        """
        super().__init__()

        # Check type consistency
        check_consistency(hidden_size, int)
        check_consistency(func, nn.Module, subclass=True)
        # Assignment
        self._nn = nn.Linear(hidden_size, hidden_size)
        self._func = func()

    def forward(self, x):
        r"""
        Forward pass of the block. It performs a sum of local average and an
        affine transformation of the field.

        :param torch.Tensor x: Features shaped
            ``(batch, n_points, hidden_size)``.
        :return: Same-shaped activated affine update plus mean over points.
        :rtype: torch.Tensor
        """
        return self._func(self._nn(x) + torch.mean(x, dim=1, keepdim=True))
