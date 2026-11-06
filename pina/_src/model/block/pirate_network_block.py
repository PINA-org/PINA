"""Module for the PirateNet block class."""

import torch
from pina._src.core.utils import check_consistency, check_positive_integer


class PirateNetBlock(torch.nn.Module):
    """
    The inner block of Physics-Informed residual adaptive network (PirateNet).

    The block consists of three dense layers with dual gating operations and an
    adaptive residual connection. The trainable ``alpha`` parameter controls
    the contribution of the residual connection.

    .. seealso::

        **Original reference**:
        Wang, S., Sankaran, S., Stinis., P., Perdikaris, P. (2025).
        *Simulating Three-dimensional Turbulence with Physics-informed Neural
        Networks*.
        DOI: `arXiv preprint arXiv:2507.08972.
        <https://arxiv.org/abs/2507.08972>`_

    :Example:

        >>> import torch
        >>> from pina.model.block import PirateNetBlock
        >>> block = PirateNetBlock(inner_size=64, activation=torch.nn.Tanh)
        >>> x = torch.randn(10, 64)
        >>> U, V = torch.randn(10, 64), torch.randn(10, 64)
        >>> out = block(x, U, V)
    """

    def __init__(self, inner_size, activation):
        """
        Initialization of the :class:`PirateNetBlock` class.

        :param int inner_size: The number of hidden units in the dense layers.
        :param type[torch.nn.Module] activation: No-argument activation class.
        :raises AssertionError: If ``inner_size`` is not a positive integer.
        :raises ValueError: If ``activation`` is not a module subclass.
        """
        super().__init__()

        # Check consistency
        check_consistency(activation, torch.nn.Module, subclass=True)
        check_positive_integer(inner_size, strict=True)

        # Initialize the linear transformations of the dense layers
        self.linear1 = torch.nn.Linear(inner_size, inner_size)
        self.linear2 = torch.nn.Linear(inner_size, inner_size)
        self.linear3 = torch.nn.Linear(inner_size, inner_size)

        # Initialize the scales of the dense layers
        self.scale1 = torch.nn.Parameter(torch.zeros(inner_size))
        self.scale2 = torch.nn.Parameter(torch.zeros(inner_size))
        self.scale3 = torch.nn.Parameter(torch.zeros(inner_size))

        # Initialize the adaptive residual connection parameter
        self._alpha = torch.nn.Parameter(torch.zeros(1))

        # Initialize the activation function
        self.activation = activation()

    def forward(self, x, U, V):
        """
        Forward pass of the PirateNet block. It computes the output of the block
        by applying the dense layers with scaling, and combines the results with
        the input using the adaptive residual connection.

        :param x: The input tensor with final dimension ``inner_size``.
        :type x: torch.Tensor | LabelTensor
        :param torch.Tensor U: The first shared gating tensor. It must have the
            same shape as ``x``.
        :param torch.Tensor V: The second shared gating tensor. It must have the
            same shape as ``x``.
        :return: The gated/residual result with the same shape as ``x``. The
            trainable residual coefficient starts at zero, so the initial
            block acts as the identity.
        :rtype: torch.Tensor | LabelTensor
        """
        # Compute the output of the first dense layer with scaling
        f = self.activation(self.linear1(x) * torch.exp(self.scale1))
        z1 = f * U + (1 - f) * V

        # Compute the output of the second dense layer with scaling
        g = self.activation(self.linear2(z1) * torch.exp(self.scale2))
        z2 = g * U + (1 - g) * V

        # Compute the output of the block
        h = self.activation(self.linear3(z2) * torch.exp(self.scale3))
        return self._alpha * h + (1 - self._alpha) * x

    @property
    def alpha(self):
        """
        Return the alpha parameter.

        :return: The alpha parameter controlling the residual connection.
        :rtype: torch.nn.Parameter
        """
        return self._alpha
