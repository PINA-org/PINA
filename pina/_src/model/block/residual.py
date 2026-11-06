"""Module for residual blocks and enhanced linear layers."""

import torch
from torch import nn
from pina._src.core.utils import check_consistency


class ResidualBlock(nn.Module):
    """
    Residual block class.

    .. seealso::

        **Original reference**: He, Kaiming, et al.
        *Deep residual learning for image recognition.*
        Proceedings of the IEEE conference on computer vision and pattern
        recognition. 2016.
        DOI: `<https://arxiv.org/pdf/1512.03385.pdf>`_.

    :Example:

        >>> import torch
        >>> from pina.model.block import ResidualBlock
        >>> block = ResidualBlock(
        ...     input_dim=10, output_dim=10, hidden_dim=20
        ... )
        >>> x = torch.randn(5, 10)
        >>> out = block(x)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        spectral_norm=False,
        activation=torch.nn.ReLU(),
    ):
        """
        Initialization of the :class:`ResidualBlock` class.

        :param int input_dim: The input dimension.
        :param int output_dim: The output dimension.
        :param int hidden_dim: The hidden dimension.
        :param bool spectral_norm: If ``True``, the spectral normalization is
            applied to the feedforward layers. Default is ``False``.
        :param torch.nn.Module activation: The activation function.
            Default is a :class:`torch.nn.ReLU` instance.
        :raises ValueError: If argument types fail validation.

        """
        super().__init__()
        # check consistency
        check_consistency(spectral_norm, bool)
        check_consistency(input_dim, int)
        check_consistency(output_dim, int)
        check_consistency(hidden_dim, int)
        check_consistency(activation, torch.nn.Module)

        # assign variables
        self._spectral_norm = spectral_norm
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._activation = activation

        # create layers
        self._l1 = self._spect_norm(nn.Linear(input_dim, hidden_dim))
        self._l2 = self._spect_norm(nn.Linear(hidden_dim, output_dim))
        self._l3 = self._spect_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        """
        Forward pass.

        :param torch.Tensor x: Input with shape ``(..., input_dim)``.
        :return: Nonlinear branch plus learned linear skip, with shape
            ``(..., output_dim)``.
        :rtype: torch.Tensor
        """
        y = self._activation(self._l1(x))
        y = self._l2(y)
        x = self._l3(x)
        return y + x

    def _spect_norm(self, x):
        """
        Perform spectral normalization on the network layers.

        :param torch.nn.Module x: A :class:`torch.nn.Linear` layer.
        :return: The layer, reparameterized in place with spectral normalization
            if enabled, or unchanged otherwise.
        :rtype: torch.nn.Module
        """
        return nn.utils.spectral_norm(x) if self._spectral_norm else x


class EnhancedLinear(torch.nn.Module):
    """
    Enhanced Linear layer class.

    This class is a wrapper for enhancing a linear layer with activation and/or
    dropout.

    :Example:

        >>> import torch
        >>> from pina.model.block import EnhancedLinear
        >>> layer = EnhancedLinear(
        ...     torch.nn.Linear(10, 20), torch.nn.ReLU()
        ... )
        >>> x = torch.randn(5, 10)
        >>> out = layer(x)
    """

    def __init__(self, layer, activation=None, dropout=None):
        """
        Initialization of the :class:`EnhancedLinear` class.

        :param torch.nn.Module layer: The linear layer to be enhanced.
        :param activation: Activation module; ``None`` omits activation.
            Default is ``None``.
        :type activation: torch.nn.Module | None
        :param dropout: Dropout probability; ``None`` omits dropout.
            Default is ``None``.
        :type dropout: float | None
        :raises ValueError: If types fail validation or dropout is outside
            ``[0, 1]``.

        :Example:

            >>> import torch
            >>> from pina.model.block import EnhancedLinear
            >>> linear_layer = torch.nn.Linear(10, 20)
            >>> activation = torch.nn.ReLU()
            >>> dropout_prob = 0.5
            >>> enhanced_linear = EnhancedLinear(
            ...     linear_layer,
            ...     activation,
            ...     dropout_prob
            ... )
        """
        super().__init__()

        # check consistency
        check_consistency(layer, nn.Module)
        if activation is not None:
            check_consistency(activation, nn.Module)
        if dropout is not None:
            check_consistency(dropout, float)

        # assign forward
        if (dropout is None) and (activation is None):
            self._model = torch.nn.Sequential(layer)

        elif (dropout is None) and (activation is not None):
            self._model = torch.nn.Sequential(layer, activation)

        elif (dropout is not None) and (activation is None):
            self._model = torch.nn.Sequential(layer, self._drop(dropout))

        elif (dropout is not None) and (activation is not None):
            self._model = torch.nn.Sequential(
                layer, activation, self._drop(dropout)
            )

    def forward(self, x):
        """
        Forward pass.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self._model(x)

    def _drop(self, p):
        """
        Apply dropout with probability p.

        :param float p: Dropout probability.
        :raises ValueError: If ``p`` is outside ``[0, 1]``.
        :return: Dropout layer with the specified probability.
        :rtype: torch.nn.Dropout
        """
        return torch.nn.Dropout(p)
