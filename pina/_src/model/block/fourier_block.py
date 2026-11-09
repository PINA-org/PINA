"""Module for the Fourier Neural Operator Block class."""

import torch
from torch import nn
from pina._src.core.utils import check_consistency

from pina._src.model.block.spectral import (
    SpectralConvBlock1D,
    SpectralConvBlock2D,
    SpectralConvBlock3D,
)


class FourierBlock1D(nn.Module):
    """
    The inner block of the Fourier Neural Operator for 1-dimensional input
    tensors.

    The module computes the spectral convolution of the input with a linear
    kernel in the fourier space, and then it maps the input back to the physical
    space. The output is then added to a Linear transformation of the input in
    the physical space. Finally an activation function is applied to the output.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_

    :Example:

        >>> import torch
        >>> from pina.model.block import FourierBlock1D
        >>> block = FourierBlock1D(
        ...     input_numb_fields=2, output_numb_fields=2, n_modes=16
        ... )
        >>> x = torch.randn(10, 2, 50)
        >>> out = block(x)
    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        activation=torch.nn.Tanh,
    ):
        r"""
        Initialization of the :class:`FourierBlock1D` class.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param int n_modes: The number of modes to select for each dimension.
            It must not exceed ``Nx // 2 + 1``.
        :param type[torch.nn.Module] activation: No-argument activation class.
            Default is :class:`torch.nn.Tanh`.
        :raises ValueError: If channel types or activation results are invalid.
        """

        super().__init__()

        # check type consistency
        check_consistency(activation(), nn.Module)

        # assign variables
        self._spectral_conv = SpectralConvBlock1D(
            input_numb_fields=input_numb_fields,
            output_numb_fields=output_numb_fields,
            n_modes=n_modes,
        )
        self._activation = activation()
        self._linear = nn.Conv1d(input_numb_fields, output_numb_fields, 1)

    def forward(self, x):
        """
        Forward pass of the block. It performs a spectral convolution and a
        linear transformation of the input. Then, it sums the results.

        :param torch.Tensor x: Real input shaped
            ``(batch, input_channels, Nx)``.
        :return: Activated spectral plus pointwise convolution, shaped
            ``(batch, output_channels, Nx)``.
        :rtype: torch.Tensor
        """
        return self._activation(self._spectral_conv(x) + self._linear(x))


class FourierBlock2D(nn.Module):
    """
    The inner block of the Fourier Neural Operator for 2-dimensional input
    tensors.

    The module computes the spectral convolution of the input with a linear
    kernel in the fourier space, and then it maps the input back to the physical
    space. The output is then added to a Linear transformation of the input in
    the physical space. Finally an activation function is applied to the output.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_

    :Example:

        >>> import torch
        >>> from pina.model.block import FourierBlock2D
        >>> block = FourierBlock2D(
        ...     input_numb_fields=2, output_numb_fields=2, n_modes=[8, 8]
        ... )
        >>> x = torch.randn(10, 2, 50, 50)
        >>> out = block(x)
    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        activation=torch.nn.Tanh,
    ):
        r"""
        Initialization of the :class:`FourierBlock2D` class.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param n_modes: The number of modes to select for each dimension.
            A scalar is repeated over both axes. Counts must fit ``Nx`` and
            ``Ny // 2 + 1``; to avoid overlapping signed bands use
            ``2 * n_modes[0] <= Nx``.
        :type n_modes: int | list[int] | tuple[int, int]
        :param type[torch.nn.Module] activation: No-argument activation class.
            Default is :class:`torch.nn.Tanh`.
        :raises ValueError: If types or the mode-list length are invalid.
        """
        super().__init__()

        # check type consistency
        check_consistency(activation(), nn.Module)

        # assign variables
        self._spectral_conv = SpectralConvBlock2D(
            input_numb_fields=input_numb_fields,
            output_numb_fields=output_numb_fields,
            n_modes=n_modes,
        )
        self._activation = activation()
        self._linear = nn.Conv2d(input_numb_fields, output_numb_fields, 1)

    def forward(self, x):
        """
        Forward pass of the block. It performs a spectral convolution and a
        linear transformation of the input. Then, it sums the results.

        :param torch.Tensor x: Real input shaped
            ``(batch, input_channels, Nx, Ny)``.
        :return: Activated spectral plus pointwise convolution, shaped
            ``(batch, output_channels, Nx, Ny)``.
        :rtype: torch.Tensor
        """
        return self._activation(self._spectral_conv(x) + self._linear(x))


class FourierBlock3D(nn.Module):
    """
    The inner block of the Fourier Neural Operator for 3-dimensional input
    tensors.

    The module computes the spectral convolution of the input with a linear
    kernel in the fourier space, and then it maps the input back to the physical
    space. The output is then added to a Linear transformation of the input in
    the physical space. Finally an activation function is applied to the output.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_

    :Example:

        >>> import torch
        >>> from pina.model.block import FourierBlock3D
        >>> block = FourierBlock3D(
        ...     input_numb_fields=2, output_numb_fields=2, n_modes=[4, 4, 4]
        ... )
        >>> x = torch.randn(10, 2, 20, 20, 20)
        >>> out = block(x)
    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        activation=torch.nn.Tanh,
    ):
        r"""
        Initialization of the :class:`FourierBlock3D` class.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param n_modes: The number of modes to select for each dimension.
            A scalar is repeated over all three axes. Counts must fit ``Nx``,
            ``Ny``, and ``Nz // 2 + 1``. To avoid overlapping signed bands, use
            at most half the extent on each of the first two axes.
        :type n_modes: int | list[int] | tuple[int, int, int]
        :param type[torch.nn.Module] activation: No-argument activation class.
            Default is :class:`torch.nn.Tanh`.
        :raises ValueError: If types or the mode-list length are invalid.
        """
        super().__init__()

        # check type consistency
        check_consistency(activation(), nn.Module)

        # assign variables
        self._spectral_conv = SpectralConvBlock3D(
            input_numb_fields=input_numb_fields,
            output_numb_fields=output_numb_fields,
            n_modes=n_modes,
        )
        self._activation = activation()
        self._linear = nn.Conv3d(input_numb_fields, output_numb_fields, 1)

    def forward(self, x):
        """
        Forward pass of the block. It performs a spectral convolution and a
        linear transformation of the input. Then, it sums the results.

        :param torch.Tensor x: Real input shaped
            ``(batch, input_channels, Nx, Ny, Nz)``.
        :return: Activated spectral plus pointwise convolution, shaped
            ``(batch, output_channels, Nx, Ny, Nz)``.
        :rtype: torch.Tensor
        """
        return self._activation(self._spectral_conv(x) + self._linear(x))
