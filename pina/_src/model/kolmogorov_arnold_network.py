"""Kolmogorov-Arnold networks built from independent spline blocks."""

import torch
from pina._src.model.block.kan_block import KANBlock
from pina._src.core.utils import check_consistency


class KolmogorovArnoldNetwork(torch.nn.Module):
    """
    Implementation of Kolmogorov-Arnold Network (KAN).

    The model consists of a sequence of KAN blocks, where each block applies a
    spline transformation to the input, optionally combined with a linear
    transformation of a base activation function.

    .. seealso::

        **Original reference**:
        Liu Z., Wang Y., Vaidya S., Ruehle F., Halverson J., Soljacic M.,
        Hou T., Tegmark M. (2025).
        *KAN: Kolmogorov-Arnold Networks*.
        DOI: `arXiv preprint arXiv:2404.19756.
        <https://arxiv.org/abs/2404.19756>`_

    :Example:

        >>> import torch
        >>> from pina.model import KolmogorovArnoldNetwork
        >>> model = KolmogorovArnoldNetwork(
        ...     layers=[2, 16, 1], spline_order=3, n_knots=10
        ... )
        >>> x = torch.randn(10, 2)
        >>> out = model(x)
    """

    def __init__(
        self,
        layers,
        spline_order=3,
        n_knots=10,
        grid_range=[-1, 1],
        base_function=torch.nn.SiLU,
        use_base_linear=True,
        use_bias=True,
        init_scale_spline=1e-2,
        init_scale_base=1.0,
    ):
        """
        Initialization of the :class:`KolmogorovArnoldNetwork` class.

        :param layers: A list of integers specifying the sizes of each layer,
            including input and output dimensions.
        :type layers: list[int] | tuple[int, ...]
        :param int spline_order: The order of each spline basis function.
            Basis degree is ``spline_order - 1``. Default is ``3`` (quadratic).
        :param int n_knots: The number of knots for each spline basis function.
            Default is ``10``; at least twice the order knots are constructed.
        :param grid_range: The range for the spline knots. It must be either a
            list or tuple of two numeric endpoints. Default is ``[-1, 1]``.
        :type grid_range: list[int | float] | tuple[int | float, int | float]
        :param type[torch.nn.Module] base_function: The activation class to be
            applied to the input before the linear transformation. Default is
            :class:`torch.nn.SiLU`.
        :param bool use_base_linear: Whether to include a linear transformation
            of the base function output. Default is ``True``.
        :param bool use_bias: Whether to include a bias term in the output.
            Default is ``True``.
        :param init_scale_spline: The scale for initializing each spline
            control points. Default is ``1e-2``.
        :type init_scale_spline: float | int
        :param init_scale_base: The scale for initializing the base linear
            weights. Default is ``1.0``.
        :type init_scale_base: float | int
        :raises ValueError: If ``grid_range`` is not of length 2.
        :raises ValueError: If fewer than two layer widths are supplied or type
            validation fails in this model or its blocks.
        :raises AssertionError: If layer widths, order, or knot count are not
            positive integers.
        """
        super().__init__()

        # Check consistency -- all other checks are performed in KANBlock
        check_consistency(layers, int)
        if len(layers) < 2:
            raise ValueError(
                "`Provide at least two elements for layers (input and output)."
            )

        # Initialize KAN blocks
        self.kan_layers = torch.nn.ModuleList(
            [
                KANBlock(
                    input_dimensions=layers[i],
                    output_dimensions=layers[i + 1],
                    spline_order=spline_order,
                    n_knots=n_knots,
                    grid_range=grid_range,
                    base_function=base_function,
                    use_base_linear=use_base_linear,
                    use_bias=use_bias,
                    init_scale_spline=init_scale_spline,
                    init_scale_base=init_scale_base,
                )
                for i in range(len(layers) - 1)
            ]
        )

    def forward(self, x):
        """
        Forward pass of the KolmogorovArnoldNetwork model. It passes the input
        through each KAN block in the network and returns the final output.

        :param x: Input of shape ``(batch, layers[0])``. Spline blocks require
            exactly two dimensions.
        :type x: torch.Tensor | LabelTensor
        :return: Predictions of shape ``(batch, layers[-1])``.
        :rtype: torch.Tensor | LabelTensor
        """
        for layer in self.kan_layers:
            x = layer(x)

        return x
