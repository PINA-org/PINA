"""Module for the Kolmogorov-Arnold Network block."""

import torch
from pina._src.model.vectorized_spline import VectorizedSpline
from pina._src.core.utils import check_consistency, check_positive_integer


class KANBlock(torch.nn.Module):
    """
    The inner block of the Kolmogorov-Arnold Network (KAN).

    The block applies a spline transformation to the input, optionally combined
    with a linear transformation of a base activation function. The output is
    aggregated across input dimensions to produce the final output.

    .. seealso::

        **Original reference**:
        Liu Z., Wang Y., Vaidya S., Ruehle F., Halverson J., Soljacic M.,
        Hou T., Tegmark M. (2025).
        *KAN: Kolmogorov-Arnold Networks*.
        DOI: `arXiv preprint arXiv:2404.19756.
        <https://arxiv.org/abs/2404.19756>`_

    :Example:

        >>> import torch
        >>> from pina.model.block import KANBlock
        >>> block = KANBlock(input_dimensions=2, output_dimensions=1)
        >>> x = torch.randn(10, 2)
        >>> out = block(x)
    """

    def __init__(
        self,
        input_dimensions,
        output_dimensions,
        spline_order=3,
        n_knots=10,
        grid_range=[0, 1],
        base_function=torch.nn.SiLU,
        use_base_linear=True,
        use_bias=True,
        init_scale_spline=1e-2,
        init_scale_base=1.0,
    ):
        """
        Initialization of the :class:`KANBlock` class.

        :param int input_dimensions: The number of input features.
        :param int output_dimensions: The number of output features.
        :param int spline_order: The order of each spline basis function.
            Basis degree is ``spline_order - 1``. Default is ``3`` (quadratic).
        :param int n_knots: The number of knots for each spline basis function.
            Default is ``10``; at least twice the order knots are constructed.
        :param grid_range: The range for the spline knots. It must be either a
            list or tuple of numeric endpoints. Default is ``[0, 1]``.
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
        :raises ValueError: If activation, flag, endpoint, or scale types fail
            validation. Endpoint ordering is not validated.
        :raises AssertionError: If dimensions, order, or knot count are not
            positive integers.
        """
        super().__init__()

        # Check consistency
        check_consistency(base_function, torch.nn.Module, subclass=True)
        check_positive_integer(input_dimensions, strict=True)
        check_positive_integer(output_dimensions, strict=True)
        check_positive_integer(spline_order, strict=True)
        check_positive_integer(n_knots, strict=True)
        check_consistency(use_base_linear, bool)
        check_consistency(use_bias, bool)
        check_consistency(init_scale_spline, (int, float))
        check_consistency(init_scale_base, (int, float))
        check_consistency(grid_range, (int, float))

        # Raise error if grid_range is not valid
        if len(grid_range) != 2:
            raise ValueError("Grid must be a list or tuple with two elements.")

        # Knots for the spline basis functions
        initial_knots = torch.ones(spline_order) * grid_range[0]
        final_knots = torch.ones(spline_order) * grid_range[1]

        # Number of internal knots
        n_internal = max(0, n_knots - 2 * spline_order)

        # Internal knots are uniformly spaced in the grid range
        internal_knots = torch.linspace(
            grid_range[0], grid_range[1], n_internal + 2
        )[1:-1]

        # Define the knots
        knots = torch.cat((initial_knots, internal_knots, final_knots))
        knots = knots.unsqueeze(0).repeat(input_dimensions, 1)

        # Define the control points for the spline basis functions
        control_points = (
            torch.randn(
                input_dimensions,
                output_dimensions,
                knots.shape[-1] - spline_order,
            )
            * init_scale_spline
        )

        # Define the vectorized spline module
        self.spline = VectorizedSpline(
            order=spline_order, knots=knots, control_points=control_points
        )

        # Initialize the base function
        self.base_function = base_function()

        # Initialize the base linear weights if needed
        if use_base_linear:
            self.base_weight = torch.nn.Parameter(
                torch.randn(output_dimensions, input_dimensions)
                * (init_scale_base / (input_dimensions**0.5))
            )
        else:
            self.register_parameter("base_weight", None)

        # Initialize the bias term if needed
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_dimensions))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Forward pass of the Kolmogorov-Arnold block. The input is passed through
        the spline transformation, optionally combined with a linear
        transformation of the base function output, and then aggregated across
        input dimensions to produce the final output.

        :param x: Input of shape ``(batch, input_dimensions)``.
        :type x: torch.Tensor | LabelTensor
        :return: Input-summed spline/base contributions of shape
            ``(batch, output_dimensions)``. The scalar ``1 -> 1`` spline case
            squeezes an axis internally. With base linear terms, broadcasting
            couples batch rows before summation, returning ``(batch, 1)``;
            without them the result has shape ``(batch,)``.
        :rtype: torch.Tensor | LabelTensor
        """
        y = self.spline(x)

        if self.base_weight is not None:
            base_x = self.base_function(x)
            base_out = torch.einsum("bi,oi->bio", base_x, self.base_weight)
            y = y + base_out

        # aggregate contributions from all input dimensions
        y = y.sum(dim=1)

        if self.bias is not None:
            y = y + self.bias

        return y
