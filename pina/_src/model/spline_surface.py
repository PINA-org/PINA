"""Module for the bivariate B-Spline surface model class."""

import torch
from .spline import Spline
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency, check_positive_integer


class SplineSurface(torch.nn.Module):
    r"""
    The bivariate B-Spline surface model class.

    A bivariate B-spline surface is a parametric surface defined as the tensor
    product of two univariate B-spline curves:

    .. math::

        S(x, y) = \sum_{i=1}^{n_x} \sum_{j=1}^{n_y} B_{i,k}(x) B_{j,s}(y)
        C_{i,j}, \quad x \in [x_1, x_m], y \in [y_1, y_l]

    where:

    - :math:`C \in \mathbb{R}^{n_x \times n_y}` is the matrix of learnable
      control coefficients. Its entries :math:`C_{i,j}` influence the shape of
      the surface but are not generally interpolated, except under certain knot
      multiplicities.
    - :math:`B_{i,k}(x)` and :math:`B_{j,s}(y)` are the B-spline basis functions
      defined over two orthogonal directions, with orders :math:`k` and
      :math:`s`, respectively.
    - :math:`X = \{ x_1, x_2, \dots, x_m \}` and
      :math:`Y = \{ y_1, y_2, \dots, y_l \}` are the non-decreasing knot
      vectors along the two directions.

    :Example:

        >>> import torch
        >>> from pina.model import SplineSurface
        >>> model = SplineSurface(
        ...     orders=[3, 3], control_points=torch.zeros(3, 3)
        ... )
        >>> x = torch.tensor([[0.5, 0.3], [0.2, 0.8]])
        >>> out = model(x)
    """

    def __init__(self, orders, knots_u=None, knots_v=None, control_points=None):
        """
        Initialization of the :class:`SplineSurface` class.

        :param orders: The two spline orders along the parametric
            direction. Each order defines the degree of the corresponding basis
            as ``degree = order - 1``.
        :type orders: list[int] | tuple[int, int]
        :param knots_u: The knots of the spline along the first direction.
            For details on valid formats and initialization modes, see the
            :class:`pina.model.Spline` class. ``None`` derives auto knots on
            ``[0, 1]`` from the control matrix. Default is ``None``.
        :type knots_u: torch.Tensor | dict[str, int | float | str] | None
        :param knots_v: The knots of the spline along the second direction.
            For details on valid formats and initialization modes, see the
            :class:`pina.model.Spline` class. ``None`` derives auto knots on
            ``[0, 1]`` from the control matrix. Default is ``None``.
        :type knots_v: torch.Tensor | dict[str, int | float | str] | None
        :param control_points: The floating control points defining the
            surface geometry. It must be a two-dimensional tensor of shape
            ``[len(knots_u) - orders[0], len(knots_v) - orders[1]]``.
            ``None`` initializes zero trainable controls only when both knot
            specifications are supplied. Default is ``None``.
        :type control_points: torch.Tensor | None
        :raises ValueError: If ``orders`` is not a list of integers.
        :raises ValueError: If ``knots_u`` is neither a torch.Tensor nor a
            dictionary, when provided.
        :raises ValueError: If ``knots_v`` is neither a torch.Tensor nor a
            dictionary, when provided.
        :raises ValueError: If ``control_points`` is not a torch.Tensor,
            when provided.
        :raises ValueError: If ``orders`` is not a list of two elements.
        :raises ValueError: If either knot specification and the control matrix
            are both ``None``, or the control shape disagrees with the knots.
        :raises AssertionError: If either order is not a positive integer.
        """
        super().__init__()

        # Check consistency
        check_consistency(orders, int)
        check_consistency(control_points, (type(None), torch.Tensor))
        check_consistency(knots_u, (type(None), torch.Tensor, dict))
        check_consistency(knots_v, (type(None), torch.Tensor, dict))

        # Check orders is a list of two elements
        if len(orders) != 2:
            raise ValueError("orders must be a list of two elements.")

        # Raise error if neither knots nor control points are provided
        if (knots_u is None or knots_v is None) and control_points is None:
            raise ValueError(
                "control_points cannot be None if knots_u or knots_v is None."
            )

        # Initialize knots_u if not provided
        if knots_u is None and control_points is not None:
            knots_u = {
                "n": control_points.shape[0] + orders[0],
                "min": 0,
                "max": 1,
                "mode": "auto",
            }

        # Initialize knots_v if not provided
        if knots_v is None and control_points is not None:
            knots_v = {
                "n": control_points.shape[1] + orders[1],
                "min": 0,
                "max": 1,
                "mode": "auto",
            }

        # Create two univariate b-splines
        self.spline_u = Spline(order=orders[0], knots=knots_u)
        self.spline_v = Spline(order=orders[1], knots=knots_v)
        self.control_points = control_points

        # Delete unneeded parameters
        delattr(self.spline_u, "_control_points")
        delattr(self.spline_v, "_control_points")

    def forward(self, x):
        """
        Forward pass for the :class:`SplineSurface` model.

        :param x: Coordinates of shape ``(..., n, 2)``; coordinate columns are
            interpreted positionally, regardless of labels.
        :type x: torch.Tensor | LabelTensor
        :return: Plain spline values of shape ``(..., n, 1)``.
        :rtype: torch.Tensor
        """
        return torch.einsum(
            "...bi, ...bj, ij -> ...b",
            self.spline_u.basis(x.as_subclass(torch.Tensor)[..., 0]),
            self.spline_v.basis(x.as_subclass(torch.Tensor)[..., 1]),
            self.control_points,
        ).unsqueeze(-1)

    def derivative(self, x, degree_u, degree_v):
        """
        Compute the partial derivatives of the spline at the given points.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :param int degree_u: The degree of the derivative along the first
            parameter direction.
        :param int degree_v: The degree of the derivative along the second
            parameter direction.
        :raises AssertionError: If either degree is not a nonnegative integer.
        :return: The mixed derivative. For plain ``(n, 2)`` input, shape is
            ``(n,)``; labeled column extraction retains a singleton axis and
            normally yields ``(n, 1)``.
        :rtype: torch.Tensor
        """
        # Check consistency
        check_positive_integer(degree_u, strict=False)
        check_positive_integer(degree_v, strict=False)

        # Split input into u and v components
        if isinstance(x, LabelTensor):
            u = x[x.labels[0]].as_subclass(torch.Tensor)
            v = x[x.labels[1]].as_subclass(torch.Tensor)
        else:
            u = x[..., 0]
            v = x[..., 1]

        # Compute basis derivatives
        der_u = self.spline_u._basis_derivative(u, degree=degree_u)
        der_v = self.spline_v._basis_derivative(v, degree=degree_v)

        return torch.einsum(
            "...bi, ...bj, ij -> ...b", der_u, der_v, self.control_points
        )

    def gradient(self, x):
        """
        Convenience method to compute the gradient of the spline surface.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: Concatenated partial derivatives. Labeled ``(n, 2)`` inputs
            yield ``(n, 2)``; plain inputs yield flattened shape ``(2 * n,)``.
        :rtype: torch.Tensor
        """
        # Compute partial derivatives
        du = self.derivative(x, degree_u=1, degree_v=0)
        dv = self.derivative(x, degree_u=0, degree_v=1)

        return torch.cat((du, dv), dim=-1)

    def laplacian(self, x):
        """
        Convenience method to compute the laplacian of the spline surface.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: The sum of second partial derivatives, with the same shape
            convention as :meth:`derivative`.
        :rtype: torch.Tensor
        """
        # Compute second partial derivatives
        ddu = self.derivative(x, degree_u=2, degree_v=0)
        ddv = self.derivative(x, degree_u=0, degree_v=2)

        return ddu + ddv

    @property
    def knots(self):
        """
        The knots of the univariate splines defining the spline surface.

        :return: The knots.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return self.spline_u.knots, self.spline_v.knots

    @knots.setter
    def knots(self, value):
        """
        Set the knots of the spline surface.

        :param value: A tuple (knots_u, knots_v) containing the knots for both
            parametric directions.
        :type value: tuple[torch.Tensor | dict[str, int | float | str],
            torch.Tensor | dict[str, int | float | str]]
        :raises ValueError: If value is not a tuple of two elements.
        """
        # Check value is a tuple of two elements
        if not (isinstance(value, tuple) and len(value) == 2):
            raise ValueError("Knots must be a tuple of two elements.")

        knots_u, knots_v = value
        self.spline_u.knots = knots_u
        self.spline_v.knots = knots_v

    @property
    def control_points(self):
        """
        The control points of the spline.

        :return: The stored trainable two-dimensional control matrix.
        :rtype: torch.nn.Parameter
        """
        return self._control_points

    @control_points.setter
    def control_points(self, control_points):
        """
        Set the control points of the spline surface.

        :param control_points: The bidimensional control points
            tensor, where each dimension refers to a direction in the parameter
            space. ``None`` initializes zero trainable controls from the knots.
        :type control_points: torch.Tensor | None
        :raises ValueError: If in any direction there are not enough knots to
            define the control points, due to the relation:
            #knots = order + #control_points.
        :raises ValueError: If ``control_points`` is not of the correct shape.
        """
        # Save correct shape of control points
        __valid_shape = (
            len(self.spline_u.knots) - self.spline_u.order,
            len(self.spline_v.knots) - self.spline_v.order,
        )

        # If control points are not provided, initialize them
        if control_points is None:

            # Check that there are enough knots to define control points
            if (
                len(self.spline_u.knots) < self.spline_u.order + 1
                or len(self.spline_v.knots) < self.spline_v.order + 1
            ):
                raise ValueError(
                    f"Not enough knots to define control points. Got "
                    f"{len(self.spline_u.knots)} knots along u and "
                    f"{len(self.spline_v.knots)} knots along v, but need at "
                    f"least {self.spline_u.order + 1} and "
                    f"{self.spline_v.order + 1}, respectively."
                )

            # Initialize control points to zero
            control_points = torch.zeros(__valid_shape)

        # Check control points
        if control_points.shape != __valid_shape:
            raise ValueError(
                f"control_points must be of the correct shape. "
                f"Expected {__valid_shape}, got {control_points.shape}."
            )

        # Register control points as a learnable parameter
        self._control_points = torch.nn.Parameter(
            control_points, requires_grad=True
        )
