"""Vectorized univariate B-spline model with per-spline knots."""

import warnings
import torch
from pina._src.core.utils import check_consistency, check_positive_integer


class VectorizedSpline(torch.nn.Module):
    r"""
    The vectorized B-spline model class.

    A :class:`VectorizedSpline` represents a vector spline, i.e., a collection
    of independent univariate B-splines evaluated in parallel. Each univariate
    spline has its own knot vector and its own control points, and acts on one
    input feature.

    Given ``s`` univariate splines, the vector spline maps an input
    :math:`x = (x^{(1)}, \dots, x^{(s)}) \in \mathbb{R}^s` to an output obtained
    by evaluating each univariate spline on its corresponding scalar input
    :math:`x^{(j)}`.

    For the :math:`j`-th univariate spline of order :math:`k`, the output is
    defined as

    .. math::

        S^{(j)}(x^{(j)}) = \sum_{i=1}^{n_j} B_{i,k}^{(j)}(x^{(j)}) C_i^{(j)},

    where:

    - :math:`C^{(j)}` are the control points of the :math:`j`-th univariate
      spline. In the scalar-output case, :math:`C^{(j)} \in \mathbb{R}^{n_j}`.
      More generally, each univariate spline may have output dimension
      :math:`o`, so :math:`C^{(j)} \in \mathbb{R}^{o \times n_j}`.
    - :math:`B_{i,k}^{(j)}(x)` are the B-spline basis functions of order
      :math:`k`, i.e., piecewise polynomials of degree :math:`k-1`, associated
      with the knot vector of the :math:`j`-th univariate spline.
    - :math:`X^{(j)} = \{x_1^{(j)}, x_2^{(j)}, \dots, x_{m_j}^{(j)}\}` is the
      non-decreasing knot vector of the :math:`j`-th univariate spline.

    If the first and last knots of a given univariate spline are repeated
    :math:`k` times, then that univariate spline interpolates its first and last
    control points.

    The full vector spline evaluates all univariate splines in parallel. If each
    univariate spline has output dimension :math:`o`, then before optional
    aggregation the output has shape ``[batch, s, o]``.

    .. note::

        Each univariate spline is forced to be zero outside the interval defined
        by the first and last knots of its own knot vector.

    .. note::

        This class does not represent a single multivariate spline
        :math:`\mathbb{R}^s \to \mathbb{R}^o` with a genuinely multivariate
        basis. Instead, it represents a vector of splines built from ``s``
        independent univariate splines, one for each input feature.

    .. note::

        When using the :meth:`derivative` method of this class, derivatives are
        computed directly in vectorized form and returned with the correct
        shape. In contrast, when relying on ``autograd``, derivatives must be
        computed with suitable Jacobian/vector-Jacobian products. Autograd does
        support this structure; this method returns coordinatewise derivatives
        directly without constructing a full Jacobian.

    :Example:

    >>> from pina.model import VectorizedSpline
    >>> import torch

    >>> knt1 = torch.tensor([
    ...     [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
    ...     [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
    ... ])
    >>> spline1 = VectorizedSpline(order=3, knots=knt1, control_points=None)

    >>> knt2 = {"n": 7, "min": 0.0, "max": 2.0, "mode": "auto", "n_splines": 2}
    >>> spline2 = VectorizedSpline(order=3, knots=knt2, control_points=None)

    >>> knt3 = torch.tensor([
    ...     [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
    ...     [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
    ... ])
    >>> ctrl3 = torch.tensor([
    ...     [0.0, 1.0, 3.0, 2.0],
    ...     [1.0, 0.0, 2.0, 1.0],
    ... ])
    >>> spline3 = VectorizedSpline(order=3, knots=knt3, control_points=ctrl3)
    """

    def __init__(
        self,
        order=4,
        knots=None,
        control_points=None,
        aggregate_output=None,
    ):
        """
        Initialization of the :class:`VectorizedSpline` class.

        :param int order: The order of each univariate spline. The corresponding
            basis functions are polynomials of degree ``order - 1``.
            Default is ``4``.
        :param knots: The knots of the spline. If a tensor is provided, it must
            have shape ``[s, n]``, where ``s`` is the number of univariate
            splines and ``n`` is the number of knots per univariate spline. If a
            dictionary is provided, it must contain the keys ``"n"``, ``"min"``,
            ``"max"``, ``"mode"``, and ``"n_splines"``. Here, ``"n"`` specifies
            the number of knots for each univariate spline, ``"min"`` and
            ``"max"`` define the interval, ``"mode"`` selects the sampling
            strategy, and ``"n_splines"`` specifies the number of univariate
            splines. The supported modes are ``"uniform"``, where the knots are
            evenly spaced over :math:`[min, max]`, and ``"auto"``, where knots
            are constructed to ensure that each univariate spline interpolates
            the first and last control points. In this case, the number of knots
            is adjusted if :math:`n < 2 * order`. With ``None``, knots are
            initialized automatically over :math:`[0, 1]` ensuring interpolation
            of the first and last control points using the supplied control
            count. Knots and controls cannot both be ``None``.
            Default is ``None``.
        :type knots: torch.Tensor | dict[str, int | float | str] | None
        :param control_points: The floating control points tensor. The
            tensor must be either of shape ``[s, o, c]`` or ``[s, c]``, where
            each univariate spline has ``c`` control points and output dimension
            ``o``. In the latter case, the control points are expanded to shape
            ``[s, 1, c]``. ``None`` creates zero-initialized trainable controls
            from the knots. Supplied controls are also made trainable.
            Default is ``None``.
        :type control_points: torch.Tensor | None
        :param aggregate_output: If ``None``, the output of each univariate
            spline is returned separately, resulting in an output of shape
            ``[batch, s, o]``, where ``s`` is the number of univariate splines
            and ``o`` is the output dimension of each univariate spline. If set
            to ``"mean"`` or ``"sum"``, the output is aggregated accordingly
            across the last dimension, resulting in an output of shape
            ``[batch, s]``. Even with ``None``, ``s=o=1`` squeezes the output
            axis. Default is ``None``.
        :type aggregate_output: str | None
        :raises AssertionError: If ``order`` is not a positive integer.
        :raises ValueError: If ``knots`` is neither a torch.Tensor nor a
            dictionary, when provided.
        :raises ValueError: If ``aggregate_output`` is not None, "mean", or
            "sum".
        :raises ValueError: If ``control_points`` is not a torch.Tensor,
            when provided.
        :raises ValueError: If both ``knots`` and ``control_points`` are None.
        :raises ValueError: If ``knots`` is not two-dimensional, after
            processing.
        :raises ValueError: If ``control_points``, after expansion when
            two-dimensional, is not three-dimensional.
        :raises ValueError: If, for each univariate spline, the number of
            ``knots`` is not equal to the sum of ``order`` and the number of
            ``control_points.``
        :raises ValueError: If the number of univariate splines in ``knots`` and
             ``control_points`` do not match.

        .. warning::

            Too few controls emits ``UserWarning``. Knot rows are sorted, but
            control coefficients are not reordered with them.
        """

        super().__init__()

        # Check consistency
        check_positive_integer(value=order, strict=True)
        check_consistency(knots, (type(None), torch.Tensor, dict))
        check_consistency(control_points, (type(None), torch.Tensor))

        # Raise error if neither knots nor control points are provided
        if knots is None and control_points is None:
            raise ValueError("knots and control_points cannot both be None.")

        # Raise error if aggregate_output is not None, "mean", or "sum"
        if aggregate_output not in (None, "mean", "sum"):
            raise ValueError(
                f"aggregate_output must be None, 'mean', or 'sum'."
                f" Got {aggregate_output}."
            )

        # Initialize knots if not provided
        if knots is None and control_points is not None:
            knots = {
                "n": control_points.shape[-1] + order,
                "min": 0,
                "max": 1,
                "n_splines": control_points.shape[0],
                "mode": "auto",
            }

        # Initialization - knots and control points managed by their setters
        self.order = order
        self.knots = knots
        self.control_points = control_points
        self.aggregate_output = aggregate_output

        # Check dimensionality of control points
        if self.control_points.ndim != 3:
            raise ValueError("control_points must be three-dimensional.")

        # Raise error if #knots != order + #control_points
        if self.knots.shape[-1] != self.order + self.control_points.shape[-1]:
            raise ValueError(
                f" The number of knots per spline must be equal to order + the"
                f" number of control points. Got {self.knots.shape[-1]} knots"
                f" per spline, {self.control_points.shape[-1]} control points,"
                f" and {self.order} order."
            )

        # Raise warning if spline is degenerate
        if self.control_points.shape[-1] < self.order:
            warnings.warn(
                "The number of control points per spline is smaller than the"
                " spline order. This creates a degenerate spline with limited"
                " flexibility.",
                UserWarning,
            )

        # Raise error if knots and control points have different # of splines
        if self.knots.shape[0] != self.control_points.shape[0]:
            raise ValueError(
                f"The number of splines must be the same for knots and"
                f" control points. Got {self.knots.shape[0]} splines for knots"
                f" and {self.control_points.shape[0]} splines for control"
                f" points."
            )

        # Precompute boundary interval index
        self.register_buffer(
            "_boundary_interval_idx", self._compute_boundary_interval()
        )

        # Precompute denominators used in derivative formulas
        self._compute_derivative_denominators()

    def _compute_boundary_interval(self):
        """
        Precompute the index of the rightmost non-degenerate interval to improve
        performance, eliminating the need to perform a search loop in the basis
        function on each call.

        :return: The index of the rightmost non-degenerate interval for each
            univariate spline.
        :rtype: torch.Tensor
        """
        # Compute the differences between consecutive knots for each spline
        diffs = self._knots[:, 1:] - self._knots[:, :-1]
        valid = diffs > 0

        # Initialize idx tensor to store the last valid interval for each spline
        idx = torch.zeros(
            self._knots.shape[0], dtype=torch.long, device=self._knots.device
        )

        # For each spline, find the last idx where interval is non-degenerate
        for s in range(self._knots.shape[0]):
            valid_s = torch.nonzero(valid[s], as_tuple=False)
            idx[s] = valid_s[-1, 0] if valid_s.numel() > 0 else 0

        return idx

    def _compute_derivative_denominators(self):
        """
        Precompute the denominators used in the derivatives for all orders up to
        the spline order to avoid redundant calculations.
        """
        # Precompute for order 2 to k
        for i in range(2, self.order + 1):

            # Denominators for the derivative recurrence relations
            left_den = self.knots[:, i - 1 : -1] - self.knots[:, :-i]
            right_den = self.knots[:, i:] - self.knots[:, 1 : -i + 1]

            # If consecutive knots are equal, set left and right factors to zero
            left_fac = torch.where(
                torch.abs(left_den) > 1e-10,
                (i - 1) / left_den,
                torch.zeros_like(left_den),
            )
            right_fac = torch.where(
                torch.abs(right_den) > 1e-10,
                (i - 1) / right_den,
                torch.zeros_like(right_den),
            )

            # Register buffers
            self.register_buffer(f"_left_factor_order_{i}", left_fac)
            self.register_buffer(f"_right_factor_order_{i}", right_fac)

    def basis(self, x, collection=False):
        """
        Evaluate the B-spline basis functions for each univariate spline.

        This method applies the Cox-de Boor recursion in vectorized form across
        all univariate splines of the vector spline.

        :param torch.Tensor x: Coordinates of shape ``(batch, s)``, cast to
            plain tensors with the knot dtype, but not automatically moved.
        :param bool collection: Whether to return all order-specific bases,
            with a dummy ``None`` at index zero. Default is ``False``.
        :raises ValueError: If ``collection`` is not a boolean.
        :raises ValueError: If ``x`` is not two-dimensional.
        :raises ValueError: If the number of input features does not match
            the number of univariate splines.
        :return: Bases of shape ``(batch, s, n_controls)``, or an order-indexed
            list when collection is requested.
        :rtype: torch.Tensor | list[torch.Tensor | None]
        """
        # Check consistency
        check_consistency(collection, bool)

        # Ensure x is a tensor of the same dtype as knots
        x = x.as_subclass(torch.Tensor).to(dtype=self.knots.dtype)

        # Raise error if x does not have shape (batch, s)
        if x.ndim != 2:
            raise ValueError(
                f"The input must have shape (batch, s). Got {x.shape}."
            )

        # Raise error if x has different number of splines than knots
        if x.shape[1] != self.knots.shape[0]:
            raise ValueError(
                f"The number of input features must be the same as the number"
                f" of univariate splines. Got {x.shape[1]} input features,"
                f" but {self.knots.shape[0]} univariate splines."
            )

        # Add a final dimension to x for broadcasting
        x = x.unsqueeze(-1)

        # Add an initial dimension to knots for broadcasting
        knots = self.knots.unsqueeze(0)

        # Base case of recursion: indicator functions for the intervals
        basis = (x >= knots[..., :-1]) & (x < knots[..., 1:])
        basis = basis.to(x.dtype)

        # Extract left and right knots of the boundary interval for each spline
        range_tensor = torch.arange(self.knots.shape[0], device=x.device)
        knot_left = self.knots[range_tensor, self._boundary_interval_idx]
        knot_right = self.knots[range_tensor, self._boundary_interval_idx + 1]

        # Identify points at the rightmost boundary
        at_rightmost_boundary = (
            x.squeeze(-1) >= knot_left.unsqueeze(0)
        ) & torch.isclose(
            x.squeeze(-1), knot_right.unsqueeze(0), rtol=1e-8, atol=1e-10
        )

        # Ensure the correct value is set at the rightmost boundary
        if torch.any(at_rightmost_boundary):
            b_idx, s_idx = torch.nonzero(at_rightmost_boundary, as_tuple=True)
            basis[b_idx, s_idx, self._boundary_interval_idx[s_idx]] = 1.0

        # If returning the whole collection, initialize list
        if collection:
            basis_collection = [None, basis]

        # Cox-de Boor recursion -- iterative case
        for i in range(1, self.order):

            # Compute the denominators for both terms of the recursion
            denom1 = knots[..., i:-1] - knots[..., : -(i + 1)]
            denom2 = knots[..., i + 1 :] - knots[..., 1:-i]

            # Ensure no division by zero
            denom1 = torch.where(
                denom1.abs() < 1e-8, torch.ones_like(denom1), denom1
            )
            denom2 = torch.where(
                denom2.abs() < 1e-8, torch.ones_like(denom2), denom2
            )

            # Compute the two terms of the recursion
            term1 = ((x - knots[..., : -(i + 1)]) / denom1) * basis[..., :-1]
            term2 = ((knots[..., i + 1 :] - x) / denom2) * basis[..., 1:]

            # Combine terms to get the new basis
            basis = term1 + term2

            if collection:
                basis_collection.append(basis)

        return basis_collection if collection else basis

    def forward(self, x):
        """
        Forward pass for the :class:`VectorizedSpline` model. Each univariate
        spline is evaluated independently on its corresponding input feature.

        The input is expected to have shape ``[batch, s]``, where ``s`` is the
        number of univariate splines. The output has shape ``[batch, s, o]``,
        where ``o`` is the output dimension of each univariate spline, unless an
        aggregation method is specified. If both ``s`` and ``o`` are 1, the
        output is aggregated across the last dimension, resulting in an output
        of shape ``[batch, s]``. If ``aggregate_output`` is set to ``"mean"`` or
        ``"sum"``, the output is aggregated across the last dimension, resulting
        in an output of shape ``[batch, s]``.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        # Compute the basis functions at x
        basis = self.basis(x)

        # Compute the output for each spline
        out = torch.einsum("bsc,soc->bso", basis, self.control_points)

        # Aggregate output if needed
        if self.aggregate_output == "mean":
            out = out.mean(dim=-1)
        elif self.aggregate_output == "sum":
            out = out.sum(dim=-1)
        elif out.shape[1] == 1 and out.shape[2] == 1:
            out = out.squeeze(-1)

        return out

    def derivative(self, x, degree):
        """
        Compute the ``degree``-th derivative of each univariate spline at the
        given input points.

        The output has shape ``[batch, s, o]``, where ``o`` is the output
        dimension of each univariate spline, unless an aggregation method is
        specified. If both ``s`` and ``o`` are 1, the output is aggregated
        across the last dimension, resulting in an output of shape
        ``[batch, s]``. If ``aggregate_output`` is set to ``"mean"`` or
        ``"sum"``, the output is aggregated across the last dimension, resulting
        in an output of shape ``[batch, s]``.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :param int degree: The derivative degree to compute.
        :raises AssertionError: If ``degree`` is not a nonnegative integer.
        :raises ValueError: If input shape is not ``(batch, n_splines)``.
        :return: The derivative tensor.
        :rtype: torch.Tensor
        """
        # Check consistency
        check_positive_integer(degree, strict=False)

        # Compute basis derivative
        der = self._basis_derivative(x.as_subclass(torch.Tensor), degree=degree)

        # Compute the output for each spline
        out = torch.einsum("bsc,soc->bso", der, self.control_points)

        # Aggregate output if needed
        if self.aggregate_output == "mean":
            out = out.mean(dim=-1)
        elif self.aggregate_output == "sum":
            out = out.sum(dim=-1)
        elif out.shape[1] == 1 and out.shape[2] == 1:
            out = out.squeeze(-1)

        return out

    def _basis_derivative(self, x, degree):
        """
        Compute the ``degree``-th derivative of the vectorized spline basis
        functions at the given input points using an iterative approach.

        :param torch.Tensor x: The points to be evaluated.
        :param int degree: The derivative degree to compute.
        :return: The derivative of the basis functions of order ``self.order``.
        :rtype: torch.Tensor
        """
        # Compute the whole basis collection
        basis = self.basis(x, collection=True)

        # Derivatives initialization (dummy at index 0 for convenience)
        derivatives = [None] + [basis[o] for o in range(1, self.order + 1)]

        # Iterate over derivative degrees
        for _ in range(1, degree + 1):

            # Current degree derivatives (with dummy at index 0 for convenience)
            current_der = [None] * (self.order + 1)
            current_der[1] = torch.zeros_like(derivatives[1])

            # Iterate over basis orders
            for o in range(2, self.order + 1):

                # Retrieve precomputed factors
                left_fac = getattr(self, f"_left_factor_order_{o}")
                right_fac = getattr(self, f"_right_factor_order_{o}")

                # derivatives[o - 1] has shape [b, s, m]
                # Slice previous derivatives to align
                left_part = derivatives[o - 1][..., :-1]
                right_part = derivatives[o - 1][..., 1:]

                # Broadcast factors over batch dims
                left_fac = left_fac.unsqueeze(0)
                right_fac = right_fac.unsqueeze(0)

                # Compute current derivatives
                current_der[o] = left_fac * left_part - right_fac * right_part

            # Update derivatives for next degree
            derivatives = current_der

        return derivatives[self.order]

    @property
    def control_points(self):
        """
        The control points of the spline.

        :return: The stored trainable tensor of shape ``(s, o, c)``.
        :rtype: torch.nn.Parameter
        """
        return self._control_points

    @control_points.setter
    def control_points(self, control_points):
        """
        Set the control points of the spline.

        :param control_points: The control points tensor. The
            tensor must be either of shape ``[s, o, c]`` or ``[s, c]``, where
            each univariate spline has ``c`` control points and output dimension
            ``o``. In the latter case, the control points are expanded to shape
            ``[s, 1, c]``. ``None`` initializes zero controls using current
            knot counts. This setter does not validate the final shape.
        :type control_points: torch.Tensor | None
        :raises ValueError: If there are not enough knots to define the control
            points, due to the relation: #knots = order + #control_points.
        """
        # If control points are not provided, initialize them
        if control_points is None:

            # Check that there are enough knots to define control points
            if self.knots.shape[-1] < self.order + 1:
                raise ValueError(
                    f"Not enough knots to define control points. Got"
                    f" {self.knots.shape[-1]} knots for each univariate spline,"
                    f" but need at least {self.order + 1}."
                )

            # Initialize control points to zero
            control_points = torch.zeros(
                self.knots.shape[0], 1, self.knots.shape[-1] - self.order
            )

        # If a the control points are 2D, add an output dimension of size 1
        if control_points.ndim == 2:
            control_points = control_points.unsqueeze(1)

        # Set control points
        self._control_points = torch.nn.Parameter(
            control_points, requires_grad=True
        )

    @property
    def knots(self):
        """
        The knots of the spline.

        :return: The knots.
        :rtype: torch.Tensor
        """
        return self._knots

    @knots.setter
    def knots(self, value):
        """
        Set the knots of the spline.

        :param value: The knots of the spline. If a tensor is provided, it must
            have shape ``[s, n]``, where ``s`` is the number of univariate
            splines and ``n`` is the number of knots per univariate spline. If a
            dictionary is provided, it must contain the keys ``"n"``, ``"min"``,
            ``"max"``, ``"mode"``, and ``"n_splines"``. Here, ``"n"`` specifies
            the number of knots for each univariate spline, ``"min"`` and
            ``"max"`` define the interval, ``"mode"`` selects the sampling
            strategy, and ``"n_splines"`` specifies the number of univariate
            splines. The supported modes are ``"uniform"``, where the knots are
            evenly spaced over :math:`[min, max]`, and ``"auto"``, where knots
            are constructed to ensure that each univariate spline interpolates
            the first and last control points. In this case, the number of knots
            is adjusted if :math:`n < 2 * order`. ``None`` is not supported by
            this setter. Rows are sorted, derivative buffers refreshed, and
            existing control coefficients left unchanged.
        :type value: torch.Tensor | dict[str, int | float | str]
        :raises ValueError: If a dictionary is provided but does not contain
            the required keys.
        :raises ValueError: If the mode specified in the dictionary is invalid.
        :raises ValueError: If knots is not two-dimensional after processing.
        """
        # If a dictionary is provided, initialize knots accordingly
        if isinstance(value, dict):

            # Check that required keys are present
            required_keys = {"n", "min", "max", "mode", "n_splines"}
            if not required_keys.issubset(value.keys()):
                raise ValueError(
                    f"When providing knots as a dictionary, the following "
                    f"keys must be present: {required_keys}. Got "
                    f"{value.keys()}."
                )

            # Save number of splines for later use
            n_splines = value["n_splines"]

            # Uniform sampling of knots
            if value["mode"] == "uniform":
                value = torch.linspace(value["min"], value["max"], value["n"])

            # Automatic sampling of interpolating knots
            elif value["mode"] == "auto":

                # Repeat the first and last knots 'order' times
                initial_knots = torch.ones(self.order) * value["min"]
                final_knots = torch.ones(self.order) * value["max"]

                # Number of internal knots
                n_internal = value["n"] - 2 * self.order

                # If no internal knots are needed, just concatenate boundaries
                if n_internal <= 0:
                    value = torch.cat((initial_knots, final_knots))

                # Else, sample internal knots uniformly and exclude boundaries
                # Recover the correct number of internal knots when slicing by
                # adding 2 to n_internal
                else:
                    internal_knots = torch.linspace(
                        value["min"], value["max"], n_internal + 2
                    )[1:-1]
                    value = torch.cat(
                        (initial_knots, internal_knots, final_knots)
                    )

            # Raise error if mode is invalid
            else:
                raise ValueError(
                    f"Invalid mode for knots initialization. Got "
                    f"{value['mode']}, but expected 'uniform' or 'auto'."
                )

            # Repeat the knot vector for each spline
            value = value.unsqueeze(0).repeat(n_splines, 1)

        # Set knots
        self.register_buffer("_knots", value.sort(dim=-1).values)

        # Check dimensionality of knots
        if self.knots.ndim != 2:
            raise ValueError("knots must be two-dimensional.")

        # Recompute boundary interval when knots change
        if hasattr(self, "_boundary_interval_idx"):
            self.register_buffer(
                "_boundary_interval_idx", self._compute_boundary_interval()
            )

        # Recompute derivative denominators when knots change
        self._compute_derivative_denominators()
