"""Module for the Radial Basis Function Interpolation layer."""

import math
import warnings
from itertools import combinations_with_replacement
import torch
from pina._src.core.utils import check_consistency


def linear(r):
    """
    Linear radial basis function.

    :param torch.Tensor r: Distance between points.
    :return: The linear radial basis function.
    :rtype: torch.Tensor
    """
    return -r


def thin_plate_spline(r, eps=1e-7):
    """
    Thin plate spline radial basis function.

    :param torch.Tensor r: Distance between points.
    :param float eps: Minimum distance before evaluating the logarithm.
        Default is ``1e-7``.
    :return: The thin plate spline radial basis function.
    :rtype: torch.Tensor
    """
    r = torch.clamp(r, min=eps)
    return r**2 * torch.log(r)


def cubic(r):
    """
    Cubic radial basis function.

    :param torch.Tensor r: Distance between points.
    :return: The cubic radial basis function.
    :rtype: torch.Tensor
    """
    return r**3


def quintic(r):
    """
    Quintic radial basis function.

    :param torch.Tensor r: Distance between points.
    :return: The quintic radial basis function.
    :rtype: torch.Tensor
    """
    return -(r**5)


def multiquadric(r):
    """
    Multiquadric radial basis function.

    :param torch.Tensor r: Distance between points.
    :return: The multiquadric radial basis function.
    :rtype: torch.Tensor
    """
    return -torch.sqrt(r**2 + 1)


def inverse_multiquadric(r):
    """
    Inverse multiquadric radial basis function.

    :param torch.Tensor r: Distance between points.
    :return: The inverse multiquadric radial basis function.
    :rtype: torch.Tensor
    """
    return 1 / torch.sqrt(r**2 + 1)


def inverse_quadratic(r):
    """
    Inverse quadratic radial basis function.

    :param torch.Tensor r: Distance between points.
    :return: The inverse quadratic radial basis function.
    :rtype: torch.Tensor
    """
    return 1 / (r**2 + 1)


def gaussian(r):
    """
    Gaussian radial basis function.

    :param torch.Tensor r: Distance between points.
    :return: The gaussian radial basis function.
    :rtype: torch.Tensor
    """
    return torch.exp(-(r**2))


radial_functions = {
    "linear": linear,
    "thin_plate_spline": thin_plate_spline,
    "cubic": cubic,
    "quintic": quintic,
    "multiquadric": multiquadric,
    "inverse_multiquadric": inverse_multiquadric,
    "inverse_quadratic": inverse_quadratic,
    "gaussian": gaussian,
}

scale_invariant = {"linear", "thin_plate_spline", "cubic", "quintic"}

min_degree_funcs = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
}


class RBFBlock(torch.nn.Module):
    """
    Radial Basis Function (RBF) interpolation layer.

    The user needs to fit the model with the data, before using it to
    interpolate new points. The layer is not trainable.

    .. note::
        It follows :class:`scipy.interpolate.RBFInterpolator`
        and it is inspired from the implementation in `torchrbf.
        <https://github.com/ArmanMaesumi/torchrbf>`_

    :Example:

        >>> import torch
        >>> from pina.model.block import RBFBlock
        >>> block = RBFBlock(kernel="gaussian", epsilon=1.0)
        >>> y = torch.randn(50, 2)
        >>> d = torch.randn(50, 1)
        >>> block.fit(y, d)
        >>> x = torch.randn(10, 2)
        >>> out = block(x)
    """

    def __init__(
        self,
        neighbors=None,
        smoothing=0.0,
        kernel="thin_plate_spline",
        epsilon=None,
        degree=None,
    ):
        """
        Initialization of the :class:`RBFBlock` class.

        :param neighbors: Neighbor count, currently unsupported by ``fit``.
            ``None`` uses all observations. Default is ``None``.
        :type neighbors: int | None
        :param smoothing: Diagonal regularization, scalar or per-observation
            tensor shaped ``(n_points,)``. ``0.0`` requests exact interpolation.
            Default is ``0.0``.
        :type smoothing: int | float | torch.Tensor
        :param str kernel: The radial basis function to use.
            The available kernels are: ``linear``, ``thin_plate_spline``,
            ``cubic``, ``quintic``, ``multiquadric``, ``inverse_multiquadric``,
            ``inverse_quadratic``, or ``gaussian``.
            Default is ``"thin_plate_spline"``.
        :param epsilon: Input distance scale. ``None`` selects ``1.0`` for
            scale-invariant kernels and raises for other kernels.
            Default is ``None``.
        :type epsilon: float | None
        :param degree: The polynomial degree. Some kernels require a
            minimum degree of the polynomial to ensure that the RBF is well
            defined. These minimum degrees are specified in the
            ``min_degree_funcs`` dictionary. ``None`` uses the larger of zero
            and that minimum; ``-1`` omits polynomial terms. A below-minimum
            degree emits a warning but is retained. Default is ``None``.
        :type degree: int | None
        :raises ValueError: If argument types, kernel name, epsilon selection,
            or degree below ``-1`` fail validation.

        .. note::

            Fitted tensors are ordinary attributes, not registered buffers.
            They are neither moved by ``to`` nor saved in ``state_dict``.
        """

        super().__init__()
        check_consistency(neighbors, (int, type(None)))
        check_consistency(smoothing, (int, float, torch.Tensor))
        check_consistency(kernel, str)
        check_consistency(epsilon, (float, type(None)))
        check_consistency(degree, (int, type(None)))

        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.powers = None
        # initialize data points and values
        self.y = None
        self.d = None
        # initialize attributes for the fitted model
        self._shift = None
        self._scale = None
        self._coeffs = None

    @property
    def smoothing(self):
        """
        The smoothing parameter for the interpolation.

        :return: The smoothing parameter.
        :rtype: int | float | torch.Tensor
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        """
        Set the smoothing parameter for the interpolation.

        :param value: Scalar or per-observation diagonal regularization.
        :type value: int | float | torch.Tensor
        """
        self._smoothing = value

    @property
    def kernel(self):
        """
        The Radial basis function.

        :return: The radial basis function.
        :rtype: str
        """
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        """
        Set the radial basis function.

        :param str value: The radial basis function.
        :raises ValueError: If the name is not a supported lowercase kernel key.
        """
        if value not in radial_functions:
            raise ValueError(f"Unknown kernel: {value}")
        self._kernel = value.lower()

    @property
    def epsilon(self):
        """
        The shape parameter that scales the input to the RBF.

        :return: The shape parameter.
        :rtype: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        """
        Set the shape parameter.

        :param value: Shape scale; ``None`` selects one for scale-invariant
            kernels and is rejected otherwise.
        :type value: float | None
        :raises ValueError: If the kernel requires an epsilon and it is not
            specified.
        """
        if value is None:
            if self.kernel in scale_invariant:
                value = 1.0
            else:
                raise ValueError("Must specify `epsilon` for this kernel.")
        else:
            value = float(value)
        self._epsilon = value

    @property
    def degree(self):
        """
        The degree of the polynomial.

        :return: The degree of the polynomial.
        :rtype: int
        """
        return self._degree

    @degree.setter
    def degree(self, value):
        """
        Set the degree of the polynomial.

        :param value: Polynomial degree; ``None`` selects the larger of zero
            and the kernel minimum.
        :type value: int | None
        :raises ValueError: If the degree is less than ``-1``.

        .. warning::

            A value below the kernel minimum emits ``UserWarning`` but is kept,
            despite the warning text claiming it will be raised to the minimum.
        """
        min_degree = min_degree_funcs.get(self.kernel, -1)
        if value is None:
            value = max(min_degree, 0)
        else:
            value = int(value)
            if value < -1:
                raise ValueError("`degree` must be at least -1.")
            if value < min_degree:
                warnings.warn(
                    "`degree` is too small for this kernel. Setting to "
                    f"{min_degree}.",
                    UserWarning,
                )
        self._degree = value

    def _check_data(self, y, d):
        """
        Check the data consistency.

        :param torch.Tensor y: Centers of shape ``(n_points, n_dimensions)``.
        :param torch.Tensor d: Values with matching leading point count.
        :raises ValueError: If the data is not consistent.
        """
        if y.ndim != 2:
            raise ValueError("y must be a 2-dimensional tensor.")

        if d.shape[0] != y.shape[0]:
            raise ValueError(
                "The first dim of d must have the same length as "
                "the first dim of y."
            )

        if isinstance(self.smoothing, (int, float)):
            self.smoothing = (
                torch.full((y.shape[0],), self.smoothing).float().to(y.device)
            )

    def fit(self, y, d):
        """
        Fit the RBF interpolator and retain references to training data.

        :param torch.Tensor y: Centers of shape ``(n_points, n_dimensions)``.
        :param torch.Tensor d: Values of shape ``(n_points, ...)``.
        :raises NotImplementedError: If the neighbors are not ``None``.
        :raises ValueError: If the data is not compatible with the requested
            degree.
        """
        self._check_data(y, d)

        self.y = y
        self.d = d

        if self.neighbors is None:
            nobs = self.y.shape[0]
        else:
            raise NotImplementedError("Neighbors currently not supported")

        powers = RBFBlock.monomial_powers(self.y.shape[1], self.degree).to(
            y.device
        )
        if powers.shape[0] > nobs:
            raise ValueError(
                "The data is not compatible with the requested degree."
            )

        if self.neighbors is None:
            self._shift, self._scale, self._coeffs = RBFBlock.solve(
                self.y,
                self.d.reshape((self.y.shape[0], -1)),
                self.smoothing,
                self.kernel,
                self.epsilon,
                powers,
            )

        self.powers = powers

    def forward(self, x):
        """
        Forward pass.

        :param torch.Tensor x: Queries shaped ``(n_queries, n_dimensions)``.
        :raises ValueError: If the input is not a 2-dimensional tensor.
        :raises ValueError: If the second dimension of the input is not the same
            as the second dimension of the data.
        :return: Values of shape ``(n_queries, *d.shape[1:])``. Fit must have
            completed beforehand; no explicit fitted-state validation occurs.
        :rtype: torch.Tensor
        """
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional tensor.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(
                "Expected the second dim of `x` to have length "
                f"{self.y.shape[1]}."
            )

        kernel_func = radial_functions[self.kernel]

        yeps = self.y * self.epsilon
        xeps = x * self.epsilon
        xhat = (x - self._shift) / self._scale

        kv = RBFBlock.kernel_vector(xeps, yeps, kernel_func)
        p = RBFBlock.polynomial_matrix(xhat, self.powers)
        vec = torch.cat([kv, p], dim=1)
        out = torch.matmul(vec, self._coeffs)
        out = out.reshape((nx,) + self.d.shape[1:])
        return out

    @staticmethod
    def kernel_vector(x, y, kernel_func):
        """
        Evaluate for all points ``x`` the radial functions with center ``y``.

        :param torch.Tensor x: The tensor of points.
        :param torch.Tensor y: The tensor of centres.
        :param Callable kernel_func: Map distance tensors to kernel values.
        :return: The radial function values.
        :rtype: torch.Tensor
        """
        return kernel_func(torch.cdist(x, y))

    @staticmethod
    def polynomial_matrix(x, powers):
        """
        Evaluate monomials of power ``powers`` at points ``x``.

        :param torch.Tensor x: The tensor of points.
        :param torch.Tensor powers: The tensor of powers for each monomial.
        :return: The monomial values.
        :rtype: torch.Tensor
        """
        x_ = torch.repeat_interleave(x, repeats=powers.shape[0], dim=0)
        powers_ = powers.repeat(x.shape[0], 1)
        return torch.prod(x_**powers_, dim=1).view(x.shape[0], powers.shape[0])

    @staticmethod
    def kernel_matrix(x, kernel_func):
        """
        Return the radial function values for all pairs of points in ``x``.

        :param torch.Tensor x: The tensor of points.
        :param Callable kernel_func: Map distance tensors to kernel values.
        :return: The radial function values.
        :rtype: torch.Tensor
        """
        return kernel_func(torch.cdist(x, x))

    @staticmethod
    def monomial_powers(ndim, degree):
        """
        Return the powers for each monomial in a polynomial.

        :param int ndim: The number of variables in the polynomial.
        :param int degree: The degree of the polynomial.
        :return: The powers for each monomial.
        :rtype: torch.Tensor
        """
        nmonos = math.comb(degree + ndim, ndim)
        out = torch.zeros((nmonos, ndim), dtype=torch.int32)
        count = 0
        for deg in range(degree + 1):
            for mono in combinations_with_replacement(range(ndim), deg):
                for var in mono:
                    out[count, var] += 1
                count += 1
        return out

    @staticmethod
    def build(y, d, smoothing, kernel, epsilon, powers):
        """
        Build the RBF linear system.

        :param torch.Tensor y: The tensor of data points.
        :param torch.Tensor d: The tensor of data values.
        :param torch.Tensor smoothing: The tensor of smoothing parameters.
        :param str kernel: The radial basis function to use.
        :param float epsilon: The shape parameter that scales the input to the
            RBF.
        :param torch.Tensor powers: The tensor of powers for each monomial.
        :return: The left-hand side and right-hand side of the linear system,
            and the shift and scale parameters.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        p = d.shape[0]
        s = d.shape[1]
        r = powers.shape[0]
        kernel_func = radial_functions[kernel]

        mins = torch.min(y, dim=0).values
        maxs = torch.max(y, dim=0).values
        shift = (maxs + mins) / 2
        scale = (maxs - mins) / 2

        scale[scale == 0.0] = 1.0

        yeps = y * epsilon
        yhat = (y - shift) / scale

        lhs = torch.empty((p + r, p + r), device=d.device).float()
        lhs[:p, :p] = RBFBlock.kernel_matrix(yeps, kernel_func)
        lhs[:p, p:] = RBFBlock.polynomial_matrix(yhat, powers)
        lhs[p:, :p] = lhs[:p, p:].T
        lhs[p:, p:] = 0.0
        lhs[:p, :p] += torch.diag(smoothing)

        rhs = torch.empty((r + p, s), device=d.device).float()
        rhs[:p] = d
        rhs[p:] = 0.0
        return lhs, rhs, shift, scale

    @staticmethod
    def solve(y, d, smoothing, kernel, epsilon, powers):
        """
        Build and solve the RBF linear system.

        :param torch.Tensor y: The tensor of data points.
        :param torch.Tensor d: The tensor of data values.
        :param torch.Tensor smoothing: The tensor of smoothing parameters.

        :param str kernel: The radial basis function to use.
        :param float epsilon: The shape parameter that scaled the input to the
            RBF.
        :param torch.Tensor powers: The tensor of powers for each monomial.
        :raises ValueError: If the linear system is singular.
        :return: The shift and scale parameters, and the coefficients of the
            interpolator.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """

        lhs, rhs, shift, scale = RBFBlock.build(
            y, d, smoothing, kernel, epsilon, powers
        )
        try:
            coeffs = torch.linalg.solve(lhs, rhs)
        except RuntimeError as e:
            msg = "Singular matrix."
            nmonos = powers.shape[0]
            if nmonos > 0:
                pmat = RBFBlock.polynomial_matrix((y - shift) / scale, powers)
                rank = torch.linalg.matrix_rank(pmat)
                if rank < nmonos:
                    msg = (
                        "Singular matrix. The matrix of monomials evaluated at "
                        "the data point coordinates does not have full column "
                        f"rank ({rank}/{nmonos})."
                    )

            raise ValueError(msg) from e

        return shift, scale, coeffs
