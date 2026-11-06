"""Module for the Ellipsoid Domain."""

from copy import deepcopy
import torch
from pina._src.domain.base_domain import BaseDomain
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency


class EllipsoidDomain(BaseDomain):
    """
    Implementation of the ellipsoid domain.

    .. seealso::

        **Original reference**: Dezert, Jean, and Musso, Christian.
        *An efficient method for generating points uniformly distributed
        in hyperellipsoids.*
        Proceedings of the Workshop on Estimation, Tracking and Fusion:
        A Tribute to Yaakov Bar-Shalom. 2001.

    :Example:

        >>> from pina.domain import EllipsoidDomain
        >>> ellipsoid_domain = EllipsoidDomain({'x':[-1, 1], 'y':[-1, 1]})
        >>> ellipsoid_domain = EllipsoidDomain({'x':[-1, 1], 'y':1.0})
    """

    def __init__(self, ellipsoid_dict, sample_surface=False):
        """
        Initialization of the :class:`EllipsoidDomain` class.

        :param ellipsoid_dict: A dictionary where the keys are the variable
            names and the values are the domain extrema. The domain extrema can
            be either a list or tuple with two elements or a single number. If
            the domain extrema is a single number, the variable is fixed to that
            value. ``None`` initializes a domain without variables. Floating
            fixed values are recommended for gradient-compatible sampling.
        :type ellipsoid_dict: dict[str, int | float | list[int | float] |
            tuple[int | float, int | float]] | None
        :param bool sample_surface: If ``True``, only the surface of the
            ellipsoid is considered part of the domain. Default is ``False``.
        :raises ValueError: If ``sample_surface`` is not a boolean.
        :raises TypeError: If the ellipsoid dictionary is not a dictionary.
        :raises ValueError: If the ellipsoid dictionary is empty.
        :raises ValueError: If the ellipsoid dictionary contains variables with
            invalid ranges.
        :raises ValueError: If the ellipsoid dictionary contains values that are
            neither numbers nor lists/tuples of numbers of length 2.
        """
        # Initialization
        super().__init__(variables_dict=ellipsoid_dict)
        self.sample_surface = sample_surface
        self._sample_modes = ("random",)
        self.compute_center_axes()

    def compute_center_axes(self):
        """
        Recompute stored centers and semiaxes from the current ranged bounds.

        Both are one-dimensional float32 labeled tensors in sorted variable
        order. With no ranged variables, both attributes are set to ``None``.
        """
        if self._range:
            rng_vars = sorted(self._range.keys())
            vals = torch.tensor(
                [self._range[k] for k in rng_vars], dtype=torch.float
            )
            self._centers = LabelTensor(vals.mean(dim=1), rng_vars)
            self._axes = LabelTensor(
                (vals - self._centers.unsqueeze(1))[:, -1],
                rng_vars,
            )
        else:
            self._centers = None
            self._axes = None

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the ellipsoid.

        :param LabelTensor point: A single point of shape ``(1, n_variables)``
            with labels matching the domain variables. Multiple rows are not
            checked independently: normalized squared distances are summed.
        :param bool check_border: If ``True``, the boundary is considered inside
            the volume domain. Ignored in surface-only mode, which checks the
            ellipsoid equation using :func:`torch.allclose`. Default is
            ``False``.
        :raises ValueError: If ``point`` is not a :class:`LabelTensor`.
        :raises ValueError: If the labels of ``point`` differ from the variables
            of the domain.
        :return: Whether the point satisfies the ellipsoid equation and matches
            all fixed coordinates exactly.
        :rtype: bool
        """
        # Checks on point
        check_consistency(point, LabelTensor)
        if set(self.variables) != set(point.labels):
            raise ValueError(
                "Point labels differ from constructor dictionary labels. "
                f"Got {sorted(point.labels)}, expected {self.variables}."
            )

        # Fixed variable checks
        fixed_check = all(
            (point.extract([k]) == v).all() for k, v in self._fixed.items()
        )

        # If there are no range variables, return fixed variable check
        if not self._range:
            return fixed_check

        # Compute the equation defining the ellipsoid
        rng = sorted(self._range.keys())
        squared_axis = self._axes[rng].pow(2)
        delta = (point[rng] - self._centers[rng]).pow(2)
        eqn = torch.sum(delta / squared_axis) - 1.0

        # Range variable check on the surface
        if self._sample_surface:
            range_check = torch.allclose(eqn, torch.zeros_like(eqn))
            return fixed_check and range_check

        # Range variable check in the volume
        range_check = (eqn <= 0) if check_border else (eqn < 0)

        return fixed_check and range_check.item()

    def update(self, domain):
        """
        Merge variable definitions into a deep copy and recompute its axes.

        Matching definitions are overwritten, including fixed/ranged category
        changes. The receiver's surface mode is preserved; inputs are unchanged.

        :param EllipsoidDomain domain: The domain whose labels are to be merged
            into the current one.
        :raises TypeError: If the provided domain is not of an instance of
            :class:`EllipsoidDomain`.
        :return: A copy of the receiver with merged bounds, centers, and axes.
        :rtype: EllipsoidDomain
        """
        updated = super().update(domain)
        updated.compute_center_axes()

        return updated

    def sample(self, n, mode="random", variables="all"):
        """
        Sampling routine.

        :param int n: The number of samples to generate.
        :param str mode: The sampling method. Available modes: ``random`` for
            random sampling. Default is ``random``.
        :param variables: The nonempty coordinate selection, or ``"all"`` for
            every variable. Duplicates are removed and columns sorted by name.
            Selecting fewer ranged axes samples an ellipsoid in those axes,
            rather than projecting samples from the full ellipsoid.
            Default is ``"all"``.
        :type variables: str | list[str] | tuple[str, ...] | set[str]
        :raises AssertionError: If ``n`` is not a positive integer.
        :raises ValueError: If the sampling mode is invalid.
        :raises ValueError: If ``variables`` is neither ``all``, a string, nor a
            list, tuple, or set of strings.
        :raises ValueError: If any of the specified variables is unknown.
        :raises ValueError: If the variable selection is empty.
        :return: Points of shape ``(n, len(variables))`` after selection
            normalization, with fixed coordinates appended as needed. Volume
            sampling scales uniform sphere directions by a random radius;
            surface sampling applies the axes to unit sphere directions and
            is not generally uniform in ellipsoid surface area. Fixed-only
            integer values can produce an integer tensor.
        :rtype: LabelTensor

        :Example:

            >>> from pina.domain import EllipsoidDomain
            >>> domain = EllipsoidDomain({'x': [0, 1], 'y': [0, 1]})
            >>> points = domain.sample(n=5)
            >>> tuple(points.shape), points.labels
            ((5, 2), ['x', 'y'])
        """
        # Validate sampling settings
        variables = self._validate_sampling(n, mode, variables)

        # Separate range and fixed variables
        range_vars = [v for v in variables if v in self._range]
        fixed_vars = [v for v in variables if v in self._fixed]

        # If there are no range variables, return fixed variables only
        if not range_vars:
            vals = [torch.full((n, 1), self._fixed[v]) for v in fixed_vars]
            result = torch.cat(vals, dim=1)
            result = result.as_subclass(LabelTensor)
            result.labels = fixed_vars
            return result

        # Sample points
        pts = self._sample_range(n, range_vars)
        labels = range_vars

        # Add fixed vars
        if fixed_vars:
            fixed_vals = [
                torch.full((pts.shape[0], 1), self._fixed[v])
                for v in fixed_vars
            ]
            pts = torch.cat([pts] + fixed_vals, dim=1)
            labels = range_vars + fixed_vars

        # Prepare output
        pts = pts.as_subclass(LabelTensor)
        pts.labels = labels

        return pts[sorted(pts.labels)]

    def _sample_range(self, n, variables):
        """
        Sample points and rescale to fit within the specified bounds.

        :param int n: The number of points to sample.
        :param list[str] variables: The nonempty selected ranged axes, in output
            order. Names must exist in the stored centers and semiaxes.
        :raises ValueError: If a selected axis is unknown.
        :return: Points of shape ``(n, len(variables))`` in the ellipsoid
            defined by the selected axes, or on its surface when enabled.
        :rtype: torch.Tensor | LabelTensor
        """
        # Extract the dimension
        dim = len(variables)

        # Extract centers and axes of the variables to sample
        centers = self._centers[variables]
        axes = self._axes[variables]

        # Find random directions on the unit sphere
        pts = torch.randn(size=(n, dim))
        norm = torch.linalg.vector_norm(pts, dim=1, keepdim=True)
        direction = pts / norm.clamp_min(1e-12)

        # Radius is set to one if sampling on the surface
        if self._sample_surface:
            radius = torch.ones((n, 1))

        # Otherwise, scale radius to lie within the sphere. Important: exponent
        # 1/dim is used to avoid shrinkage of the ellipsoid in higher dims.
        else:
            radius = torch.rand((n, 1)).pow(1.0 / dim)

        # Rescale the points to lie within the ellipsoid
        pts = direction * radius * axes + centers

        return pts

    def partial(self):
        """
        Return the boundary of the domain as a new domain object.

        :return: The boundary of the domain.
        :rtype: EllipsoidDomain
        """
        boundary = deepcopy(self)
        boundary.sample_surface = True

        return boundary

    @property
    def sample_surface(self):
        """
        Whether only the surface of the ellipsoid is considered part of the
        domain.

        :return: ``True`` if only the surface is considered part of the domain,
            ``False`` otherwise.
        :rtype: bool
        """
        return self._sample_surface

    @sample_surface.setter
    def sample_surface(self, value):
        """
        Setter for the sample_surface property.

        :param bool value: The new value for the sample_surface property.
        :raises ValueError: If ``value`` is not a boolean.
        """
        check_consistency(value, bool)
        self._sample_surface = value
