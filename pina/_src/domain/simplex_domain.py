"""Module for the Simplex Domain."""

from copy import deepcopy
import torch
from pina._src.domain.base_domain import BaseDomain
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency


class SimplexDomain(BaseDomain):
    """
    Implementation of the simplex domain.

    :Example:

        >>> import torch
        >>> from pina import LabelTensor
        >>> from pina.domain import SimplexDomain
        >>> vertices = [
        ...     LabelTensor(torch.tensor([v]), ["x", "y"])
        ...     for v in [[0., 0.], [1., 1.], [0., 1.]]
        ... ]
        >>> simplex_domain = SimplexDomain(vertices)
    """

    def __init__(self, simplex_matrix, sample_surface=False):
        """
        Initialization of the :class:`SimplexDomain` class.

        :param simplex_matrix: The ``d + 1`` vertices, each of shape ``(1, d)``
            with matching ordered coordinate labels. Vertices are stacked and
            converted to float32. Row counts and affine independence are not
            explicitly validated.
        :type simplex_matrix: list[LabelTensor] | tuple[LabelTensor, ...]
        :param bool sample_surface: Whether to sample only boundary faces.
            Does not change membership testing. Default is ``False``.
        :raises ValueError: If any element of ``simplex_matrix`` is not a
            :class:`LabelTensor`.
        :raises TypeError: If ``simplex_matrix`` is not a list or tuple.
        :raises ValueError: If ``sample_surface`` is not a boolean.
        :raises ValueError: If the labels of the vertices do not match.
        :raises ValueError: If the number of vertices is not equal to the
            dimension of the simplex plus one.
        :raises IndexError: If the vertex sequence is empty.
        """
        super().__init__()

        # Initialization
        self._sample_modes = ("random",)
        self.sample_surface = sample_surface
        self.vert_matrix = simplex_matrix

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the simplex.

        :param LabelTensor point: A single point of shape ``(1, d)`` with labels
            matching the domain variables and a dtype compatible with the
            float32 vertex matrix.
        :param bool check_border: If ``True``, the boundary is considered inside
            the domain, with barycentric bounds extended by ``1e-6``. Otherwise
            every coordinate must lie between ``1e-6`` and ``1 - 1e-6``.
            Default is ``False``.
        :raises ValueError: If ``point`` is not a :class:`LabelTensor`.
        :raises ValueError: If the labels of ``point`` differ from the variables
            of the domain.
        :raises RuntimeError: If the vertex system is singular, or the point's
            shape or dtype is incompatible with the linear solve.
        :return: A scalar boolean tensor indicating membership in the volume,
            regardless of ``sample_surface``.
        :rtype: torch.Tensor
        """
        # Checks on point
        check_consistency(point, LabelTensor)
        if set(self.variables) != set(point.labels):
            raise ValueError(
                "Point labels differ from constructor vertices labels. "
                f"Got {sorted(point.labels)}, expected {self.variables}."
            )

        # Shift the point by the last vertex
        shift_point = point[self.variables] - self._vert_matrix[-1]
        shift_point = shift_point.tensor.reshape(-1, 1)

        # Shift the vertices by the last vertex
        shift_vert = (self._vert_matrix[:-1] - self._vert_matrix[-1]).T

        # Compute barycentric coordinates
        coords = torch.linalg.solve(shift_vert, shift_point)
        last_coord = 1.0 - torch.sum(coords)
        coords = torch.vstack([coords, last_coord])

        # If check_border is False -- use tolerance for numerical errors
        if not check_border:
            return torch.all(coords > 1e-6) & torch.all(coords < 1 - 1e-6)

        return torch.all(coords >= -1e-6) & torch.all(coords <= 1 + 1e-6)

    def update(self, domain):
        """
        Update the current domain by substituting the simplex vertices with
        those contained in ``domain``. Only domains of the same type can be used
        for update.

        :param SimplexDomain domain: The domain whose vertices are to be set
            into the current one.
        :raises TypeError: If the domain is not a :class:`SimplexDomain` object.
        :return: A deep copy of the receiver with replacement vertices and the
            receiver's surface-sampling mode preserved. Inputs are unchanged.
        :rtype: SimplexDomain
        """
        # Raise an error if the domain types do not match
        if not isinstance(domain, type(self)):
            raise TypeError(
                f"Cannot update domain of type {type(self)} "
                f"with domain of type {type(domain)}."
            )

        # Compute new vertex matrix
        vert_matrix = []
        for v in domain.vert_matrix:
            vert = v.reshape(1, -1)
            vert.labels = domain.variables
            vert_matrix.append(vert)

        # Replace geometry
        updated = deepcopy(self)
        updated.vert_matrix = vert_matrix

        return updated

    def sample(self, n, mode="random", variables="all"):
        """
        Sampling routine.

        :param int n: The number of samples to generate.
        :param str mode: The sampling method. Available modes: ``random`` for
            random sampling. Default is ``random``.
        :param variables: The nonempty coordinate selection, or ``"all"`` for
            every variable. Duplicates are removed and columns sorted by name.
            Selected coordinates are projections of full-simplex samples.
            Default is ``"all"``.
        :type variables: str | list[str] | tuple[str, ...] | set[str]
        :raises AssertionError: If ``n`` is not a positive integer.
        :raises ValueError: If the sampling mode is invalid.
        :raises ValueError: If ``variables`` is neither ``all``, a string, nor a
            list, tuple, or set of strings.
        :raises ValueError: If any of the specified variables is unknown.
        :raises IndexError: If the variable selection is empty.
        :return: Points of shape ``(n, len(variables))`` after selection
            normalization, using normalized exponential barycentric weights.
            Surface mode selects faces uniformly by index, not by face area.
        :rtype: LabelTensor

        :Example:

            >>> import torch
            >>> from pina import LabelTensor
            >>> from pina.domain import SimplexDomain
            >>> vertices = [
            ...     LabelTensor(torch.tensor([v]), ["x", "y"])
            ...     for v in [[0., 0.], [1., 1.], [0., 1.]]
            ... ]
            >>> points = SimplexDomain(vertices).sample(n=5)
            >>> tuple(points.shape), points.labels
            ((5, 2), ['x', 'y'])
        """
        # Validate sampling settings
        variables = self._validate_sampling(n, mode, variables)

        # Extract vertex matrix for the requested variables
        vert_matrix = self._vert_matrix[variables].tensor

        # Sample barycentric coordinates using the Dirichlet distribution over
        # the simplex. This can be efficiently done by using samples obtained
        # via: -log(U(0,1)) ~ Exp(1) ~ Gamma(1, 1) ~ Dirichlet(1, ..., 1).
        coords = -torch.rand((n, vert_matrix.shape[0])).clamp_min(1e-12).log()

        # If only the surface is to be sampled
        if self._sample_surface:

            # Pick one face of the simplex at random for each point and set the
            # corresponding barycentric coordinate to zero.
            face_idx = torch.randint(0, vert_matrix.shape[0], (n,))
            coords.scatter_(1, face_idx.view(-1, 1), 0.0)

        # Normalize the coords
        coords = coords / coords.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # Prepare output
        pts = (coords @ vert_matrix).as_subclass(LabelTensor)
        pts.labels = variables

        return pts[sorted(pts.labels)]

    def partial(self):
        """
        Return the boundary of the domain as a new domain object.

        :return: A deep copy with surface sampling enabled. Membership tests
            still use the volume and the caller's boundary flag.
        :rtype: SimplexDomain
        """
        boundary = deepcopy(self)
        boundary.sample_surface = True

        return boundary

    @property
    def variables(self):
        """
        The list of variables of the domain.

        :return: The list of variables of the domain.
        :rtype: list[str]
        """
        return sorted(self._vert_matrix.labels)

    @property
    def domain_dict(self):
        """
        The dictionary representing the domain. For the simplex domain, the keys
        are of the form ``"v0"``, ``"v1"``, ..., ``"vn"``, each identifying a
        vertex of the simplex.

        :return: A new mapping to vertex rows, with shape ``(d,)`` per value.
        :rtype: dict[str, LabelTensor]
        """
        return {
            f"v{i}": self._vert_matrix[i]
            for i in range(self._vert_matrix.shape[0])
        }

    @property
    def range(self):
        """
        Return an empty dictionary since the simplex domain does not have range
        variables. Implemented to comply with the :class:`BaseDomain` interface.

        :return: A new empty mapping; vertices define the geometry instead.
        :rtype: dict[str, tuple[int | float, int | float]]
        """
        return {}

    @property
    def fixed(self):
        """
        Return an empty dictionary since the simplex domain does not have fixed
        variables. Implemented to comply with the :class:`BaseDomain` interface.

        :return: A new empty mapping; vertices define the geometry instead.
        :rtype: dict[str, int | float]
        """
        return {}

    @property
    def sample_surface(self):
        """
        Whether sampling is restricted to simplex boundary faces.

        :return: Whether only boundary faces are sampled. Membership testing
            ignores this setting.
        :rtype: bool
        """
        return self._sample_surface

    @sample_surface.setter
    def sample_surface(self, value):
        """
        Setter for the sample_surface property.

        :param bool value: Whether to restrict sampling to boundary faces.
            Membership testing is unaffected.
        :raises ValueError: If ``value`` is not a boolean.
        """
        check_consistency(value, bool)
        self._sample_surface = value

    @property
    def vert_matrix(self):
        """
        The vertex matrix of the simplex.

        :return: The stored float32 matrix of shape ``(d + 1, d)``. Mutating it
            directly changes the geometry without setter validation.
        :rtype: LabelTensor
        """
        return self._vert_matrix

    @vert_matrix.setter
    def vert_matrix(self, value):
        """
        Setter for the vertex matrix.

        :param value: The ``d + 1`` vertices, each of shape ``(1, d)``, with
            matching ordered labels. Vertices are stacked and cast to float32;
            row counts and affine independence are not explicitly validated.
        :type value: list[LabelTensor] | tuple[LabelTensor, ...]
        :raises ValueError: If any element of ``value`` is not a
            :class:`LabelTensor`.
        :raises TypeError: If ``value`` is not a list or tuple.
        :raises ValueError: If the labels of the vertices do not match.
        :raises ValueError: If the number of vertices is not equal to the
            dimension of the simplex plus one.
        :raises IndexError: If the vertex sequence is empty.
        """
        # Check consistency
        check_consistency(value, LabelTensor)
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                "The simplex matrix must be a list or tuple of LabelTensor."
            )

        # Check that all labels match
        matrix_labels = value[0].labels
        if not all(vert.labels == matrix_labels for vert in value):
            raise ValueError("Labels of all vertices must match.")

        # Check dimensionality
        if len(value) != len(matrix_labels) + 1:
            raise ValueError(
                "An n-dimensional simplex needs n+1 vertices in R^n."
            )

        self._vert_matrix = LabelTensor.vstack(value).to(torch.float32)
