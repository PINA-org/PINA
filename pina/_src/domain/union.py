"""Module for the Union operation."""

import random
from pina._src.domain.base_operation import BaseOperation
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency


class Union(BaseOperation):
    r"""
    Implementation of the union operation defined on a list of domains.

    Given multiple sets :math:`A_1, A_2, \ldots, A_n`, define their union as:

    .. math::

        \bigcup_{i=1}^{n} A_i = \{x \mid \exists i: x \in A_i \}

    :Example:

        >>> from pina.domain import CartesianDomain, Union
        >>> cartesian1 = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
        >>> cartesian2 = CartesianDomain({'x': [0, 1], 'y': [1, 2]})
        >>> union = Union([cartesian1, cartesian2])
    """

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the union of the domains.

        :param LabelTensor point: A single point with shape ``(1, n_variables)``
            and labels matching the constituent domains.
        :param bool check_border: The boundary flag passed to every constituent;
            each applies its own membership rules. Default is ``False``.
        :raises ValueError: If ``point`` is not a :class:`LabelTensor`.
        :raises ValueError: If the labels of ``point`` differ from the variables
            of the domain.
        :return: Whether the point is inside the domain or not.
        :rtype: bool
        """
        # Checks on point
        check_consistency(point, LabelTensor)
        if set(self.variables) != set(point.labels):
            raise ValueError(
                "Point labels differ from domain's dictionary labels. "
                f"Got {sorted(point.labels)}, expected {self.variables}."
            )

        return any(g.is_inside(point, check_border) for g in self.geometries)

    def sample(self, n, mode="random", variables="all"):
        """
        Concatenate nearly equal sample allocations from the constituents.

        Python's :mod:`random` shuffles geometry order and assigns any remainder
        to the first geometries in that order. Allocation is not weighted by
        volume, and points in overlapping regions are not deduplicated.

        :param int n: The number of samples to generate.
        :param str mode: A method supported by every constituent. Available
            modes are listed in ``sample_modes``. Default is ``"random"``.
        :param variables: The nonempty coordinate selection, or ``"all"`` for
            every variable. Duplicate names are removed and columns sorted.
            Default is ``"all"``.
        :type variables: str | list[str] | tuple[str, ...] | set[str]
        :raises AssertionError: If ``n`` is not a positive integer.
        :raises ValueError: If the sampling mode is invalid.
        :raises ValueError: If ``variables`` is neither ``all``, a string, nor a
            list, tuple, or set of strings.
        :raises ValueError: If any of the specified variables is unknown.
        :raises TypeError: If there are no constituent geometries.
        :return: Concatenated samples with one column per selected variable.
            Random modes return ``n`` rows. Grid and Chebyshev child samplers
            can expand their allocations, so the row count may exceed ``n``.
        :rtype: LabelTensor
        """
        # Validate sampling settings
        variables = self._validate_sampling(n, mode, variables)

        # Compute number of points per geometry and remainder
        num_pts, remainder = divmod(n, len(self.geometries))

        # Shuffle indices
        shuffled_geometries = random.sample(
            range(len(self.geometries)), len(self.geometries)
        )

        # Precompute per-geometry allocations following the shuffled order
        alloc = [num_pts + (i < remainder) for i in range(len(self.geometries))]
        samples = []

        # Iterate over geometries in shuffled order
        for idx, gi in enumerate(shuffled_geometries):

            # If no points to allocate (possible if len(self.geometries) > n)
            if alloc[idx] == 0:
                continue

            # Sample points
            pts = self.geometries[gi].sample(alloc[idx], mode, variables)
            samples.append(pts)

        return LabelTensor.cat(samples, dim=0)

    def partial(self):
        """
        Return the boundary of the domain resulting from the operation.

        :raises NotImplementedError: The :meth:`partial` method is not
            implemented for union domains. Please operate on the individual
            domains instead.
        """
        raise NotImplementedError(
            "The partial method is not implemented for union domains. "
            "Please operate on the individual domains instead."
        )
