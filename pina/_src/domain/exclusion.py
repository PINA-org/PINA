"""Module for the Exclusion set-operation."""

import random
from pina._src.domain.base_operation import BaseOperation
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency


class Exclusion(BaseOperation):
    r"""
    Implementation of the exclusion operation defined on a list of domains.

    Given multiple sets :math:`A_1, A_2, \ldots, A_n`, define their exclusion
    as:

    .. math::

        \bigcup_{i=1}^{n} \big(A_i \setminus \bigcup_{j \neq i} A_j \big)

    In other words, the exclusion operation returns the set of points that
    belong to exactly one of the input sets.

    In case of two sets, the exclusion corresponds to the symmetric difference.

    No check is performed to ensure that the resulting domain is non-empty.

    :Example:

        >>> from pina.domain import CartesianDomain, Exclusion
        >>> cartesian1 = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
        >>> cartesian2 = CartesianDomain({'x': [0, 1], 'y': [0.5, 1.5]})
        >>> exclusion = Exclusion([cartesian1, cartesian2])
    """

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the exclusion of the domains.

        :param LabelTensor point: A single point with shape ``(1, n_variables)``
            and labels matching the constituent domains.
        :param bool check_border: The boundary flag passed to every constituent;
            each applies its own membership rules. Default is ``False``.
        :raises ValueError: If ``point`` is not a :class:`LabelTensor`.
        :raises ValueError: If the labels of ``point`` differ from the variables
            of the domain.
        :return: Whether exactly one constituent contains the point. The result
            can be a scalar boolean tensor when constituent tests use tensors.
        :rtype: bool | torch.Tensor
        """
        # Checks on point
        check_consistency(point, LabelTensor)
        if set(self.variables) != set(point.labels):
            raise ValueError(
                "Point labels differ from domain's dictionary labels. "
                f"Got {sorted(point.labels)}, expected {self.variables}."
            )

        # Check if the point belongs to any of the geometries
        inside_flags = [
            g.is_inside(point, check_border) for g in self.geometries
        ]

        return sum(inside_flags) == 1

    def sample(self, n, mode="random", variables="all"):
        """
        Reject proposed points unless exactly one constituent contains them.

        Quotas are nearly equal across geometries shuffled with Python's
        :mod:`random`. Proposals use full coordinates and default
        boundary-excluding membership checks. Requested columns are selected
        only after acceptance. Quotas are not volume-weighted.

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
        :raises IndexError: If the variable selection is empty.
        :return: Accepted points of shape ``(n, len(variables))`` after
            selection normalization. Repeated points are not removed.
        :rtype: LabelTensor

        .. warning::

            There is no iteration limit. Any geometry assigned a positive quota
            must be able to propose points in its exclusive region. Otherwise,
            sampling can loop indefinitely even if other geometries have
            nonempty exclusive regions.
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

            # Sampled points for the current geometry
            sampled_points = []

            # Sample until we have enough points
            while len(sampled_points) < alloc[idx]:

                # Sample a sufficiently large number of points
                batch_size = 2 * (alloc[idx] - len(sampled_points))
                pts = self.geometries[gi].sample(batch_size, mode)

                # Filter points inside the intersection
                for p in pts:
                    p = p.reshape(1, -1)
                    p.labels = pts.labels
                    if self.is_inside(p):
                        sampled_points.append(p[variables])
                        if len(sampled_points) >= alloc[idx]:
                            break

            # Sample points
            samples.append(LabelTensor.cat(sampled_points, dim=0))

        return LabelTensor.cat(samples, dim=0)

    def partial(self):
        """
        Return the boundary of the domain resulting from the operation.

        :raises NotImplementedError: The :meth:`partial` method is not
            implemented for exclusion domains. Please operate on the individual
            domains instead.
        """
        raise NotImplementedError(
            "The partial method is not implemented for exclusion domains. "
            "Please operate on the individual domains instead."
        )
