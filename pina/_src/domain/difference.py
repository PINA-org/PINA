"""Module for the Difference operation."""

from pina._src.domain.base_operation import BaseOperation
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency


class Difference(BaseOperation):
    r"""
    Implementation of the difference operation defined on a list of domains.

    Given two sets :math:`A` and :math:`B`, define their difference as:

    .. math::

        A \setminus B = \{x \mid x \in A \land x \not\in B\}

    For multiple sets :math:`A_1, A_2, \ldots, A_n`, define their difference as
    the set of points that belong to the first set but not to any of the
    remaining sets.

    No check is performed to ensure that the resulting domain is non-empty.

    :Example:

        >>> from pina.domain import CartesianDomain, Difference
        >>> cartesian1 = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
        >>> cartesian2 = CartesianDomain({'x': [0, 1], 'y': [0.5, 1.5]})
        >>> difference = Difference([cartesian1, cartesian2])
    """

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the difference of the domains.

        :param LabelTensor point: A single point with shape ``(1, n_variables)``
            and labels matching the constituent domains.
        :param bool check_border: The boundary flag passed to every constituent,
            including subtracted domains. Including their boundaries removes
            those boundaries from the difference. Default is ``False``.
        :raises ValueError: If ``point`` is not a :class:`LabelTensor`.
        :raises ValueError: If the labels of ``point`` differ from the variables
            of the domain.
        :raises IndexError: If there are no constituent geometries.
        :return: Whether the point belongs to the first geometry and none of
            the remaining geometries. A false scalar tensor from the first
            geometry can be returned unchanged.
        :rtype: bool | torch.Tensor
        """
        # Checks on point
        check_consistency(point, LabelTensor)
        if set(self.variables) != set(point.labels):
            raise ValueError(
                "Point labels differ from domain's dictionary labels. "
                f"Got {sorted(point.labels)}, expected {self.variables}."
            )

        # Check if the point is inside the first geometry and not in any other
        inside_first = self.geometries[0].is_inside(point, check_border)
        inside_others = any(
            g.is_inside(point, check_border) for g in self.geometries[1:]
        )

        return inside_first and not inside_others

    def sample(self, n, mode="random", variables="all"):
        """
        Sample the first geometry and reject points in any later geometry.

        Proposals use full coordinates and default boundary-excluding membership
        checks. Requested columns are selected only after acceptance.

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

            There is no iteration limit. A small remaining region can be slow;
            an empty difference, or a sampler unable to propose accepted points,
            can make this method loop indefinitely.
        """
        # Validate sampling settings
        variables = self._validate_sampling(n, mode, variables)

        # Allocate list for samples
        samples = []

        # Sample until we have enough points
        while len(samples) < n:

            # Sample a sufficiently large number of points
            batch_size = 2 * (n - len(samples))
            pts = self.geometries[0].sample(batch_size, mode)

            # Filter points inside the intersection
            for p in pts:
                p = p.reshape(1, -1)
                p.labels = pts.labels
                if self.is_inside(p):
                    samples.append(p[variables])
                    if len(samples) >= n:
                        break

        return LabelTensor.cat(samples, dim=0)

    def partial(self):
        """
        Return the boundary of the domain resulting from the operation.

        :raises NotImplementedError: The :meth:`partial` method is not
            implemented for difference domains. Please operate on the individual
            domains instead.
        """
        raise NotImplementedError(
            "The partial method is not implemented for difference domains. "
            "Please operate on the individual domains instead."
        )
