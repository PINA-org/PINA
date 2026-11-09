"""Module for the Cartesian Domain."""

import torch
from pina._src.domain.base_domain import BaseDomain
from pina._src.domain.union import Union
from pina._src.core.utils import torch_lhs, chebyshev_roots, check_consistency
from pina._src.core.label_tensor import LabelTensor


class CartesianDomain(BaseDomain):
    """
    Implementation of the hypercube domain, obtained as the cartesian product of
    one-dimensional intervals.

    :Example:

        >>> from pina.domain import CartesianDomain
        >>> cartesian_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
        >>> cartesian_domain = CartesianDomain({'x': [0, 1], 'y': 1.0})
    """

    def __init__(self, cartesian_dict):
        """
        Initialization of the :class:`CartesianDomain` class.

        :param cartesian_dict: A dictionary where the keys are the variable
            names and the values are the domain extrema. The domain extrema can
            be either a list or tuple with two elements or a single number. If
            the domain extrema is a single number, the variable is fixed to that
            value. ``None`` initializes a domain without variables. Floating
            fixed values are recommended for gradient-compatible sampling.
        :type cartesian_dict: dict[str, int | float | list[int | float] |
            tuple[int | float, int | float]] | None
        :raises TypeError: If the cartesian dictionary is not a dictionary.
        :raises ValueError: If the cartesian dictionary is empty.
        :raises ValueError: If the cartesian dictionary contains variables with
            invalid ranges.
        :raises ValueError: If the cartesian dictionary contains values that are
            neither numbers nor lists/tuples of numbers of length 2.
        """
        # Initialization
        super().__init__(variables_dict=cartesian_dict)
        self._sample_modes = ("random", "grid", "chebyshev", "lh", "latin")

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the domain.

        :param LabelTensor point: One or more points with shape
            ``(..., n_variables)`` and labels matching the domain variables.
        :param bool check_border: If ``True``, the boundary is considered inside
            the domain. Default is ``False``.
        :raises ValueError: If ``point`` is not a :class:`LabelTensor`.
        :raises ValueError: If the labels of ``point`` differ from the variables
            of the domain.
        :return: Whether every supplied point satisfies all fixed values exactly
            and all interval bounds with the requested boundary inclusion.
        :rtype: bool
        """
        # Checks on point
        check_consistency(point, LabelTensor)
        if set(self.variables) != set(point.labels):
            raise ValueError(
                "Point labels differ from domain's dictionary labels. "
                f"Got {sorted(point.labels)}, expected {self.variables}."
            )

        # Fixed variable checks
        fixed_check = all(
            (point.extract([k]) == v).all() for k, v in self._fixed.items()
        )

        # If there are no range variables, return fixed variable check
        if not self._range:
            return fixed_check

        # Ranged variable checks -- check_border True
        if check_border:
            range_check = all(
                (
                    (point.extract([k]) >= low) & (point.extract([k]) <= high)
                ).all()
                for k, (low, high) in self._range.items()
            )

        # Ranged variable checks -- check_border False
        else:
            range_check = all(
                ((point.extract([k]) > low) & (point.extract([k]) < high)).all()
                for k, (low, high) in self._range.items()
            )

        return fixed_check and range_check

    def sample(self, n, mode="random", variables="all"):
        """
        The sampling routine.

        :param int n: The positive count for random modes, or resolution per
            selected ranged axis for grid and Chebyshev modes.
        :param str mode: The sampling method. Available modes: ``random`` for
            random sampling; ``latin`` or ``lh`` for latin hypercube sampling;
            ``chebyshev`` for chebyshev sampling; ``grid`` for grid sampling.
            Default is ``random``.
        :param variables: The nonempty coordinate selection, or ``"all"`` for
            every variable. Duplicates are removed and columns sorted by name.
            Default is ``"all"``.
        :type variables: str | list[str] | tuple[str, ...] | set[str]
        :raises AssertionError: If ``n`` is not a positive integer.
        :raises ValueError: If the sampling mode is invalid.
        :raises ValueError: If ``variables`` is neither ``all``, a string, nor a
            list, tuple, or set of strings.
        :raises ValueError: If any of the specified variables is unknown.
        :raises ValueError: If the variable selection is empty.
        :return: The sampled points with shape ``(count, len(variables))`` after
            selection normalization. Random modes return ``n`` rows. Grid and
            Chebyshev modes return ``n ** r`` rows for ``r`` selected ranged
            variables, or ``n`` rows when only fixed variables are selected.
            Fixed-only integer values can produce an integer tensor.
        :rtype: LabelTensor

        .. note::
            When multiple variables are involved, the total number of sampled
            points may differ depending on the chosen ``mode``.
            If ``mode`` is ``grid`` or ``chebyshev``, points are sampled
            independently for each variable and then combined, resulting in a
            total number of points equal to ``n`` raised to the power of the
            number of selected ranged variables. Fixed-only sampling returns
            ``n`` rows. If ``mode`` is ``random``, ``lh`` or ``latin``,
            all variables are sampled together, and the total number of points
            remains ``n``.

        .. warning::
            Grid sampling includes both interval endpoints when ``n > 1``.
            Random sampling can include lower endpoints, Latin hypercube
            sampling can include upper endpoints, and floating-point rounding
            can affect endpoint inclusion in other modes.

        :Example:

            >>> from pina.domain import CartesianDomain
            >>> domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
            >>> tuple(domain.sample(n=3).shape)
            (3, 2)
            >>> points = domain.sample(n=3, mode='grid')
            >>> tuple(points.shape), points.labels
            ((9, 2), ['x', 'y'])
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

        # Create a tensor of bounds for the range variables
        bounds = torch.as_tensor([self._range[v] for v in range_vars])

        # Sample for mode random or latin hypercube
        if mode in {"random", "lh", "latin"}:
            pts = self._sample_range(n, mode, bounds)

        # Sample for mode grid or chebyshev
        else:
            grids = [
                self._sample_range(
                    n, mode, torch.as_tensor([self._range[v]])
                ).reshape(-1)
                for v in range_vars
            ]
            pts = torch.cartesian_prod(*grids).reshape(-1, len(grids))

        # Add fixed vars
        if fixed_vars:
            fixed_vals = [
                torch.full((pts.shape[0], 1), self._fixed[v])
                for v in fixed_vars
            ]
            pts = torch.cat([pts] + fixed_vals, dim=1)
            labels = range_vars + fixed_vars
        else:
            labels = range_vars

        # Create the result as a LabelTensor
        pts = pts.as_subclass(LabelTensor)
        pts.labels = labels

        return pts[sorted(pts.labels)]

    def _sample_range(self, n, mode, bounds):
        """
        Sample points and rescale to fit within the specified bounds.

        :param int n: The number of points to sample.
        :param str mode: One of ``"random"``, ``"grid"``, ``"chebyshev"``,
            ``"lh"``, or ``"latin"``. No default is supplied by this helper.
        :param torch.Tensor bounds: Lower/upper bounds of shape ``(r, 2)``.
            Grid and Chebyshev calls normally supply one axis at a time.
        :raises KeyError: If ``mode`` is not a supported sampler name.
        :return: Rescaled coordinates of shape ``(n, r)``. Grid and Chebyshev
            coordinates are broadcast across bounds if multiple axes are given.
        :rtype: torch.Tensor
        """
        # Define a dictionary of sampling methods
        samplers = {
            "random": lambda: torch.rand(size=(n, bounds.shape[0])),
            "chebyshev": lambda: chebyshev_roots(n)
            .mul(0.5)
            .add(0.5)
            .reshape(-1, 1),
            "grid": lambda: torch.linspace(0, 1, n).reshape(-1, 1),
            "lh": lambda: torch_lhs(n, bounds.shape[0]),
            "latin": lambda: torch_lhs(n, bounds.shape[0]),
        }

        # Sample points in [0, 1]^d and rescale to the desired bounds
        pts = samplers[mode]()

        return pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    def partial(self):
        """
        Return the boundary of the domain as a :class:`Union` object.

        :return: Two faces per ranged variable, with that variable fixed to its
            lower or upper bound. Fixed-only domains produce an empty union.
        :rtype: Union
        """
        faces = []

        # Iterate over ranged variables
        for var, (low, high) in self._range.items():

            # Fix the variable to its low value to get the lower face
            lower = CartesianDomain({**self._fixed, **self._range, var: low})

            # Fix the variable to its high value to get the upper face
            higher = CartesianDomain({**self._fixed, **self._range, var: high})

            faces.extend([lower, higher])

        return Union(faces)
