"""Module for the Stride class."""

import torch


class Stride:
    """
    Stride class for continuous convolution.

    :Example:

        >>> from pina._src.model.block.stride import Stride
        >>> stride_dict = {
        ...     "domain": [4, 4],
        ...     "start": [-4, 2],
        ...     "jump": [2, 2],
        ...     "direction": [1, 1],
        ... }
        >>> stride = Stride(stride_dict)
    """

    def __init__(self, dict_):
        """
        Initialization of the :class:`Stride` class.

        :param dict[str, list[int | float]] dict_: Ordered mapping of domain
            extents, starting coordinates, jumps, and directions. Values are
            unpacked in insertion order, regardless of key spelling.
            Conventionally the keys are the domain size ``domain``,
            the starting position of the filter ``start``, the jump size for the
            filter ``jump``, and the direction of the filter ``direction``.
            Zero jumps and corresponding zero directions are replaced in place
            by the domain extent and one, respectively.
        :raises IndexError: If lengths or zero jump/direction positions differ.
        :raises ValueError: If a domain extent is negative, a direction is not
            ``-1``, ``0``, or ``1``, or there are not exactly four values.
        """

        self._dict_stride = dict_
        self._stride_continuous = None
        self._stride_discrete = self._create_stride_discrete(dict_)

    def _create_stride_discrete(self, my_dict):
        """
        Create a tensor of positions where to apply the filter.

        :param dict[str, list[int | float]] my_dict: Dictionary having as keys
            the domain size
            ``domain``, the starting position of the filter ``start``, the jump
            size for the filter ``jump``, and the direction of the filter
            ``direction``.
        :raises IndexError: Values in the dict must have all same length.
        :raises ValueError: If a domain extent is negative.
        :raises ValueError: Direction must be either equal to ``1``, ``-1`` or
            ``0``.
        :raises IndexError: Direction and jumps must be zero in the same index.
        :return: The positions for the filter
        :rtype: torch.Tensor

        :Example:

            >>> from pina._src.model.block.stride import Stride
            >>> stride_dict = {
            ...     "domain": [4, 4],
            ...     "start": [-4, 2],
            ...     "jump": [2, 2],
            ...     "direction": [1, 1],
            ... }
            >>> stride = Stride(stride_dict)
        """
        # we must check boundaries of the input as well
        domain, start, jumps, direction = my_dict.values()

        # checking
        if not all(len(s) == len(domain) for s in my_dict.values()):
            raise IndexError("Values in the dict must have all same length")

        if not all(v >= 0 for v in domain):
            raise ValueError("Domain values must be greater than 0")

        if not all(v in (0, -1, 1) for v in direction):
            raise ValueError("Direction must be either equal to 1, -1 or 0")

        seq_jumps = [i for i, e in enumerate(jumps) if e == 0]
        seq_direction = [i for i, e in enumerate(direction) if e == 0]

        if seq_direction != seq_jumps:
            raise IndexError(
                "Direction and jumps must have zero in the same index"
            )

        if seq_jumps:
            for i in seq_jumps:
                jumps[i] = domain[i]
                direction[i] = 1

        # creating the stride grid
        values_mesh = [
            torch.arange(0, i, step).float() for i, step in zip(domain, jumps)
        ]

        values_mesh = [
            single * dim for single, dim in zip(values_mesh, direction)
        ]

        mesh = torch.meshgrid(values_mesh)
        coordinates_mesh = [x.reshape(-1, 1) for x in mesh]

        stride = torch.cat(coordinates_mesh, dim=1) + torch.tensor(start)

        return stride
