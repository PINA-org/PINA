"""Module for utility functions for the convolutional layer."""

import torch


def check_point(x, current_stride, dim):
    """
    Check if the point is in the current stride.

    :param torch.Tensor x: Points of shape ``(..., d + 1)`` with field values
        in the last column, which is ignored.
    :param torch.Tensor current_stride: Filter origin of shape ``(d,)``.
    :param torch.Tensor dim: Filter extents of shape ``(d,)``.
    :return: Boolean mask of shape ``x.shape[:-1]``, including lower bounds
        and excluding upper bounds.
    :rtype: torch.Tensor
    """
    max_stride = current_stride + dim
    indeces = torch.logical_and(
        x[..., :-1] < max_stride, x[..., :-1] >= current_stride
    ).all(dim=-1)
    return indeces


def map_points_(x, filter_position):
    """
    The mapping function for n-dimensional case.

    :param torch.Tensor x: Coordinates of shape ``(..., d)``, modified in place.
    :param torch.Tensor filter_position: Broadcastable origin to subtract.
    :return: The data mapped in-place.
    :rtype: torch.Tensor
    """
    x.add_(-filter_position)

    return x


def optimizing(f):
    """
    Track initialization directions while calling the function every time.

    :param Callable f: Function called with the original arguments.
    :return: Wrapper with mutable ``has_run`` and ``has_run_inverse`` flags.
        Neither flag skips subsequent calls.
    :rtype: Callable
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function.

        :param tuple[Any, ...] args: Positional arguments forwarded unchanged.
        :param dict[str, Any] kwargs: Forwarded keywords; ``type_`` must be
            present. ``"forward"`` sets ``has_run_inverse`` and ``"inverse"``
            sets ``has_run``; other values just call the function.
        :raises KeyError: If ``type_`` is absent.
        :return: The wrapped function's result.
        :rtype: Any
        """
        if kwargs["type_"] == "forward":
            if not wrapper.has_run_inverse:
                wrapper.has_run_inverse = True
                return f(*args, **kwargs)

        if kwargs["type_"] == "inverse":
            if not wrapper.has_run:
                wrapper.has_run = True
                return f(*args, **kwargs)

        return f(*args, **kwargs)

    wrapper.has_run_inverse = False
    wrapper.has_run = False

    return wrapper
