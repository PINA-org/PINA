"""Module for utility functions."""

import types
from functools import reduce
import torch

from pina._src.core.label_tensor import LabelTensor


# Codacy error unused parameters
def custom_warning_format(
    message, category, filename, lineno, file=None, line=None
):
    """
    Format a warning using its filename, category name, and message.

    :param message: The warning message or warning instance.
    :type message: str | Warning
    :param type[Warning] category: The warning category class.
    :param str filename: The filename included in the formatted warning.
    :param int lineno: The source line number; unused.
    :param file: The optional output stream; unused. ``None`` does not select
        any output stream because this function only formats text. Default is
        ``None``.
    :type file: TextIO | None
    :param line: The optional source text; unused, including when ``None``.
        Default is ``None``.
    :type line: str | None
    :return: The string ``"filename: CategoryName: message\\n"``. No line number
        or source text is included, and no output is written here.
    :rtype: str
    """
    return f"{filename}: {category.__name__}: {message}\n"


def check_consistency(object_, object_instance, subclass=False):
    """
    Check if an object maintains inheritance consistency.

    This function checks whether a given object is an instance of a specified
    class or, if ``subclass=True``, whether it is a subclass of the specified
    class.

    Lists, sets, and tuples are checked element by element at one level. Other
    objects, including other iterables, are checked as single values. Empty
    supported collections pass without checking any elements. Instance mode
    rejects class objects even when the expected type would otherwise allow
    them.

    :param Any object_: The value, class, or supported collection to check.
    :param object_instance: The expected class, tuple of accepted classes, or
        PEP 604 union supported by Python's instance/subclass checks.
    :type object_instance: type | tuple[type | types.UnionType, ...] |
        types.UnionType
    :param bool subclass: If ``True``, check whether each value is a subclass
        of ``object_instance`` instead of an instance. Default is ``False``.
    :raises ValueError: If a checked value has the wrong instance/subclass type,
        an instance is supplied in subclass mode, or a class is supplied in
        instance mode.
    :raises TypeError: If the expected class specification is invalid for
        :func:`isinstance` or :func:`issubclass` when those checks are reached.

    :Example:

        >>> from pina import LabelTensor
        >>> from pina.utils import check_consistency
        >>> import torch
        >>> tensor = LabelTensor(torch.randn(3, 2), labels=["x", "y"])
        >>> check_consistency(tensor, LabelTensor)
    """
    if not isinstance(object_, (list, set, tuple)):
        object_ = [object_]

    for obj in object_:
        is_class = isinstance(obj, type)
        expected_type_name = (
            object_instance.__name__
            if isinstance(object_instance, type)
            else str(object_instance)
        )

        if subclass:
            if not is_class:
                raise ValueError(
                    f"You passed {repr(obj)} "
                    f"(an instance of {type(obj).__name__}), "
                    f"but a {expected_type_name} class was expected. "
                    f"Please pass a {expected_type_name} class or a "
                    "derived one."
                )
            if not issubclass(obj, object_instance):
                raise ValueError(
                    f"You passed {obj.__name__} class, but a "
                    f"{expected_type_name} class was expected. "
                    f"Please pass a {expected_type_name} class or a "
                    "derived one."
                )
        else:
            if is_class:
                raise ValueError(
                    f"You passed {obj.__name__} class, but a "
                    f"{expected_type_name} instance was expected. "
                    f"Please pass a {expected_type_name} instance."
                )
            if not isinstance(obj, object_instance):
                raise ValueError(
                    f"You passed {repr(obj)} "
                    f"(an instance of {type(obj).__name__}), "
                    f"but a {expected_type_name} instance was expected. "
                    f"Please pass a {expected_type_name} instance."
                )


def labelize_forward(forward, input_variables, output_variables):
    """
    Wrap a forward callable with label-based input and output handling.

    Inputs are always extracted in the requested variable order; this wrapper
    does not toggle label handling off. The forward result is cast to
    :class:`pina.LabelTensor` and its last-dimension labels are assigned. The
    wrapper does not preserve the original function's metadata or signature.

    :param Callable forward: The forward function of a :class:`torch.nn.Module`.
    :param list[str] input_variables: The input column names, in the order
        expected by the forward callable. Nonempty names select those columns;
        an empty list fails during extraction.
    :param list[str] output_variables: The unique output column names. Their
        count must match the last dimension of the forward result.
    :return: The decorated forward function.
    :rtype: Callable
    """

    def wrapper(x, *args, **kwargs):
        """
        Decorated forward function.

        :param LabelTensor x: The labeled input providing the selected columns.
        :param tuple[Any, ...] args: Additional positional arguments forwarded
            unchanged to the wrapped callable.
        :param dict[str, Any] kwargs: Additional keyword arguments forwarded
            unchanged to the wrapped callable; their meaning depends on it.
        :raises AttributeError: If the input has no ``extract`` method or the
            forward result has no tensor ``as_subclass`` method.
        :raises ValueError: If selected labels are unknown or output labels
            repeat or have an incompatible count.
        :raises KeyError: If input selection needs missing label metadata.
        :raises IndexError: If the input selection is empty.
        :return: The forward result cast to a labeled tensor, with the requested
            last-dimension labels and its autograd connection retained.
        :rtype: LabelTensor
        """
        x = x.extract(input_variables)
        output = forward(x, *args, **kwargs)
        # keep it like this, directly using LabelTensor(...) raises errors
        # when compiling the code
        output = output.as_subclass(LabelTensor)
        output.labels = output_variables
        return output

    return wrapper


def merge_tensors(tensors):
    """
    Merge a list of :class:`~pina.label_tensor.LabelTensor` instances into a
    single :class:`~pina.label_tensor.LabelTensor` tensor, by applying
    iteratively the cartesian product.

    :param tensors: The nonempty sequence of two-dimensional tensors to merge.
        For shapes ``(Ni, Fi)``, the result has ``product(Ni)`` rows and
        ``sum(Fi)`` columns. Only last-dimension labels are propagated by the
        pairwise merges.
    :type tensors: list[LabelTensor] | tuple[LabelTensor, ...]
    :raises ValueError: If the list of tensors is empty.
    :raises RuntimeError: If tensors cannot be repeated or concatenated with the
        required shapes.
    :raises TypeError: If concatenation cannot combine their component labels.
    :return: The Cartesian-product tensor, with columns in input sequence order.
        A single input is returned unchanged.
    :rtype: LabelTensor

    :Example:

        >>> from pina import LabelTensor
        >>> from pina.utils import merge_tensors
        >>> import torch
        >>> t1 = LabelTensor(torch.tensor([[1.0], [2.0]]), labels=["x"])
        >>> t2 = LabelTensor(torch.tensor([[3.0], [4.0]]), labels=["y"])
        >>> merged = merge_tensors([t1, t2])
        >>> merged.labels
        ['x', 'y']
    """
    if tensors:
        return reduce(merge_two_tensors, tensors[1:], tensors[0])
    raise ValueError("Expected at least one tensor")


def merge_two_tensors(tensor1, tensor2):
    """
    Merge two :class:`~pina.label_tensor.LabelTensor` instances into a single
    :class:`~pina.label_tensor.LabelTensor` tensor, by applying the cartesian
    product.

    :param LabelTensor tensor1: The first tensor, with shape ``(n1, f1)``.
    :param LabelTensor tensor2: The second tensor, with shape ``(n2, f2)``.
    :raises RuntimeError: If repetition or concatenation rejects the shapes.
    :raises TypeError: If concatenation cannot combine the component labels.
    :return: The merged tensor with shape ``(n1 * n2, f1 + f2)``. Rows of the
        first tensor repeat for each row of the second. Only their last-axis
        labels are carried into the result.
    :rtype: LabelTensor
    """
    n1 = tensor1.shape[0]
    n2 = tensor2.shape[0]

    tensor1 = LabelTensor(tensor1.repeat(n2, 1), labels=tensor1.labels)
    tensor2 = LabelTensor(
        tensor2.repeat_interleave(n1, dim=0), labels=tensor2.labels
    )
    return tensor1.append(tensor2)


def torch_lhs(n, dim):
    """
    Sample a Latin hypercube in :math:`(0, 1]` using PyTorch randomness.

    Each coordinate dimension independently permutes the ``n`` equal strata and
    subtracts uniform random offsets from their upper endpoints. Allocations use
    PyTorch's default device and floating dtype.

    :param int n: The number of points to sample. Zero returns an empty tensor.
    :param int dim: The number of coordinate dimensions, at least one.
    :raises TypeError: If ``n`` or ``dim`` is not an integer.
    :raises ValueError: If ``dim`` is less than one.
    :raises RuntimeError: If ``n`` is negative and tensor allocation rejects it.
    :return: The sampled coordinates with shape ``(n, dim)``. Upper stratum
        endpoints can be included; lower endpoints are excluded by the formula.
    :rtype: torch.Tensor

    :Example:

        >>> from pina._src.core.utils import torch_lhs
        >>> samples = torch_lhs(10, 2)
        >>> samples.shape
        torch.Size([10, 2])
    """

    if not isinstance(n, int):
        raise TypeError("number of point n must be int")

    if not isinstance(dim, int):
        raise TypeError("dim must be int")

    if dim < 1:
        raise ValueError("dim must be greater than one")

    samples = torch.rand(size=(n, dim))

    perms = torch.tile(torch.arange(1, n + 1), (dim, 1))

    for row in range(dim):
        idx_perm = torch.randperm(perms.shape[-1])
        perms[row, :] = perms[row, idx_perm]

    perms = perms.T

    samples = (perms - samples) / n

    return samples


def is_function(f):
    """
    Check whether an object is callable.

    :param Any f: The object to check, including functions, classes, or objects
        implementing ``__call__``.
    :return: The result of :func:`callable`.
    :rtype: bool
    """
    return callable(f)


def chebyshev_roots(n):
    """
    Compute sorted roots of the first-kind Chebyshev polynomial of degree ``n``.

    :param int n: The number of roots. Zero produces an empty tensor.
        Integrality is assumed rather than explicitly validated.
    :raises RuntimeError: If a negative ``n`` makes :func:`torch.arange` fail.
    :return: The roots in ascending order, with shape ``(n,)`` for integral
        ``n``. Values lie inside ``(-1, 1)`` up to floating-point rounding and
        use PyTorch's default floating dtype and allocation device.
    :rtype: torch.Tensor

    :Example:

        >>> from pina._src.core.utils import chebyshev_roots
        >>> roots = chebyshev_roots(5)
        >>> roots.shape
        torch.Size([5])
    """
    pi = torch.acos(torch.zeros(1)).item() * 2
    k = torch.arange(n)
    nodes = torch.sort(torch.cos(pi * (k + 0.5) / n))[0]
    return nodes


def check_positive_integer(value, strict=True):
    """
    Check if the value is a positive integer.

    :param int value: The value to check.
    :param bool strict: Whether to require ``value > 0``. If ``False``, zero is
        also allowed. Default is ``True``.
    :raises AssertionError: If ``value`` is not an integer, is negative, or is
        zero while ``strict`` is true.

    .. note::

        The check uses :func:`isinstance` with ``int``, so booleans count as
        integers. Assertions are disabled when Python runs with optimization.
    """
    if strict:
        assert (
            isinstance(value, int) and value > 0
        ), f"Expected a strictly positive integer, got {value}."
    else:
        assert (
            isinstance(value, int) and value >= 0
        ), f"Expected a non-negative integer, got {value}."


def in_range(value, range_vals, strict=True):
    """
    Check if a value is within a specified range.

    :param value: The numeric value to compare with the bounds.
    :type value: int | float
    :param list[int | float] range_vals: Exactly two numeric bounds, lower then
        upper. Bound ordering is not validated.
    :param bool strict: Whether both comparisons exclude the endpoints. If
        ``False``, both endpoints are included. Default is ``True``.
    :raises ValueError: If numeric type validation fails for the value or
        bounds.
    :raises AssertionError: If ``range_vals`` is not a list of length two.
    :return: Whether the value is strictly between the bounds, or within the
        closed interval when ``strict`` is ``False``.
    :rtype: bool

    :Example:

        >>> from pina._src.core.utils import in_range
        >>> in_range(5, [0, 10])
        True
        >>> in_range(10, [0, 10], strict=False)
        True
    """
    # Validate inputs
    check_consistency(value, (float, int))
    check_consistency(range_vals, (float, int))
    assert (
        isinstance(range_vals, list) and len(range_vals) == 2
    ), "range_vals must be a list of two integers [lower, upper]"
    lower, upper = range_vals

    # Check the range
    if strict:
        return lower < value < upper

    return lower <= value <= upper
