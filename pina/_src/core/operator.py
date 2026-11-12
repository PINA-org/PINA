"""Module for vectorized differential operators implementation.

Differential operators are used to define differential problems and are
implemented to run efficiently on various accelerators, including CPU, GPU, TPU,
and MPS.

Each differential operator takes the following inputs:
- A tensor on which the operator is applied.
- A tensor with respect to which the operator is computed.
- The names of the output variables for which the operator is evaluated.
- The names of the variables with respect to which the operator is computed.

Each differential operator has its fast version, which performs no internal
checks on input and output tensors. For these methods, the user is always
required to specify both ``components`` and ``d`` as lists of strings.
"""

import torch

from pina._src.core.label_tensor import LabelTensor
from pina._src.core.logging import get_logger

logger = get_logger(__name__)


def _check_values(output_, input_, components, d):
    """
    Perform checks on arguments of differential operators.

    :param LabelTensor output_: The output tensor on which the operator is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        operator is computed.
    :param components: The names of the output variables for which to compute
        the operator. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which the
        operator is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises RuntimeError: If derivative labels are missing from the ``input_``.
    :raises RuntimeError: If component labels are missing from the ``output_``.
    :return: The components and d lists.
    :rtype: tuple[list[str], list[str]]
    """
    # Check if the input is a LabelTensor
    if not isinstance(input_, LabelTensor):
        raise TypeError("Input must be a LabelTensor.")

    # Check if the output is a LabelTensor
    if not isinstance(output_, LabelTensor):
        raise TypeError("Output must be a LabelTensor.")

    # If no labels are provided, use all labels
    d = d or input_.labels
    components = components or output_.labels

    # Convert to list if not already
    d = d if isinstance(d, list) else [d]
    components = components if isinstance(components, list) else [components]

    # Check if all labels are present in the input tensor
    if not all(di in input_.labels for di in d):
        raise RuntimeError("Derivative labels missing from input tensor.")

    # Check if all labels are present in the output tensor
    if not all(c in output_.labels for c in components):
        raise RuntimeError("Component label missing from output tensor.")

    return components, d


def _scalar_grad(output_, input_, d):
    """
    Compute the gradient of a scalar-valued ``output_``.

    :param LabelTensor output_: The output tensor on which the gradient is
        computed. It must be a column tensor.
    :param LabelTensor input_: The input tensor with respect to which the
        gradient is computed.
    :param d: The indices of the input variables with respect to which the
        gradient is computed. If ``None``, all input variables are kept.
    :type d: list[int] | None
    :return: The computed gradient tensor.
    :rtype: torch.Tensor
    """
    # Skip LabelTensor dispatch so autograd returns a plain tensor
    with torch._C.DisableTorchFunctionSubclass():
        grad_out = torch.autograd.grad(
            outputs=output_,
            inputs=input_,
            grad_outputs=torch.ones_like(output_),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

    return grad_out if d is None else grad_out[..., d]


def _scalar_laplacian(output_, input_, d):
    """
    Compute the laplacian of a scalar-valued ``output_``.

    :param LabelTensor output_: The output tensor on which the laplacian is
        computed. It must be a column tensor.
    :param LabelTensor input_: The input tensor with respect to which the
        laplacian is computed.
    :param list[str] d: The names of the input variables with respect to
        which the laplacian is computed. It must be a subset of the input
        labels. If ``None``, all input variables are considered.
    :return: The computed laplacian tensor.
    :rtype: LabelTensor
    """
    first_grad = fast_grad(
        output_=output_, input_=input_, components=output_.labels, d=d
    )
    second_grad = fast_grad(
        output_=first_grad, input_=input_, components=first_grad.labels, d=d
    )
    labels_to_extract = [f"d{c}d{d_}" for c, d_ in zip(first_grad.labels, d)]
    return torch.sum(
        second_grad.extract(labels_to_extract), dim=-1, keepdim=True
    )


def fast_grad(output_, input_, components, d):
    """
    Compute the gradient of the ``output_`` with respect to the ``input``.

    Unlike ``grad``, this function performs no internal checks on input and
    output tensors. The user is required to specify both ``components`` and
    ``d`` as lists of strings. It is designed to enhance computation speed.

    This operator supports both vector-valued and scalar-valued functions with
    one or multiple input coordinates.

    :param LabelTensor output_: The output tensor on which the gradient is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        gradient is computed.
    :param list[str] components: The names of the output variables for which to
        compute the gradient. It must be a subset of the output labels.
    :param list[str] d: The names of the input variables with respect to which
        the gradient is computed. It must be a subset of the input labels.
    :return: The computed gradient tensor.
    :rtype: LabelTensor
    """
    logger.debug(
        "operator.fast_grad: components=%s, d=%s.", components, d
    )
    # Work on the plain output tensor to skip LabelTensor overhead
    out = output_.tensor
    labels = output_.labels

    # A one-column output is treated as its single component
    if out.shape[-1] == 1:
        components = labels[:1]

    # Skip indexing the gradient when all inputs are requested in order
    in_labels = input_.labels
    d_idx = None if d == in_labels else [in_labels.index(i) for i in d]
    cols = [labels.index(c) for c in components]
    grads = [_scalar_grad(out[..., j : j + 1], input_, d_idx) for j in cols]
    grads = grads[0] if len(grads) == 1 else torch.cat(grads, dim=-1)
    logger.debug("operator.fast_grad: -> shape=%s.", tuple(grads.shape))

    return LabelTensor(
        grads, labels=[f"d{c}d{i}" for c in components for i in d]
    )


def fast_div(output_, input_, components, d):
    """
    Compute the divergence of the ``output_`` with respect to ``input``.

    Unlike ``div``, this function performs no internal checks on input and
    output tensors. The user is required to specify both ``components`` and
    ``d`` as lists of strings. It is designed to enhance computation speed.

    This operator supports vector-valued functions with multiple input
    coordinates.

    :param LabelTensor output_: The output tensor on which the divergence is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        divergence is computed.
    :param list[str] components: The names of the output variables for which to
        compute the divergence. It must be a subset of the output labels.
    :param list[str] d: The names of the input variables with respect to which
        the divergence is computed. It must be a subset of the input labels.
    :rtype: LabelTensor
    """
    logger.debug("operator.fast_div: components=%s, d=%s.", components, d)
    grad_out = fast_grad(
        output_=output_, input_=input_, components=components, d=d
    )
    tensors_to_sum = [
        grad_out.extract(f"d{c}d{d_}") for c, d_ in zip(components, d)
    ]

    return LabelTensor.summation(tensors_to_sum)


def fast_laplacian(output_, input_, components, d, method="std"):
    """
    Compute the laplacian of the ``output_`` with respect to ``input``.

    Unlike ``laplacian``, this function performs no internal checks on input and
    output tensors. The user is required to specify both ``components`` and
    ``d`` as lists of strings. It is designed to enhance computation speed.

    This operator supports both vector-valued and scalar-valued functions with
    one or multiple input coordinates.

    :param LabelTensor output_: The output tensor on which the laplacian is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        laplacian is computed.
    :param list[str] components: The names of the output variables for which to
        compute the laplacian. It must be a subset of the output labels.
    :param list[str] d: The names of the input variables with respect to which
        the laplacian is computed. It must be a subset of the input labels.
    :param str method: The method used to compute the Laplacian. Available
        methods are ``std`` and ``divgrad``. The ``std`` method computes the
        trace of the Hessian matrix, while the ``divgrad`` method computes the
        divergence of the gradient. Default is ``std``.
    :return: The computed laplacian tensor.
    :rtype: LabelTensor
    :raises ValueError: If the passed method is neither ``std`` nor ``divgrad``.
    """
    logger.debug(
        "operator.fast_laplacian: method=%s, components=%s.",
        method,
        components,
    )
    # Scalar laplacian
    if output_.shape[-1] == 1:
        return LabelTensor(
            _scalar_laplacian(output_=output_, input_=input_, d=d),
            labels=[f"dd{c}" for c in components],
        )

    # Initialize the result tensor and its labels
    labels = [f"dd{c}" for c in components]
    result = torch.empty(
        input_.shape[0], len(components), device=output_.device
    )

    # Vector laplacian
    if method == "std":
        result = torch.cat(
            [
                _scalar_laplacian(
                    output_=output_.extract(c), input_=input_, d=d
                )
                for c in components
            ],
            dim=-1,
        )

    elif method == "divgrad":
        grads = fast_grad(
            output_=output_, input_=input_, components=components, d=d
        )
        result = torch.cat(
            [
                fast_div(
                    output_=grads,
                    input_=input_,
                    components=[f"d{c}d{i}" for i in d],
                    d=d,
                )
                for c in components
            ],
            dim=-1,
        )

    else:
        raise ValueError(
            "Invalid method. Available methods are ``std`` and ``divgrad``."
        )

    return LabelTensor(result, labels=labels)


def fast_advection(output_, input_, velocity_field, components, d):
    """
    Perform the advection operation on the ``output_`` with respect to the
    ``input``. This operator supports vector-valued functions with multiple
    input coordinates.

    Unlike ``advection``, this function performs no internal checks on input and
    output tensors. The user is required to specify both ``components`` and
    ``d`` as lists of strings. It is designed to enhance computation speed.

    :param LabelTensor output_: The output tensor on which the advection is
        computed. It includes both the velocity and the quantity to be advected.
    :param LabelTensor input_: the input tensor with respect to which advection
        is computed.
    :param list[str] velocity_field: The name of the output variables used as
        velocity field. It must be chosen among the output labels.
    :param list[str] components: The names of the output variables for which to
        compute the advection. It must be a subset of the output labels.
    :param list[str] d: The names of the input variables with respect to which
        the advection is computed. It must be a subset of the input labels.
    :return: The computed advection tensor.
    :rtype: LabelTensor
    """
    logger.debug(
        "operator.fast_advection: components=%s, d=%s, velocity_field=%s.",
        components,
        d,
        velocity_field,
    )
    # Add a dimension to the velocity field for following operations
    velocity = output_.extract(velocity_field).unsqueeze(-1)

    # Compute the gradient
    grads = fast_grad(
        output_=output_, input_=input_, components=components, d=d
    )

    # Reshape into [..., len(filter_components), len(d)]
    tmp = grads.reshape(*output_.shape[:-1], len(components), len(d))

    # Transpose to [..., len(d), len(filter_components)]
    tmp = tmp.transpose(-1, -2)

    adv = (tmp * velocity).sum(dim=tmp.tensor.ndim - 2)
    return LabelTensor(adv, labels=[f"adv_{c}" for c in components])


def grad(output_, input_, components=None, d=None):
    """
    Compute the gradient of the ``output_`` with respect to the ``input``.

    This operator supports both vector-valued and scalar-valued functions with
    one or multiple input coordinates.

    :param LabelTensor output_: The output tensor on which the gradient is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        gradient is computed.
    :param components: The names of the output variables for which to compute
        the gradient. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which the
        gradient is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises RuntimeError: If derivative labels are missing from the ``input_``.
    :raises RuntimeError: If component labels are missing from the ``output_``.
    :return: The computed gradient tensor.
    :rtype: LabelTensor

    :Example:

        >>> from pina import LabelTensor
        >>> import torch
        >>> x = LabelTensor(torch.rand(5, 1), labels=["x"])
        >>> u = LabelTensor(x ** 2, labels=["u"])
        >>> grad(u, x, components=["u"], d=["x"])
        LabelTensor([[2.0 * x] ...], labels=["dudx"])
    """
    logger.debug("operator.grad: components=%s, d=%s.", components, d)
    components, d = _check_values(
        output_=output_, input_=input_, components=components, d=d
    )
    return fast_grad(output_=output_, input_=input_, components=components, d=d)


def div(output_, input_, components=None, d=None):
    """
    Compute the divergence of the ``output_`` with respect to ``input``.

    This operator supports vector-valued functions with multiple input
    coordinates.

    :param LabelTensor output_: The output tensor on which the divergence is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        divergence is computed.
    :param components: The names of the output variables for which to compute
        the divergence. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which the
        divergence is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type components: str | list[str]
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises ValueError: If the length of ``components`` and ``d`` do not match.
    :return: The computed divergence tensor.
    :rtype: LabelTensor

    :Example:

        >>> from pina import LabelTensor
        >>> import torch
        >>> x = LabelTensor(torch.rand(5, 2), labels=["x", "y"])
        >>> u = LabelTensor(torch.cat([x.extract("x")**2,
        ...     x.extract("y")**2], dim=-1), labels=["u", "v"])
        >>> div(u, x, components=["u", "v"], d=["x", "y"])
        LabelTensor([...], labels=["div"])
    """
    logger.debug("operator.div: components=%s, d=%s.", components, d)
    components, d = _check_values(
        output_=output_, input_=input_, components=components, d=d
    )

    # Components and d must be of the same length
    if len(components) != len(d):
        raise ValueError(
            "Divergence requires components and d to be of the same length."
        )

    return fast_div(output_=output_, input_=input_, components=components, d=d)


def laplacian(output_, input_, components=None, d=None, method="std"):
    """
    Compute the laplacian of the ``output_`` with respect to ``input``.

    This operator supports both vector-valued and scalar-valued functions with
    one or multiple input coordinates.

    :param LabelTensor output_: The output tensor on which the laplacian is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        laplacian is computed.
    :param components: The names of the output variables for which to
        compute the laplacian. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which
        the laplacian is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :param str method: The method used to compute the Laplacian. Available
        methods are ``std`` and ``divgrad``. The ``std`` method computes the
        trace of the Hessian matrix, while the ``divgrad`` method computes the
        divergence of the gradient. Default is ``std``.
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises ValueError: If the passed method is neither ``std`` nor ``divgrad``.
    :return: The computed laplacian tensor.
    :rtype: LabelTensor

    :Example:

        >>> from pina import LabelTensor
        >>> import torch
        >>> x = LabelTensor(torch.rand(5, 1), labels=["x"])
        >>> u = LabelTensor(x ** 3, labels=["u"])
        >>> laplacian(u, x, components=["u"], d=["x"])
        LabelTensor([...], labels=["ddu"])
    """
    logger.debug(
        "operator.laplacian: method=%s, components=%s.", method, components
    )
    components, d = _check_values(
        output_=output_, input_=input_, components=components, d=d
    )

    return fast_laplacian(
        output_=output_,
        input_=input_,
        components=components,
        d=d,
        method=method,
    )


def advection(output_, input_, velocity_field, components=None, d=None):
    """
    Perform the advection operation on the ``output_`` with respect to the
    ``input``. This operator supports vector-valued functions with multiple
    input coordinates.

    :param LabelTensor output_: The output tensor on which the advection is
        computed. It includes both the velocity and the quantity to be advected.
    :param LabelTensor input_: the input tensor with respect to which advection
        is computed.
    :param velocity_field: The name of the output variables used as velocity
        field. It must be chosen among the output labels.
    :type velocity_field: str | list[str]
    :param components: The names of the output variables for which to compute
        the advection. It must be a subset of the output labels.
        If ``None``, all output variables are considered. Default is ``None``.
    :type components: str | list[str]
    :param d: The names of the input variables with respect to which the
        advection is computed. It must be a subset of the input labels.
        If ``None``, all input variables are considered. Default is ``None``.
    :type d: str | list[str]
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises RuntimeError: If the velocity field is not a subset of the output
        labels.
    :raises RuntimeError: If the dimensionality of the velocity field does not
        match that of the input tensor.
    :return: The computed advection tensor.
    :rtype: LabelTensor

    :Example:

        >>> from pina import LabelTensor
        >>> import torch
        >>> x = LabelTensor(torch.rand(5, 2), labels=["x", "y"])
        >>> u = LabelTensor(torch.cat([x.extract("x")**2,
        ...     torch.zeros(5, 2)], dim=-1), labels=["u", "v", "w", "z"])
        >>> advection(u, x, velocity_field=["v", "w"],
        ...     components=["u"], d=["x", "y"])
        LabelTensor([...], labels=["adv_u"])
    """
    logger.debug(
        "operator.advection: components=%s, d=%s, velocity_field=%s.",
        components,
        d,
        velocity_field,
    )
    components, d = _check_values(
        output_=output_, input_=input_, components=components, d=d
    )

    # Map velocity_field to a list if it is a string
    if isinstance(velocity_field, str):
        velocity_field = [velocity_field]

    # Check if all the velocity_field labels are present in the output labels
    if not all(vi in output_.labels for vi in velocity_field):
        raise RuntimeError("Velocity labels missing from output tensor.")

    # Check if the velocity has the same dimensionality as the input tensor
    if len(velocity_field) != len(d):
        raise RuntimeError(
            "Velocity dimensionality does not match input dimensionality."
        )

    return fast_advection(
        output_=output_,
        input_=input_,
        velocity_field=velocity_field,
        components=components,
        d=d,
    )
