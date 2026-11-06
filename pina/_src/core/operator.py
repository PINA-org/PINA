"""Automatic-differentiation operators for labeled tensors.

Inputs and outputs use variable names on their last dimension. Inputs must
require gradients, and outputs must be connected to those inputs through
autograd. Derivative calls retain the graph and build higher-order graphs.
All-ones vector-Jacobian products are used, so the usual pointwise derivative
interpretation assumes independent samples along the leading dimensions.

Public wrappers normalize selectors and check labels; fast variants skip those
checks and require explicit lists. They still perform delegated PyTorch and
label-metadata validation. Calls are not cached: compute an operator once for
the needed components and reuse or extract its result.

Constant or disconnected outputs are not replaced by zero derivatives. Autograd
can reject them, or return an unused gradient that the implementation cannot
index. This also affects second derivatives of linear functions.
"""

import torch
from pina._src.core.label_tensor import LabelTensor


def _check_values(output_, input_, components, d):
    """
    Perform checks on arguments of differential operators.

    :param LabelTensor output_: The output tensor on which the operator is
        computed.
    :param LabelTensor input_: The input tensor with respect to which the
        operator is computed.
    :param components: The names of the output variables for which to compute
        the operator. It must be a subset of the output labels.
        ``None`` or an empty selection uses all last-dimension output labels.
        Lists are preserved; a scalar string is wrapped in a one-item list.
    :type components: str | list[str] | None
    :param d: The names of the input variables with respect to which the
        operator is computed. It must be a subset of the input labels.
        ``None`` or an empty selection uses all last-dimension input labels.
        Lists are preserved; a scalar string is wrapped in a one-item list.
    :type d: str | list[str] | None
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises RuntimeError: If derivative labels are missing from the ``input_``.
    :raises RuntimeError: If component labels are missing from the ``output_``.
    :return: The normalized component and coordinate lists, in that order.
        Nonempty input lists and fallback label lists are returned by reference.
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
        computed, with shape ``(..., 1)`` and an autograd connection to input.
    :param LabelTensor input_: The input tensor with respect to which the
        gradient is computed, with shape ``(..., n_inputs)`` and gradient
        tracking already enabled.
    :param list[str] d: The names of the input variables with respect to
        which the gradient is computed. It must be a subset of the input
        labels. This helper does not normalize ``None`` or scalar selectors.
    :raises RuntimeError: If autograd cannot differentiate the supplied tensors.
    :raises TypeError: If autograd returns ``None`` for an unused input and the
        result cannot be indexed.
    :raises ValueError: If a coordinate name is absent from input labels.
    :return: The selected vector-Jacobian product with shape
        ``(*input_.shape[:-1], len(d))``. No derivative metadata is attached by
        this helper.
    :rtype: torch.Tensor | LabelTensor
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
        computed, with shape ``(..., 1)`` and one stored component label.
    :param LabelTensor input_: The input tensor with respect to which the
        laplacian is computed.
    :param list[str] d: The names of the input variables with respect to
        which the laplacian is computed. It must be a subset of the input
        labels. ``None`` is not normalized by this helper.
    :raises RuntimeError: If first or second differentiation fails, including a
        constant first derivative with no higher-order autograd graph.
    :raises TypeError: If an unused gradient is returned as ``None``.
    :raises ValueError: If a selected coordinate is unknown.
    :return: The sum of diagonal second derivatives over ``d``, with shape
        ``(*input_.shape[:-1], 1)``. The sum is not assigned derivative labels
        until the caller wraps it.
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

    Unlike :func:`grad`, skip argument normalization and operator label checks.
    The user must supply nonempty component and coordinate lists. Delegated
    autograd and label extraction still validate their own requirements.

    This operator supports both vector-valued and scalar-valued functions with
    one or multiple input coordinates.

    :param LabelTensor output_: The output tensor on which the gradient is
        computed, with labeled last dimension and leading sample dimensions.
    :param LabelTensor input_: The input tensor with respect to which the
        gradient is computed, with ``requires_grad=True``. Outputs must retain
        an autograd connection to this exact tensor.
    :param list[str] components: The names of the output variables for which to
        compute the gradient. It must be a subset of the output labels.
    :param list[str] d: The names of the input variables with respect to which
        the gradient is computed. It must be a subset of the input labels.
    :raises RuntimeError: If differentiation fails or vector components cannot
        be concatenated.
    :raises TypeError: If an unused gradient is returned as ``None``.
    :raises ValueError: If a requested label is unknown or derivative labels
        cannot be assigned, for example because they repeat.
    :return: The gradient with shape
        ``(*input_.shape[:-1], len(components) * len(d))`` for vector outputs.
        Columns have labels ``"d{component}d{coordinate}"`` in component-major,
        coordinate-minor order. For a one-column output, ``components`` is
        ignored and that column is differentiated over ``d``.
    :rtype: LabelTensor
    """
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

    return LabelTensor(
        grads, labels=[f"d{c}d{i}" for c in components for i in d]
    )


def fast_div(output_, input_, components, d):
    """
    Compute the divergence of the ``output_`` with respect to ``input``.

    Unlike :func:`div`, skip selector normalization and length validation.
    Terms pair components with coordinates in list order. Unequal list lengths
    are silently truncated by :func:`zip`; pass equal nonempty lists for a
    conventional divergence.

    This operator supports vector-valued functions with multiple input
    coordinates.

    :param LabelTensor output_: The output tensor on which the divergence is
        computed, connected to ``input_`` through autograd.
    :param LabelTensor input_: The input tensor with respect to which the
        divergence is computed, with gradient tracking enabled.
    :param list[str] components: The names of the output variables for which to
        compute the divergence. It must be a subset of the output labels.
    :param list[str] d: The names of the input variables with respect to which
        the divergence is computed. It must be a subset of the input labels.
    :raises RuntimeError: If autograd differentiation fails.
    :raises TypeError: If an unused gradient is returned as ``None``.
    :raises ValueError: If labels are unknown or no paired terms can be summed.
    :return: The sum of paired derivatives with one final column. Its label is
        the derivative labels joined by ``"+"``, such as ``"dudx+dvdy"``.
    :rtype: LabelTensor
    """
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

    Unlike :func:`laplacian`, skip selector normalization and operator label
    checks. Explicit nonempty lists are required; delegated validation still
    applies.

    This operator supports both vector-valued and scalar-valued functions with
    one or multiple input coordinates.

    :param LabelTensor output_: The output tensor on which the laplacian is
        computed, connected to ``input_`` through an autograd graph supporting
        second derivatives.
    :param LabelTensor input_: The input tensor with respect to which the
        laplacian is computed, with gradient tracking enabled.
    :param list[str] components: The names of the output variables for which to
        compute the laplacian. It must be a subset of the output labels.
    :param list[str] d: The names of the input variables with respect to which
        the laplacian is computed. It must be a subset of the input labels.
    :param str method: The method used to compute the Laplacian. Available
        methods are ``"std"`` for componentwise Hessian traces and
        ``"divgrad"`` for divergence of each component's gradient. For a
        one-column output, the scalar Hessian-trace path is always used and
        ``method`` is ignored, even if invalid. Default is ``"std"``.
    :raises ValueError: If a vector output uses an unsupported method, labels
        are unknown, or the result's component labels are invalid.
    :raises RuntimeError: If first or second differentiation fails.
    :raises TypeError: If an unused gradient is returned as ``None``.
    :return: The componentwise Laplacian with shape
        ``(*input_.shape[:-1], len(components))`` and labels
        ``"dd{component}"``.
        For a one-column output, the selected component list must contain one
        name for the scalar result's metadata to fit.
    :rtype: LabelTensor
    """
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

    Unlike :func:`advection`, skip selector normalization and velocity-label
    dimension checks. Pass explicit nonempty component and coordinate lists.

    :param LabelTensor output_: The output tensor on which the advection is
        computed. It includes both the velocity and the quantity to be advected.
    :param LabelTensor input_: The input tensor with respect to which advection
        is computed, with gradient tracking enabled and an autograd connection
        to the output.
    :param velocity_field: The names of the output variables used as
        velocity field, ordered to match ``d``. Its length should equal
        ``len(d)``; broadcasting may otherwise change the operation or fail.
    :type velocity_field: str | list[str] | tuple[str, ...]
    :param list[str] components: The names of the output variables for which to
        compute the advection. It must be a subset of the output labels.
    :param list[str] d: The names of the input variables with respect to which
        the advection is computed. It must be a subset of the input labels.
    :raises ValueError: If requested labels are unknown or result labels repeat.
    :raises RuntimeError: If differentiation, reshaping, or velocity
        broadcasting fails.
    :raises TypeError: If an unused gradient is returned as ``None``.
    :return: The sum of velocity times each selected component's gradient over
        coordinates, with shape ``(*output_.shape[:-1], len(components))`` and
        labels ``"adv_{component}"``. Leading sample dimensions must be
        compatible with the gradient's leading dimensions.
    :rtype: LabelTensor
    """
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
        computed, with labeled last dimension and an autograd connection to
        ``input_``.
    :param LabelTensor input_: The input tensor with respect to which the
        gradient is computed, with labeled last dimension and gradient tracking
        already enabled.
    :param components: The names of the output variables for which to compute
        the gradient. It must be a subset of the output labels.
        ``None`` or an empty selection uses all output labels. Default is
        ``None``.
    :type components: str | list[str] | None
    :param d: The names of the input variables with respect to which the
        gradient is computed. It must be a subset of the input labels.
        ``None`` or an empty selection uses all input labels. Default is
        ``None``.
    :type d: str | list[str] | None
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises RuntimeError: If derivative labels are missing from the ``input_``.
    :raises RuntimeError: If component labels are missing from the ``output_``.
    :raises RuntimeError: If autograd cannot differentiate the supplied graph.
    :raises TypeError: If an unused gradient is returned as ``None``.
    :raises ValueError: If derivative labels cannot be assigned, for example
        because the selected coordinates repeat.
    :return: The gradient with shape
        ``(*input_.shape[:-1], len(components) * len(d))`` for vector outputs.
        Columns use ``"d{component}d{coordinate}"`` labels in component-major,
        coordinate-minor order. A scalar output produces ``len(d)`` columns;
        after selector validation its component list is ignored.
    :rtype: LabelTensor

    :Example:

        >>> from pina import LabelTensor
        >>> from pina.operator import grad
        >>> import torch
        >>> x = LabelTensor(torch.tensor([[1.0], [2.0]]), ["x"])
        >>> x = x.requires_grad_()
        >>> u = LabelTensor(x ** 2, labels=["u"])
        >>> derivatives = grad(u, x, components=["u"], d=["x"])
        >>> derivatives.labels
        ['dudx']
        >>> torch.allclose(derivatives.tensor, 2 * x.tensor)
        True
    """
    components, d = _check_values(
        output_=output_, input_=input_, components=components, d=d
    )
    return fast_grad(output_=output_, input_=input_, components=components, d=d)


def div(output_, input_, components=None, d=None):
    """
    Compute the divergence of the ``output_`` with respect to ``input``.

    Pair selected components with selected coordinates in list order and sum
    their derivatives. Unlike :func:`fast_div`, require equal list lengths.

    :param LabelTensor output_: The output tensor on which the divergence is
        computed, connected to ``input_`` through autograd.
    :param LabelTensor input_: The input tensor with respect to which the
        divergence is computed, with gradient tracking enabled.
    :param components: The names of the output variables for which to compute
        the divergence. It must be a subset of the output labels.
        ``None`` or an empty selection uses all output labels. Default is
        ``None``.
    :type components: str | list[str] | None
    :param d: The names of the input variables with respect to which the
        divergence is computed. It must be a subset of the input labels.
        ``None`` or an empty selection uses all input labels. Default is
        ``None``.
    :type d: str | list[str] | None
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises ValueError: If the length of ``components`` and ``d`` do not match.
    :raises RuntimeError: If selected labels are missing or autograd cannot
        differentiate the supplied graph.
    :raises TypeError: If an unused gradient is returned as ``None``.
    :return: The paired derivative sum with shape ``(*input_.shape[:-1], 1)``.
        Its label joins derivative names with ``"+"``, such as ``"dudx+dvdy"``.
    :rtype: LabelTensor

    :Example:

        >>> from pina import LabelTensor
        >>> from pina.operator import div
        >>> import torch
        >>> x = LabelTensor(torch.tensor([[1.0, 2.0]]), ["x", "y"])
        >>> x = x.requires_grad_()
        >>> u = LabelTensor(torch.cat([x.extract("x")**2,
        ...     x.extract("y")**2], dim=-1), labels=["u", "v"])
        >>> divergence = div(u, x, components=["u", "v"], d=["x", "y"])
        >>> divergence.labels
        ['dudx+dvdy']
        >>> expected = 2 * x.tensor.sum(-1, keepdim=True)
        >>> torch.allclose(divergence.tensor, expected)
        True
    """
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
        computed, connected to ``input_`` through an autograd graph supporting
        second derivatives.
    :param LabelTensor input_: The input tensor with respect to which the
        laplacian is computed, with gradient tracking enabled.
    :param components: The names of the output variables for which to
        compute the laplacian. It must be a subset of the output labels.
        ``None`` or an empty selection uses all output labels. Default is
        ``None``.
    :type components: str | list[str] | None
    :param d: The names of the input variables with respect to which
        the laplacian is computed. It must be a subset of the input labels.
        ``None`` or an empty selection uses all input labels. Default is
        ``None``.
    :type d: str | list[str] | None
    :param str method: The method used to compute the Laplacian. Available
        methods are ``"std"`` for componentwise Hessian traces and
        ``"divgrad"`` for divergence of each component's gradient. For a
        one-column output, the scalar Hessian-trace path is always used and
        ``method`` is ignored, even if invalid. Default is ``"std"``.
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises ValueError: If a vector output uses an unsupported method or result
        labels cannot be assigned.
    :raises RuntimeError: If selected labels are missing or first or second
        differentiation fails, including constant first derivatives without a
        higher-order graph.
    :raises TypeError: If an unused gradient is returned as ``None``.
    :return: The componentwise Laplacian with shape
        ``(*input_.shape[:-1], len(components))`` and labels
        ``"dd{component}"``.
    :rtype: LabelTensor

    :Example:

        >>> from pina import LabelTensor
        >>> from pina.operator import laplacian
        >>> import torch
        >>> x = LabelTensor(torch.tensor([[1.0], [2.0]]), ["x"])
        >>> x = x.requires_grad_()
        >>> u = LabelTensor(x ** 3, labels=["u"])
        >>> second_derivatives = laplacian(u, x, components=["u"], d=["x"])
        >>> second_derivatives.labels
        ['ddu']
        >>> torch.allclose(second_derivatives.tensor, 6 * x.tensor)
        True
    """
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
    :param LabelTensor input_: The input tensor with respect to which advection
        is computed, with gradient tracking enabled and an autograd connection
        to the output.
    :param velocity_field: The name of the output variables used as velocity
        field. Names must belong to the output labels and be ordered to match
        ``d``, with exactly one velocity component per selected coordinate.
    :type velocity_field: str | list[str] | tuple[str, ...]
    :param components: The names of the output variables for which to compute
        the advection. It must be a subset of the output labels.
        ``None`` or an empty selection uses all output labels, including the
        velocity fields. Default is ``None``.
    :type components: str | list[str] | None
    :param d: The names of the input variables with respect to which the
        advection is computed. It must be a subset of the input labels.
        ``None`` or an empty selection uses all input labels. Default is
        ``None``.
    :type d: str | list[str] | None
    :raises TypeError: If the input tensor is not a LabelTensor.
    :raises TypeError: If the output tensor is not a LabelTensor.
    :raises RuntimeError: If the velocity field is not a subset of the output
        labels.
    :raises RuntimeError: If the dimensionality of the velocity field does not
        match ``len(d)``.
    :raises RuntimeError: If component or coordinate labels are missing, or
        differentiation, reshaping, or velocity broadcasting fails.
    :raises TypeError: If an unused gradient is returned as ``None``.
    :raises ValueError: If result labels cannot be assigned.
    :return: The velocity-weighted sum of component gradients with shape
        ``(*output_.shape[:-1], len(components))`` and labels
        ``"adv_{component}"``.
    :rtype: LabelTensor

    :Example:

        >>> from pina import LabelTensor
        >>> from pina.operator import advection
        >>> import torch
        >>> x = LabelTensor(torch.tensor([[1.0, 2.0]]), ["x", "y"])
        >>> x = x.requires_grad_()
        >>> field = (x ** 2).sum(-1, keepdim=True)
        >>> u = LabelTensor(
        ...     torch.cat([torch.ones_like(x.tensor), field], dim=-1),
        ...     labels=["vx", "vy", "u"],
        ... )
        >>> transported = advection(
        ...     u, x, velocity_field=["vx", "vy"], components=["u"]
        ... )
        >>> transported.labels
        ['adv_u']
        >>> expected = 2 * x.tensor.sum(-1, keepdim=True)
        >>> torch.allclose(transported.tensor, expected)
        True
    """
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
