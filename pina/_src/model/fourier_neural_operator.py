"""Module for the Fourier Neural Operator model class."""

import warnings
import torch
from torch import nn
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency
from pina._src.model.block.fourier_block import (
    FourierBlock1D,
    FourierBlock2D,
    FourierBlock3D,
)
from pina._src.model.kernel_neural_operator import KernelNeuralOperator


class FourierIntegralKernel(torch.nn.Module):
    """
    Fourier Integral Kernel model class.

    This class implements the Fourier Integral Kernel network, which
    performs global convolution in the Fourier space.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K., Liu,
        B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_

    :Example:

        >>> import torch
        >>> from pina.model import (
        ...     FourierIntegralKernel
        ... )
        >>> model = FourierIntegralKernel(
        ...     input_numb_fields=2, output_numb_fields=2, n_modes=16,
        ...     dimensions=1
        ... )
        >>> x = torch.randn(10, 50, 2)
        >>> out = model(x)
    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        dimensions=3,
        padding=8,
        padding_type="constant",
        inner_size=20,
        n_layers=2,
        func=nn.Tanh,
        layers=None,
    ):
        """
        Initialization of the :class:`FourierIntegralKernel` class.

        :param int input_numb_fields: The number of input fields.
        :param int output_numb_fields: Requested output width. The current
            construction does not use it: the final width is ``layers[-1]``.
        :param n_modes: Retained modes per spatial axis, as a scalar, an
            axis-wise list, or a list of axis-wise lists per block. Counts must
            fit the padded Fourier grid. Tuples are not normalized like lists.
        :type n_modes: int | list[int] | list[list[int]] | tuple[int, ...]
        :param int dimensions: The number of dimensions. It can be set to ``1``,
            ``2``, or ``3``. Default is ``3``.
        :param padding: Right-side padding per axis, in PyTorch's reversed
            spatial-axis order. Default is ``8``.
        :type padding: int | list[int] | tuple[int, ...]
        :param str padding_type: PyTorch padding mode, such as ``"constant"``,
            ``"reflect"``, ``"replicate"``, or ``"circular"``.
            Default is ``"constant"``.
        :param int inner_size: The inner size. Default is ``20``.
        :param int n_layers: The number of layers. Default is ``2``.
        :param func: No-argument activation classes. A list must match the
            number of block widths and is mutated by appending Identity; its
            original entries activate all blocks. A single class activates all
            except the final block, which uses Identity.
            Default is :class:`torch.nn.Tanh`.
        :type func: type[torch.nn.Module] | list[type[torch.nn.Module]]
        :param layers: The output widths of successive Fourier blocks.
            If ``None``, ``n_layers`` of dimension ``inner_size`` are used.
            Otherwise, it overrides the values passed to ``n_layers`` and
            ``inner_size``. Default is ``None``.
        :type layers: list[int] | tuple[int, ...] | None
        :raises RuntimeError: If the number of layers and functions are
            inconsistent.
        :raises RuntimeError: If the number of layers and modes are
            inconsistent.
        :raises ValueError: If argument types fail validation.
        :raises NotImplementedError: If dimensions are not one, two, or three.

        .. note::

            Although tuples pass width validation, construction concatenates
            ``layers`` with lists and therefore requires a list in practice.
        """
        super().__init__()

        # check type consistency
        self._check_consistency(
            dimensions,
            padding,
            padding_type,
            inner_size,
            n_layers,
            func,
            layers,
            n_modes,
        )

        # assign padding
        self._padding = padding

        # initialize fourier layer for each dimension
        fourier_layer = self._get_fourier_block(dimensions)

        # Here we build the FNO kernels by stacking Fourier Blocks

        # 1. Assign output dimensions for each FNO layer
        if layers is None:
            layers = [inner_size] * n_layers

        # 2. Assign activation functions for each FNO layer
        if isinstance(func, list):
            if len(layers) != len(func):
                raise RuntimeError(
                    "Inconsistent number of layers and functions."
                )
            _functions = func
        else:
            _functions = [func for _ in range(len(layers) - 1)]
        _functions.append(torch.nn.Identity)

        # 3. Assign modes functions for each FNO layer
        if isinstance(n_modes, list):
            if all(isinstance(i, list) for i in n_modes) and len(layers) != len(
                n_modes
            ):
                raise RuntimeError("Inconsistent number of layers and modes.")
            if all(isinstance(i, int) for i in n_modes):
                n_modes = [n_modes] * len(layers)
        else:
            n_modes = [n_modes] * len(layers)

        # 4. Build the FNO network
        tmp_layers = [input_numb_fields] + layers + [output_numb_fields]
        self._layers = nn.Sequential(
            *[
                fourier_layer(
                    input_numb_fields=tmp_layers[i],
                    output_numb_fields=tmp_layers[i + 1],
                    n_modes=n_modes[i],
                    activation=_functions[i],
                )
                for i in range(len(layers))
            ]
        )

        # 5. Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding] * dimensions
        self._ipad = [-pad if pad > 0 else None for pad in padding[:dimensions]]
        self._padding_type = padding_type
        self._pad = [
            val for pair in zip([0] * dimensions, padding) for val in pair
        ]

    def forward(self, x):
        """
        Forward pass for the :class:`FourierIntegralKernel` model.

        :param x: The input tensor for performing the computation. Depending
            on the ``dimensions`` in the initialization, it expects a tensor
            with the following shapes:

            * 1D tensors: ``[batch, X, channels]``
            * 2D tensors: ``[batch, X, Y, channels]``
            * 3D tensors: ``[batch, X, Y, Z, channels]``
        :type x: torch.Tensor | LabelTensor
        :return: Plain tensor with the input spatial extents and final channel
            width ``layers[-1]`` (or ``inner_size`` with default widths).
        :rtype: torch.Tensor

        .. warning::

            Labeled input emits a warning and is cast to a plain tensor.
        """
        if isinstance(x, LabelTensor):
            warnings.warn(
                "LabelTensor passed as input is not allowed,"
                " casting LabelTensor to Torch.Tensor"
            )
            x = x.as_subclass(torch.Tensor)
        # permuting the input [batch, channels, x, y, ...]
        permutation_idx = [0, x.ndim - 1, *list(range(1, x.ndim - 1))]
        x = x.permute(permutation_idx)

        # padding the input
        x = torch.nn.functional.pad(x, pad=self._pad, mode=self._padding_type)

        # apply fourier layers
        x = self._layers(x)

        # remove padding
        idxs = [slice(None), slice(None)] + [slice(pad) for pad in self._ipad]
        x = x[idxs]

        # permuting back [batch, x, y, ..., channels]
        permutation_idx = [0, *list(range(2, x.ndim)), 1]
        x = x.permute(permutation_idx)

        return x

    @staticmethod
    def _check_consistency(
        dimensions,
        padding,
        padding_type,
        inner_size,
        n_layers,
        func,
        layers,
        n_modes,
    ):
        """
        Check the consistency of the input parameters.


        :param int dimensions: The number of dimensions.
        :param padding: The padding size or axis-wise sizes.
        :type padding: int | list[int] | tuple[int, ...]
        :param str padding_type: The padding strategy.
        :param int inner_size: The inner size.
        :param int n_layers: The number of layers.
        :param func: The activation classes.
        :type func: type[torch.nn.Module] | list[type[torch.nn.Module]]
        :param layers: Explicit block widths, or ``None`` for inferred widths.
        :type layers: list[int] | tuple[int, ...] | None
        :param n_modes: The number of modes.
        :type n_modes: int | list[int] | list[list[int]] | tuple[int, ...]
        :raises ValueError: If the input is not consistent.
        """
        check_consistency(dimensions, int)
        check_consistency(padding, int)
        check_consistency(padding_type, str)
        check_consistency(inner_size, int)
        check_consistency(n_layers, int)
        check_consistency(func, nn.Module, subclass=True)

        if layers is not None:
            if isinstance(layers, (tuple, list)):
                check_consistency(layers, int)
            else:
                raise ValueError("layers must be tuple or list of int.")
        if not isinstance(n_modes, (list, tuple, int)):
            raise ValueError(
                "n_modes must be a int or list or tuple of valid modes."
                " More information on the official documentation."
            )

    @staticmethod
    def _get_fourier_block(dimensions):
        """
        Retrieve the Fourier Block class based on the number of dimensions.

        :param int dimensions: The number of dimensions.
        :raises NotImplementedError: If the number of dimensions is not 1, 2,
            or 3.
        :return: The Fourier Block class.
        :rtype: type[FourierBlock1D] | type[FourierBlock2D] |
            type[FourierBlock3D]
        """
        if dimensions == 1:
            return FourierBlock1D
        if dimensions == 2:
            return FourierBlock2D
        if dimensions == 3:
            return FourierBlock3D
        raise NotImplementedError("FNO implemented only for 1D/2D/3D data.")


class FNO(KernelNeuralOperator):
    """
    Fourier Neural Operator model class.

    The Fourier Neural Operator (FNO) is a general architecture for learning
    operators, which  map functions to functions. It can be trained both with
    Supervised and Physics_Informed learning strategies. The Fourier Neural
    Operator performs global convolution in the Fourier space.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
        *Fourier neural operator for parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_

    :Example:

        >>> import torch
        >>> from pina.model import FNO, FeedForward
        >>> lifting_net = FeedForward(input_dimensions=3, output_dimensions=20)
        >>> projecting_net = FeedForward(20, 1)
        >>> model = FNO(
        ...     lifting_net=lifting_net, projecting_net=projecting_net,
        ...     n_modes=4, dimensions=2
        ... )
        >>> x = torch.randn(10, 2, 50, 3)
        >>> out = model(x)
    """

    def __init__(
        self,
        lifting_net,
        projecting_net,
        n_modes,
        dimensions=3,
        padding=8,
        padding_type="constant",
        inner_size=20,
        n_layers=2,
        func=nn.Tanh,
        layers=None,
    ):
        """
        Initialization of the :class:`FNO` class.

        :param torch.nn.Module lifting_net: The lifting neural network mapping
            the input to its hidden dimension. It must have parameters; a
            random CPU tensor shaped like its first parameter probes its output
            width during construction, without disabling training or autograd.
        :param torch.nn.Module projecting_net: The projection neural network
            mapping the hidden representation to the output function.
        :param n_modes: Scalar, axis-wise, or per-block axis-wise mode counts,
            as for :class:`FourierIntegralKernel`.
        :type n_modes: int | list[int] | list[list[int]] | tuple[int, ...]
        :param int dimensions: The number of dimensions. It can be set to ``1``,
            ``2``, or ``3``. Default is ``3``.
        :param padding: Right-side padding in reversed spatial-axis order.
            Default is ``8``.
        :type padding: int | list[int] | tuple[int, ...]
        :param str padding_type: PyTorch padding mode.
            Default is ``"constant"``.
        :param int inner_size: The inner size. Default is ``20``.
        :param int n_layers: The number of layers. Default is ``2``.
        :param func: Activation classes for the Fourier blocks. A list must
            match the block count and is mutated by appending Identity. A
            single class uses Identity in the last block instead.
            Default is :class:`torch.nn.Tanh`.
        :type func: type[torch.nn.Module] | list[type[torch.nn.Module]]
        :param layers: The output widths of successive Fourier blocks.
            If ``None``, ``n_layers`` of dimension ``inner_size`` are used.
            Otherwise, it overrides the values passed to ``n_layers`` and
            ``inner_size``. Default is ``None``.
        :type layers: list[int] | None
        :raises ValueError: If kernel argument types fail validation.
        :raises RuntimeError: If activation or mode counts mismatch block
            widths.
        :raises NotImplementedError: If dimensions are not one, two, or three.

        .. note::

            The projection must accept the final block width. Its parameter
            shape does not set the kernel's output width.
        """
        lifting_operator_out = lifting_net(
            torch.rand(size=next(lifting_net.parameters()).size())
        ).shape[-1]
        super().__init__(
            lifting_operator=lifting_net,
            projection_operator=projecting_net,
            integral_kernels=FourierIntegralKernel(
                input_numb_fields=lifting_operator_out,
                output_numb_fields=next(projecting_net.parameters()).size(),
                n_modes=n_modes,
                dimensions=dimensions,
                padding=padding,
                padding_type=padding_type,
                inner_size=inner_size,
                n_layers=n_layers,
                func=func,
                layers=layers,
            ),
        )

    def forward(self, x):
        """
        Forward pass for the :class:`FNO` model.

        The ``lifting_net`` maps the input to the hidden dimension.
        Then, several layers of Fourier blocks are applied. Finally, the
        ``projection_net`` maps the hidden representation to the output
        function.

        :param x: The input tensor for performing the computation. Depending
            on the ``dimensions`` in the initialization, it expects a tensor
            with the following shapes:

            * 1D tensors: ``[batch, X, channels]``
            * 2D tensors: ``[batch, X, Y, channels]``
            * 3D tensors: ``[batch, X, Y, Z, channels]``

        :type x: torch.Tensor | LabelTensor
        :return: Plain tensor with input spatial extents and output channels
            determined by the projection network. Input labels are discarded.
        :rtype: torch.Tensor
        """

        if isinstance(x, LabelTensor):
            x = x.as_subclass(torch.Tensor)
        return super().forward(x)
