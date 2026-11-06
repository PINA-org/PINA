"""Module for the Feed Forward model class."""

import torch
from torch import nn
from pina._src.core.utils import check_consistency
from pina._src.model.block.residual import EnhancedLinear


class FeedForward(torch.nn.Module):
    """
    Feed Forward neural network model class, also known as Multi-layer
    Perceptron.

    :Example:

        >>> import torch
        >>> from pina.model import FeedForward
        >>> model = FeedForward(input_dimensions=2, output_dimensions=1)
        >>> x = torch.randn(10, 2)
        >>> out = model(x)
        >>> out.shape
        torch.Size([10, 1])
    """

    def __init__(
        self,
        input_dimensions,
        output_dimensions,
        inner_size=20,
        n_layers=2,
        func=nn.Tanh,
        layers=None,
        bias=True,
    ):
        """
        Initialization of the :class:`FeedForward` class.

        :param int input_dimensions: The number of input components.
            The expected tensor shape is :math:`(*, d)`, where *
            represents any number of preceding dimensions (including none), and
            :math:`d` corresponds to ``input_dimensions``.
        :param int output_dimensions: The number of output components .
            The expected tensor shape is :math:`(*, d)`, where *
            represents any number of preceding dimensions (including none), and
            :math:`d` corresponds to ``output_dimensions``.
        :param int inner_size: The number of neurons for each hidden layer.
            Default is ``20``.
        :param int n_layers: The number of hidden layers. Default is ``2``.
        :param func: No-argument activation classes or factories. A list must
            match the actual hidden width count, including when ``layers``
            overrides ``n_layers``. A single factory is reused for hidden
            layers only; ``None`` omits activation.
            Default is :class:`torch.nn.Tanh`.
        :type func: type[torch.nn.Module] | list[type[torch.nn.Module] | None] |
            None
        :param layers: The list of the dimension of inner layers.
            If ``None``, ``n_layers`` of dimension ``inner_size`` are used.
            Otherwise, it overrides the values passed to ``n_layers`` and
            ``inner_size``. Default is ``None``.
        :type layers: list[int] | None
        :param bool bias: If ``True`` bias is considered for the basis function
            neural network. Default is ``True``.
        :raises ValueError: If the input dimension is not an integer.
        :raises ValueError: If the output dimension is not an integer.
        :raises RuntimeError: If the number of layers and functions are
            inconsistent.
        """
        super().__init__()

        if not isinstance(input_dimensions, int):
            raise ValueError("input_dimensions expected to be int.")
        self.input_dimension = input_dimensions

        if not isinstance(output_dimensions, int):
            raise ValueError("output_dimensions expected to be int.")
        self.output_dimension = output_dimensions
        if layers is None:
            layers = [inner_size] * n_layers

        tmp_layers = layers.copy()
        tmp_layers.insert(0, self.input_dimension)
        tmp_layers.append(self.output_dimension)

        self.layers = []
        for i in range(len(tmp_layers) - 1):
            self.layers.append(
                nn.Linear(tmp_layers[i], tmp_layers[i + 1], bias=bias)
            )

        if isinstance(func, list):
            self.functions = func
        else:
            self.functions = [func for _ in range(len(self.layers) - 1)]

        if len(self.layers) != len(self.functions) + 1:
            raise RuntimeError("Incosistent number of layers and functions")

        unique_list = []
        for layer, func_ in zip(self.layers[:-1], self.functions):
            unique_list.append(layer)
            if func_ is not None:
                unique_list.append(func_())
        unique_list.append(self.layers[-1])

        self.model = nn.Sequential(*unique_list)

    def forward(self, x):
        """
        Forward pass for the :class:`FeedForward` model.

        :param x: The input tensor with shape ``(..., input_dimensions)``.
        :type x: torch.Tensor | LabelTensor
        :raises RuntimeError: If input dimensions or dtype do not match layers.
        :return: Predictions of shape ``(..., output_dimensions)``. Leading
            dimensions are preserved; output labels are not assigned here.
        :rtype: torch.Tensor | LabelTensor
        """
        return self.model(x)


class ResidualFeedForward(torch.nn.Module):
    """
    Residual Feed Forward neural network model class.

    The model is composed of a series of linear layers with a residual
    connection between themm as presented in the following:

    .. seealso::

        **Original reference**: Wang, S., Teng, Y., and Perdikaris, P. (2021).
        *Understanding and mitigating gradient flow pathologies in
        physics-informed neural networks*.
        SIAM Journal on Scientific Computing 43.5 (2021): A3055-A3081.
        DOI: `10.1137/20M1318043
        <https://epubs.siam.org/doi/abs/10.1137/20M1318043>`_

    :Example:

        >>> import torch
        >>> from pina.model import ResidualFeedForward
        >>> model = ResidualFeedForward(
        ...     input_dimensions=2, output_dimensions=1
        ... )
        >>> x = torch.randn(10, 2)
        >>> out = model(x)
    """

    def __init__(
        self,
        input_dimensions,
        output_dimensions,
        inner_size=20,
        n_layers=2,
        func=nn.Tanh,
        bias=True,
        transformer_nets=None,
    ):
        """
        Initialization of the :class:`ResidualFeedForward` class.

        :param int input_dimensions: The number of input components.
            The expected tensor shape is :math:`(*, d)`, where *
            represents any number of preceding dimensions (including none), and
            :math:`d` corresponds to ``input_dimensions``.
        :param int output_dimensions: The number of output components .
            The expected tensor shape is :math:`(*, d)`, where *
            represents any number of preceding dimensions (including none), and
            :math:`d` corresponds to ``output_dimensions``.
        :param int inner_size: The number of neurons for each hidden layer.
            Default is ``20``.
        :param int n_layers: The number of hidden layers. Default is ``2``.
        :param type[torch.nn.Module] func: No-argument activation class used in
            each hidden layer.
            Lists pass element validation but fail construction when called.
            Default is :class:`torch.nn.Tanh`.
        :param bool bias: If ``True`` bias is considered for the basis function
            neural network. Default is ``True``.
        :param transformer_nets: The two :class:`torch.nn.Module` acting as
            transformer network. The input dimension of both networks must be
            equal to ``input_dimensions``, and the output dimension must be
            equal to ``inner_size``. If ``None``, two
            :class:`~pina.model.block.EnhancedLinear` layers with Tanh are used,
            independently of ``func``.
            Default is ``None``.
        :type transformer_nets: list[torch.nn.Module] |
            tuple[torch.nn.Module, torch.nn.Module] | None
        :raises ValueError: If argument types or transformer dimensions fail
            validation. Custom transformers are probed with random CPU inputs
            during construction, potentially updating their training state.
        :raises RuntimeError: If the number of layers and functions are
            inconsistent.
        """
        super().__init__()

        # check type consistency
        check_consistency(input_dimensions, int)
        check_consistency(output_dimensions, int)
        check_consistency(inner_size, int)
        check_consistency(n_layers, int)
        check_consistency(func, torch.nn.Module, subclass=True)
        check_consistency(bias, bool)

        transformer_nets = self._check_transformer_nets(
            transformer_nets, input_dimensions, inner_size
        )

        # assign variables
        self.transformer_nets = nn.ModuleList(transformer_nets)

        # build layers
        layers = [inner_size] * n_layers

        layers = layers.copy()
        layers.insert(0, input_dimensions)

        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1], bias=bias))
        self.last_layer = nn.Linear(
            layers[len(layers) - 1], output_dimensions, bias=bias
        )

        if isinstance(func, list):
            self.functions = func()
        else:
            self.functions = [func() for _ in range(len(self.layers))]

        if len(self.layers) != len(self.functions):
            raise RuntimeError("Incosistent number of layers and functions")

        unique_list = []
        for layer, func_ in zip(self.layers, self.functions):
            unique_list.append(EnhancedLinear(layer=layer, activation=func_))
        self.inner_layers = torch.nn.Sequential(*unique_list)

    def forward(self, x):
        """
        Forward pass for the :class:`ResidualFeedForward` model.

        :param x: The input tensor of shape ``(..., input_dimensions)``.
        :type x: torch.Tensor | LabelTensor
        :raises RuntimeError: If input or gating shapes cannot be combined.
        :return: Predictions of shape ``(..., output_dimensions)``.
        :rtype: torch.Tensor | LabelTensor
        """
        # enhance the input with transformer
        input_ = []
        for nets in self.transformer_nets:
            input_.append(nets(x))

        # skip connections pass
        for layer in self.inner_layers.children():
            x = layer(x)
            x = (1.0 - x) * input_[0] + x * input_[1]

        # last layer
        return self.last_layer(x)

    @staticmethod
    def _check_transformer_nets(transformer_nets, input_dimensions, inner_size):
        """
        Check the transformer networks consistency.

        :param transformer_nets: The two :class:`torch.nn.Module` acting as
            transformer network. ``None`` creates two default Linear/Tanh
            networks. Custom networks are probed on ``(10, input_dimensions)``
            random inputs without disabling gradients or training mode.
        :type transformer_nets: list[torch.nn.Module] |
            tuple[torch.nn.Module, torch.nn.Module] | None
        :param int input_dimensions: The number of input components.
        :param int inner_size: The number of neurons for each hidden layer.
        :raises ValueError: If the passed ``transformer_nets`` is not a list of
            length two.
        :raises ValueError: If the passed ``transformer_nets`` is not a list of
            :class:`torch.nn.Module`.
        :raises ValueError: If the input dimension of the transformer network
            is incompatible with the input dimension of the model.
        :raises ValueError: If the output dimension of the transformer network
            is incompatible with the inner size of the model.
        :raises RuntimeError: If the container is neither a list, tuple, nor
            ``None``.
        :return: The two :class:`torch.nn.Module` acting as transformer network.
        :rtype: list[torch.nn.Module] | tuple[torch.nn.Module, torch.nn.Module]
        """
        # check transformer nets
        if transformer_nets is None:
            transformer_nets = [
                EnhancedLinear(
                    nn.Linear(
                        in_features=input_dimensions, out_features=inner_size
                    ),
                    nn.Tanh(),
                ),
                EnhancedLinear(
                    nn.Linear(
                        in_features=input_dimensions, out_features=inner_size
                    ),
                    nn.Tanh(),
                ),
            ]
        elif isinstance(transformer_nets, (list, tuple)):
            if len(transformer_nets) != 2:
                raise ValueError(
                    "transformer_nets needs to be a list of len two."
                )
            for net in transformer_nets:
                if not isinstance(net, nn.Module):
                    raise ValueError(
                        "transformer_nets needs to be a list of "
                        "torch.nn.Module."
                    )
                x = torch.rand(10, input_dimensions)
                try:
                    out = net(x)
                except RuntimeError as e:
                    raise ValueError(
                        "transformer network input incompatible with "
                        "input_dimensions."
                    ) from e
                if out.shape[-1] != inner_size:
                    raise ValueError(
                        "transformer network output incompatible with "
                        "inner_size."
                    )
        else:
            raise RuntimeError(
                "Runtime error for transformer nets, check official "
                "documentation."
            )
        return transformer_nets
