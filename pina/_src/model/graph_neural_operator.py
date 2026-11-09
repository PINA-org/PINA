"""Module for the Graph Neural Operator model class."""

import torch
from torch.nn import Tanh
from pina._src.model.block.gno_block import GNOBlock
from pina._src.model.kernel_neural_operator import KernelNeuralOperator


class GraphNeuralKernel(torch.nn.Module):
    """
    Graph Neural Operator kernel model class.

    This class implements the Graph Neural Operator kernel network.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A. (2020).
        *Neural Operator: Graph Kernel Network for Partial Differential
        Equations*.
        DOI: `arXiv preprint arXiv:2003.03485
        <https://arxiv.org/abs/2003.03485>`_

    :Example:

        >>> import torch
        >>> from pina._src.model.graph_neural_operator import (
        ...     GraphNeuralKernel
        ... )
        >>> model = GraphNeuralKernel(width=16, edge_features=4)
        >>> x = torch.randn(10, 16)
        >>> edge_index = torch.randint(0, 10, (2, 20))
        >>> edge_attr = torch.randn(20, 4)
        >>> out = model(x, edge_index, edge_attr)
    """

    def __init__(
        self,
        width,
        edge_features,
        n_layers=2,
        internal_n_layers=0,
        internal_layers=None,
        inner_size=None,
        internal_func=None,
        external_func=None,
        shared_weights=False,
    ):
        """
        Initialization of the :class:`GraphNeuralKernel` class.

        :param int width: The width of the kernel.
        :param int edge_features: The number of edge features.
        :param int n_layers: The number of kernel layers. Default is ``2``.
        :param int internal_n_layers: The number of layers of the neural network
            inside each kernel layer. Default is ``0``.
        :param internal_layers: The number of neurons for each layer of the
            neural network inside each kernel layer. ``None`` uses
            ``internal_n_layers`` and ``inner_size``. Default is ``None``.
        :type internal_layers: list[int] | tuple[int, ...] | None
        :param inner_size: Hidden width of the edge network. ``None`` uses
            ``width`` unless explicit widths are supplied. Default is ``None``.
        :type inner_size: int | None
        :param internal_func: The activation class used
            inside each kernel layer. If ``None``, it uses the
            :class:`torch.nn.Tanh` activation. Default is ``None``.
        :type internal_func: type[torch.nn.Module] | None
        :param external_func: The activation class applied to
            the output of the each kernel layer. If ``None``, it uses the
            :class:`torch.nn.Tanh` activation. Default is ``None``.
        :type external_func: type[torch.nn.Module] | None
        :param bool shared_weights: If ``True``, the weights of each kernel
            layer are shared. Default is ``False``.
        """
        super().__init__()
        if external_func is None:
            external_func = Tanh
        if internal_func is None:
            internal_func = Tanh

        if shared_weights:
            self.layers = GNOBlock(
                width=width,
                edges_features=edge_features,
                n_layers=internal_n_layers,
                layers=internal_layers,
                inner_size=inner_size,
                internal_func=internal_func,
                external_func=external_func,
            )
            self.n_layers = n_layers
            self._forward_func = self._forward_shared
        else:
            self.layers = torch.nn.ModuleList(
                [
                    GNOBlock(
                        width=width,
                        edges_features=edge_features,
                        n_layers=internal_n_layers,
                        layers=internal_layers,
                        inner_size=inner_size,
                        internal_func=internal_func,
                        external_func=external_func,
                    )
                    for _ in range(n_layers)
                ]
            )
            self._forward_func = self._forward_unshared

    def _forward_unshared(self, x, edge_index, edge_attr):
        """
        Forward pass for the Graph Neural Kernel with unshared weights.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge index.
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x

    def _forward_shared(self, x, edge_index, edge_attr):
        """
        Forward pass for the Graph Neural Kernel with shared weights.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge index.
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        for _ in range(self.n_layers):
            x = self.layers(x, edge_index, edge_attr)
        return x

    def forward(self, x, edge_index, edge_attr):
        """
        The forward pass of the Graph Neural Kernel.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge index.
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self._forward_func(x, edge_index, edge_attr)


class GraphNeuralOperator(KernelNeuralOperator):
    """
    Graph Neural Operator model class.

    The Graph Neural Operator is a general architecture for learning operators,
    which map functions to functions. It can be trained both with Supervised
    and Physics-Informed learning strategies. The Graph Neural Operator performs
    graph convolution by means of a Graph Neural Kernel.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K.,
        Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A. (2020).
        *Neural Operator: Graph Kernel Network for Partial Differential
        Equations*.
        DOI: `arXiv preprint arXiv:2003.03485.
        <https://arxiv.org/abs/2003.03485>`_

    :Example:

        >>> import torch
        >>> from pina.model import GraphNeuralOperator, FeedForward
        >>> lifting_net = torch.nn.Linear(3, 16)
        >>> projecting_net = FeedForward(16, 1)
        >>> model = GraphNeuralOperator(
        ...     lifting_operator=lifting_net,
        ...     projection_operator=projecting_net,
        ...     edge_features=4
        ... )
    """

    def __init__(
        self,
        lifting_operator,
        projection_operator,
        edge_features,
        n_layers=10,
        internal_n_layers=0,
        inner_size=None,
        internal_layers=None,
        internal_func=None,
        external_func=None,
        shared_weights=True,
    ):
        """
        Initialization of the :class:`GraphNeuralOperator` class.

        :param torch.nn.Module lifting_operator: The lifting neural network
            mapping the input to its hidden dimension. Its ``out_features``
            attribute must give the kernel width, as for ``torch.nn.Linear``.
        :param torch.nn.Module projection_operator: The projection neural
            network mapping the hidden representation to the output function.
        :param int edge_features: The number of edge features.
        :param int n_layers: The number of kernel layers. Default is ``10``.
        :param int internal_n_layers: The number of layers of the neural network
            inside each kernel layer. Default is ``0``.
        :param inner_size: Hidden width of the edge network. ``None`` uses the
            lifting output width unless explicit widths are given.
            Default is ``None``.
        :type inner_size: int | None
        :param internal_layers: The number of neurons for each layer of the
            neural network inside each kernel layer. ``None`` uses
            ``internal_n_layers`` and ``inner_size``. Default is ``None``.
        :type internal_layers: list[int] | tuple[int, ...] | None
        :param internal_func: The activation class used
            inside each kernel layer. If ``None``, it uses the
            :class:`torch.nn.Tanh` activation. Default is ``None``.
        :type internal_func: type[torch.nn.Module] | None
        :param external_func: The activation class applied to
            the output of the each kernel layer. If ``None``, it uses the
            :class:`torch.nn.Tanh` activation. Default is ``None``.
        :type external_func: type[torch.nn.Module] | None
        :param bool shared_weights: If ``True``, the weights of each kernel
            layer are shared. Default is ``True``.
        """

        if internal_func is None:
            internal_func = Tanh
        if external_func is None:
            external_func = Tanh

        super().__init__(
            lifting_operator=lifting_operator,
            integral_kernels=GraphNeuralKernel(
                width=lifting_operator.out_features,
                edge_features=edge_features,
                internal_n_layers=internal_n_layers,
                inner_size=inner_size,
                internal_layers=internal_layers,
                external_func=external_func,
                internal_func=internal_func,
                n_layers=n_layers,
                shared_weights=shared_weights,
            ),
            projection_operator=projection_operator,
        )

    def forward(self, x):
        """
        The forward pass of the Graph Neural Operator.

        :param x: A graph with node features ``(N, input_width)``, integer
            connectivity ``(2, E)``, and edge features ``(E, edge_features)``.
        :type x: pina.Graph | torch_geometric.data.Data
        :return: Projected node features of shape ``(N, output_width)``; the
            input graph is not modified.
        :rtype: torch.Tensor | LabelTensor
        """
        x, edge_index, edge_attr = x.x, x.edge_index, x.edge_attr
        x = self.lifting_operator(x)
        x = self.integral_kernels(x, edge_index, edge_attr)
        x = self.projection_operator(x)
        return x
