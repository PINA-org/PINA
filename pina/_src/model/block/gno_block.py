"""Module for the Graph Neural Operator Block class."""

import torch
from torch_geometric.nn import MessagePassing


class GNOBlock(MessagePassing):
    """
    The inner block of the Graph Neural Operator, based on Message Passing.

    :Example:

        >>> import torch
        >>> from pina.model.block import GNOBlock
        >>> block = GNOBlock(width=16, edges_features=4)
        >>> x = torch.randn(10, 16)
        >>> edge_index = torch.randint(0, 10, (2, 30))
        >>> edge_attr = torch.randn(30, 4)
        >>> out = block(x, edge_index, edge_attr)
    """

    def __init__(
        self,
        width,
        edges_features,
        n_layers=2,
        layers=None,
        inner_size=None,
        internal_func=None,
        external_func=None,
    ):
        """
        Initialization of the :class:`GNOBlock` class.

        :param int width: The width of the kernel.
        :param int edge_features: The number of edge features.
        :param int n_layers: The number of kernel layers. Default is ``2``.
        :param layers: A list specifying the number of neurons for each layer
            of the neural network. If not ``None``, it overrides the
            ``inner_size`` and ``n_layers``parameters. Default is ``None``.
        :type layers: list[int] | tuple[int]
        :param int inner_size: The size of the inner layer. Default is ``None``.
        :param torch.nn.Module internal_func: The activation function applied to
            the output of each layer. If ``None``, it uses the
            :class:`torch.nn.Tanh` activation. Default is ``None``.
        :param torch.nn.Module external_func: The activation function applied to
            the output of the block. If ``None``, it uses the
            :class:`torch.nn.Tanh`. activation. Default is ``None``.
        """

        from ...model.feed_forward import FeedForward

        super().__init__(aggr="mean")  # Uses PyG's default aggregation
        self.width = width

        if layers is None and inner_size is None:
            inner_size = width

        self.dense = FeedForward(
            input_dimensions=edges_features,
            output_dimensions=width**2,
            n_layers=n_layers,
            layers=layers,
            inner_size=inner_size,
            func=internal_func,
        )

        self.W = torch.nn.Linear(width, width)
        self.func = external_func()

    def message(self, x_j, edge_attr):
        """
        Construct the messages sent along each edge.

        The edge features are mapped to a ``width x width`` matrix that
        multiplies the source node features. This is the non-fused
        counterpart of :meth:`message_and_aggregate`, which is used when a
        plain :obj:`torch.Tensor` edge index is passed to ``propagate``.

        :param torch.Tensor x_j: The source node features of each edge.
        :param torch.Tensor edge_attr: The edge features.
        :return: The messages for each edge.
        :rtype: torch.Tensor
        """
        x_ = self.dense(edge_attr).view(-1, self.width, self.width)
        return torch.einsum("bij,bj->bi", x_, x_j)

    def message_and_aggregate(self, edge_index, x, edge_attr):
        """
        Combine messages and perform aggregation.

        This fused implementation is used when the fused message passing
        path is available (e.g. with a sorted edge index or a sparse
        adjacency). It is equivalent to :meth:`message` followed by
        aggregation.

        :param torch.Tensor edge_index: The edge index.
        :param torch.Tensor x: The node feature matrix.
        :param torch.Tensor edge_attr: The edge features.
        :return: The aggregated messages.
        :rtype: torch.Tensor
        """
        # Edge features are transformed into a matrix of shape
        # [num_edges, width, width]
        x_ = self.dense(edge_attr).view(-1, self.width, self.width)
        # Messages are computed as the product of the edge features
        messages = torch.einsum("bij,bj->bi", x_, x[edge_index[0]])
        # Aggregation is performed using the mean (set in the constructor)
        return self.aggregate(messages, edge_index[1])

    def edge_update(self, edge_attr):
        """
        Update edge features.

        :param torch.Tensor edge_attr: The edge features.
        :return: The updated edge features.
        :rtype: torch.Tensor
        """
        return edge_attr

    def update(self, aggr_out, x):
        """
        Update node features.

        :param torch.Tensor aggr_out: The aggregated messages.
        :param torch.Tensor x: The node feature matrix.
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        return aggr_out + self.W(x)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the block.

        :param torch.Tensor x: The node features.
        :param torch.Tensor edge_index: The edge indeces.
        :param torch.Tensor edge_attr: The edge features.
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        return self.func(self.propagate(edge_index, x=x, edge_attr=edge_attr))
