"""Module for the Radial Field Network block."""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops
from pina._src.core.utils import check_positive_integer
from pina._src.model.feed_forward import FeedForward


class RadialFieldNetworkBlock(MessagePassing):
    """
    Implementation of the Radial Field Network block.

    This block is used to perform message-passing between nodes and edges in a
    graph neural network, following the scheme proposed by Köhler et al. in
    2020. It serves as an inner block in a larger graph neural network
    architecture.

    Each message multiplies the recipient-minus-sender feature difference by
    a learned scalar MLP of its norm. Self loops are removed before propagation.
    Messages are then aggregated using an aggregation scheme
    (e.g., sum, mean, min, max, or product).

    The update step is performed by a simple addition of the incoming messages
    to the node features.

    .. seealso::

        **Original reference** Köhler, J., Klein, L., Noé, F. (2020).
        *Equivariant Flows: Exact Likelihood Generative Learning for Symmetric
        Densities*.
        In International Conference on Machine Learning.
        DOI: `<https://doi.org/10.48550/arXiv.2006.02425>`_.

    :Example:

        >>> import torch
        >>> from pina.model.block.message_passing import (
        ...     RadialFieldNetworkBlock
        ... )
        >>> block = RadialFieldNetworkBlock(node_feature_dim=16)
        >>> x = torch.randn(10, 16)
        >>> edge_index = torch.randint(0, 10, (2, 30))
        >>> out = block(x, edge_index)
    """

    def __init__(
        self,
        node_feature_dim,
        hidden_dim=64,
        n_layers=2,
        activation=torch.nn.Tanh,
        aggr="add",
        node_dim=-2,
        flow="source_to_target",
    ):
        """
        Initialization of the :class:`RadialFieldNetworkBlock` class.

        :param int node_feature_dim: The dimension of the node features.
        :param int hidden_dim: The dimension of the hidden features.
            Default is ``64``.
        :param int n_layers: Hidden layer count. Default is ``2``.
        :param type[torch.nn.Module] activation: No-argument activation class.
            Default is :class:`torch.nn.Tanh`.
        :param str aggr: The aggregation scheme to use for message passing.
            Available options include ``"add"``, ``"mean"``, ``"min"``,
            ``"max"``, and ``"mul"``.
            See :class:`torch_geometric.nn.MessagePassing` for more details.
            Default is ``"add"``.
        :param int node_dim: The propagation axis. Default is ``-2``.
        :param str flow: The direction of message passing. Available options
            are ``"source_to_target"`` and ``"target_to_source"``.
            The ``"source_to_target"`` flow means that messages are sent from
            the source node to the target node, while ``"target_to_source"``
            flow means that messages are sent from the target node to the
            source node. See :class:`torch_geometric.nn.MessagePassing` for more
            details. Default is ``"source_to_target"``.
        :raises AssertionError: If ``node_feature_dim`` is not a positive
            integer.
        :raises AssertionError: If ``hidden_dim`` is not a positive integer.
        :raises AssertionError: If ``n_layers`` is not a positive integer.
        :raises ValueError: If activation or PyG propagation options are
            invalid.
        """
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)

        # Check values
        check_positive_integer(node_feature_dim, strict=True)
        check_positive_integer(hidden_dim, strict=True)
        check_positive_integer(n_layers, strict=True)

        # Layer for processing node features
        self.radial_net = FeedForward(
            input_dimensions=1,
            output_dimensions=1,
            inner_size=hidden_dim,
            n_layers=n_layers,
            func=activation,
        )

    def forward(self, x, edge_index):
        """
        Forward pass of the block, triggering the message-passing routine.

        :param x: Node features of shape ``(N, node_feature_dim)``. The radial
            norm uses axis one, so batched higher-rank inputs are not supported.
        :type x: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: Integer connectivity shaped ``(2, E)``.
        :return: Updated node features of shape ``(N, node_feature_dim)``.
        :rtype: torch.Tensor
        """
        edge_index, _ = remove_self_loops(edge_index)
        return self.propagate(edge_index=edge_index, x=x)

    def message(self, x_i, x_j):
        """
        Compute the message to be passed between nodes and edges.

        :param x_i: The node features of the recipient nodes.
        :type x_i: torch.Tensor | LabelTensor
        :param x_j: The node features of the sender nodes.
        :type x_j: torch.Tensor | LabelTensor
        :return: The message to be passed.
        :rtype: torch.Tensor
        """
        r = x_i - x_j
        return self.radial_net(torch.norm(r, dim=1, keepdim=True)) * r

    def update(self, message, x):
        """
        Update the node features with the received messages.

        :param torch.Tensor message: The message to be passed.
        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        return x + message
