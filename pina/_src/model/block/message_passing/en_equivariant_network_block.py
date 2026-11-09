"""Module for the E(n) Equivariant Graph Neural Network block."""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from pina._src.core.utils import check_positive_integer, check_consistency
from pina._src.model.feed_forward import FeedForward


class EnEquivariantNetworkBlock(MessagePassing):
    """
    Implementation of the E(n) Equivariant Graph Neural Network block.
    This block is used to perform message-passing between nodes and edges in a
    graph neural network, following the scheme proposed by Satorras et al. in
    2021. It serves as an inner block in a larger graph neural network
    architecture.

    The message between two nodes connected by an edge is computed by applying a
    multi-layer perceptron to sender and recipient node features and edge data,
    together with the squared euclidean distance between the sender and
    recipient node positions, followed by a non-linear activation function.
    Messages are then aggregated using an aggregation scheme (e.g., sum, mean,
    min, max, or product).

    The update step is performed by applying another MLP to the concatenation of
    the incoming messages and the node features. Here, also the node
    positions are updated by adding the incoming messages divided by the
    degree of the recipient node.

    When velocities are used, a scalar MLP of the incoming node features scales
    each velocity. Position updates add this scaled velocity and the normalized
    position messages. Degree normalization always uses ``edge_index[1]``, even
    for reversed message flow.

    .. seealso::

        **Original reference** Satorras, V. G., Hoogeboom, E., Welling, M.
        (2021). *E(n) Equivariant Graph Neural Networks.*
        In International Conference on Machine Learning.
        DOI: `<https://doi.org/10.48550/arXiv.2102.09844>`_.

    :Example:

        >>> import torch
        >>> from pina.model.block.message_passing import (
        ...     EnEquivariantNetworkBlock
        ... )
        >>> block = EnEquivariantNetworkBlock(
        ...     node_feature_dim=16, edge_feature_dim=4, pos_dim=3
        ... )
    """

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        pos_dim,
        use_velocity=False,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        activation=torch.nn.SiLU,
        aggr="add",
        node_dim=-2,
        flow="source_to_target",
    ):
        """
        Initialization of the :class:`EnEquivariantNetworkBlock` class.

        :param int node_feature_dim: The dimension of the node features.
        :param int edge_feature_dim: The dimension of the edge features.
        :param int pos_dim: The dimension of the position features.
        :param bool use_velocity: Whether to use velocity features in the
            message passing. Default is ``False``.
        :param int hidden_dim: The dimension of the hidden features.
            Default is ``64``.
        :param int n_message_layers: The number of layers in the message
            network. Default is ``2``.
        :param int n_update_layers: The number of layers in the update network.
            Default is ``2``.
        :param type[torch.nn.Module] activation: No-argument activation class.
            Default is :class:`torch.nn.SiLU`.
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
        :raises AssertionError: If dimensions or layer counts are not positive
            integers, except for edge width, which may be zero.
        :raises ValueError: If ``use_velocity``, activation, or PyG options fail
            validation.
        """
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)

        # Check values
        check_positive_integer(node_feature_dim, strict=True)
        check_positive_integer(edge_feature_dim, strict=False)
        check_positive_integer(pos_dim, strict=True)
        check_positive_integer(hidden_dim, strict=True)
        check_positive_integer(n_message_layers, strict=True)
        check_positive_integer(n_update_layers, strict=True)
        check_consistency(use_velocity, bool)

        # Initialization
        self.use_velocity = use_velocity

        # Layer for computing the message
        self.message_net = FeedForward(
            input_dimensions=2 * node_feature_dim + edge_feature_dim + 1,
            output_dimensions=pos_dim,
            inner_size=hidden_dim,
            n_layers=n_message_layers,
            func=activation,
        )

        # Layer for updating the node features
        self.update_feat_net = FeedForward(
            input_dimensions=node_feature_dim + pos_dim,
            output_dimensions=node_feature_dim,
            inner_size=hidden_dim,
            n_layers=n_update_layers,
            func=activation,
        )

        # Layer for updating the node positions
        # The output dimension is set to 1 for equivariant updates
        self.update_pos_net = FeedForward(
            input_dimensions=pos_dim,
            output_dimensions=1,
            inner_size=hidden_dim,
            n_layers=n_update_layers,
            func=activation,
        )

        # If velocity is used, instantiate layer for velocity updates
        if self.use_velocity:
            self.update_vel_net = FeedForward(
                input_dimensions=node_feature_dim,
                output_dimensions=1,
                inner_size=hidden_dim,
                n_layers=n_update_layers,
                func=activation,
            )

    def forward(self, x, pos, edge_index, edge_attr=None, vel=None):
        """
        Forward pass of the block, triggering the message-passing routine.

        :param x: Node features shaped ``(N, node_feature_dim)``.
        :type x: torch.Tensor | LabelTensor
        :param pos: Node coordinates shaped ``(N, pos_dim)``.
        :type pos: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: Integer connectivity shaped ``(2, E)``.
        :param edge_attr: Features shaped ``(E, edge_feature_dim)``. ``None``
            omits edge inputs and requires zero edge width. Default is ``None``.
        :type edge_attr: torch.Tensor | LabelTensor | None
        :param vel: Velocities shaped ``(N, pos_dim)``; required when velocity
            updates are enabled, ignored otherwise. Default is ``None``.
        :type vel: torch.Tensor | LabelTensor | None
        :raises ValueError: If velocities are enabled and ``vel`` is ``None``.
        :return: Updated features and positions, plus velocities when enabled;
            each tensor retains its corresponding input shape.
        :rtype: tuple[torch.Tensor, torch.Tensor] |
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        if self.use_velocity and vel is None:
            raise ValueError(
                "Velocity features are enabled, but no velocity is passed."
            )

        return self.propagate(
            edge_index=edge_index, x=x, pos=pos, edge_attr=edge_attr, vel=vel
        )

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        """
        Compute the message to be passed between nodes and edges.

        :param x_i: The node features of the recipient nodes.
        :type x_i: torch.Tensor | LabelTensor
        :param x_j: The node features of the sender nodes.
        :type x_j: torch.Tensor | LabelTensor
        :param pos_i: The node coordinates of the recipient nodes.
        :type pos_i: torch.Tensor | LabelTensor
        :param pos_j: The node coordinates of the sender nodes.
        :type pos_j: torch.Tensor | LabelTensor
        :param edge_attr: Per-edge features, or ``None`` to omit them.
        :type edge_attr: torch.Tensor | LabelTensor | None
        :return: Position messages and learned invariant messages, both shaped
            ``(E, pos_dim)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        # Compute the euclidean distance between the sender and recipient nodes
        diff = pos_i - pos_j
        dist = torch.norm(diff, dim=-1, keepdim=True) ** 2

        # Compute the message input
        if edge_attr is None:
            input_ = torch.cat((x_i, x_j, dist), dim=-1)
        else:
            input_ = torch.cat((x_i, x_j, dist, edge_attr), dim=-1)

        # Compute the messages and their equivariant counterpart
        m_ij = self.message_net(input_)
        message = diff * self.update_pos_net(m_ij)

        return message, m_ij

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregate the messages at the nodes during message passing.

        This method receives a tuple of tensors corresponding to the messages
        to be aggregated. Both messages are aggregated separately according to
        the specified aggregation scheme.

        :param tuple[torch.Tensor, torch.Tensor] inputs: Two message tensors to
            aggregate.
        :param index: The indices of target nodes for each message. This tensor
            specifies which node each message is aggregated into.
        :type index: torch.Tensor | LabelTensor
        :param ptr: Optional tensor to specify the slices of messages for each
            node. ``None`` uses ``index`` alone. Default is ``None``.
        :type ptr: torch.Tensor | LabelTensor | None
        :param dim_size: Output node count. ``None`` lets PyG infer it from
            indices. Default is ``None``.
        :type dim_size: int | None
        :return: Tuple of aggregated tensors corresponding to (aggregated
            messages for position updates, aggregated messages for feature
            updates).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        # Unpack the messages from the inputs
        message, m_ij = inputs

        # Aggregate messages as usual using self.aggr method
        agg_message = super().aggregate(message, index, ptr, dim_size)
        agg_m_ij = super().aggregate(m_ij, index, ptr, dim_size)

        return agg_message, agg_m_ij

    def update(self, aggregated_inputs, x, pos, edge_index, vel):
        """
        Update node features, positions, and optionally velocities.

        :param tuple[torch.Tensor, torch.Tensor] aggregated_inputs: Aggregated
            position and feature messages, respectively.
        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :param pos: The euclidean coordinates of the nodes.
        :type pos: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge indices.
        :param vel: The velocity of the nodes.
        :type vel: torch.Tensor | LabelTensor | None
        :return: Updated features and positions, plus scaled velocities when
            enabled. Normalization uses target degrees clamped to at least one.
        :rtype: tuple[torch.Tensor, torch.Tensor] |
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # aggregated_inputs is tuple (agg_message, agg_m_ij)
        agg_message, agg_m_ij = aggregated_inputs

        # Degree for normalization of position updates
        c = degree(edge_index[1], pos.shape[0]).unsqueeze(-1).clamp(min=1)

        # If velocity is used, update it and use it to update positions
        if self.use_velocity:
            vel = self.update_vel_net(x) * vel

        # Update node features with aggregated m_ij
        x = self.update_feat_net(torch.cat((x, agg_m_ij), dim=-1))

        # Update positions with aggregated messages m_ij and velocities
        pos = pos + agg_message / c + (vel if self.use_velocity else 0)

        return (x, pos, vel) if self.use_velocity else (x, pos)
