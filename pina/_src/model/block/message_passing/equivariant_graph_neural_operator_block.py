"""Module for the Equivariant Graph Neural Operator block."""

import torch
from pina._src.core.utils import check_positive_integer
from pina._src.model.block.message_passing.en_equivariant_network_block import (
    EnEquivariantNetworkBlock,
)


class EquivariantGraphNeuralOperatorBlock(torch.nn.Module):
    """
    A single block of the Equivariant Graph Neural Operator (EGNO).

    This block combines a temporal convolution with an equivariant graph neural
    network (EGNN) layer. It preserves equivariance while modeling complex
    interactions between nodes in a graph over time.

    .. seealso::

        **Original reference**
        Xu, M., Han, J., Lou, A., Kossaifi, J., Ramanathan, A., Azizzadenesheli,
        K., Leskovec, J., Ermon, S., Anandkumar, A. (2024).
        *Equivariant Graph Neural Operator for Modeling 3D Dynamics*
        DOI: `arXiv preprint arXiv:2401.11037.
        <https://arxiv.org/abs/2401.11037>`_

    :Example:

        >>> import torch
        >>> from pina.model.block.message_passing import (
        ...     EquivariantGraphNeuralOperatorBlock
        ... )
        >>> block = EquivariantGraphNeuralOperatorBlock(
        ...     node_feature_dim=16, edge_feature_dim=4, pos_dim=3, modes=8
        ... )
    """

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        pos_dim,
        modes,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        activation=torch.nn.SiLU,
        aggr="add",
        node_dim=-2,
        flow="source_to_target",
    ):
        """
        Initialization of the :class:`EquivariantGraphNeuralOperatorBlock`
        class.

        :param int node_feature_dim: The dimension of the node features.
        :param int edge_feature_dim: The dimension of the edge features.
        :param int pos_dim: The dimension of the position features.
        :param int modes: The number of Fourier modes to use in the temporal
            convolution, clipped to ``time_steps // 2 + 1`` during each call.
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
        :raises AssertionError: If ``modes`` is not a positive integer.
        :raises AssertionError: If EGNN dimensions or layer counts fail positive
            integer validation; edge width may be zero.
        :raises ValueError: If activation or PyG propagation options are
            invalid.
        """
        super().__init__()

        # Check consistency
        check_positive_integer(modes, strict=True)

        # Initialization
        self.modes = modes

        # Temporal convolution weights - real and imaginary parts
        self.weight_scalar_r = torch.nn.Parameter(
            torch.rand(node_feature_dim, node_feature_dim, modes)
        )
        self.weight_scalar_i = torch.nn.Parameter(
            torch.rand(node_feature_dim, node_feature_dim, modes)
        )
        self.weight_vector_r = torch.nn.Parameter(torch.rand(2, 2, modes) * 0.1)
        self.weight_vector_i = torch.nn.Parameter(torch.rand(2, 2, modes) * 0.1)

        # EGNN block
        self.egnn = EnEquivariantNetworkBlock(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            pos_dim=pos_dim,
            use_velocity=True,
            hidden_dim=hidden_dim,
            n_message_layers=n_message_layers,
            n_update_layers=n_update_layers,
            activation=activation,
            aggr=aggr,
            node_dim=node_dim,
            flow=flow,
        )

    def forward(self, x, pos, vel, edge_index, edge_attr=None):
        """
        Forward pass of the Equivariant Graph Neural Operator block.

        :param x: The node feature tensor of shape
            ``[time_steps, num_nodes, node_feature_dim]``.
        :type x: torch.Tensor | LabelTensor
        :param pos: The node position tensor (Euclidean coordinates) of shape
            ``[time_steps, num_nodes, pos_dim]``.
        :type pos: torch.Tensor | LabelTensor
        :param vel: The node velocity tensor of shape
            ``[time_steps, num_nodes, pos_dim]``.
        :type vel: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: Connectivity shaped
            ``(2, time_steps * num_edges)``
            on the flattened time/node axis. Each temporal copy's indices must
            be offset by ``time_index * num_nodes``.
        :param edge_attr: The edge feature tensor of shape
            ``[time_steps, num_edges, edge_feature_dim]``. ``None`` omits edge
            inputs and requires zero edge width. Default is ``None``.
        :type edge_attr: torch.Tensor | LabelTensor | None
        :return: The updated node features, positions, and velocities, each with
            the same shape as the inputs.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # Prepare features
        center = pos.mean(dim=1, keepdim=True)
        vector = torch.stack((pos - center, vel), dim=-1)

        # Compute temporal convolution
        x = x + self._convolution(
            x, "mni, iom -> mno", self.weight_scalar_r, self.weight_scalar_i
        )
        vector = vector + self._convolution(
            vector,
            "mndi, iom -> mndo",
            self.weight_vector_r,
            self.weight_vector_i,
        )

        # Split position and velocity
        pos, vel = vector.unbind(dim=-1)
        pos = pos + center

        # Reshape to (time * nodes, feature) for egnn
        x = x.reshape(-1, x.shape[-1])
        pos = pos.reshape(-1, pos.shape[-1])
        vel = vel.reshape(-1, vel.shape[-1])
        if edge_attr is not None:
            edge_attr = edge_attr.reshape(-1, edge_attr.shape[-1])

        x, pos, vel = self.egnn(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            vel=vel,
        )

        # Reshape back to (time, nodes, feature)
        x = x.reshape(center.shape[0], -1, x.shape[-1])
        pos = pos.reshape(center.shape[0], -1, pos.shape[-1])
        vel = vel.reshape(center.shape[0], -1, vel.shape[-1])

        return x, pos, vel

    def _convolution(self, x, einsum_idx, real, img):
        """
        Compute the temporal convolution.

        :param torch.Tensor x: The input features.
        :param str einsum_idx: The indices for the einsum operation.
        :param torch.Tensor real: The real part of the convolution weights.
        :param torch.Tensor img: The imaginary part of the convolution weights.
        :return: The convolved features.
        :rtype: torch.Tensor
        """
        # Number of modes to use
        modes = min(self.modes, (x.shape[0] // 2) + 1)

        # Build complex weights
        weights = torch.complex(real[..., :modes], img[..., :modes])

        # Convolution in Fourier space
        fourier = torch.fft.rfftn(x, dim=[0])[:modes]
        out = torch.einsum(einsum_idx, fourier, weights)

        return torch.fft.irfftn(out, s=x.shape[0], dim=0)
