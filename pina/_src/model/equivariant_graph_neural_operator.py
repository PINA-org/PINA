"""Module for the Equivariant Graph Neural Operator model."""

import torch
from pina._src.core.utils import check_positive_integer
from pina._src.model.block.message_passing.equivariant_graph_neural_operator_block import (
    EquivariantGraphNeuralOperatorBlock,
)


class EquivariantGraphNeuralOperator(torch.nn.Module):
    """
    Equivariant Graph Neural Operator (EGNO) for modeling 3D dynamics.

    EGNO is a graph-based neural operator that preserves equivariance with
    respect to 3D transformations while modeling temporal and spatial
    interactions between nodes. It combines:

        1. Temporal convolution in the Fourier domain to capture long-range
           temporal dependencies efficiently.
        2. Equivariant Graph Neural Network (EGNN) layers to model interactions
           between nodes while respecting geometric symmetries.

    This design allows EGNO to learn complex spatiotemporal dynamics of
    physical systems, molecules, or particles while enforcing physically
    meaningful constraints.

    .. seealso::

        **Original reference**
        Xu, M., Han, J., Lou, A., Kossaifi, J., Ramanathan, A., Azizzadenesheli,
        K., Leskovec, J., Ermon, S., Anandkumar, A. (2024).
        *Equivariant Graph Neural Operator for Modeling 3D Dynamics*
        DOI: `arXiv preprint arXiv:2401.11037.
        <https://arxiv.org/abs/2401.11037>`_

    :Example:

        >>> import torch
        >>> from pina.model import EquivariantGraphNeuralOperator
        >>> model = EquivariantGraphNeuralOperator(
        ...     n_egno_layers=2, node_feature_dim=16, edge_feature_dim=4,
        ...     pos_dim=3, modes=8
        ... )
    """

    def __init__(
        self,
        n_egno_layers,
        node_feature_dim,
        edge_feature_dim,
        pos_dim,
        modes,
        time_steps=2,
        hidden_dim=64,
        time_emb_dim=16,
        max_time_idx=10000,
        n_message_layers=2,
        n_update_layers=2,
        activation=torch.nn.SiLU,
        aggr="add",
        node_dim=-2,
        flow="source_to_target",
    ):
        """
        Initialization of the :class:`EquivariantGraphNeuralOperator` class.

        :param int n_egno_layers: The number of EGNO layers.
        :param int node_feature_dim: The dimension of the node features in each
            EGNO layer.
        :param int edge_feature_dim: The dimension of the edge features in each
            EGNO layer.
        :param int pos_dim: The dimension of the position features in each
            EGNO layer.
        :param int modes: The number of Fourier modes to use in the temporal
            convolution, clipped to ``time_steps // 2 + 1`` during evaluation.
        :param int time_steps: The number of time steps to consider in the
            temporal convolution. Default is ``2``.
        :param int hidden_dim: The dimension of the hidden features in each EGNO
            layer. Default is ``64``.
        :param int time_emb_dim: The dimension of the sinusoidal time
            embeddings. Widths two or three produce nonfinite frequency values
            due to division by zero; use at least four. Default is ``16``.
        :param int max_time_idx: The maximum time index for the sinusoidal
            embeddings. Default is ``10000``.
        :param int n_message_layers: The number of layers in the message
            network of each EGNO layer. Default is ``2``.
        :param int n_update_layers: The number of layers in the update network
            of each EGNO layer. Default is ``2``.
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
        :raises AssertionError: If ``n_egno_layers`` is not a positive integer.
        :raises AssertionError: If ``time_emb_dim`` is not a positive integer.
        :raises AssertionError: If ``max_time_idx`` is not a positive integer.
        :raises AssertionError: If ``time_steps`` is not a positive integer.
        :raises AssertionError: If block dimensions, modes, or layer counts fail
            positive integer validation; edge width may be zero.
        :raises ValueError: If activation or PyG propagation options are
            invalid.
        """
        super().__init__()

        # Check consistency
        check_positive_integer(n_egno_layers, strict=True)
        check_positive_integer(time_emb_dim, strict=True)
        check_positive_integer(max_time_idx, strict=True)
        check_positive_integer(time_steps, strict=True)

        # Initialize parameters
        self.time_steps = time_steps
        self.time_emb_dim = time_emb_dim
        self.max_time_idx = max_time_idx

        # Initialize EGNO layers
        self.egno_layers = torch.nn.ModuleList()
        for _ in range(n_egno_layers):
            self.egno_layers.append(
                EquivariantGraphNeuralOperatorBlock(
                    node_feature_dim=node_feature_dim,
                    edge_feature_dim=edge_feature_dim,
                    pos_dim=pos_dim,
                    modes=modes,
                    hidden_dim=hidden_dim,
                    n_message_layers=n_message_layers,
                    n_update_layers=n_update_layers,
                    activation=activation,
                    aggr=aggr,
                    node_dim=node_dim,
                    flow=flow,
                )
            )

        # Linear layer to adjust the scalar feature dimension
        self.linear = torch.nn.Linear(
            node_feature_dim + time_emb_dim, node_feature_dim
        )

    def forward(self, graph):
        """
        Forward pass of the :class:`EquivariantGraphNeuralOperator` class.

        :param graph: The input graph object with the following attributes:

            * ``x``: Features shaped ``(num_nodes, node_feature_dim)``.
            * ``pos``: Positions shaped ``(num_nodes, pos_dim)``.
            * ``vel``: Velocities shaped ``(num_nodes, pos_dim)``.
            * ``edge_index``: Connectivity shaped ``(2, num_edges)``.
            * ``edge_attr``: Features shaped ``(num_edges, edge_feature_dim)``,
              or ``None`` with zero configured edge width.

        :type graph: torch_geometric.data.Data | pina.Graph
        :raises ValueError: If the graph does not contain ``vel``.
        :return: The output graph object with updated node features,
            positions, and velocities. It is a clone: the input is unchanged.
            ``x``, ``pos``, and ``vel`` gain a leading ``time_steps`` axis, as
            does ``edge_attr`` when present. The original ``edge_index`` and
            other graph metadata are retained, not replaced by the internally
            offset temporal connectivity.
        :rtype: torch_geometric.data.Data | pina.Graph
        """
        # Check that the graph has the required attributes
        if "vel" not in graph:
            raise ValueError("The input graph must have a 'vel' attribute.")

        # Compute the temporal embedding
        emb = self._embedding(torch.arange(self.time_steps)).to(graph.x.device)
        emb = emb.unsqueeze(1).repeat(1, graph.x.shape[0], 1)

        # Expand dimensions
        x = graph.x.unsqueeze(0).repeat(self.time_steps, 1, 1)
        x = self.linear(torch.cat((x, emb), dim=-1))
        pos = graph.pos.unsqueeze(0).repeat(self.time_steps, 1, 1)
        vel = graph.vel.unsqueeze(0).repeat(self.time_steps, 1, 1)

        # Manage edge index
        offset = torch.arange(self.time_steps).reshape(-1, 1)
        offset = offset.to(graph.x.device) * graph.x.shape[0]
        src = graph.edge_index[0].unsqueeze(0) + offset
        dst = graph.edge_index[1].unsqueeze(0) + offset
        edge_index = torch.stack([src, dst], dim=0).reshape(2, -1)

        # Manage edge attributes
        if graph.edge_attr is not None:
            edge_attr = graph.edge_attr.unsqueeze(0)
            edge_attr = edge_attr.repeat(self.time_steps, 1, 1)
        else:
            edge_attr = None

        # Iteratively apply EGNO layers
        for layer in self.egno_layers:
            x, pos, vel = layer(
                x=x,
                pos=pos,
                vel=vel,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        # Build new graph
        new_graph = graph.clone()
        new_graph.x, new_graph.pos, new_graph.vel = x, pos, vel
        if edge_attr is not None:
            new_graph.edge_attr = edge_attr

        return new_graph

    def _embedding(self, time):
        """
        Generate sinusoidal temporal embeddings.

        :param torch.Tensor time: One-dimensional CPU time coordinates.
        :return: Embeddings shaped ``(len(time), time_emb_dim)``. Odd widths
            have a final zero column; widths two and three divide by zero in
            frequency construction and can produce nonfinite values.
        :rtype: torch.Tensor
        """
        # Compute the sinusoidal embeddings
        half_dim = self.time_emb_dim // 2
        logs = torch.log(torch.as_tensor(self.max_time_idx)) / (half_dim - 1)
        freqs = torch.exp(-torch.arange(half_dim) * logs)
        args = torch.as_tensor(time)[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Apply padding if the embedding dimension is odd
        if self.time_emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1), mode="constant")

        return emb
