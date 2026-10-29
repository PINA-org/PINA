"""Module to build Graph objects and perform operations on them."""

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.sampler import NodeSamplerInput, SamplerOutput
from torch_geometric.utils import to_undirected
from torch_geometric.utils.loop import remove_self_loops
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency, is_function


class Graph(Data):
    """
    Extends :class:`~torch_geometric.data.Data` class to include additional
    checks and functionalities.

    :Example:

        >>> import torch
        >>> from pina.graph import Graph
        >>> pos = torch.randn(5, 2)
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4],
        ...                            [1, 2, 3, 4, 0]])
        >>> graph = Graph(pos=pos, edge_index=edge_index, x=pos)
        >>> graph.num_nodes
        5
    """

    def __new__(
        cls,
        **kwargs,
    ):
        """
        Create a new instance of the :class:`~pina.graph.Graph` class by
        checking the consistency of the input data and storing the attributes.

        :param dict kwargs: Parameters used to initialize the
            :class:`~pina.graph.Graph` object.
        :return: A new instance of the :class:`~pina.graph.Graph` class.
        :rtype: Graph
        """
        # create class instance
        instance = Data.__new__(cls)

        # check the consistency of types defined in __init__, the others are not
        # checked (as in pyg Data object)
        instance._check_type_consistency(**kwargs)

        return instance

    def __init__(
        self,
        x=None,
        edge_index=None,
        pos=None,
        edge_attr=None,
        undirected=False,
        **kwargs,
    ):
        """
        Initialize the object by setting the node features, edge index,
        edge attributes, and positions. The edge index is preprocessed to make
        the graph undirected if required. For more details, see the
        :meth:`torch_geometric.data.Data`

        :param x: Optional tensor of node features ``(N, F)`` where ``F`` is the
            number of features per node.
        :type x: torch.Tensor, LabelTensor
        :param torch.Tensor edge_index: A tensor of shape ``(2, E)``
            representing the indices of the graph's edges.
        :param pos: A tensor of shape ``(N, D)`` representing the positions of
            ``N`` points in ``D``-dimensional space.
        :type pos: torch.Tensor | LabelTensor
        :param edge_attr: Optional tensor of edge_featured ``(E, F')`` where
            ``F'`` is the number of edge features
        :type edge_attr: torch.Tensor | LabelTensor
        :param bool undirected: Whether to make the graph undirected
        :param dict kwargs: Additional keyword arguments passed to the
            :class:`~torch_geometric.data.Data` class constructor.
        """
        # preprocessing
        self._preprocess_edge_index(edge_index, undirected)

        # calling init
        super().__init__(
            x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, **kwargs
        )

    def _check_type_consistency(self, **kwargs):
        """
        Check the consistency of the types of the input data.

        :param dict kwargs: Attributes to be checked for consistency.
        """
        # default types, specified in cls.__new__, by default they are Nont
        # if specified in **kwargs they get override
        x, pos, edge_index, edge_attr = None, None, None, None
        if "pos" in kwargs:
            pos = kwargs["pos"]
            self._check_pos_consistency(pos)
        if "edge_index" in kwargs:
            edge_index = kwargs["edge_index"]
            self._check_edge_index_consistency(edge_index)
        if "x" in kwargs:
            x = kwargs["x"]
            # self._check_x_consistency(x, pos)
        if "edge_attr" in kwargs:
            edge_attr = kwargs["edge_attr"]
            self._check_edge_attr_consistency(edge_attr, edge_index)
        if "undirected" in kwargs:
            undirected = kwargs["undirected"]
            check_consistency(undirected, bool)

    @staticmethod
    def _check_pos_consistency(pos):
        """
        Check if the position tensor is consistent.
        :param torch.Tensor pos: The position tensor.
        :raises ValueError: If the position tensor is not consistent.
        """
        if pos is not None:
            check_consistency(pos, (torch.Tensor, LabelTensor))
            if pos.ndim != 2:
                raise ValueError("pos must be a 2D tensor.")

    @staticmethod
    def _check_edge_index_consistency(edge_index):
        """
        Check if the edge index is consistent.

        :param torch.Tensor edge_index: The edge index tensor.
        :raises ValueError: If the edge index tensor is not consistent.
        """
        check_consistency(edge_index, (torch.Tensor, LabelTensor))
        if edge_index.ndim != 2:
            raise ValueError("edge_index must be a 2D tensor.")
        if edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")

    @staticmethod
    def _check_edge_attr_consistency(edge_attr, edge_index):
        """
        Check if the edge attribute tensor is consistent in type and shape
        with the edge index.

        :param edge_attr: The edge attribute tensor.
        :type edge_attr: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge index tensor.
        :raises ValueError: If the edge attribute tensor is not consistent.
        """
        if edge_attr is not None:
            check_consistency(edge_attr, (torch.Tensor, LabelTensor))
            if edge_attr.ndim != 2:
                raise ValueError("edge_attr must be a 2D tensor.")
            if edge_attr.size(0) != edge_index.size(1):
                raise ValueError(
                    "edge_attr must have shape "
                    "[num_edges, num_edge_features], expected "
                    f"num_edges {edge_index.size(1)} "
                    f"got {edge_attr.size(0)}."
                )

    @staticmethod
    def _check_x_consistency(x, pos=None):
        """
        Check if the input tensor x is consistent with the position tensor
        `pos`.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :param pos: The position tensor.
        :type pos: torch.Tensor | LabelTensor
        :raises ValueError: If the input tensor is not consistent.
        """
        if x is not None:
            check_consistency(x, (torch.Tensor, LabelTensor))
            if x.ndim != 2:
                raise ValueError("x must be a 2D tensor.")
            if pos is not None:
                if x.size(0) != pos.size(0):
                    raise ValueError("Inconsistent number of nodes.")

    @staticmethod
    def _preprocess_edge_index(edge_index, undirected):
        """
        Preprocess the edge index to make the graph undirected (if required).

        :param torch.Tensor edge_index: The edge index.
        :param bool undirected: Whether the graph is undirected.
        :return: The preprocessed edge index.
        :rtype: torch.Tensor
        """
        if undirected:
            edge_index = to_undirected(edge_index)
        return edge_index

    def extract(self, labels, attr="x"):
        """
        Perform extraction of labels from the attribute specified by `attr`.

        :param labels: Labels to extract
        :type labels: list[str] | tuple[str] | str | dict
        :return: Batch object with extraction performed on x
        :rtype: PinaBatch
        """
        # Extract labels from LabelTensor object
        tensor = getattr(self, attr).extract(labels)
        # Set the extracted tensor as the new attribute
        setattr(self, attr, tensor)
        return self

    def create_subgraph(self, sampler, batch_size=1, seed_nodes=None):
        """
        Create subgraphs of this graph by sampling seed nodes through a
        :class:`torch_geometric.sampler.BaseSampler`.

        The seed nodes are split into chunks of ``batch_size`` nodes and each
        chunk is passed to ``sampler.sample_from_nodes``. Every sampler output
        is converted into a subgraph: node-level attributes are sliced with
        the sampled nodes, edge-level attributes with the sampled edges, and
        graph-level attributes are copied as they are.

        :param sampler: The sampler used to sample the subgraphs. It must
            expose a ``sample_from_nodes`` method returning a
            :class:`torch_geometric.sampler.SamplerOutput`.
        :type sampler: torch_geometric.sampler.BaseSampler
        :param int batch_size: The number of seed nodes used for each
            subgraph. Default is ``1``.
        :param seed_nodes: The nodes to sample the subgraphs from. Default is
            ``None``, meaning all the nodes of the graph.
        :type seed_nodes: torch.Tensor | list[int] | None
        :raises TypeError: If ``sampler`` does not expose a callable
            ``sample_from_nodes`` method or if it does not return a
            :class:`torch_geometric.sampler.SamplerOutput`.
        :raises ValueError: If ``batch_size`` is not a positive integer or if
            the graph has an edge-level attribute while the sampler output
            does not provide the sampled edge ids.
        :return: The sampled subgraphs.
        :rtype: list[Graph]
        """
        if isinstance(sampler, type) or not callable(
            getattr(sampler, "sample_from_nodes", None)
        ):
            raise TypeError(
                "sampler must expose a callable 'sample_from_nodes' method."
            )
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if seed_nodes is None:
            seed_nodes = torch.arange(self.num_nodes)
        else:
            seed_nodes = torch.as_tensor(seed_nodes, dtype=torch.long)

        subgraphs = []
        for i, chunk in enumerate(seed_nodes.split(batch_size)):
            sampler_output = sampler.sample_from_nodes(
                NodeSamplerInput(input_id=torch.tensor([i]), node=chunk)
            )
            if not isinstance(sampler_output, SamplerOutput):
                raise TypeError(
                    "sampler.sample_from_nodes must return a "
                    "torch_geometric.sampler.SamplerOutput object."
                )

            subgraph = Graph()
            for key, value in self.items():
                if key in ("edge_index", "num_nodes"):
                    continue
                if self.is_node_attr(key):
                    subgraph[key] = value[sampler_output.node]
                elif self.is_edge_attr(key):
                    if sampler_output.edge is None:
                        raise ValueError(
                            f"The graph attribute '{key}' is edge-level but "
                            "the sampler output does not provide the sampled "
                            "edge ids ('edge')."
                        )
                    subgraph[key] = value[sampler_output.edge]
                else:
                    subgraph[key] = value
            subgraph.edge_index = torch.stack(
                [sampler_output.row, sampler_output.col]
            )
            subgraph.num_nodes = sampler_output.node.numel()
            subgraph.n_id = sampler_output.node
            subgraph.e_id = sampler_output.edge
            subgraph.seed_n_id = chunk
            subgraphs.append(subgraph)
        return subgraphs


class GraphBuilder:
    """
    A class that allows an easy definition of :class:`Graph` instances.

    :Example:

        >>> import torch
        >>> from pina.graph import GraphBuilder
        >>> pos = torch.randn(5, 2)
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4],
        ...                            [1, 2, 3, 4, 0]])
        >>> graph = GraphBuilder(pos=pos, edge_index=edge_index, loop=True)
        >>> isinstance(graph, Graph)
        True
    """

    def __new__(
        cls,
        pos,
        edge_index,
        x=None,
        edge_attr=False,
        custom_edge_func=None,
        loop=True,
        **kwargs,
    ):
        """
        Compute the edge attributes and create a new instance of the
        :class:`~pina.graph.Graph` class.

        :param pos: A tensor of shape ``(N, D)`` representing the positions of
            ``N`` points in ``D``-dimensional space.
        :type pos: torch.Tensor or LabelTensor
        :param edge_index: A tensor of shape ``(2, E)`` representing the indices
            of the graph's edges.
        :type edge_index: torch.Tensor
        :param x: Optional tensor of node features of shape ``(N, F)``, where
            ``F`` is the number of features per node.
        :type x: torch.Tensor | LabelTensor, optional
        :param bool edge_attr: Whether to compute the edge attributes.
        :param custom_edge_func: A custom function to compute edge attributes.
            If provided, overrides ``edge_attr``.
        :type custom_edge_func: Callable, optional
        :param bool loop: Whether to include self-loops.
        :param kwargs: Additional keyword arguments passed to the
            :class:`~pina.graph.Graph` class constructor.
        :return: A :class:`~pina.graph.Graph` instance constructed using the
            provided information.
        :rtype: Graph
        """
        if not loop:
            edge_index = remove_self_loops(edge_index)[0]
        edge_attr = cls._create_edge_attr(
            pos, edge_index, edge_attr, custom_edge_func or cls._build_edge_attr
        )
        return Graph(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            **kwargs,
        )

    @staticmethod
    def _create_edge_attr(pos, edge_index, edge_attr, func):
        """
        Create the edge attributes based on the input parameters.

        :param pos: Positions of the points.
        :type pos: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: Edge indices.
        :param bool edge_attr: Whether to compute the edge attributes.
        :param Callable func: Function to compute the edge attributes.
        :raises ValueError: If ``func`` is not a function.
        :return: The edge attributes.
        :rtype: torch.Tensor | LabelTensor | None
        """
        check_consistency(edge_attr, bool)
        if edge_attr:
            if is_function(func):
                return func(pos, edge_index)
            raise ValueError("custom_edge_func must be a function.")
        return None

    @staticmethod
    def _build_edge_attr(pos, edge_index):
        """
        Default function to compute the edge attributes.

        :param pos: Positions of the points.
        :type pos: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: Edge indices.
        :return: The edge attributes.
        :rtype: torch.Tensor
        """
        return (
            (pos[edge_index[0]] - pos[edge_index[1]])
            .abs()
            .as_subclass(torch.Tensor)
        )


class RadiusGraph(GraphBuilder):
    """
    Extends the :class:`~pina.graph.GraphBuilder` class to compute
    ``edge_index`` based on a radius. Each point is connected to all the points
    within the radius.

    :Example:

        >>> import torch
        >>> from pina.graph import RadiusGraph
        >>> pos = torch.randn(5, 2)
        >>> graph = RadiusGraph(pos=pos, radius=0.5)
        >>> isinstance(graph, Graph)
        True
    """

    def __new__(cls, pos, radius, **kwargs):
        """
        Instantiate the :class:`~pina.graph.Graph` class by computing the
        ``edge_index`` based on the radius provided.

        :param pos: A tensor of shape ``(N, D)`` representing the positions of
            ``N`` points in ``D``-dimensional space.
        :type pos: torch.Tensor | LabelTensor
        :param float radius: The radius within which points are connected.
        :param dict kwargs: The additional keyword arguments to be passed to
            :class:`GraphBuilder` and :class:`Graph` classes.
        :return: A :class:`~pina.graph.Graph` instance with the computed
            ``edge_index``.
        :rtype: Graph
        """
        edge_index = cls.compute_radius_graph(pos, radius)
        return super().__new__(cls, pos=pos, edge_index=edge_index, **kwargs)

    @staticmethod
    def compute_radius_graph(points, radius):
        """
        Computes the ``edge_index`` based on the radius. Each point is connected
        to all the points within the radius.

        :param points: A tensor of shape ``(N, D)`` representing the positions
            of ``N`` points in ``D``-dimensional space.
        :type points: torch.Tensor | LabelTensor
        :param float radius: The radius within which points are connected.
        :return: A tensor of shape ``(2, E)``, with ``E`` number of edges,
            representing the edge indices of the graph.
        :rtype: torch.Tensor
        """
        dist = torch.cdist(points, points, p=2)
        return (
            torch.nonzero(dist <= radius, as_tuple=False)
            .t()
            .as_subclass(torch.Tensor)
        )


class KNNGraph(GraphBuilder):
    """
    Extends the :class:`~pina.graph.GraphBuilder` class to compute
    ``edge_index`` based on a K-nearest neighbors algorithm.

    :Example:

        >>> import torch
        >>> from pina.graph import KNNGraph
        >>> pos = torch.randn(10, 2)
        >>> graph = KNNGraph(pos=pos, neighbours=3)
        >>> isinstance(graph, Graph)
        True
    """

    def __new__(cls, pos, neighbours, **kwargs):
        """
        Instantiate the :class:`~pina.graph.Graph` class by computing the
        ``edge_index`` based on the K-nearest neighbors algorithm.

        :param pos: A tensor of shape ``(N, D)`` representing the positions of
            ``N`` points in ``D``-dimensional space.
        :type pos: torch.Tensor | LabelTensor
        :param int neighbours: The number of nearest neighbors to consider when
            building the graph.
        :param dict kwargs: The additional keyword arguments to be passed to
            :class:`GraphBuilder` and :class:`Graph` classes.

        :return: A :class:`~pina.graph.Graph` instance with the computed
            ``edge_index``.
        :rtype: Graph
        """

        edge_index = cls.compute_knn_graph(pos, neighbours)
        return super().__new__(cls, pos=pos, edge_index=edge_index, **kwargs)

    @staticmethod
    def compute_knn_graph(points, neighbours):
        """
        Computes the ``edge_index`` based on the K-nearest neighbors algorithm.

        :param points: A tensor of shape ``(N, D)`` representing the positions
            of ``N`` points in ``D``-dimensional space.
        :type points: torch.Tensor | LabelTensor
        :param int neighbours: The number of nearest neighbors to consider when
            building the graph.
        :return: A tensor of shape ``(2, E)``, with ``E`` number of edges,
            representing the edge indices of the graph.
        :rtype: torch.Tensor
        """
        dist = torch.cdist(points, points, p=2)
        knn_indices = torch.topk(dist, k=neighbours, largest=False).indices
        row = torch.arange(points.size(0)).repeat_interleave(neighbours)
        col = knn_indices.flatten()
        return torch.stack([row, col], dim=0).as_subclass(torch.Tensor)


class LabelBatch(Batch):
    """
    Extends the :class:`~torch_geometric.data.Batch` class to include
    :class:`~pina.label_tensor.LabelTensor` objects.

    :Example:

        >>> import torch
        >>> from pina.graph import Graph, LabelBatch
        >>> graphs = [Graph(pos=torch.randn(3, 2),
        ...     edge_index=torch.tensor([[0, 1], [1, 0]])) for _ in range(2)]
        >>> batch = LabelBatch.from_data_list(graphs)
        >>> batch.num_graphs
        2
    """

    @classmethod
    def from_data_list(cls, data_list):
        """
        Create a Batch object from a list of :class:`~torch_geometric.data.Data`
        or :class:`~pina.graph.Graph` objects.

        :param data_list: List of :class:`~torch_geometric.data.Data` or
            :class:`~pina.graph.Graph` objects.
        :type data_list: list[Data] | list[Graph]
        :return: A :class:`~torch_geometric.data.Batch` object containing
            the input data.
        :rtype: :class:`~torch_geometric.data.Batch`
        """
        # Store the labels of Data/Graph objects (all data have the same labels)
        # If the data do not contain labels, labels is an empty dictionary,
        # therefore the labels are not stored
        labels = {
            k: v.labels
            for k, v in data_list[0].items()
            if isinstance(v, LabelTensor)
        }

        # Create a Batch object from the list of Data objects
        batch = super().from_data_list(data_list)

        # Put the labels back in the Batch object
        for k, v in labels.items():
            batch[k].labels = v
        return batch
