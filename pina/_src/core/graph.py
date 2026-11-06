"""Module to build Graph objects and perform operations on them."""

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.utils.loop import remove_self_loops
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency, is_function


class Graph(Data):
    """
    Extends :class:`~torch_geometric.data.Data` class to include additional
    checks and functionalities.

    Construct graphs with keyword arguments. Positions, connectivity, edge
    attributes, and the undirected flag receive selected validation; node
    features and arbitrary extra attributes are left to PyG. Tensor attributes
    are stored without cloning.

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
        Allocate a graph after validating explicitly supplied attributes.

        :param dict[str, Any] kwargs: Constructor attributes. ``pos`` is checked
            for tensor type and rank when non-``None``; ``edge_index`` is
            checked for tensor type and shape ``(2, E)`` whenever supplied.
            Non-``None`` edge features are checked for shape ``(E, F)`` against
            connectivity;
            ``undirected`` is checked for boolean type. ``x`` and other keys are
            not validated here. See :meth:`__init__` for defaults and meanings.
        :raises ValueError: If a checked type or shape is invalid.
        :raises AttributeError: If non-``None`` edge attributes are supplied
            without connectivity to determine their expected length.
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
        Initialization of the :class:`Graph` class.

        :param Any x: The node features, typically a tensor or
            :class:`pina.LabelTensor` with leading node dimension ``N``. Neither
            their type, rank, nor agreement with ``pos`` is checked. ``None``
            stores no node features. Default is ``None``.
        :param edge_index: The connectivity tensor of shape ``(2, E)``, normally
            containing integer node indices. Dtype and index bounds are not
            checked. Omitting this keyword leaves connectivity unset; explicitly
            supplying ``None`` fails allocation-time validation. Default is
            ``None``.
        :type edge_index: torch.Tensor | LabelTensor
        :param pos: The positions of ``N`` nodes in ``D`` dimensions, with shape
            ``(N, D)``. ``None`` leaves positions unset. Default is ``None``.
        :type pos: torch.Tensor | LabelTensor | None
        :param edge_attr: The edge features of shape ``(E, F)``. Non-``None``
            values require connectivity with ``E`` columns. ``None`` leaves edge
            features unset. Default is ``None``.
        :type edge_attr: torch.Tensor | LabelTensor | None
        :param bool undirected: Whether to request undirected preprocessing.
            Its result is currently discarded, so stored connectivity is not
            changed by this flag. Default is ``False``.
        :param dict[str, Any] kwargs: Additional PyG attributes, such as ``y``
            for targets or ``num_nodes`` for an explicit node count. Arbitrary
            keys and values are forwarded to :class:`torch_geometric.data.Data`
            without PINA validation; omitted keys are not added by PINA.
        :raises ValueError: If allocation-time validation rejects a supplied
            position, connectivity, edge-feature, or undirected value.
        :raises AttributeError: If edge features are supplied without
            connectivity.
        :raises TypeError: If the graph is constructed with positional
            arguments.

        .. warning::

            ``undirected=True`` computes a new edge index but does not store it.
            Supply symmetric connectivity explicitly when it is required.
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

        Validate only explicitly supplied position, connectivity, edge-feature,
        and undirected values. Node-feature validation is not invoked.

        :param dict[str, Any] kwargs: The constructor attributes. Missing keys
            are skipped; ``pos=None`` and ``edge_attr=None`` disable their
            corresponding checks, but ``edge_index=None`` is rejected.
        :raises ValueError: If a checked type or shape is invalid.
        :raises AttributeError: If edge attributes require missing connectivity.
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

        :param pos: The position tensor, expected to have shape ``(N, D)``.
            ``None`` skips validation.
        :type pos: torch.Tensor | LabelTensor | None
        :raises ValueError: If ``pos`` has an unsupported type or is not
            two-dimensional.
        """
        if pos is not None:
            check_consistency(pos, (torch.Tensor, LabelTensor))
            if pos.ndim != 2:
                raise ValueError("pos must be a 2D tensor.")

    @staticmethod
    def _check_edge_index_consistency(edge_index):
        """
        Check if the edge index is consistent.

        Dtype and node-index bounds are not validated.

        :param edge_index: The connectivity tensor with shape ``(2, E)``.
        :type edge_index: torch.Tensor | LabelTensor
        :raises ValueError: If the type is unsupported, the rank is not two,
            or the leading dimension does not have size two.
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

        :param edge_attr: The edge features with shape ``(E, F)``. ``None``
            skips validation.
        :type edge_attr: torch.Tensor | LabelTensor | None
        :param edge_index: The connectivity tensor of shape ``(2, E)``. ``None``
            is usable only when edge features are also ``None``.
        :type edge_index: torch.Tensor | LabelTensor | None
        :raises ValueError: If edge features have an unsupported type, are not
            two-dimensional, or their leading size differs from the edge count.
        :raises AttributeError: If edge features are present but connectivity is
            ``None``.
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
        Validate two-dimensional node features and their node count.

        This helper is not called during normal graph construction.

        :param x: The node-feature tensor. ``None`` skips all checks.
        :type x: torch.Tensor | LabelTensor | None
        :param pos: The positions used for comparing node counts. ``None``
            disables that comparison. Default is ``None``.
        :type pos: torch.Tensor | LabelTensor | None
        :raises ValueError: If features have an unsupported type, are not
            two-dimensional, or their node count differs from the positions.
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

        :param edge_index: The connectivity to process. ``None`` is returned
            unchanged when ``undirected`` is ``False``.
        :type edge_index: torch.Tensor | LabelTensor | None
        :param bool undirected: Whether to call
            :func:`torch_geometric.utils.to_undirected`.
        :return: The symmetrized edge index when requested, otherwise the
            original value. This helper does not update the graph itself.
        :rtype: torch.Tensor | LabelTensor | None
        """
        if undirected:
            edge_index = to_undirected(edge_index)
        return edge_index

    def extract(self, labels, attr="x"):
        """
        Extract labels from one graph attribute and replace it in place.

        Delegate to :meth:`pina.LabelTensor.extract`. Other graph attributes and
        connectivity are left unchanged, without a consistency check.

        :param labels: The requested component labels, or a mapping from stored
            dimension names to scalar labels, label lists, or ranges.
        :type labels: str | int | list[Hashable] | tuple[Hashable, ...] |
            dict[Hashable, str | int | list[Hashable] | builtins.range]
        :param str attr: The attribute containing a
            :class:`pina.LabelTensor` to extract from. Default is ``"x"``.
        :raises AttributeError: If the attribute is missing or does not provide
            an ``extract`` method, or its stored labels cannot be searched.
        :raises ValueError: If the dimension or component labels are unknown,
            repeated, or incompatible with the extracted shape.
        :raises KeyError: If last-dimension extraction needs missing metadata.
        :raises IndexError: If a component selection is empty.
        :return: This graph, with the selected attribute replaced.
        :rtype: Graph
        """
        # Extract labels from LabelTensor object
        tensor = getattr(self, attr).extract(labels)
        # Set the extracted tensor as the new attribute
        setattr(self, attr, tensor)
        return self

    def create_subgraph(self, start_node, num_hops=5):
        """
        Create a subgraph around a starting node.

        Follow incoming edges using PyG's ``"source_to_target"`` flow to find
        nodes within ``num_hops`` of the seed. Return the induced subgraph on
        those nodes with consecutive local node indices. Node- and edge-level
        :class:`LabelTensor` attributes are rewrapped using their original
        last-dimension labels; metadata for other dimensions is not rebuilt.
        Other attributes follow PyG's shallow-copy subgraph behavior.

        :param start_node: The seed node index. A tensor must hold one value
            convertible to an integer.
        :type start_node: int | torch.Tensor
        :param int num_hops: The number of hops around the starting node.
            Zero or negative values select only the seed node. Default is ``5``.
        :raises TypeError: If connectivity is missing and cannot be unpacked
            by PyG, or ``num_hops`` cannot be used as a range bound.
        :raises IndexError: If node indices are outside the usable index range.
        :return: The subgraph centered on ``start_node``.
        :rtype: Graph
        """
        subset = k_hop_subgraph(
            node_idx=int(start_node),
            num_hops=num_hops,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes,
        )[0]
        subgraph = self.subgraph(subset)

        for key, value in self.items():
            if not isinstance(value, LabelTensor) or key not in subgraph:
                continue
            if self.is_node_attr(key) or self.is_edge_attr(key):
                setattr(
                    subgraph,
                    key,
                    LabelTensor(getattr(subgraph, key), value.labels),
                )

        return subgraph


class GraphBuilder:
    """
    Factory for :class:`Graph` instances with optional edge features.

    Construction returns a :class:`Graph`, not a builder instance. Existing
    self-loops can be removed, but are not added. The default edge features are
    absolute coordinate differences between edge endpoints, not scalar lengths.

    :Example:

        >>> import torch
        >>> from pina._src.core.graph import Graph, GraphBuilder
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

        :param pos: The node positions of shape ``(N, D)``. ``None`` leaves
            positions unset; default edge-feature computation requires a tensor.
        :type pos: torch.Tensor | LabelTensor | None
        :param torch.Tensor edge_index: The connectivity tensor of shape
            ``(2, E)``, containing integer node indices.
        :param Any x: The optional node features forwarded unchanged to
            :class:`Graph`. ``None`` leaves features unset. Default is ``None``.
        :param bool edge_attr: Whether to compute edge features. When ``False``,
            ``custom_edge_func`` is ignored and no edge features are stored.
            Default is ``False``.
        :param custom_edge_func: The callable receiving ``(pos, edge_index)``
            after loop removal and returning edge features of shape ``(E, F)``
            or ``None``. It is used only when ``edge_attr`` is ``True``.
            ``None`` or another falsey value selects :meth:`_build_edge_attr`,
            which computes absolute coordinate differences. Default is ``None``.
        :type custom_edge_func: Callable | None
        :param bool loop: Whether to retain existing self-loops. ``False``
            removes them before feature computation; ``True`` does not add
            missing loops. Default is ``True``.
        :param dict[str, Any] kwargs: Additional :class:`Graph` options.
            ``undirected`` (bool) requests preprocessing. Default is ``False``.
            Other options become arbitrary PyG attributes, such as ``y`` or
            ``num_nodes``. Defaults and validation follow
            :meth:`Graph.__init__`.
        :raises ValueError: If ``edge_attr`` is not a boolean, an enabled custom
            edge function is not callable, or graph attribute validation fails.
        :raises IndexError: If feature computation indexes outside ``pos``.
        :raises AttributeError: If computed edge features require missing
            connectivity during graph validation.
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

        :param pos: The node positions with shape ``(N, D)``, or ``None`` when
            computation is disabled or the custom function accepts no positions.
        :type pos: torch.Tensor | LabelTensor | None
        :param torch.Tensor edge_index: The edge indices with shape ``(2, E)``.
        :param bool edge_attr: Whether to compute the edge attributes.
        :param Any func: The object to call as ``func(pos, edge_index)`` when
            computation is enabled; ignored otherwise.
        :raises ValueError: If ``edge_attr`` is not a boolean or computation is
            enabled and ``func`` is not callable.
        :return: The callable's unchecked result when enabled, otherwise
            ``None``. The factory subsequently validates tensor feature results
            through the graph constructor.
        :rtype: Any
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
        Compute absolute coordinate differences for each directed edge.

        :param pos: The node positions with shape ``(N, D)``.
        :type pos: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge indices with shape ``(2, E)``.
        :raises IndexError: If endpoint indices are out of range or invalid for
            indexing.
        :return: A plain tensor of shape ``(E, D)``, containing elementwise
            absolute source-minus-target coordinates, without label metadata.
        :rtype: torch.Tensor
        """
        return (
            (pos[edge_index[0]] - pos[edge_index[1]])
            .abs()
            .as_subclass(torch.Tensor)
        )


class RadiusGraph(GraphBuilder):
    """
    Extends :class:`pina._src.core.graph.GraphBuilder` to compute
    ``edge_index`` by thresholding all pairwise Euclidean distances. Both
    directions are included for each pair whose distance is at most the radius.
    Nonnegative radii also select self-loops, retained unless ``loop=False``.
    Construction returns a :class:`Graph`, not a radius-builder instance.

    :Example:

        >>> import torch
        >>> from pina.graph import Graph, RadiusGraph
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
        :param radius: The inclusive Euclidean distance threshold. Negative
            values produce no edges.
        :type radius: float | int
        :param dict[str, Any] kwargs: Options forwarded to
            :meth:`GraphBuilder.__new__` and :class:`Graph`:

            * ``x`` (Any): The node features; ``None`` leaves them unset.
              Default is ``None``.
            * ``edge_attr`` (bool): Whether to compute edge features. Default
              is ``False``.
            * ``custom_edge_func`` (Callable | None): The enabled edge-feature
              callable taking ``(pos, edge_index)``. ``None`` selects absolute
              coordinate differences. Used only when ``edge_attr=True``.
              Default is ``None``.
            * ``loop`` (bool): Whether to keep selected self-loops. Default is
              ``True``.
            * ``undirected`` (bool): The graph preprocessing request, whose
              result is currently discarded. Default is ``False``.

            Other keys become PyG attributes, such as ``y`` and ``num_nodes``.
        :raises RuntimeError: If :func:`torch.cdist` rejects the position shape
            or dtype.
        :raises ValueError: If edge-feature options or graph validation fail.
        :return: A :class:`~pina.graph.Graph` instance with the computed
            ``edge_index``.
        :rtype: Graph
        """
        edge_index = cls.compute_radius_graph(pos, radius)
        return super().__new__(cls, pos=pos, edge_index=edge_index, **kwargs)

    @staticmethod
    def compute_radius_graph(points, radius):
        """
        Select all ordered point pairs within an inclusive distance threshold.

        Pairwise Euclidean distances are computed with :func:`torch.cdist`.
        Nonnegative radii include self-loops; this helper does not remove them.

        :param points: A tensor of shape ``(N, D)`` representing the positions
            of ``N`` points in ``D``-dimensional space.
        :type points: torch.Tensor | LabelTensor
        :param radius: The inclusive radius. Negative values produce no edges.
        :type radius: float | int
        :raises RuntimeError: If :func:`torch.cdist` rejects the shape or dtype.
        :return: A tensor of shape ``(2, E)``, with ``E`` number of edges,
            representing source and target node indices. The result is a plain
            integer tensor on the positions' device, including both directions.
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
    Extends :class:`pina._src.core.graph.GraphBuilder` to compute
    ``edge_index`` based on a K-nearest neighbors algorithm.

    Each source node selects its closest targets, including itself among the
    candidates. Edges need not be symmetric. Removing loops does not replace
    them with additional neighbors. Construction returns a :class:`Graph`, not
    a nearest-neighbor-builder instance.

    :Example:

        >>> import torch
        >>> from pina.graph import Graph, KNNGraph
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
        :param int neighbours: The number of targets per source before optional
            loop removal, including self-distance among the candidates. Valid
            values are from ``0`` to ``N``; zero creates no edges.
        :param dict[str, Any] kwargs: Options forwarded to
            :meth:`GraphBuilder.__new__` and :class:`Graph`:

            * ``x`` (Any): The node features; ``None`` leaves them unset.
              Default is ``None``.
            * ``edge_attr`` (bool): Whether to compute edge features. Default
              is ``False``.
            * ``custom_edge_func`` (Callable | None): The enabled edge-feature
              callable taking ``(pos, edge_index)``. ``None`` selects absolute
              coordinate differences. Used only when ``edge_attr=True``.
              Default is ``None``.
            * ``loop`` (bool): Whether to keep selected self-loops. Removed
              loops are not replaced by other neighbors. Default is ``True``.
            * ``undirected`` (bool): The graph preprocessing request, whose
              result is currently discarded. Default is ``False``.

            Other keys become PyG attributes, such as ``y`` and ``num_nodes``.
        :raises RuntimeError: If pairwise distances or neighbor selection fail,
            the neighbor count is outside ``[0, N]``, or non-CPU positions cause
            a device mismatch when constructing edges.
        :raises ValueError: If edge-feature options or graph validation fail.
        :return: A :class:`~pina.graph.Graph` instance with the computed
            ``edge_index``.
        :rtype: Graph
        """

        edge_index = cls.compute_knn_graph(pos, neighbours)
        return super().__new__(cls, pos=pos, edge_index=edge_index, **kwargs)

    @staticmethod
    def compute_knn_graph(points, neighbours):
        """
        Select the closest targets independently for every source node.

        Compute all Euclidean distances and take the smallest ``neighbours``
        entries per row. Self-distance participates in selection; ties follow
        :func:`torch.topk` ordering and need not select the node itself. No
        symmetrization or loop removal is performed here.

        :param points: A tensor of shape ``(N, D)`` representing the positions
            of ``N`` points in ``D``-dimensional space.
        :type points: torch.Tensor | LabelTensor
        :param int neighbours: The number of selected targets per source, in
            ``[0, N]``. Zero produces an empty edge index.
        :raises RuntimeError: If distance computation fails, the neighbor count
            is out of range, or source and target indices have different
            devices.
        :raises TypeError: If the neighbor count is not accepted as an integer
            by :func:`torch.topk`.
        :return: A tensor of shape ``(2, E)``, with ``E`` number of edges,
            representing source and target indices grouped by source node.
            The result is a plain integer tensor.
        :rtype: torch.Tensor

        .. warning::

            Source indices are created on the CPU without matching the position
            device. Use CPU positions with this implementation.
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
        >>> from pina._src.core.graph import Graph, LabelBatch
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

        Read last-dimension labels only from labeled attributes of the first
        graph. After PyG batching, assign those labels to the corresponding
        batch attributes. Labels on other dimensions are not reconstructed and
        label agreement across graphs is not checked.

        :param list[torch_geometric.data.Data] data_list: The nonempty graph
            list. :class:`Graph` instances are also accepted. Attribute shapes
            must satisfy PyG's batching rules.
        :raises IndexError: If the list is empty.
        :raises RuntimeError: If PyG cannot concatenate the graph attributes.
        :raises ValueError: If the first graph's last-dimension labels cannot
            be assigned to the resulting batch attribute, including ``None``
            for an unlabeled last dimension.
        :return: The batched graph with PyG node assignments and offset edge
            indices. Its dynamic class derives from :class:`LabelBatch` and
            the input graph class.
        :rtype: LabelBatch
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
