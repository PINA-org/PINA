"""Module for the Graph Input-Target Condition class."""

from pina._src.condition.graph_condition import GraphCondition


class GraphInputTargetCondition(GraphCondition):
    """
    Supervised condition defined by graph input and output node fields.

    The condition inherits the graph storage and batching behaviour from
    :class:`~pina._src.condition.graph_condition.GraphCondition`. The input
    and output data are stored directly on the graph using the configured
    keys, which default to ``"x"`` and ``"y"`` respectively.

    :Example:

        >>> from pina.condition import GraphInputTargetCondition
        >>> from pina.graph import Graph
        >>> import torch

        >>> graph = Graph(
        ...     x=torch.randn(10, 2),
        ...     y=torch.randn(10, 1),
        ...     pos=torch.randn(10, 2),
        ...     edge_index=torch.randint(0, 10, (2, 30)),
        ... )
        >>> condition = GraphInputTargetCondition(graph)
    """

    def __init__(self, graph, input_key="x", output_key="y"):
        """
        Initialization of the :class:`GraphInputTargetCondition` class.

        :param Graph graph: The graph containing the input and output fields.
        :param str input_key: The graph attribute holding the input data.
            Default is ``"x"``.
        :param str output_key: The graph attribute holding the output data.
            Default is ``"y"``.
        """
        GraphCondition.__init__(self, graph, key=input_key)
        self.output_key = output_key

    def evaluate(self, batch, solver):
        """
        Evaluate the residual of the condition on the given graph batch.

        :param Graph batch: The graph batch containing input and output data.
        :param BaseSolver solver: The solver used to perform the forward pass.
        :return: The non-aggregated residual tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        return solver.forward(batch) - getattr(batch, self.output_key)
