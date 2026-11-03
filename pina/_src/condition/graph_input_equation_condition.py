"""Module for the Graph Input-Equation Condition class."""

from pina._src.condition.graph_condition import GraphCondition
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency
from pina._src.equation.base_equation import BaseEquation


class GraphInputEquationCondition(GraphCondition):
    """
    Equation condition evaluated on node features of sampled subgraphs.

    Subgraphs are sampled and stored by :meth:`store_data`. Residual
    evaluation enables gradients on their selected node features, evaluates
    the solver on the graph, and passes the features, predictions, and
    unknown parameters to the equation.
    """

    def __init__(self, graph, equation, input_key="x"):
        """
        Initialization of the :class:`GraphInputEquationCondition`.

        :param Graph graph: The graph from which subgraphs are sampled.
        :param BaseEquation equation: The equation used to compute the
            residual from node features and model predictions.
        :param str input_key: The name of the graph attribute containing
            the input node features. Default is ``"x"``.
        :raises ValueError: If the equation is not an instance of
            :class:`BaseEquation`.
        """
        check_consistency(equation, BaseEquation)
        GraphCondition.__init__(self, graph, key=input_key)
        self._equation = equation

    @property
    def equation(self):
        """
        Return the equation associated with this condition.

        :return: The equation used to compute the condition residual.
        :rtype: BaseEquation
        """
        return self._equation

    def evaluate(self, batch, solver):
        """
        Compute the equation residual for the graph batch's node features.

        Enable gradient tracking on the feature attribute selected by
        ``input_key`` and pass the graph batch to the solver. Evaluate the
        equation using these features, the model predictions, and the
        solver's unknown parameters. No loss aggregation is applied.

        :param Graph batch: The graph batch containing the input node
            features under the attribute selected by ``input_key``.
        :param BaseSolver solver: The solver providing the model forward
            pass and the unknown parameters used by the equation.
        :raises AttributeError: If the batch lacks the selected feature
            attribute.
        :raises RuntimeError: If the input features do not support gradient
            tracking, for example because they have an integer dtype.
        :return: The equation residual evaluated on the graph batch.
        :rtype: torch.Tensor | LabelTensor
        """
        samples = getattr(batch, self.key).requires_grad_(True)
        output = solver.forward(batch)
        return self.equation.residual(samples, output, solver._params)
