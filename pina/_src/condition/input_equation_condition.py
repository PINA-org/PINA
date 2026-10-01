"""Module for the Input-Equation Condition class."""

from torch_geometric.data import Data

from pina._src.condition.base_condition import BaseCondition, _unwrap_single
from pina._src.condition.graph_condition import GraphCondition, _is_graph
from pina._src.condition.tensor_condition import TensorCondition
from pina._src.core.graph import Graph
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency
from pina._src.equation.base_equation import BaseEquation


class InputEquationCondition(BaseCondition):
    """
    The class :class:`InputEquationCondition` represents a condition defined
    by an ``input`` and an ``equation``. The ``equation`` is evaluated on the
    ``input`` data, which can be a :class:`~pina.label_tensor.LabelTensor` or
    a :class:`~pina.graph.Graph`.

    :Example:

    >>> from pina import Condition
    >>> from pina.graph import Graph
    >>> import torch

    >>> pos = LabelTensor(torch.randn(50, 2), labels=["x", "y"])
    >>> edge_index = torch.randint(0, 50, (2, 250))
    >>> graph = Graph(pos=pos, edge_index=edge_index)

    >>> def equation(input_, output):
    >>>     return input_.extract(["x"]) * output
    >>> condition = Condition(input=graph, equation=equation)
    """

    # Available fields, input, and equation data types
    __fields__ = ["input", "equation"]
    _avail_input_cls = (LabelTensor, Data, Graph)
    _avail_equation_cls = BaseEquation

    def __new__(cls, input, equation):
        """
        Check the types of ``input`` and ``equation`` data and instantiate the
        appropriate condition variant accordingly.

        :param input: The input data associated with the condition.
        :param equation: The equation associated with the condition.
        :raises ValueError: If ``input`` is not of an available type.
        :return: A tensor or graph variant of
            :class:`InputEquationCondition`.
        :rtype: InputEquationCondition
        """
        # Check input type - if iterable, ensure it is either Data or Graph
        if isinstance(input, (list, tuple)):
            check_consistency(input, (Data, Graph))
        else:
            check_consistency(input, cls._avail_input_cls)

        # Check equation type
        check_consistency(equation, cls._avail_equation_cls)

        # Instantiate the variant matching the data type
        variant = (
            _GraphInputEquationCondition
            if _is_graph(input)
            else _TensorInputEquationCondition
        )
        return super().__new__(variant)

    def evaluate(self, batch, solver):
        """
        Evaluate the equation of the condition on the given batch using the
        solver.

        :param dict batch: The batch containing the data required by the
            condition evaluation.
        :param BaseSolver solver: The solver used to perform the forward pass
            and compute the residual.
        :return: The non-aggregated residual tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        samples = batch["input"].requires_grad_(True)
        output = solver.forward(samples)
        return self.equation.residual(samples, output, solver._params)

    @property
    def input(self):
        """
        The input data associated with the condition.

        :return: The input data.
        :rtype: LabelTensor | Graph | Data
        """
        return _unwrap_single(self.data.input)

    @input.setter
    def input(self, value):
        """
        Set the input data associated with the condition.

        :param value: The new input data.
        :type value: LabelTensor | Graph | Data
        """
        self.data.input = value

    @property
    def equation(self):
        """
        The equation associated with the condition.

        :return: The equation.
        :rtype: BaseEquation
        """
        return self.data.equation

    @equation.setter
    def equation(self, value):
        """
        Set the equation associated with the condition.

        :param value: The new equation.
        :type value: BaseEquation
        :raises ValueError: If ``value`` is not an instance of
            :class:`~pina.equation.base_equation.BaseEquation`.
        """
        # Check consistency
        check_consistency(value, self._avail_equation_cls)
        self.data.equation = value


class _TensorInputEquationCondition(TensorCondition, InputEquationCondition):
    """
    Tensor variant of :class:`InputEquationCondition`.
    """


class _GraphInputEquationCondition(GraphCondition, InputEquationCondition):
    """
    Graph variant of :class:`InputEquationCondition`.
    """
