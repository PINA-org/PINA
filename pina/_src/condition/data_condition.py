"""Module for the Data Condition class."""

import torch
from torch_geometric.data import Data

from pina._src.condition.base_condition import BaseCondition, _unwrap_single
from pina._src.condition.graph_condition import GraphCondition, _is_graph
from pina._src.condition.tensor_condition import TensorCondition
from pina._src.core.graph import Graph
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency


class DataCondition(BaseCondition):
    """
    The class :class:`DataCondition` represents a condition defined by an
    ``input`` data, and optional ``conditional_variables``. It is used only
    for data-driven problems, where the equations are not needed.

    :Example:

    >>> import torch
    >>> from pina import Condition, LabelTensor

    >>> pos = LabelTensor(torch.randn(50, 2), labels=["x", "y"])
    >>> zv = LabelTensor(torch.randn(50, 1), labels=["zv"])

    >>> # data-driven condition with conditional variables
    >>> condition = Condition(input=pos, conditional_variables=zv)

    >>> # data-driven condition without conditional variables
    >>> condition = Condition(input=pos)
    """

    # Available fields and input data types
    __fields__ = ["input", "conditional_variables"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph)

    def __new__(cls, input, conditional_variables=None):
        """
        Check the types of the condition data and instantiate the appropriate
        condition variant accordingly.

        :param input: The input data associated with the condition.
        :param conditional_variables: The conditional variables associated
            with the condition. Default is ``None``.
        :raises ValueError: If ``input`` is not of an available type, or if
            ``conditional_variables`` is not a :class:`LabelTensor` or a
            :class:`torch.Tensor`.
        :return: A tensor or graph variant of :class:`DataCondition`.
        :rtype: DataCondition
        """
        # Check input type - if iterable, ensure it is either Data or Graph
        if isinstance(input, (list, tuple)):
            check_consistency(input, (Data, Graph))
        else:
            check_consistency(input, cls._avail_input_cls)

        # Check conditional variables type, if provided - lists and tuples are
        # not supported here (iterable inputs are only allowed for graphs)
        if conditional_variables is not None:
            if isinstance(conditional_variables, (list, tuple)):
                raise ValueError(
                    "Conditional variables must be provided as a "
                    "torch.Tensor or LabelTensor, not as a list or tuple."
                )
            check_consistency(
                conditional_variables, (torch.Tensor, LabelTensor)
            )

        # Instantiate the variant matching the data type
        variant = (
            _GraphDataCondition if _is_graph(input) else _TensorDataCondition
        )
        return super().__new__(variant)

    def evaluate(self, batch, solver):
        """
        Evaluate the residual of the condition on the given batch using the
        solver.

        Since the class :class:`DataCondition` is only used for data-driven
        problems where the equations are not needed, the ``evaluate`` method
        simply returns the forward pass of the solver on the batch input.

        :param dict batch: The batch containing the data required by the
            condition evaluation.
        :param BaseSolver solver: The solver used to perform the forward pass
            and compute the residual.
        :return: The non-aggregated forward pass output.
        :rtype: torch.Tensor | LabelTensor
        """
        return solver.forward(batch["input"])

    @property
    def input(self):
        """
        The input data associated with the condition.

        :return: The input data.
        :rtype: torch.Tensor | LabelTensor | Graph | Data
        """
        return _unwrap_single(self.data.input)

    @property
    def conditional_variables(self):
        """
        The conditional variables associated with the condition.

        :return: The conditional variables.
        :rtype: LabelTensor | torch.Tensor | None
        """
        if hasattr(self.data, "conditional_variables"):
            return self.data.conditional_variables

        return None


class _TensorDataCondition(TensorCondition, DataCondition):
    """
    Tensor variant of :class:`DataCondition`.
    """


class _GraphDataCondition(GraphCondition, DataCondition):
    """
    Graph variant of :class:`DataCondition`.
    """
