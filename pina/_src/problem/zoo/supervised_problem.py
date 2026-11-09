"""Formulation of a Supervised Problem in PINA."""

from torch_geometric.data import Data

from pina._src.condition.graph_input_target_condition import (
    GraphInputTargetCondition,
)
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.core.graph import Graph
from pina._src.problem.base_problem import BaseProblem


class SupervisedProblem(BaseProblem):
    """
    Definition of a supervised-learning problem.

    This class provides a simple way to define a supervised problem using the
    :class:`~pina.condition.InputTargetCondition` for tensors or
    :class:`~pina.condition.GraphInputTargetCondition` for a graph. Graph inputs
    are modified in place by assigning the target to their ``y`` attribute.

    :Example:

        >>> import torch
        >>> from pina.problem.zoo import SupervisedProblem
        >>> input_data = torch.rand((100, 10))
        >>> output_data = torch.rand((100, 10))
        >>> problem = SupervisedProblem(input_data, output_data)
    """

    # TODO: This is necessary to override the abstract properties of
    # BaseProblem, but it is not an ideal solution. We should consider
    # a different desgin to manage input and output variables.
    conditions = {}
    output_variables = None
    input_variables = None

    def __init__(
        self, input_, output_, input_variables=None, output_variables=None
    ):
        """
        Initialization of the :class:`SupervisedProblem` class.

        :param input_: Input samples, normally ``(n, n_inputs)`` tensors, or a
            graph with an ``x`` node-feature field. The original object is kept.
        :type input_: torch.Tensor | LabelTensor | pina.graph.Graph |
            torch_geometric.data.Data
        :param output_: Target values, normally ``(n, n_outputs)`` tensors. For
            tensor inputs the sample count must match. For graph inputs this is
            assigned directly to ``input_.y`` without target validation.
        :type output_: torch.Tensor | LabelTensor
        :param input_variables: The input variable names, stored unchanged.
            ``None`` leaves the names unspecified; no inference is performed.
            Default is ``None``.
        :type input_variables: list[str] | None
        :param output_variables: The output variable names, stored unchanged.
            ``None`` leaves the names unspecified; no inference is performed.
            Default is ``None``.
        :type output_variables: list[str] | None
        :raises ValueError: If a tensor condition rejects input/target types or
            their sample counts differ.
        """
        # Set input and output variables
        self.input_variables = input_variables
        self.output_variables = output_variables

        # Set the condition
        if isinstance(input_, (Graph, Data)):
            input_.y = output_
            self.conditions["data"] = GraphInputTargetCondition(graph=input_)
        else:
            self.conditions["data"] = InputTargetCondition(
                input=input_, target=output_
            )
        super().__init__()
