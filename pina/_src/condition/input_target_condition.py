"""Module for the Input-Target Condition class."""

import torch
from torch_geometric.data import Data

from pina._src.condition.base_condition import BaseCondition, _unwrap_single
from pina._src.condition.graph_condition import GraphCondition, _is_graph
from pina._src.condition.tensor_condition import TensorCondition
from pina._src.core.graph import Graph
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency


class InputTargetCondition(BaseCondition):
    """
    The class :class:`InputTargetCondition` represents a supervised condition
    defined by both ``input`` and ``target`` data. The model is trained to
    reproduce the ``target`` values given the ``input``. Supported data types
    include :class:`torch.Tensor`, :class:`~pina.label_tensor.LabelTensor`,
    :class:`~pina.graph.Graph`, or :class:`~torch_geometric.data.Data`.

    :Example:

    >>> from pina import Condition, LabelTensor
    >>> from pina.graph import Graph
    >>> import torch

    >>> pos = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> edge_index = torch.randint(0, 100, (2, 300))
    >>> graph = Graph(pos=pos, edge_index=edge_index)

    >>> input = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> condition = Condition(input=input, target=graph)
    """

    # Available fields, input, and target data types
    __fields__ = ["input", "target"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph)
    _avail_target_cls = (torch.Tensor, LabelTensor, Data, Graph)

    def __new__(cls, input, target):
        """
        Check the types of ``input`` and ``target`` data and instantiate the
        appropriate condition variant accordingly.

        :param input: The input data associated with the condition.
        :param target: The target data associated with the condition.
        :raises ValueError: If ``input`` or ``target`` is not of an available
            type.
        :return: A tensor or graph variant of
            :class:`InputTargetCondition`.
        :rtype: InputTargetCondition
        """
        # Check input type - if iterable, ensure it is either Data or Graph
        if isinstance(input, (list, tuple)):
            check_consistency(input, (Data, Graph))
        else:
            check_consistency(input, cls._avail_input_cls)

        # Check target type - if iterable, ensure it is either Data or Graph
        if isinstance(target, (list, tuple)):
            check_consistency(target, (Data, Graph))
        else:
            check_consistency(target, cls._avail_target_cls)

        # Instantiate the variant matching the data types
        variant = (
            _GraphInputTargetCondition
            if _is_graph(input) or _is_graph(target)
            else _TensorInputTargetCondition
        )
        return super().__new__(variant)

    def evaluate(self, batch, solver):
        """
        Evaluate the residual of the condition on the given batch using the
        solver.

        This method computes the non-aggregated, element-wise residual of the
        condition. A forward pass of the solver's model is performed on the
        input samples, and the condition residual is evaluated accordingly.

        The returned tensor is not reduced, preserving the per-sample residual
        values.

        :param dict batch: The batch containing the data required by the
            condition evaluation.
        :param BaseSolver solver: The solver used to perform the forward pass
            and compute the residual.
        :return: The non-aggregated residual tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        return solver.forward(batch["input"]) - batch["target"]

    @property
    def input(self):
        """
        The input data associated with the condition.

        :return: The input data.
        :rtype: torch.Tensor | LabelTensor | Graph | Data
        """
        return _unwrap_single(self.data.input)

    @property
    def target(self):
        """
        The target data associated with the condition.

        :return: The target data.
        :rtype: torch.Tensor | LabelTensor | Graph | Data
        """
        return _unwrap_single(self.data.target)


class _TensorInputTargetCondition(TensorCondition, InputTargetCondition):
    """
    Tensor variant of :class:`InputTargetCondition`.
    """


class _GraphInputTargetCondition(GraphCondition, InputTargetCondition):
    """
    Graph variant of :class:`InputTargetCondition`.
    """
