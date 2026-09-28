"""Module for the Input-Equation Condition class."""

import torch

from pina._src.condition.tensor_condition import TensorCondition
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency
from pina._src.equation.base_equation import BaseEquation


class InputEquationCondition(TensorCondition):
    """
    Condition defined by tensor ``input`` data and an ``equation``.

    The condition splits the input into batches during :meth:`store_data`.
    ``self.data`` is a list of dictionaries, one per batch, and
    :meth:`materialize` returns the batch selected by a single id.
    """

    __fields__ = ["input", "equation"]
    _avail_input_cls = (torch.Tensor, LabelTensor)
    _avail_equation_cls = BaseEquation

    def __init__(self, input, equation):
        super().__init__()
        check_consistency(input, self._avail_input_cls)
        check_consistency(equation, self._avail_equation_cls)
        self._input = input
        self._equation = equation

    def evaluate(self, batch, solver):
        samples = batch["input"].requires_grad_(True)
        output = solver.forward(samples)
        return self.equation.residual(samples, output, solver._params)

    @property
    def input(self):
        return self._input

    @property
    def equation(self):
        return self._equation
