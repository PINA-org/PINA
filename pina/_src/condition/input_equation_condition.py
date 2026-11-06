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
        """
        Initialization of the :class:`InputEquationCondition` class.

        :param input: The input samples at which the equation is evaluated.
            The first dimension indexes the samples.
        :type input: torch.Tensor | LabelTensor
        :param pina.equation.BaseEquation equation: The equation used to compute
            residual at the input samples.
        :raises ValueError: If the input or equation has an unsupported type.
        """
        check_consistency(input, self._avail_input_cls)
        check_consistency(equation, self._avail_equation_cls)
        self._input = input
        self._equation = equation

    def evaluate(self, batch, solver):
        """
        Compute the equation residual for a batch of input samples.

        Gradient tracking is enabled on ``batch["input"]`` before the
        forward pass. The equation receives the input samples, model output,
        and the solver's unknown parameters. The residual is returned
        without loss aggregation.

        :param dict[str, Any] batch: The batch containing a tensor or
            :class:`pina.LabelTensor` under ``"input"``. Stored equation entries
            are ignored; the condition's own equation is used.
        :param pina.solver.BaseSolver solver: The solver providing the forward
            pass and the unknown parameters used by the equation.
        :raises KeyError: If the batch does not contain the ``"input"`` key.
        :raises RuntimeError: If the input tensor does not support gradient
            tracking, for example because it has an integer dtype.
        :return: The equation residual evaluated at the input samples. Shape
            and component count are determined by the equation.
        :rtype: torch.Tensor | LabelTensor
        """
        samples = batch["input"].requires_grad_(True)
        output = solver.forward(samples)
        return self.equation.residual(samples, output, solver._params)

    @property
    def input(self):
        """
        Return the input samples associated with the condition.

        :return: The original input samples, before batching.
        :rtype: torch.Tensor | LabelTensor
        """
        return self._input

    @property
    def equation(self):
        """
        Return the equation associated with the condition.

        :return: The equation used to compute the condition residual.
        :rtype: pina.equation.BaseEquation
        """
        return self._equation
