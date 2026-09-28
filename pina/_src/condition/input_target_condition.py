"""Module for the Input-Target Condition class."""

import torch

from pina._src.condition.tensor_condition import TensorCondition
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency


class InputTargetCondition(TensorCondition):
    """
    Supervised condition defined by tensor ``input`` and ``target`` data.

    The condition splits the data into batches during :meth:`store_data`.
    ``self.data`` is a list of dictionaries, one per batch, and
    :meth:`materialize` returns the batch selected by a single id.
    """

    __fields__ = ["input", "target"]
    _avail_input_cls = (torch.Tensor, LabelTensor)
    _avail_target_cls = (torch.Tensor, LabelTensor)

    def __init__(self, input, target):
        super().__init__()
        check_consistency(input, self._avail_input_cls)
        check_consistency(target, self._avail_target_cls)
        if input.shape[0] != target.shape[0]:
            raise ValueError(
                "Input and target must have the same number of samples. "
                f"Got {input.shape[0]} and {target.shape[0]}."
            )
        self._input = input
        self._target = target

    def evaluate(self, batch, solver):
        return solver.forward(batch["input"]) - batch["target"]

    @property
    def input(self):
        return self._input

    @property
    def target(self):
        return self._target
