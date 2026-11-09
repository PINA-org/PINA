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
        """
        Initialization of the :class:`InputTargetCondition` class.

        :param input: The input samples passed to the model. The first
            dimension indexes the samples.
        :type input: torch.Tensor | LabelTensor
        :param target: The target values corresponding to the input samples.
            The first dimension must match that of the input.
        :type target: torch.Tensor | LabelTensor
        :raises ValueError: If the input or target has an unsupported type,
            or their numbers of samples differ.
        """
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
        """
        Compute the prediction residual for a batch of paired samples.

        The residual is the model prediction minus the target values.
        No reduction or loss aggregation is applied.

        :param dict[str, Any] batch: The batch containing tensor input samples
            under ``"input"`` and tensor target values under ``"target"``.
        :param pina.solver.BaseSolver solver: The solver providing the forward
            pass.
        :raises KeyError: If the batch lacks the ``"input"`` or ``"target"``
            key.
        :raises RuntimeError: If the prediction and target shapes are
            incompatible for subtraction.
        :return: The difference between predictions and targets, with the shape
            resulting from their broadcasted subtraction.
        :rtype: torch.Tensor | LabelTensor
        """
        return solver.forward(batch["input"]) - batch["target"]

    @property
    def input(self):
        """
        Return the input samples associated with the condition.

        :return: The original input samples, before batching.
        :rtype: torch.Tensor | LabelTensor
        """
        return self._input

    @property
    def target(self):
        """
        Return the target values associated with the condition.

        :return: The original target values, before batching.
        :rtype: torch.Tensor | LabelTensor
        """
        return self._target
