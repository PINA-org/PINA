"""Module for the Equation Interface."""

from abc import ABCMeta, abstractmethod


class EquationInterface(metaclass=ABCMeta):
    """
    Abstract interface for all equations.
    """

    @abstractmethod
    def residual(self, input_, output_, params_=None):
        """
        Evaluate the equation residual at the given inputs.

        :param LabelTensor input_: Labeled input points, normally of shape
            ``(n, n_inputs)``. Differential equations require gradient tracking.
        :param LabelTensor output_: Labeled predictions, normally of shape
            ``(n, n_outputs)``, connected to ``input_`` when differentiated.
        :param params_: The named unknown parameters for an
            :class:`~pina.problem.InverseProblem`, or ``None`` when unused.
            Default is ``None``.
        :type params_: dict[str, torch.nn.Parameter] | None
        :return: The residual tensor. Its shape and labels depend on the
            concrete equation and any reduction.
        :rtype: torch.Tensor | LabelTensor
        """

    @abstractmethod
    def to(self, device):
        """
        Move supported tensor attributes to the specified device.

        :param device: The target device accepted by :meth:`torch.Tensor.to`.
        :type device: torch.device | str | int
        :return: The instance moved to the specified device.
        :rtype: EquationInterface
        """
