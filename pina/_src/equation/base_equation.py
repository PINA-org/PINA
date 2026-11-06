"""Module for the Base Equation."""

from abc import ABCMeta, abstractmethod
import torch


class BaseEquation(metaclass=ABCMeta):
    """
    Base class for all equations, implementing common functionality.

    Equations are fundamental components in PINA, representing mathematical
    constraints that must be satisfied by the model outputs. They can be passed
    to
    :class:`~pina.condition.DomainEquationCondition`
    or :class:`~pina.condition.InputEquationCondition`
    objects to define the conditions under which the model is trained.

    All specific equation types should inherit from this class and implement its
    abstract methods.

    This class is not meant to be instantiated directly.
    """

    @abstractmethod
    def residual(self, input_, output_, params_):
        """
        Evaluate the equation residual at the given inputs.

        :param LabelTensor input_: Labeled input points, normally of shape
            ``(n, n_inputs)``. Differential equations require gradient tracking.
        :param LabelTensor output_: Labeled predictions, normally of shape
            ``(n, n_outputs)``, connected to ``input_`` when differentiated.
        :param params_: The named unknown parameters for an
            :class:`~pina.problem.InverseProblem`, or ``None`` when unused.
            This abstract method has no default argument.
        :type params_: dict[str, torch.nn.Parameter] | None
        :return: The residual tensor. Its shape and labels depend on the
            concrete equation and any reduction.
        :rtype: torch.Tensor | LabelTensor
        """

    def to(self, device):
        """
        Move directly stored tensors to a device, modifying this equation.

        Tensor attributes and tensors immediately inside list/dictionary
        attributes are moved. Those containers are replaced; nested containers,
        tuples, constituent equation objects, and callable closures are not
        traversed.

        :param device: The target device accepted by :meth:`torch.Tensor.to`.
        :type device: torch.device | str | int
        :return: This same equation instance after moving supported attributes.
        :rtype: BaseEquation
        """
        # Iterate over all attributes of the Equation
        for key, val in self.__dict__.items():

            # Move tensors in dictionaries to the specified device
            if isinstance(val, dict):
                self.__dict__[key] = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in val.items()
                }

            # Move tensors in lists to the specified device
            elif isinstance(val, list):
                self.__dict__[key] = [
                    v.to(device) if torch.is_tensor(v) else v for v in val
                ]

            # Move tensor attributes to the specified device
            elif torch.is_tensor(val):
                self.__dict__[key] = val.to(device)

        return self
