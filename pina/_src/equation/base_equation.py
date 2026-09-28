"""Module for the Base Equation."""

from abc import ABCMeta, abstractmethod
import torch


class BaseEquation(metaclass=ABCMeta):
    """
    Base class for all equations, implementing common functionality.

    Equations are fundamental components in PINA, representing mathematical
    constraints that must be satisfied by the model outputs. They can be passed
    to
    :class:`~pina.condition.domain_equation_condition.DomainEquationCondition`
    objects to define the conditions under which the model is trained.

    All specific equation types should inherit from this class and implement its
    abstract methods.

    This class is not meant to be instantiated directly.
    """

    @abstractmethod
    def residual(self, input_, output_, params_):
        """
        Evaluate the equation residual at the given inputs.

        :param LabelTensor input_: The input points where the residual is
            computed.
        :param LabelTensor output_: The output tensor, potentially produced by a
            :class:`torch.nn.Module` instance.
        :param dict params_: An optional dictionary of unknown parameters, used
            in :class:`~pina.problem.inverse_problem.InverseProblem` settings.
            If the equation is not related to an inverse problem, this should be
            set to ``None``. Default is ``None``.
        :return: The residual values of the equation.
        :rtype: LabelTensor
        """

    def to(self, device):
        """
        Move all tensor attributes to the specified device.

        :param torch.device device: The target device to move the tensors to.
        :return: The instance moved to the specified device.
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
