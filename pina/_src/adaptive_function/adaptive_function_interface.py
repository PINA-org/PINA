"""Module for the Adaptive Function Interface."""

from abc import ABCMeta, abstractmethod


class AdaptiveFunctionInterface(metaclass=ABCMeta):
    """
    Abstract interface for all adaptive functions.
    """

    @abstractmethod
    def forward(self, x):
        """
        Compute the transformation of the adaptive function on the input.

        :param x: The input tensor to evaluate the adaptive function. Its shape
            must be compatible with the underlying activation.
        :type x: torch.Tensor | LabelTensor
        :return: The scaled activation of the scaled and shifted input. For
            the built-in adaptive activations, the output has the input shape.
        :rtype: torch.Tensor | LabelTensor
        """

    @property
    @abstractmethod
    def alpha(self):
        """
        The output scaling parameter of the adaptive function.

        :return: The alpha parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """

    @property
    @abstractmethod
    def beta(self):
        """
        The input scaling parameter of the adaptive function.

        :return: The beta parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """

    @property
    @abstractmethod
    def gamma(self):
        """
        The input shifting parameter of the adaptive function.

        :return: The gamma parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """
