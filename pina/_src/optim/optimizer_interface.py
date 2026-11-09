"""Module for the Optimizer Interface."""

from abc import ABCMeta, abstractmethod


class OptimizerInterface(metaclass=ABCMeta):
    """
    Abstract interface for all optimizers.
    """

    @abstractmethod
    def hook(self, parameters):
        """
        Initialize the optimizer for the supplied model parameters.

        :param parameters: The parameters to optimize, such as the iterator
            returned by :meth:`torch.nn.Module.parameters`, or parameter-group
            dictionaries accepted by the concrete backend.
        :type parameters: Iterable[torch.Tensor] | Iterable[dict[str, Any]]
        """

    @property
    @abstractmethod
    def instance(self):
        """
        The underlying optimizer object.

        :return: The backend optimizer, or ``None`` before initialization when
            the implementation uses deferred construction.
        :rtype: object
        """
