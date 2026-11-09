"""Module for the Operation Interface."""

from abc import ABCMeta, abstractmethod
from pina._src.domain.domain_interface import DomainInterface


class OperationInterface(DomainInterface, metaclass=ABCMeta):
    """
    Abstract interface for all set operations defined on geometric domains.
    """

    @property
    @abstractmethod
    def geometries(self):
        """
        The list of domains on which to perform the set operation.

        :return: The list of domains on which to perform the set operation.
        :rtype: list[BaseDomain] | tuple[BaseDomain, ...]
        """

    @geometries.setter
    @abstractmethod
    def geometries(self, values):
        """
        Setter for the ``geometries`` property.

        :param values: The geometries to be set.
        :type values: list[BaseDomain] | tuple[BaseDomain, ...]
        """
