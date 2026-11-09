"""Module for all set-based operations Base class."""

from copy import deepcopy
from abc import ABCMeta
from pina._src.domain.operation_interface import OperationInterface
from pina._src.domain.base_domain import BaseDomain
from pina._src.core.utils import check_consistency


class BaseOperation(OperationInterface, BaseDomain, metaclass=ABCMeta):
    """
    Base class for all set operation defined on geometric domains, implementing
    common functionality.

    All specific operation types should inherit from this class and implement
    the abstract methods defined in both the following interfaces:
    :class:`~pina.domain.OperationInterface`, and
    :class:`~pina.domain.DomainInterface`.

    This class is not meant to be instantiated directly.
    """

    def __init__(self, geometries):
        """
        Initialization of the :class:`BaseOperation` class.

        :param geometries: The list of domains on which to perform the set
            operation, including nested operations. The container is stored by
            reference. Empty containers are accepted, but sampling and updates
            generally require at least one geometry.
        :type geometries: list[BaseDomain] | tuple[BaseDomain, ...]
        :raises TypeError: If geometries is neither a list nor a tuple.
        :raises ValueError: If geometries elements are not instances of
            :class:`~pina.domain.BaseDomain`.
        :raises NotImplementedError: If the geometries have different variable
            names or dimensions.
        """
        super().__init__()
        self.geometries = geometries

    def update(self, domain):
        """
        Apply a compatible domain update to every constituent geometry.

        The receiver is deep-copied, then each original geometry's ``update``
        method supplies the corresponding new constituent.

        :param BaseDomain domain: The geometry definitions to apply. Must be an
            instance of the first constituent's class.
        :raises NotImplementedError: If the geometries involved in the operation
            are of different types.
        :raises TypeError: If the passed domain is not of the same type of all
            the geometries involved in the operation.
        :raises IndexError: If there are no constituent geometries.
        :return: A new operation with updated constituent geometries.
        :rtype: BaseOperation
        """
        # Check all geometries are of the same type
        domain_type = type(self.geometries[0])
        if not all(isinstance(g, domain_type) for g in self.geometries):
            raise NotImplementedError(
                f"The {self.__class__.__name__} of geometries of different"
                " types does not support the update operation. All geometries"
                " must be of the same type."
            )

        # Check domain type consistency
        if not isinstance(domain, domain_type):
            raise TypeError(
                f"Cannot update the {self.__class__.__name__} of domains of"
                f" type {domain_type} with domain of type {type(domain)}."
            )

        # Update each geometry
        updated = deepcopy(self)
        updated.geometries = [geom.update(domain) for geom in self.geometries]

        return updated

    @property
    def sample_modes(self):
        """
        The list of available sampling modes.

        :raises TypeError: If there are no constituent geometries.
        :return: A new list of modes supported by every constituent, with no
            guaranteed ordering.
        :rtype: list[str]
        """
        return list(
            set.intersection(
                *map(set, [g.sample_modes for g in self.geometries])
            )
        )

    @property
    def variables(self):
        """
        The list of variables of the domain.

        :return: A new sorted list of constituent variable names.
        :rtype: list[str]
        """
        return sorted({v for g in self.geometries for v in g.variables})

    @property
    def domain_dict(self):
        """
        Returns a dictionary representation of the operation domain.

        :return: A new mapping with ``"type"`` holding the operation class name
            and ``"geometries"`` holding the constituents' domain dictionaries.
            Nested operations produce recursive dictionaries.
        :rtype: dict[str, str | list[dict[str, Any]]]
        """
        return {
            "type": self.__class__.__name__,
            "geometries": [geom.domain_dict for geom in self.geometries],
        }

    @property
    def geometries(self):
        """
        The domains on which to perform the set operation.

        :return: The stored constituent container, including nested operations.
            Mutating a returned list bypasses setter validation.
        :rtype: list[BaseDomain] | tuple[BaseDomain, ...]
        """
        return self._geometries

    @property
    def range(self):
        """
        The range variables of each geometry.

        :return: A new ``"geometry_i"`` mapping to each constituent's ranged
            definitions. Nested operations introduce further mapping levels.
        :rtype: dict[str, dict[str, Any]]
        """
        return {f"geometry_{i}": g.range for i, g in enumerate(self.geometries)}

    @property
    def fixed(self):
        """
        The fixed variables of each geometry.

        :return: A new ``"geometry_i"`` mapping to each constituent's fixed
            definitions. Nested operations introduce further mapping levels.
        :rtype: dict[str, dict[str, Any]]
        """
        return {f"geometry_{i}": g.fixed for i, g in enumerate(self.geometries)}

    @geometries.setter
    def geometries(self, values):
        """
        Setter for the ``geometries`` property.

        :param values: The constituent domains, including nested operations,
            stored by reference after validation. Empty containers are accepted.
        :type values: list[BaseDomain] | tuple[BaseDomain, ...]
        :raises TypeError: If values is neither a list nor a tuple.
        :raises ValueError: If values elements are not instances of
            :class:`~pina.domain.BaseDomain`.
        :raises NotImplementedError: If the geometries have different variable
            names or dimensions.
        """
        # Check geometries are list or tuple
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                "geometries must be either a list or a tuple of BaseDomain."
            )

        # Check consistency
        check_consistency(values, (BaseDomain, BaseOperation))

        # Check geometries
        for v in values:
            if v.variables != values[0].variables:
                raise NotImplementedError(
                    f"The {self.__class__.__name__} of geometries living in "
                    "different ambient spaces is not well-defined. "
                    "All geometries must share the same dimensions and labels."
                )

        self._geometries = values
