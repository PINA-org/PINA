"""Module for the Domain Interface."""

from abc import ABCMeta, abstractmethod


class DomainInterface(metaclass=ABCMeta):
    """
    Abstract interface for all geometric domains.
    """

    @abstractmethod
    def is_inside(self, point, check_border):
        """
        Check if a point is inside the domain.

        :param LabelTensor point: A single point with labeled coordinates
            matching the domain variables.
        :param bool check_border: Whether to include boundaries in membership
            tests. Concrete domains may apply tolerances or surface-only rules.
        :return: Whether the point is inside the domain or not.
        :rtype: bool | torch.Tensor
        """

    @abstractmethod
    def update(self, domain):
        """
        Return a new domain incorporating compatible geometry definitions.

        :param DomainInterface domain: The compatible domain providing updated
            definitions. Merge or replacement semantics depend on the concrete
            domain implementation.
        :return: A new domain with the updated geometry.
        :rtype: DomainInterface
        """

    @abstractmethod
    def sample(self, n, mode, variables):
        """
        The sampling routine.

        :param int n: The positive sample count or per-axis resolution,
            depending on the concrete domain and mode.
        :param str mode: A method supported by ``sample_modes``.
        :param variables: The coordinate names to return, or ``"all"`` for
            every variable.
        :type variables: str | list[str] | tuple[str, ...] | set[str]
        :return: The sampled points with labeled coordinate columns.
        :rtype: LabelTensor
        """

    @abstractmethod
    def partial(self):
        """
        Return the boundary of the domain as a new domain object.

        :raises NotImplementedError: If boundary construction is unsupported.
        :return: A new domain representing the boundary for sampling.
        :rtype: DomainInterface
        """

    @property
    @abstractmethod
    def sample_modes(self):
        """
        The list of available sampling modes.

        :return: The list of available sampling modes.
        :rtype: list[str]
        """

    @property
    @abstractmethod
    def variables(self):
        """
        The list of variables of the domain.

        :return: The list of variables of the domain.
        :rtype: list[str]
        """

    @property
    @abstractmethod
    def domain_dict(self):
        """
        The dictionary representing the domain.

        :return: The dictionary representing the domain.
        :rtype: dict[str, Any]
        """

    @property
    @abstractmethod
    def range(self):
        """
        The range variables of the domain.

        :return: The range variables of the domain.
        :rtype: dict[str, Any]
        """

    @property
    @abstractmethod
    def fixed(self):
        """
        The fixed variables of the domain.

        :return: The fixed variables of the domain.
        :rtype: dict[str, Any]
        """
