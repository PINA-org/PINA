"""Module for the Domain-Equation Condition class."""

from pina._src.condition.base_condition import BaseCondition
from pina._src.domain.domain_interface import DomainInterface
from pina._src.equation.base_equation import BaseEquation
from pina._src.core.utils import check_consistency


class DomainEquationCondition(BaseCondition):
    """
    The class :class:`DomainEquationCondition` defines a condition based on a
    ``domain`` and an ``equation``. This condition is typically used in
    physics-informed problems, where the model is trained to satisfy a given
    ``equation`` over a specified ``domain``. The ``domain`` is used to sample
    points where the ``equation`` residual is evaluated and minimized during
    training.

    This condition stores no batches. During data-module setup, the problem
    replaces it with an :class:`pina.condition.InputEquationCondition` using
    points previously sampled by
    :meth:`pina.problem.BaseProblem.discretise_domain`.

    :Example:

        >>> from pina.domain import CartesianDomain
        >>> from pina.equation import Equation
        >>> from pina.condition import DomainEquationCondition
        >>> def dummy_equation(input_, output_):
        ...     return output_.extract("u") - input_.extract("x") ** 2
        >>> domain = CartesianDomain({"x": [0.0, 1.0]})
        >>> condition = DomainEquationCondition(
        ...     domain=domain, equation=Equation(dummy_equation)
        ... )
    """

    # Available fields, domain and equation data types
    __fields__ = ["domain", "equation"]
    _avail_domain_cls = (DomainInterface, str)
    _avail_equation_cls = BaseEquation

    def __init__(self, domain, equation):
        """
        Initialization of the :class:`DomainEquationCondition` class.

        :param domain: The sampling domain, or its name in the problem's domain
            mapping. A problem registers domain objects under condition names.
        :type domain: pina.domain.DomainInterface | str
        :param pina.equation.BaseEquation equation: The equation to be satisfied
            over the specified domain.
        :raises ValueError: If ``domain`` or ``equation`` fails its setter's
            type validation.
        """
        self.domain = domain
        self.equation = equation

    def __len__(self):
        """
        Return the number of data points in the condition.

        :raises NotImplementedError: Always raised since the number of points is
            determined by the domain sampling strategy and is not fixed.
        """
        raise NotImplementedError(
            "The number of data points in a DomainEquationCondition is not "
            "fixed and is determined by the domain sampling strategy. "
            "Therefore, the :meth:`__len__` method is not implemented for this "
            "condition."
        )

    def __getitem__(self, idx):
        """
        Reject indexing because this condition stores no batches.

        :param Any idx: The requested index; unused.
        :raises NotImplementedError: Always raised since the data points are not
            stored in a list-like structure and cannot be accessed by index.
        """
        raise NotImplementedError(
            "Data points in a DomainEquationCondition are not stored in a "
            "list-like structure and cannot be accessed by index. Therefore, "
            "the :meth:`__getitem__` method is not implemented for this "
            "condition."
        )

    def store_data(self, **kwargs):
        """
        Update the domain and equation from the supplied keyword arguments.

        Omitted fields remain unchanged. No batches are created.

        :param dict[str, Any] kwargs: Optional replacement fields:

            * ``domain`` (pina.domain.DomainInterface | str): The sampling
              domain or its registered name.
            * ``equation`` (pina.equation.BaseEquation): The residual equation.

            Other keywords are ignored. Neither field has a default replacement
            value; its existing value is retained when omitted.
        :raises ValueError: If a supplied field fails its setter's type
            validation.
        """
        # Store only the provided fields as attributes of the condition
        for field in self.__fields__:
            if field in kwargs:
                setattr(self, field, kwargs[field])

    def materialize(self, ids, device=None, batch_fn=None):
        """
        Reject materialization because this condition stores no batches.

        :param Any ids: The requested identifiers; unused.
        :param device: The requested device; unused. Default is ``None``.
        :type device: str | torch.device | int | None
        :param batch_fn: The requested construction override; unused. Default
            is ``None``.
        :type batch_fn: Callable | None
        :raises NotImplementedError: Always raised since the data points are
            not stored here. The problem replaces this condition with an
            input-equation condition after domain discretisation.
        """
        raise NotImplementedError(
            "Data points in a DomainEquationCondition are not stored and are "
            "sampled from the domain at training time. Therefore, the "
            ":meth:`materialize` method is not implemented for this condition."
        )

    def evaluate(self, batch, solver):
        """
        Reject direct evaluation of a domain-equation condition.

        :param Any batch: The supplied batch; unused.
        :param pina.solver.BaseSolver solver: The supplied solver; unused.
        :raises NotImplementedError: Always raised since any domain-equation
            condition is transformed into an input-equation condition before
            evaluation, and the residual is computed using the input-equation
            condition's evaluation method.
        """
        raise NotImplementedError(
            "Domain-equation conditions are transformed into input-equation "
            "conditions before evaluation, and the residual is computed using "
            "the input-equation condition's evaluation method. Therefore, the "
            "evaluate method is not implemented for domain-equation conditions."
        )

    @property
    def equation(self):
        """
        The equation associated with the condition.

        :return: The equation.
        :rtype: pina.equation.BaseEquation
        """
        return self._equation

    @equation.setter
    def equation(self, value):
        """
        Set the equation associated with this condition.

        :param pina.equation.BaseEquation value: The equation to associate with
            the condition.
        :raises ValueError: If ``value`` is not an instance of
            :class:`pina.equation.BaseEquation`.
        """
        # Check consistency
        check_consistency(value, self._avail_equation_cls)
        self._equation = value

    @property
    def domain(self):
        """
        The domain associated with the condition.

        :return: The sampling domain object or its registered name.
        :rtype: pina.domain.DomainInterface | str
        """
        return self._domain

    @domain.setter
    def domain(self, value):
        """
        Set the domain associated with this condition.

        :param value: The sampling domain object or its registered name.
        :type value: pina.domain.DomainInterface | str
        :raises ValueError: If ``value`` is neither a string nor an instance of
            :class:`pina.domain.DomainInterface`.
        """
        # Check consistency
        check_consistency(value, self._avail_domain_cls)
        self._domain = value
