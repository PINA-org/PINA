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

    :Example:

    >>> from pina.domain import CartesianDomain
    >>> from pina.equation import Equation
    >>> from pina import Condition

    >>> # Equation to be satisfied over the domain: # x^2 + y^2 - 1 = 0
    >>> def dummy_equation(pts):
    ...     return pts["x"]**2 + pts["y"]**2 - 1

    >>> domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
    >>> condition = Condition(domain=domain, equation=Equation(dummy_equation))
    """

    # Available fields, domain and equation data types
    __fields__ = ["domain", "equation"]
    _avail_domain_cls = (DomainInterface, str)
    _avail_equation_cls = BaseEquation

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
        Return the data point at the specified index.

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
        Store the domain and the equation for the condition. It sets the
        attributes ``domain`` and ``equation`` of the condition instance based
        on the provided keyword arguments.

        :param dict kwargs: The keyword arguments containing the data to be
            stored.
        """
        # Store domain and equation as attributes of the condition instance
        setattr(self, "domain", kwargs.get("domain"))
        setattr(self, "equation", kwargs.get("equation"))

    def materialize(self, ids, device=None, batch_fn=None):
        """
        Materialize the data points at the given ids.

        :raises NotImplementedError: Always raised since the data points are
            not stored in a list-like structure and are sampled from the
            domain at training time.
        """
        raise NotImplementedError(
            "Data points in a DomainEquationCondition are not stored and are "
            "sampled from the domain at training time. Therefore, the "
            ":meth:`materialize` method is not implemented for this condition."
        )

    def evaluate(self, batch, solver):
        """
        Evaluate the residual of the condition on the given batch using the
        solver.

        This method computes the non-aggregated, element-wise residual of the
        condition. A forward pass of the solver's model is performed on the
        input samples, and the condition residual is evaluated accordingly.

        The returned tensor is not reduced, preserving the per-sample residual
        values.

        :param dict batch: The batch containing the data required by the
            condition evaluation.
        :param BaseSolver solver: The solver used to perform the forward pass
            and compute the residual. The solver provides access to the model
            and its parameters, which may be necessary for evaluating the
            condition residual.
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
        :rtype: BaseEquation
        """
        return self._equation

    @equation.setter
    def equation(self, value):
        """
        Set the equation associated with this condition.

        :param BaseEquation value: The equation to associate with the condition.
        :raises ValueError: If ``value`` is not an instance of
            :class:`~pina.equation.base_equation.BaseEquation`.
        """
        # Check consistency
        check_consistency(value, self._avail_equation_cls)
        self._equation = value

    @property
    def domain(self):
        """
        The domain associated with the condition.

        :return: The domain.
        :rtype: DomainInterface
        """
        return self._domain

    @domain.setter
    def domain(self, value):
        """
        Set the domain associated with this condition.

        :param DomainInterface value: The domain to associate with the
            condition.
        :raises ValueError: If ``value`` is neither a string nor an instance of
            :class:`~pina.domain.domain_interface.DomainInterface`.
        """
        # Check consistency
        check_consistency(value, self._avail_domain_cls)
        self._domain = value
