import pytest

from pina.domain import CartesianDomain
from pina.equation.zoo import FixedValue
from pina.condition import DomainEquationCondition

domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
equation = FixedValue(0.0)


def _create_condition():
    return DomainEquationCondition(domain=domain, equation=equation)


def test_store_data():
    condition = _create_condition()

    assert isinstance(condition, DomainEquationCondition)
    assert condition.domain is domain
    assert condition.equation is equation
    assert not hasattr(condition, "data")


def test_get_item():
    condition = _create_condition()

    with pytest.raises(NotImplementedError):
        condition[0]


def test_materialize():
    condition = _create_condition()

    with pytest.raises(NotImplementedError):
        condition.materialize(0)


def test_evaluate():
    condition = _create_condition()

    with pytest.raises(NotImplementedError):
        condition.evaluate(None, None)
