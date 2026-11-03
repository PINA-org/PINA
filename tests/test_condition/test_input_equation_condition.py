import pytest
import torch

from pina import LabelTensor
from pina._src.core.utils import labelize_forward
from pina.condition import InputEquationCondition
from pina.equation.zoo import FixedValue

n_samples = 10
input_tensor = LabelTensor(torch.rand((n_samples, 2)), ["x", "y"])
test_equation = FixedValue(0.0)


class DummySolver:
    def __init__(self, input_vars, output_vars):
        self.forward = labelize_forward(
            forward=self.forward,
            input_variables=input_vars,
            output_variables=output_vars,
        )
        self._params = None

    def forward(self, samples):
        return samples


def test_constructor():
    condition = InputEquationCondition(
        input=input_tensor, equation=test_equation
    )

    assert isinstance(condition, InputEquationCondition)
    assert condition.equation is test_equation
    assert isinstance(condition.input, LabelTensor)

    condition = InputEquationCondition(
        input=torch.rand(n_samples, 2), equation=test_equation
    )
    assert isinstance(condition.input, torch.Tensor)
    assert not isinstance(condition.input, LabelTensor)

    with pytest.raises(ValueError):
        InputEquationCondition(input="invalid_input", equation=test_equation)

    with pytest.raises(ValueError):
        InputEquationCondition(input=input_tensor, equation="not_an_equation")


def test_get_item():
    condition = InputEquationCondition(
        input=input_tensor, equation=test_equation
    )
    condition.store_data(batch_size=1)

    item = condition[0]

    assert isinstance(item, dict)
    assert "input" in item
    assert isinstance(item["input"], LabelTensor)
    assert torch.allclose(item["input"][0], input_tensor[0])


def test_materialize():
    condition = InputEquationCondition(
        input=input_tensor, equation=test_equation
    )
    condition.store_data(batch_size=2)

    batch = condition.materialize(1)

    assert isinstance(batch, dict)
    assert "input" in batch
    assert torch.allclose(batch["input"], input_tensor[2:4])
    assert batch["input"].shape == torch.Size([2, 2])


def test_evaluate():
    condition = InputEquationCondition(
        input=input_tensor, equation=test_equation
    )
    condition.store_data()
    batch = condition.materialize(0)
    solver = DummySolver(input_vars=input_tensor.labels, output_vars=["z", "t"])

    residual = condition.evaluate(batch, solver)
    expected = solver.forward(batch["input"])

    assert torch.allclose(residual, expected)
