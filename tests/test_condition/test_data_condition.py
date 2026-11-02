import pytest
import torch

from pina import LabelTensor
from pina._src.core.utils import labelize_forward
from pina.condition import DataCondition

n_samples = 10


def _create_tensor_data(use_lt, conditional_variables):
    if use_lt:
        input_tensor = LabelTensor(torch.rand((n_samples, 3)), ["x", "y", "z"])
        cond_vars = LabelTensor(torch.rand((n_samples, 1)), ["a"])
        cond_vars = cond_vars if conditional_variables else None
        return input_tensor, cond_vars

    input_tensor = torch.rand((n_samples, 3))
    cond_vars = torch.rand((n_samples, 1))
    cond_vars = cond_vars if conditional_variables else None
    return input_tensor, cond_vars


def _assert_tensor_type(t, use_lt):
    if use_lt:
        assert isinstance(t, LabelTensor)
    else:
        assert isinstance(t, torch.Tensor) and not isinstance(t, LabelTensor)


class DummySolver:
    def __init__(self, use_lt, input_vars, cond_vars):
        if use_lt:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=input_vars,
                output_variables="z",
            )

        self.cond_vars = cond_vars
        self._params = None

    def forward(self, pts):
        factor = self.cond_vars if self.cond_vars is not None else 1.0
        return pts.mean(dim=-1, keepdim=True) * factor


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_constructor(use_lt, conditional_variables):
    input_tensor, cond_vars = _create_tensor_data(
        use_lt, conditional_variables
    )
    condition = DataCondition(
        input=input_tensor, conditional_variables=cond_vars
    )

    assert isinstance(condition, DataCondition)
    _assert_tensor_type(condition.input, use_lt)
    if cond_vars is not None:
        _assert_tensor_type(condition.conditional_variables, use_lt)

    assert torch.allclose(condition.input, input_tensor)
    if cond_vars is not None:
        assert torch.allclose(condition.conditional_variables, cond_vars)

    if use_lt:
        assert condition.input.labels == ["x", "y", "z"]
        if cond_vars is not None:
            assert condition.conditional_variables.labels == ["a"]

    with pytest.raises(ValueError):
        DataCondition(
            input="invalid_input", conditional_variables=cond_vars
        )

    with pytest.raises(ValueError):
        DataCondition(
            input=input_tensor, conditional_variables="invalid_cond_vars"
        )

    with pytest.raises(ValueError):
        DataCondition(input=[input_tensor], conditional_variables=cond_vars)

    if cond_vars is not None:
        with pytest.raises(ValueError):
            DataCondition(input=input_tensor, conditional_variables=[cond_vars])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_get_item(use_lt, conditional_variables):
    input_tensor, cond_vars = _create_tensor_data(
        use_lt, conditional_variables
    )
    condition = DataCondition(
        input=input_tensor, conditional_variables=cond_vars
    )
    condition.store_data(batch_size=1)

    index = 0
    item = condition[index]

    assert isinstance(item, dict)
    _assert_tensor_type(item["input"], use_lt)
    if cond_vars is not None:
        _assert_tensor_type(item["conditional_variables"], use_lt)

    assert torch.allclose(item["input"][0], input_tensor[index])
    if cond_vars is not None:
        assert torch.allclose(
            item["conditional_variables"][0], cond_vars[index]
        )


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_materialize(use_lt, conditional_variables):
    input_, cond_vars = _create_tensor_data(use_lt, conditional_variables)
    condition = DataCondition(input=input_, conditional_variables=cond_vars)
    condition.store_data(batch_size=2)

    batch = condition.materialize(0)

    assert isinstance(batch, dict)
    assert "input" in batch
    if cond_vars is not None:
        assert "conditional_variables" in batch

    cls = LabelTensor if use_lt else torch
    expected_input = cls.stack([input_[i] for i in range(2)])

    assert torch.allclose(batch["input"], expected_input)
    assert batch["input"].shape == expected_input.shape

    if cond_vars is not None:
        exp_cond = cls.stack([cond_vars[i] for i in range(2)])
        assert torch.allclose(batch["conditional_variables"], exp_cond)
        assert batch["conditional_variables"].shape == exp_cond.shape


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_evaluate(use_lt, conditional_variables):
    input_, cond_vars = _create_tensor_data(use_lt, conditional_variables)
    input_vars = input_.labels if use_lt else None

    condition = DataCondition(input=input_, conditional_variables=cond_vars)
    condition.store_data()
    batch = condition.materialize(0)
    solver = DummySolver(use_lt, input_vars, cond_vars)

    loss = condition.evaluate(batch, solver)
    expected = solver.forward(batch["input"])

    assert torch.allclose(loss, expected)
