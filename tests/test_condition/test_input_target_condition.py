import pytest
import torch

from pina import LabelTensor
from pina._src.core.utils import labelize_forward
from pina.condition import InputTargetCondition

n_samples = 10


def _create_tensor_data(use_lt=False):
    if use_lt:
        input_tensor = LabelTensor(torch.rand((n_samples, 3)), ["x", "y", "z"])
        target_tensor = LabelTensor(torch.rand((n_samples, 2)), ["a", "b"])
        return input_tensor, target_tensor

    input_tensor = torch.rand((n_samples, 3))
    target_tensor = torch.rand((n_samples, 2))
    return input_tensor, target_tensor


def _assert_tensor_type(t, use_lt):
    if use_lt:
        assert isinstance(t, LabelTensor)
    else:
        assert isinstance(t, torch.Tensor) and not isinstance(t, LabelTensor)


class DummySolver:
    def __init__(self, use_lt, input_vars, output_vars):
        if use_lt is True:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=input_vars,
                output_variables=output_vars,
            )

    def forward(self, pts):
        return torch.cat(
            [pts.mean(dim=-1, keepdim=True), pts.sum(dim=-1, keepdim=True)],
            dim=-1,
        )


@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor(use_lt):
    input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
    condition = InputTargetCondition(input=input_tensor, target=target_tensor)

    assert isinstance(condition, InputTargetCondition)
    _assert_tensor_type(condition.input, use_lt)
    _assert_tensor_type(condition.target, use_lt)
    assert torch.allclose(condition.input, input_tensor)
    assert torch.allclose(condition.target, target_tensor)

    if use_lt:
        assert condition.input.labels == ["x", "y", "z"]
        assert condition.target.labels == ["a", "b"]

    with pytest.raises(ValueError):
        InputTargetCondition(input="invalid_input", target=target_tensor)

    with pytest.raises(ValueError):
        InputTargetCondition(input=input_tensor, target="invalid_target")


@pytest.mark.parametrize("use_lt", [True, False])
def test_get_item(use_lt):
    input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
    condition = InputTargetCondition(input=input_tensor, target=target_tensor)
    condition.store_data(batch_size=1)

    item = condition[0]

    assert isinstance(item, dict)
    _assert_tensor_type(item["input"], use_lt)
    _assert_tensor_type(item["target"], use_lt)
    assert torch.allclose(item["input"][0], input_tensor[0])
    assert torch.allclose(item["target"][0], target_tensor[0])


@pytest.mark.parametrize("use_lt", [True, False])
def test_materialize(use_lt):
    input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
    condition = InputTargetCondition(input=input_tensor, target=target_tensor)
    condition.store_data(batch_size=2)

    batch = condition.materialize(1)

    expected_input = input_tensor[2:4]
    expected_target = target_tensor[2:4]

    assert isinstance(batch, dict)
    assert "input" in batch
    assert "target" in batch
    assert torch.allclose(batch["input"], expected_input)
    assert torch.allclose(batch["target"], expected_target)
    assert batch["input"].shape == expected_input.shape
    assert batch["target"].shape == expected_target.shape


@pytest.mark.parametrize("use_lt", [True, False])
def test_evaluate(use_lt):
    input_, target_ = _create_tensor_data(use_lt=use_lt)
    input_vars = input_.labels if use_lt else None
    output_vars = target_.labels if use_lt else None

    condition = InputTargetCondition(input=input_, target=target_)
    condition.store_data()
    batch = condition.materialize(0)
    solver = DummySolver(use_lt, input_vars, output_vars)

    residual = condition.evaluate(batch, solver)
    expected = solver.forward(batch["input"]) - batch["target"]

    assert torch.allclose(residual, expected)
