import torch
import pytest

from pina import LabelTensor

data = torch.rand((20, 3))
labels_column = {1: {"name": "space", "dof": ["x", "y", "z"]}}
labels_row = {0: {"name": "samples", "dof": range(20)}}
labels_list = ["x", "y", "z"]
labels_all = labels_column.copy()
labels_all.update(labels_row)


@pytest.mark.parametrize(
    "labels", [labels_column, labels_row, labels_all, labels_list]
)
def test_constructor(labels):
    print(LabelTensor(data, labels))


def test_wrong_constructor():
    with pytest.raises(ValueError):
        LabelTensor(data, ["a", "b"])


@pytest.mark.parametrize("labels", [labels_column, labels_all])
@pytest.mark.parametrize("labels_te", ["z", ["z"], {"space": ["z"]}])
def test_extract_column(labels, labels_te):
    tensor = LabelTensor(data, labels)
    new = tensor.extract(labels_te)
    assert new.ndim == tensor.ndim
    assert new.shape[1] == 1
    assert new.shape[0] == 20
    assert torch.all(torch.isclose(data[:, 2].reshape(-1, 1), new))


@pytest.mark.parametrize("labels", [labels_row, labels_all])
@pytest.mark.parametrize("labels_te", [{"samples": [2]}])
def test_extract_row(labels, labels_te):
    tensor = LabelTensor(data, labels)
    new = tensor.extract(labels_te)
    assert new.ndim == tensor.ndim
    assert new.shape[1] == 3
    assert new.shape[0] == 1
    assert torch.all(torch.isclose(data[2].reshape(1, -1), new))


@pytest.mark.parametrize(
    "labels_te",
    [{"samples": [2], "space": ["z"]}, {"space": "z", "samples": 2}],
)
def test_extract_2D(labels_te):
    labels = labels_all
    tensor = LabelTensor(data, labels)
    new = tensor.extract(labels_te)
    assert new.ndim == tensor.ndim
    assert new.shape[1] == 1
    assert new.shape[0] == 1
    assert torch.all(torch.isclose(data[2, 2].reshape(1, 1), new))


def test_extract_3D():
    data = torch.rand(20, 3, 4)
    labels = {
        1: {"name": "space", "dof": ["x", "y", "z"]},
        2: {"name": "time", "dof": range(4)},
    }
    labels_te = {"space": ["x", "z"], "time": range(1, 4)}

    tensor = LabelTensor(data, labels)
    new = tensor.extract(labels_te)
    tensor2 = LabelTensor(data, labels)
    assert new.ndim == tensor.ndim
    assert new.shape[0] == 20
    assert new.shape[1] == 2
    assert new.shape[2] == 3
    assert torch.all(torch.isclose(data[:, 0::2, 1:4].reshape(20, 2, 3), new))
    assert tensor2.ndim == tensor.ndim
    assert tensor2.shape == tensor.shape
    assert tensor.full_labels == tensor2.full_labels
    assert new.shape != tensor.shape


def test_concatenation_3D():
    data_1 = torch.rand(20, 3, 4)
    labels_1 = ["x", "y", "z", "w"]
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(50, 3, 4)
    labels_2 = ["x", "y", "z", "w"]
    lt2 = LabelTensor(data_2, labels_2)
    lt_cat = LabelTensor.cat([lt1, lt2])
    assert lt_cat.shape == (70, 3, 4)
    assert lt_cat.full_labels[0]["dof"] == range(70)
    assert lt_cat.full_labels[1]["dof"] == range(3)
    assert lt_cat.full_labels[2]["dof"] == ["x", "y", "z", "w"]

    data_1 = torch.rand(20, 3, 4)
    labels_1 = ["x", "y", "z", "w"]
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(20, 2, 4)
    labels_2 = ["x", "y", "z", "w"]
    lt2 = LabelTensor(data_2, labels_2)
    lt_cat = LabelTensor.cat([lt1, lt2], dim=1)
    assert lt_cat.shape == (20, 5, 4)
    assert lt_cat.full_labels[0]["dof"] == range(20)
    assert lt_cat.full_labels[1]["dof"] == range(5)
    assert lt_cat.full_labels[2]["dof"] == ["x", "y", "z", "w"]

    data_1 = torch.rand(20, 3, 2)
    labels_1 = ["x", "y"]
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(20, 3, 3)
    labels_2 = ["z", "w", "a"]
    lt2 = LabelTensor(data_2, labels_2)
    lt_cat = LabelTensor.cat([lt1, lt2], dim=2)
    assert lt_cat.shape == (20, 3, 5)
    assert lt_cat.full_labels[2]["dof"] == ["x", "y", "z", "w", "a"]
    assert lt_cat.full_labels[0]["dof"] == range(20)
    assert lt_cat.full_labels[1]["dof"] == range(3)

    data_1 = torch.rand(20, 2, 4)
    labels_1 = ["x", "y", "z", "w"]
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(20, 3, 4)
    labels_2 = ["x", "y", "z", "w"]
    lt2 = LabelTensor(data_2, labels_2)
    with pytest.raises(RuntimeError):
        LabelTensor.cat([lt1, lt2], dim=2)
    data_1 = torch.rand(20, 3, 2)
    labels_1 = ["x", "y"]
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(20, 3, 3)
    labels_2 = ["z", "w", "a"]
    lt2 = LabelTensor(data_2, labels_2)
    lt_cat = LabelTensor.cat([lt1, lt2], dim=2)
    assert lt_cat.shape == (20, 3, 5)
    assert lt_cat.full_labels[2]["dof"] == ["x", "y", "z", "w", "a"]
    assert lt_cat.full_labels[0]["dof"] == range(20)
    assert lt_cat.full_labels[1]["dof"] == range(3)


def test_summation():
    lt1 = LabelTensor(torch.ones(20, 3), labels_all)
    lt2 = LabelTensor(torch.ones(30, 3), ["x", "y", "z"])
    with pytest.raises(RuntimeError):
        LabelTensor.summation([lt1, lt2])
    lt1 = LabelTensor(torch.ones(20, 3), labels_all)
    lt2 = LabelTensor(torch.ones(20, 3), labels_all)
    lt_sum = LabelTensor.summation([lt1, lt2])
    assert lt_sum.ndim == lt_sum.ndim
    assert lt_sum.shape[0] == 20
    assert lt_sum.shape[1] == 3
    assert lt_sum.full_labels[0] == labels_all[0]
    assert lt_sum.labels == ["x+x", "y+y", "z+z"]
    assert torch.eq(lt_sum.tensor, torch.ones(20, 3) * 2).all()
    lt1 = LabelTensor(torch.ones(20, 3), labels_all)
    lt2 = LabelTensor(torch.ones(20, 3), labels_all)
    lt3 = LabelTensor(torch.zeros(20, 3), labels_all)
    lt_sum = LabelTensor.summation([lt1, lt2, lt3])
    assert lt_sum.ndim == lt_sum.ndim
    assert lt_sum.shape[0] == 20
    assert lt_sum.shape[1] == 3
    assert lt_sum.full_labels[0] == labels_all[0]
    assert lt_sum.labels == ["x+x+x", "y+y+y", "z+z+z"]
    assert torch.eq(lt_sum.tensor, torch.ones(20, 3) * 2).all()


def test_append_3D():
    data_1 = torch.rand(20, 3, 2)
    labels_1 = ["x", "y"]
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(20, 3, 2)
    labels_2 = ["z", "w"]
    lt2 = LabelTensor(data_2, labels_2)
    lt1 = lt1.append(lt2)
    assert lt1.shape == (20, 3, 4)
    assert lt1.full_labels[0]["dof"] == range(20)
    assert lt1.full_labels[1]["dof"] == range(3)
    assert lt1.full_labels[2]["dof"] == ["x", "y", "z", "w"]


def test_append_2D():
    data_1 = torch.rand(20, 2)
    labels_1 = ["x", "y"]
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(20, 2)
    labels_2 = ["z", "w"]
    lt2 = LabelTensor(data_2, labels_2)
    lt1 = lt1.append(lt2, mode="cross")
    assert lt1.shape == (400, 4)
    assert lt1.full_labels[0]["dof"] == range(400)
    assert lt1.full_labels[1]["dof"] == ["x", "y", "z", "w"]


def test_vstack_3D():
    data_1 = torch.rand(20, 3, 2)
    labels_1 = {
        1: {"dof": ["a", "b", "c"], "name": "first"},
        2: {"dof": ["x", "y"], "name": "second"},
    }
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(20, 3, 2)
    labels_1 = {
        1: {"dof": ["a", "b", "c"], "name": "first"},
        2: {"dof": ["x", "y"], "name": "second"},
    }
    lt2 = LabelTensor(data_2, labels_1)
    lt_stacked = LabelTensor.vstack([lt1, lt2])
    assert lt_stacked.shape == (40, 3, 2)
    assert lt_stacked.full_labels[0]["dof"] == range(40)
    assert lt_stacked.full_labels[1]["dof"] == ["a", "b", "c"]
    assert lt_stacked.full_labels[2]["dof"] == ["x", "y"]
    assert lt_stacked.full_labels[1]["name"] == "first"
    assert lt_stacked.full_labels[2]["name"] == "second"


def test_vstack_2D():
    data_1 = torch.rand(20, 2)
    labels_1 = {1: {"dof": ["x", "y"], "name": "second"}}
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(20, 2)
    labels_1 = {1: {"dof": ["x", "y"], "name": "second"}}
    lt2 = LabelTensor(data_2, labels_1)
    lt_stacked = LabelTensor.vstack([lt1, lt2])
    assert lt_stacked.shape == (40, 2)
    assert lt_stacked.full_labels[0]["dof"] == range(40)
    assert lt_stacked.full_labels[1]["dof"] == ["x", "y"]
    assert lt_stacked.full_labels[0]["name"] == 0
    assert lt_stacked.full_labels[1]["name"] == "second"


def test_stack_2D():
    data_1 = torch.rand(20, 3)
    labels = {1: {"dof": ["x", "y", "z"], "name": "vars"}}
    lt1 = LabelTensor(data_1, labels)
    lt2 = LabelTensor(data_1, labels)

    # Test default dim=0
    lt_stacked_0 = LabelTensor.stack([lt1, lt2], dim=0)
    assert lt_stacked_0.shape == (2, 20, 3)
    assert lt_stacked_0.full_labels[0]["dof"] == range(2)
    assert lt_stacked_0.full_labels[0]["name"] == 0
    assert lt_stacked_0.full_labels[1]["dof"] == range(20)
    assert lt_stacked_0.full_labels[1]["name"] == 1
    assert lt_stacked_0.full_labels[2]["dof"] == ["x", "y", "z"]
    assert lt_stacked_0.full_labels[2]["name"] == "vars"
    assert lt_stacked_0.stored_labels == {
        2: {"dof": ["x", "y", "z"], "name": "vars"}
    }
    assert lt_stacked_0.labels == ["x", "y", "z"]

    # Test dim=1
    lt_stacked_1 = LabelTensor.stack([lt1, lt2], dim=1)
    assert lt_stacked_1.shape == (20, 2, 3)
    assert lt_stacked_1.full_labels[0]["dof"] == range(20)
    assert lt_stacked_1.full_labels[0]["name"] == 0
    assert lt_stacked_1.full_labels[1]["dof"] == range(2)
    assert lt_stacked_1.full_labels[1]["name"] == 1
    assert lt_stacked_1.full_labels[2]["dof"] == ["x", "y", "z"]
    assert lt_stacked_1.full_labels[2]["name"] == "vars"
    assert lt_stacked_1.stored_labels == {
        2: {"dof": ["x", "y", "z"], "name": "vars"}
    }
    assert lt_stacked_1.labels == ["x", "y", "z"]

    # Test dim=2 and negative dim=-1
    lt_stacked_2 = LabelTensor.stack([lt1, lt2], dim=2)
    lt_stacked_neg = LabelTensor.stack([lt1, lt2], dim=-1)
    for lt_s in [lt_stacked_2, lt_stacked_neg]:
        assert lt_s.shape == (20, 3, 2)
        assert lt_s.full_labels[0]["dof"] == range(20)
        assert lt_s.full_labels[0]["name"] == 0
        assert lt_s.full_labels[1]["dof"] == ["x", "y", "z"]
        assert lt_s.full_labels[1]["name"] == "vars"
        assert lt_s.full_labels[2]["dof"] == range(2)
        assert lt_s.full_labels[2]["name"] == 2
        assert lt_s.stored_labels == {
            1: {"dof": ["x", "y", "z"], "name": "vars"}
        }
        assert lt_s.labels is None


def test_stack_3D():
    data_1 = torch.rand(20, 3, 2)
    labels_1 = {
        1: {"dof": ["a", "b", "c"], "name": "first"},
        2: {"dof": ["x", "y"], "name": "second"},
    }
    lt1 = LabelTensor(data_1, labels_1)
    data_2 = torch.rand(20, 3, 2)
    lt2 = LabelTensor(data_2, labels_1)

    # Test dim=0 (and dim=-4)
    lt_stacked_0 = LabelTensor.stack([lt1, lt2], dim=0)
    lt_stacked_neg4 = LabelTensor.stack([lt1, lt2], dim=-4)
    for lt_s in [lt_stacked_0, lt_stacked_neg4]:
        assert lt_s.shape == (2, 20, 3, 2)
        assert lt_s.full_labels[0]["dof"] == range(2)
        assert lt_s.full_labels[0]["name"] == 0
        assert lt_s.full_labels[1]["dof"] == range(20)
        assert lt_s.full_labels[1]["name"] == 1
        assert lt_s.full_labels[2]["dof"] == ["a", "b", "c"]
        assert lt_s.full_labels[2]["name"] == "first"
        assert lt_s.full_labels[3]["dof"] == ["x", "y"]
        assert lt_s.full_labels[3]["name"] == "second"
        assert lt_s.stored_labels == {
            2: {"dof": ["a", "b", "c"], "name": "first"},
            3: {"dof": ["x", "y"], "name": "second"},
        }
        assert lt_s.labels == ["x", "y"]

    # Test dim=1 (and dim=-3)
    lt_stacked_1 = LabelTensor.stack([lt1, lt2], dim=1)
    lt_stacked_neg3 = LabelTensor.stack([lt1, lt2], dim=-3)
    for lt_s in [lt_stacked_1, lt_stacked_neg3]:
        assert lt_s.shape == (20, 2, 3, 2)
        assert lt_s.full_labels[0]["dof"] == range(20)
        assert lt_s.full_labels[0]["name"] == 0
        assert lt_s.full_labels[1]["dof"] == range(2)
        assert lt_s.full_labels[1]["name"] == 1
        assert lt_s.full_labels[2]["dof"] == ["a", "b", "c"]
        assert lt_s.full_labels[2]["name"] == "first"
        assert lt_s.full_labels[3]["dof"] == ["x", "y"]
        assert lt_s.full_labels[3]["name"] == "second"
        assert lt_s.stored_labels == {
            2: {"dof": ["a", "b", "c"], "name": "first"},
            3: {"dof": ["x", "y"], "name": "second"},
        }
        assert lt_s.labels == ["x", "y"]

    # Test dim=2 (and dim=-2)
    lt_stacked_2 = LabelTensor.stack([lt1, lt2], dim=2)
    lt_stacked_neg2 = LabelTensor.stack([lt1, lt2], dim=-2)
    for lt_s in [lt_stacked_2, lt_stacked_neg2]:
        assert lt_s.shape == (20, 3, 2, 2)
        assert lt_s.full_labels[0]["dof"] == range(20)
        assert lt_s.full_labels[0]["name"] == 0
        assert lt_s.full_labels[1]["dof"] == ["a", "b", "c"]
        assert lt_s.full_labels[1]["name"] == "first"
        assert lt_s.full_labels[2]["dof"] == range(2)
        assert lt_s.full_labels[2]["name"] == 2
        assert lt_s.full_labels[3]["dof"] == ["x", "y"]
        assert lt_s.full_labels[3]["name"] == "second"
        assert lt_s.stored_labels == {
            1: {"dof": ["a", "b", "c"], "name": "first"},
            3: {"dof": ["x", "y"], "name": "second"},
        }
        assert lt_s.labels == ["x", "y"]

    # Test dim=3 (and dim=-1)
    lt_stacked_3 = LabelTensor.stack([lt1, lt2], dim=3)
    lt_stacked_neg1 = LabelTensor.stack([lt1, lt2], dim=-1)
    for lt_s in [lt_stacked_3, lt_stacked_neg1]:
        assert lt_s.shape == (20, 3, 2, 2)
        assert lt_s.full_labels[0]["dof"] == range(20)
        assert lt_s.full_labels[0]["name"] == 0
        assert lt_s.full_labels[1]["dof"] == ["a", "b", "c"]
        assert lt_s.full_labels[1]["name"] == "first"
        assert lt_s.full_labels[2]["dof"] == ["x", "y"]
        assert lt_s.full_labels[2]["name"] == "second"
        assert lt_s.full_labels[3]["dof"] == range(2)
        assert lt_s.full_labels[3]["name"] == 3
        assert lt_s.stored_labels == {
            1: {"dof": ["a", "b", "c"], "name": "first"},
            2: {"dof": ["x", "y"], "name": "second"},
        }
        assert lt_s.labels is None


def test_stack_errors():
    data_1 = torch.rand(20, 3)
    lt1 = LabelTensor(data_1, {1: {"dof": ["x", "y", "z"], "name": "vars"}})
    lt2 = LabelTensor(data_1, {1: {"dof": ["a", "b", "c"], "name": "vars"}})

    # Empty list
    assert LabelTensor.stack([]) == []

    # Single tensor
    assert LabelTensor.stack([lt1]) is lt1

    # Out of range dim
    with pytest.raises(IndexError):
        LabelTensor.stack([lt1, lt1], dim=3)
    with pytest.raises(IndexError):
        LabelTensor.stack([lt1, lt1], dim=-4)

    # Inconsistent labels
    with pytest.raises(RuntimeError):
        LabelTensor.stack([lt1, lt2])


def test_sorting():
    data = torch.ones(20, 5)
    data[:, 0] = data[:, 0] * 4
    data[:, 1] = data[:, 1] * 2
    data[:, 2] = data[:, 2]
    data[:, 3] = data[:, 3] * 5
    data[:, 4] = data[:, 4] * 3
    labels = ["d", "b", "a", "e", "c"]
    lt_data = LabelTensor(data, labels)
    lt_sorted = LabelTensor.sort_labels(lt_data)
    assert lt_sorted.shape == (20, 5)
    assert lt_sorted.labels == ["a", "b", "c", "d", "e"]
    assert torch.eq(lt_sorted.tensor[:, 0], torch.ones(20) * 1).all()
    assert torch.eq(lt_sorted.tensor[:, 1], torch.ones(20) * 2).all()
    assert torch.eq(lt_sorted.tensor[:, 2], torch.ones(20) * 3).all()
    assert torch.eq(lt_sorted.tensor[:, 3], torch.ones(20) * 4).all()
    assert torch.eq(lt_sorted.tensor[:, 4], torch.ones(20) * 5).all()

    data = torch.ones(20, 4, 5)
    data[:, 0, :] = data[:, 0] * 4
    data[:, 1, :] = data[:, 1] * 2
    data[:, 2, :] = data[:, 2]
    data[:, 3, :] = data[:, 3] * 3
    labels = {1: {"dof": ["d", "b", "a", "c"], "name": 1}}
    lt_data = LabelTensor(data, labels)
    lt_sorted = LabelTensor.sort_labels(lt_data, dim=1)
    assert lt_sorted.shape == (20, 4, 5)
    assert lt_sorted.full_labels[1]["dof"] == ["a", "b", "c", "d"]
    assert torch.eq(lt_sorted.tensor[:, 0, :], torch.ones(20, 5) * 1).all()
    assert torch.eq(lt_sorted.tensor[:, 1, :], torch.ones(20, 5) * 2).all()
    assert torch.eq(lt_sorted.tensor[:, 2, :], torch.ones(20, 5) * 3).all()
    assert torch.eq(lt_sorted.tensor[:, 3, :], torch.ones(20, 5) * 4).all()


@pytest.mark.parametrize(
    "labels",
    [
        [f"s{i}" for i in range(10)],
        {0: {"dof": ["a", "b", "c"]}, 1: {"dof": [f"s{i}" for i in range(10)]}},
    ],
)
def test_cat_bool(labels):
    out = torch.randn((3, 10))
    out = LabelTensor(out, labels)
    selected = out[torch.tensor([True, True, False])]
    assert selected.shape == (2, 10)
    assert selected.stored_labels[1]["dof"] == [f"s{i}" for i in range(10)]
    if isinstance(labels, dict):
        assert selected.stored_labels[0]["dof"] == ["a", "b"]


def test_getitem_int():
    data = torch.rand(20, 3)
    labels = {1: {"name": 1, "dof": ["x", "y", "z"]}}
    lt = LabelTensor(data, labels)
    new = lt[0, 0]
    assert new.ndim == 1
    assert new.shape[0] == 1
    assert torch.all(torch.isclose(data[0, 0], new))

    data = torch.rand(20, 3, 2)
    labels = {
        1: {"name": 1, "dof": ["x", "y", "z"]},
        2: {"name": 2, "dof": ["a", "b"]},
    }
    lt = LabelTensor(data, labels)
    new = lt[0, 0, 0]
    assert new.ndim == 2
    assert new.shape[0] == 1
    assert new.shape[1] == 1
    assert torch.all(torch.isclose(data[0, 0, 0], new))
    assert new.stored_labels[0]["dof"] == ["x"]
    assert new.stored_labels[1]["dof"] == ["a"]

    new = lt[0, 0, :]
    assert new.ndim == 2
    assert new.shape[0] == 1
    assert new.shape[1] == 2
    assert torch.all(torch.isclose(data[0, 0, :], new))
    assert new.stored_labels[0]["dof"] == ["x"]
    assert new.stored_labels[1]["dof"] == ["a", "b"]

    new = lt[0, :, 1]
    assert new.ndim == 2
    assert new.shape[0] == 3
    assert new.shape[1] == 1
    assert torch.all(torch.isclose(data[0, :, 1], new.squeeze()))
    assert new.stored_labels[0]["dof"] == ["x", "y", "z"]
    assert new.stored_labels[1]["dof"] == ["b"]

    labels.pop(2)
    lt = LabelTensor(data, labels)
    new = lt[0, 0, 0]
    assert new.ndim == 1
    assert new.shape[0] == 1
    assert new.stored_labels[0]["dof"] == ["x"]

    new = lt[:, 0, 0]
    assert new.ndim == 2
    assert new.shape[0] == 20
    assert new.shape[1] == 1
    assert new.stored_labels[1]["dof"] == ["x"]

    new = lt[:, 0, :]
    assert new.ndim == 3
    assert new.shape[0] == 20
    assert new.shape[1] == 1
    assert new.shape[2] == 2
    assert new.stored_labels[1]["dof"] == ["x"]
