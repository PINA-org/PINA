from types import SimpleNamespace

import pytest
import torch

from pina.condition import GraphInputTargetCondition, InputTargetCondition
from pina import DataModule, LabelTensor, Trainer
from pina.graph import RadiusGraph
from pina.model import FeedForward
from pina.optim import TorchOptimizer
from pina.problem import BaseProblem
from pina.problem.zoo import Poisson2DSquareProblem, SupervisedProblem
from pina.solver import PhysicsInformedSingleModelSolver
from pina.weighting import ScalarWeighting

# Number of samples in the synthetic datasets
n_samples = 100


# Define helper functions to create synthetic tensor data
def _create_tensor_data(n=n_samples):
    return (torch.rand((n, 4)), torch.rand((n, 2)))


# Graph problem using a graph input-target condition
class GraphProblem(BaseProblem):

    input_variables = ["a", "b", "c", "d"]
    output_variables = ["u", "v"]
    conditions = {}

    def __init__(self):
        graph = RadiusGraph(
            x=torch.rand((20, 4)),
            y=torch.rand((20, 2)),
            pos=torch.rand((20, 2)),
            radius=0.2,
        )
        self.conditions = {"data": GraphInputTargetCondition(graph=graph)}
        super().__init__()


def _create_problem(problem_type):
    if problem_type == "tensor":
        input_tensor, output_tensor = _create_tensor_data()
        problem = SupervisedProblem(input_=input_tensor, output_=output_tensor)
    elif problem_type == "graph":
        problem = GraphProblem()
    else:
        problem = Poisson2DSquareProblem()
        problem.discretise_domain(n=n_samples, mode="random")
    return problem


def _expected_split_lengths(n_batches, train_size, val_size):
    train_end = 0
    if train_size > 0 and n_batches > 0:
        train_end = max(1, int(train_size * n_batches))

    val_end = train_end
    if val_size > 0 and n_batches > train_end:
        val_end += max(1, int(val_size * n_batches))
        val_end = min(val_end, n_batches)

    return {
        "train": train_end,
        "val": val_end - train_end,
        "test": n_batches - val_end,
    }


# Fixture remove data condition from pinns, caused by external tests in suite
@pytest.fixture(autouse=True)
def remove_data_from_pinn_conditions():
    yield

    # Remove the data condition
    Poisson2DSquareProblem.conditions.pop("data", None)


@pytest.mark.parametrize("problem_type", ["tensor", "graph", "pinn"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize(
    "train_size, val_size, test_size",
    [(0.7, 0.2, 0.1), (0.8, 0.2, 0.0), (0.0, 0.8, 0.2)],
)
def test_constructor(problem_type, batch_size, train_size, val_size, test_size):

    # Build the problem and check the input-target condition type
    problem = _create_problem(problem_type)
    if problem_type == "tensor":
        assert isinstance(problem.conditions["data"], InputTargetCondition)
    elif problem_type == "graph":
        assert isinstance(
            problem.conditions["data"], GraphInputTargetCondition
        )

    # Initialize the data module
    dm = DataModule(
        problem=problem,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        batch_size=batch_size,
        batching_mode="proportional",
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # Check that the data module has been initialized correctly
    assert dm.problem == problem
    assert dm.trainer is None

    # Datasets are created during setup
    for dataset in ["train_dataset", "val_dataset", "test_dataset"]:
        assert not hasattr(dm, dataset)


@pytest.mark.parametrize("problem_type", ["tensor", "graph", "pinn"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize(
    "train_size, val_size, test_size",
    [(0.7, 0.2, 0.1), (0.8, 0.2, 0.0), (0.0, 0.8, 0.2)],
)
def test_setup(problem_type, batch_size, train_size, val_size, test_size):

    # Build the problem and check the input-target condition type
    problem = _create_problem(problem_type)
    if problem_type == "tensor":
        assert isinstance(problem.conditions["data"], InputTargetCondition)
    elif problem_type == "graph":
        assert isinstance(
            problem.conditions["data"], GraphInputTargetCondition
        )

    # Initialize the data module
    dm = DataModule(
        problem=problem,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        batch_size=batch_size,
        batching_mode="proportional",
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # Call setup
    dm.setup()

    # Expected condition names
    expected_keys = (
        {"data"} if problem_type in ["tensor", "graph"] else {"D", "boundary"}
    )
    assert set(dm.condition_name_to_idx) == expected_keys
    assert dm.idx_to_condition_name == {
        idx: name for name, idx in dm.condition_name_to_idx.items()
    }

    # Expected number of batches per split
    expected_lengths = {
        split: sum(
            _expected_split_lengths(
                len(dm.problem.conditions[name]), train_size, val_size
            )[split]
            for name in expected_keys
        )
        for split in ["train", "val", "test"]
    }
    condition_ids = {
        dm.condition_name_to_idx[name] for name in expected_keys
    }

    # Iterate over datasets
    for dataset_name, split in [
        ("train_dataset", "train"),
        ("val_dataset", "val"),
        ("test_dataset", "test"),
    ]:

        # Assert that each dataset has been created correctly
        dataset = getattr(dm, dataset_name)
        assert isinstance(dataset, torch.Tensor)
        assert dataset.dtype == torch.long
        assert dataset.ndim == 2 and dataset.shape[1] == 2
        assert len(dataset) == expected_lengths[split]

        if len(dataset):
            assert set(dataset[:, 0].tolist()) <= condition_ids
            for condition_id in dataset[:, 0].unique():
                name = dm.idx_to_condition_name[int(condition_id)]
                batch_ids = dataset[dataset[:, 0] == condition_id, 1]
                assert batch_ids.max() < len(dm.problem.conditions[name])

        loader = getattr(dm, f"{split}_dataloader")()
        condition_rows = {
            int(condition_id): dataset[dataset[:, 0] == condition_id]
            for condition_id in dataset[:, 0].unique()
        }
        assert len(loader) == max(
            (len(rows) for rows in condition_rows.values()), default=0
        )
        for step, identifiers in enumerate(loader):
            assert identifiers[:, 0].tolist() == list(condition_rows)
            for row in identifiers:
                rows = condition_rows[int(row[0])]
                torch.testing.assert_close(row, rows[step % len(rows)])


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_full_batch_groups_conditions(split):
    problem = _create_problem("pinn")
    dm = DataModule(
        problem=problem,
        train_size=float(split == "train"),
        val_size=float(split == "val"),
        test_size=float(split == "test"),
        shuffle=False,
    )
    dm.setup()

    loader = getattr(dm, f"{split}_dataloader")()
    assert len(loader) == 1
    identifiers = next(iter(loader))
    assert identifiers.shape == (len(problem.conditions), 2)
    batch = dm.transfer_batch_to_device(identifiers, "cpu", 0)
    assert [name for name, _ in batch] == list(problem.conditions)
    for name, data in batch:
        torch.testing.assert_close(
            data["input"], problem.conditions[name].input
        )


@pytest.mark.parametrize("batch_size", [None, 5])
def test_joint_minibatches_cycle_shorter_conditions(batch_size):
    problem = _create_problem("pinn")
    dm = DataModule(
        problem=problem,
        batch_size=batch_size,
        problem_setting={"D": {"batch_size": 17}},
        shuffle=False,
    )
    dm.setup()

    loader = dm.train_dataloader()
    expected_steps = max(len(c) for c in problem.conditions.values())
    assert len(loader) == expected_steps
    for step, identifiers in enumerate(loader):
        assert identifiers.shape == (len(problem.conditions), 2)
        batch = dm.transfer_batch_to_device(identifiers, "cpu", 0)
        assert [name for name, _ in batch] == list(problem.conditions)
        for (condition_id, batch_id), (name, data) in zip(identifiers, batch):
            condition = problem.conditions[name]
            assert int(condition_id) == dm.condition_name_to_idx[name]
            assert int(batch_id) == step % len(condition)
            torch.testing.assert_close(
                data["input"], condition.data[int(batch_id)]["input"]
            )

    # Iterating again starts a fresh epoch with the same cycling order.
    assert next(iter(loader))[:, 1].tolist() == [0] * len(problem.conditions)


def test_explicit_batch_size_groups_single_batches():
    problem = _create_problem("pinn")
    dm = DataModule(problem=problem, batch_size=10000)
    dm.setup()

    assert all(
        len(condition) == 1 for condition in problem.conditions.values()
    )
    assert len(dm.train_dataloader()) == 1
    assert next(iter(dm.train_dataloader())).shape == (
        len(problem.conditions), 2
    )


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [0, 1])
def test_joint_batches_respect_split_membership(shuffle, num_workers):
    problem = _create_problem("pinn")
    dm = DataModule(
        problem=problem,
        batch_size=17,
        problem_setting={"boundary": {"batch_size": 10000}},
        train_size=0.5,
        val_size=0.25,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    dm.setup()

    for split in ("train", "val", "test"):
        dataset = getattr(dm, f"{split}_dataset")
        active_ids = dataset[:, 0].unique().tolist()
        seen = set()
        for identifiers in getattr(dm, f"{split}_dataloader")():
            assert identifiers[:, 0].tolist() == active_ids
            for row in identifiers:
                assert (dataset == row).all(dim=1).any()
                seen.add(tuple(row.tolist()))
        assert seen == set(map(tuple, dataset.tolist()))

    # The boundary condition only has a training batch, not validation/test.
    boundary_id = dm.condition_name_to_idx["boundary"]
    assert boundary_id not in dm.val_dataset[:, 0]
    assert boundary_id not in dm.test_dataset[:, 0]


@pytest.mark.parametrize("batch_size", [None, 3])
def test_batch_loss_and_gradient_match_joint_objective(batch_size):
    torch.manual_seed(1)
    inputs = LabelTensor(torch.rand(10, 1), ["x"])
    target = LabelTensor(torch.rand(10, 1), ["u"])
    problem = SupervisedProblem(
        input_=inputs,
        output_=target,
        input_variables=["x"],
        output_variables=["u"],
    )
    problem.conditions = dict(problem.conditions)
    problem.conditions["other"] = InputTargetCondition(
        input=inputs[:4],
        target=LabelTensor(target[:4].as_subclass(torch.Tensor) + 1.0, ["u"]),
    )
    weights = {"data": 2.0, "other": 3.0}
    model = FeedForward(1, 1, 8, 1)
    solver = PhysicsInformedSingleModelSolver(
        problem=problem, model=model, weighting=ScalarWeighting(weights)
    )
    solver.trainer = SimpleNamespace(current_epoch=0)
    dm = DataModule(problem=problem, batch_size=batch_size, shuffle=False)
    dm.setup()
    loader = dm.train_dataloader()

    def joint_loss(entries):
        losses = {
            name: (
                problem.conditions[name].evaluate(data, solver).square().mean()
            )
            for name, data in entries
        }
        return solver.weighting.aggregate(losses)

    for step, identifiers in enumerate(loader):
        batch = dm.transfer_batch_to_device(identifiers, "cpu", 0)
        actual = joint_loss(batch)
        expected = sum(
            weights[name]
            * condition.evaluate(
                condition.materialize(step % len(condition)), solver
            ).square().mean()
            for name, condition in problem.conditions.items()
        )
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, tuple(model.parameters()))
        expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize(
    "optimizer_cls", [torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS]
)
def test_joint_minibatch_training(optimizer_cls, tmp_path):
    torch.manual_seed(1)
    inputs = LabelTensor(torch.rand(11, 1), ["x"])
    target = LabelTensor(torch.rand(11, 1), ["u"])
    problem = SupervisedProblem(
        input_=inputs,
        output_=target,
        input_variables=["x"],
        output_variables=["u"],
    )
    problem.conditions = dict(problem.conditions)
    problem.conditions["short"] = InputTargetCondition(
        input=inputs[:5], target=target[:5]
    )
    options = {"lr": 0.01}
    if optimizer_cls is torch.optim.LBFGS:
        options.update(max_iter=3, line_search_fn="strong_wolfe")
    solver = PhysicsInformedSingleModelSolver(
        problem=problem,
        model=FeedForward(1, 1, 8, 1),
        optimizer=TorchOptimizer(optimizer_cls, **options),
        weighting=ScalarWeighting({"data": 1.0, "short": 2.0}),
    )
    trainer = Trainer(
        solver=solver,
        batch_size=4,
        max_epochs=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.train()

    assert len(trainer.train_dataloader) == 3
    assert trainer.global_step == 9
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    for parameter in solver.model.parameters():
        assert torch.isfinite(parameter).all()
