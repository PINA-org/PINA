import pytest
from torch.utils.data import DataLoader

from pina import DataModule, Trainer
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem
from pina.solver import PhysicsInformedSingleModelSolver

# Define the problem, the model and the solver for testing purposes
problem = Poisson2DSquareProblem()
problem.discretise_domain(n=10, mode="random")
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
solver = PhysicsInformedSingleModelSolver(model=model, problem=problem)


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize(
    "train_size, test_size, val_size", [(0.8, 0.1, 0.1), (0.7, 0.2, 0.1)]
)
def test_constructor(
    batch_size,
    train_size,
    test_size,
    val_size,
    shuffle,
):
    trainer = Trainer(
        solver=solver,
        batch_size=batch_size,
        train_size=train_size,
        test_size=test_size,
        val_size=val_size,
        shuffle=shuffle,
    )

    # The trainer builds a data module from the provided options
    assert isinstance(trainer.data_module, DataModule)
    assert trainer.data_module.problem is problem
    assert trainer.data_module.batch_size == batch_size
    assert trainer.data_module.train_size == train_size
    assert trainer.data_module.test_size == test_size
    assert trainer.data_module.val_size == val_size
    assert trainer.data_module.shuffle == shuffle
    assert trainer.batch_size == batch_size

    # Should raise ValueError if solver is not an instance of BaseSolver
    with pytest.raises(ValueError):
        Trainer(
            solver="not_a_solver",
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            shuffle=shuffle,
        )

    # Should raise ValueError if train_size + test_size + val_size != 1.0
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=0.5,
            test_size=0.3,
            val_size=0.3,
            shuffle=shuffle,
        )

    # Should raise ValueError if shuffle is not a boolean
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            shuffle="not_a_boolean",
        )

    # Should raise AssertionError if batch_size is not a positive integer
    with pytest.raises(AssertionError):
        Trainer(
            solver=solver,
            batch_size=-1,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            shuffle=shuffle,
        )

    # Should raise RuntimeError if any domain has not been discretised
    with pytest.raises(RuntimeError):

        # Create a new problem without discretising the domain
        new_problem = Poisson2DSquareProblem()
        new_solver = PhysicsInformedSingleModelSolver(
            model=model, problem=new_problem
        )

        Trainer(
            solver=new_solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            shuffle=shuffle,
        )


def test_default_data_module():
    trainer = Trainer(solver=solver)

    assert trainer.data_module.train_size == 1.0
    assert trainer.data_module.val_size == 0.0
    assert trainer.data_module.test_size == 0.0
    assert trainer.data_module.batch_size is None
    assert trainer.data_module.shuffle is True
    assert trainer.data_module.batching_mode == "simple"
    assert trainer.data_module.num_workers == 0
    assert trainer.data_module.pin_memory is False


def test_user_datamodule_is_used():
    data_module = DataModule(
        problem=problem,
        batch_size=5,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
    )

    trainer = Trainer(solver=solver, datamodule=data_module)

    assert trainer.data_module is data_module
    assert trainer.batch_size == 5


def test_user_datamodule_with_split_options_warns():
    data_module = DataModule(problem=problem, batch_size=5)

    with pytest.warns(UserWarning):
        trainer = Trainer(
            solver=solver,
            datamodule=data_module,
            batch_size=10,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            shuffle=False,
        )

    # The data module settings win
    assert trainer.data_module is data_module
    assert trainer.data_module.batch_size == 5


def test_invalid_datamodule_type_raises():
    with pytest.raises(ValueError):
        Trainer(solver=solver, datamodule=object())


def test_datamodule_with_different_problem_raises():
    other_problem = Poisson2DSquareProblem()
    other_problem.discretise_domain(n=10, mode="random")
    data_module = DataModule(problem=other_problem)

    with pytest.raises(ValueError):
        Trainer(solver=solver, datamodule=data_module)


@pytest.mark.parametrize(
    "deprecated_kwargs",
    [
        {"batching_mode": "simple"},
        {"num_workers": 0},
        {"pin_memory": False},
        {"problem_setting": {"D": {"batch_size": 10}}},
        {"dataloader_cls": DataLoader},
        {"collate_fn": lambda batch: batch},
    ],
)
def test_deprecated_data_options_warn_and_are_ignored(deprecated_kwargs):
    with pytest.warns(DeprecationWarning):
        trainer = Trainer(solver=solver, **deprecated_kwargs)

    # Deprecated options are ignored: the data module keeps its defaults
    assert trainer.data_module.batching_mode == "simple"
    assert trainer.data_module.num_workers == 0
    assert trainer.data_module.pin_memory is False
