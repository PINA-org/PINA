import pytest
from torch.utils.data import DataLoader

from pina import Trainer
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem
from pina.solver import PhysicsInformedSingleModelSolver

# Define the problem, the model and the solver for testing purposes
problem = Poisson2DSquareProblem()
problem.discretise_domain(n=10, mode="random")
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
solver = PhysicsInformedSingleModelSolver(model=model, problem=problem)


# Define a dummy dataloader class used for testing purposes
class DummyDataLoader:
    @staticmethod
    def __new__(cls, *args, **kwargs):
        return DataLoader(*args, **kwargs)


def _update_batching_mode(batch_size, batching_mode):
    return batching_mode if batch_size else "common_batch_size"


def _update_pin_memory(batch_size, pin_memory):
    return pin_memory if batch_size else False


@pytest.mark.parametrize("batching_mode", ["common_batch_size", "proportional"])
@pytest.mark.parametrize("pin_memory", [True, False])
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
    batching_mode,
    pin_memory,
    shuffle,
):

    batching_mode = _update_batching_mode(batch_size, batching_mode)
    pin_memory = _update_pin_memory(batch_size, pin_memory)

    Trainer(
        solver=solver,
        batch_size=batch_size,
        train_size=train_size,
        test_size=test_size,
        val_size=val_size,
        batching_mode=batching_mode,
        num_workers=0,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

    # Should raise ValueError if solver is not an instance of BaseSolver
    with pytest.raises(ValueError):
        Trainer(
            solver="not_a_solver",
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batching_mode=batching_mode,
            num_workers=0,
            pin_memory=pin_memory,
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
            batching_mode=batching_mode,
            num_workers=0,
            pin_memory=pin_memory,
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
            batching_mode=batching_mode,
            num_workers=0,
            pin_memory=pin_memory,
            shuffle="not_a_boolean",
        )

    # Should raise ValueError if pin_memory is not a boolean
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batching_mode=batching_mode,
            num_workers=0,
            pin_memory="not_a_boolean",
            shuffle=shuffle,
        )

    # Should raise AssertionError if num_workers is negative
    with pytest.raises(AssertionError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batching_mode=batching_mode,
            num_workers=-1,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

    # Should raise AssertionError if batch_size is not a positive integer
    with pytest.raises(AssertionError):
        Trainer(
            solver=solver,
            batch_size=-1,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batching_mode=batching_mode,
            num_workers=0,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

    # Should raise ValueError if an invalid batching mode is provided
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batching_mode="invalid_mode",
            num_workers=0,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

    # Should raise ValueError if a mapping refers to unknown conditions
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batching_mode=batching_mode,
            num_workers=0,
            pin_memory=pin_memory,
            shuffle=shuffle,
            dataloader_cls={"unknown_condition": DataLoader},
        )

    # Should raise ValueError if a collate mapping refers to unknown
    # conditions
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batching_mode=batching_mode,
            num_workers=0,
            pin_memory=pin_memory,
            shuffle=shuffle,
            collate_fn={"unknown_condition": lambda x, y: x},
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
            batching_mode=batching_mode,
            num_workers=0,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )


@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("condition_name", ["D", "boundary"])
def test_dataloader_options_resolution(batch_size, condition_name):

    # Define the dataloader class mapping
    dataloader_cls = {condition_name: DummyDataLoader}

    # Initialize the trainer
    trainer = Trainer(
        solver=solver,
        batch_size=batch_size,
        batching_mode="common_batch_size",
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        dataloader_cls=dataloader_cls,
    )

    # Check that the dataloader class mapping has been resolved correctly
    assert isinstance(trainer.data_module.dataloader_cls, dict)
    assert trainer.data_module.dataloader_cls[condition_name] == DummyDataLoader
    assert condition_name in trainer.data_module.dataloader_cls

    # Check that the batcher receives the resolved mapping
    assert trainer.data_module.batcher.dataloader_cls == {
        condition_name: DummyDataLoader
    }


def test_batching_mode_without_batch_size_warns():

    # Should warn and still construct if batching forces common_batch_size
    with pytest.warns(UserWarning):
        trainer = Trainer(
            solver=solver,
            batch_size=None,
            batching_mode="proportional",
            num_workers=0,
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
        )

    assert trainer.batch_size is None


def test_batch_size_none_disables_workers_and_pin_memory():

    # Should warn that num_workers and pin_memory have no effect when
    # batch_size is None
    with pytest.warns(UserWarning):
        Trainer(
            solver=solver,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
        )
