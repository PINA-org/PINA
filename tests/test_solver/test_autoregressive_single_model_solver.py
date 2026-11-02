import pytest
import torch
from pina.solver import AutoregressiveSingleModelSolver
from pina import Trainer, LabelTensor, Condition
from pina.condition import TimeSeriesCondition
from pina.problem import BaseProblem
from pina.model import FeedForward


# Settings for test purposes
n_traj = 5
t_steps = 10
n_dofs = 40
n_feats = 2
n_windows = 3
unroll_length = 5


# Helper function to create tensor data
def create_scalar_data(use_lt):

    # Define the data tensor
    data = torch.rand(n_traj, t_steps, n_feats)

    # Add labels if use_lt is True
    if use_lt:
        labels = [f"feat_{i}" for i in range(n_feats)]
        return LabelTensor(data, labels=labels)
    else:
        return data


def create_vector_data(use_lt):
    data = torch.rand(n_traj, t_steps, n_dofs, n_feats)
    if use_lt:
        labels = [f"feat_{i}" for i in range(n_feats)]
        return LabelTensor(data, labels=labels)
    else:
        return data


# Define a dummy problem for testing
class DummyProblem(BaseProblem):

    # Input and output variables
    input_variables = [f"feat_{i}" for i in range(n_feats)]
    output_variables = [f"feat_{i}" for i in range(n_feats)]

    # Conditions
    conditions = {}

    def __init__(self, data):
        super().__init__()

        # Initialize the time series condition with the provided data
        self.conditions["time"] = TimeSeriesCondition(
            input=data, n_windows=n_windows, unroll_length=unroll_length
        )


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("bool_value", [True, False])
@pytest.mark.parametrize("eps", [0.0, 1.0])
@pytest.mark.parametrize(
    "create_data", [create_scalar_data, create_vector_data]
)
@pytest.mark.parametrize("aggregation_strategy", [torch.mean, torch.sum])
def test_constructor(
    use_lt, bool_value, eps, create_data, aggregation_strategy
):

    # Define the problem
    data = create_data(use_lt)
    problem = DummyProblem(data)
    model = FeedForward(n_feats, n_feats, 10, 2)

    # Define the solver
    solver = AutoregressiveSingleModelSolver(
        problem=problem,
        model=model,
        reset_weights_at_epoch_start=bool_value,
        use_lt=use_lt,
        eps=eps,
    )

    # Assert accepted condition types
    assert solver.accepted_conditions_types == (TimeSeriesCondition,)

    # Should fail if eps is not a float or int
    with pytest.raises(ValueError):
        AutoregressiveSingleModelSolver(
            problem=problem,
            model=model,
            reset_weights_at_epoch_start=bool_value,
            use_lt=use_lt,
            eps="not_a_number",
        )

    # Should fail if reset_weights_at_epoch_start is not a boolean
    with pytest.raises(ValueError):
        AutoregressiveSingleModelSolver(
            problem=problem,
            model=model,
            reset_weights_at_epoch_start="not_a_boolean",
            use_lt=use_lt,
            eps=eps,
        )


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize(
    "create_data", [create_scalar_data, create_vector_data]
)
def test_solver_train(use_lt, batch_size, compile, create_data):

    # Define the problem
    data = create_data(use_lt)
    problem = DummyProblem(data)
    model = FeedForward(n_feats, n_feats, 10, 2)

    # Define the solver
    solver = AutoregressiveSingleModelSolver(
        problem=problem,
        model=model,
        use_lt=use_lt,
    )

    # Trainer
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=1.0,
        val_size=0.0,
        test_size=0.0,
    )
    trainer.train()


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize(
    "create_data", [create_scalar_data, create_vector_data]
)
def test_solver_validation(use_lt, batch_size, compile, create_data):

    # Define the problem
    data = create_data(use_lt)
    problem = DummyProblem(data)
    model = FeedForward(n_feats, n_feats, 10, 2)

    # Define the solver
    solver = AutoregressiveSingleModelSolver(
        problem=problem,
        model=model,
        use_lt=use_lt,
    )

    # Trainer
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.9,
        val_size=0.1,
        test_size=0.0,
    )
    trainer.train()


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize(
    "create_data", [create_scalar_data, create_vector_data]
)
def test_solver_test(use_lt, batch_size, compile, create_data):

    # Define the problem
    data = create_data(use_lt)
    problem = DummyProblem(data)
    model = FeedForward(n_feats, n_feats, 10, 2)

    # Define the solver
    solver = AutoregressiveSingleModelSolver(
        problem=problem,
        model=model,
        use_lt=use_lt,
    )

    # Trainer
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )
    trainer.test()


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize(
    "create_data", [create_scalar_data, create_vector_data]
)
def test_train_load_restore(clean_tmp_dir, use_lt, create_data):

    # Define the problem
    data = create_data(use_lt)
    problem = DummyProblem(data)
    model = FeedForward(n_feats, n_feats, 10, 2)
    dir = clean_tmp_dir

    # Define the solver
    solver = AutoregressiveSingleModelSolver(
        problem=problem,
        model=model,
        use_lt=use_lt,
    )

    # Train
    trainer = Trainer(
        solver=solver,
        max_epochs=5,
        accelerator="cpu",
        batch_size=None,
        train_size=0.7,
        val_size=0.2,
        test_size=0.1,
        default_root_dir=dir,
    )
    trainer.train()

    # Restore from checkpoint
    new_trainer = Trainer(
        solver=solver,
        max_epochs=5,
        accelerator="cpu",
        default_root_dir=dir,
    )
    new_trainer.train(
        ckpt_path=trainer.checkpoint_callback.best_model_path
    )

    # Load the restored solver
    new_solver = AutoregressiveSingleModelSolver.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        problem=problem,
        model=model,
    )

    # Test points
    test_pts = LabelTensor(
        torch.rand(n_traj, t_steps, n_feats), problem.input_variables
    )

    # Assert that the predictions from the loaded solver match original ones
    assert new_solver.forward(test_pts).shape == (n_traj, t_steps, n_feats)
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts), solver.forward(test_pts)
    )
