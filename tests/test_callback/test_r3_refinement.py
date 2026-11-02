import torch
import pytest
from pina.solver import PhysicsInformedSingleModelSolver
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callback import R3Refinement
from pina.problem.zoo import Poisson2DSquareProblem


@pytest.mark.parametrize("sample_every", [1, 3])
@pytest.mark.parametrize("residual_loss", [torch.nn.MSELoss, torch.nn.L1Loss])
@pytest.mark.parametrize("condition_to_update", [None, ["D"]])
def test_constructor(sample_every, residual_loss, condition_to_update):

    # Initialize the callback
    R3Refinement(
        sample_every=sample_every,
        residual_loss=residual_loss,
        condition_to_update=condition_to_update,
    )

    # Should fail if sample_every is not a positive integer
    with pytest.raises(AssertionError):
        R3Refinement(sample_every=0)

    # Should fail if residual_loss is not a valid loss class
    with pytest.raises(ValueError):
        R3Refinement(sample_every=10, residual_loss="not_a_loss")

    # Should fail if condition_to_update is not a string or iterable of strings
    with pytest.raises(ValueError):
        R3Refinement(sample_every=10, condition_to_update=123)


@pytest.mark.parametrize("sample_every", [1, 3])
@pytest.mark.parametrize("residual_loss", [torch.nn.MSELoss, torch.nn.L1Loss])
@pytest.mark.parametrize("condition_to_update", [None, ["D"], ["boundary"]])
def test_sample(sample_every, residual_loss, condition_to_update):

    # Define the problem, model, and solver for testing
    problem = Poisson2DSquareProblem()
    problem.discretise_domain(10, "grid", domains="boundary")
    problem.discretise_domain(10, "grid", domains="D")
    model = FeedForward(
        len(problem.input_variables), len(problem.output_variables)
    )
    solver = PhysicsInformedSingleModelSolver(problem=problem, model=model)

    # Initialize the callback
    callback = R3Refinement(
        sample_every=sample_every,
        residual_loss=residual_loss,
        condition_to_update=condition_to_update,
    )

    # Initialize the trainer
    trainer = Trainer(
        solver=solver,
        callbacks=callback,
        accelerator="cpu",
        max_epochs=5,
    )

    # Initialize the conditions to update if None (mirrors on_train_start logic)
    if callback._condition_to_update is None:
        callback._condition_to_update = [
            name
            for name, cond in solver.problem.conditions.items()
            if hasattr(cond, "domain")
        ]

    # Capture initial point count from the discretised domain (before DataModule
    # setup transforms DomainEquationConditions into InputEquationConditions)
    n_points_before_train = {
        cond: len(
            problem.discretised_domains[
                solver.problem.conditions[cond].domain
            ]
        )
        for cond in callback._condition_to_update
    }

    # Train the model to trigger refinement
    trainer.train()

    # After training each condition is an InputEquationCondition whose data is
    # a list of batch dicts; sum lengths across all batches for total points.
    n_points_after_train = {
        cond: sum(
            len(batch["input"])
            for batch in trainer.solver.problem.conditions[cond].data
        )
        for cond in callback._condition_to_update
    }

    # Assert initial population size matches pre-train point counts
    assert (
        n_points_before_train == trainer.callbacks[0].initial_population_size
    )
    # Assert total point count is preserved after refinement
    assert n_points_before_train == n_points_after_train

    # Should fail if the specified condition does not exist in the problem
    with pytest.raises(RuntimeError):
        callback = R3Refinement(
            sample_every=sample_every,
            residual_loss=residual_loss,
            condition_to_update="non_existent_condition",
        )
        trainer = Trainer(
            solver=solver,
            callbacks=callback,
            accelerator="cpu",
            max_epochs=5,
        )
        callback.on_train_start(trainer, solver=solver)
