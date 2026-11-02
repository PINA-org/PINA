import torch
import pytest
from pina.problem.zoo import InversePoisson2DSquareProblem
from pina.solver import RBAPhysicsInformedSingleModelSolver
from pina.problem.zoo import Poisson2DSquareProblem
from pina import LabelTensor, Condition, Trainer
from pina.model import FeedForward
from pina.condition import (
    InputTargetCondition,
    InputEquationCondition,
    DomainEquationCondition,
)


# Helper function for direct problem definition
def define_direct_problem_model(n_pts=10):

    # Initialize direct problem
    problem = Poisson2DSquareProblem()
    problem.discretise_domain(n_pts)

    # Add input-output condition to test supervised learning
    input_pts = torch.rand(10, len(problem.input_variables))
    input_pts = LabelTensor(input_pts, problem.input_variables)
    output_pts = torch.rand(10, len(problem.output_variables))
    output_pts = LabelTensor(output_pts, problem.output_variables)
    problem.conditions["data"] = Condition(input=input_pts, target=output_pts)
    problem.conditions["data"].name = "data"

    # Initialize the model
    model = FeedForward(
        len(problem.input_variables), len(problem.output_variables)
    )

    return problem, model


# Helper function for inverse problem definition
def define_inverse_problem_model(n_pts=10):

    # Initialize inverse problem
    problem = InversePoisson2DSquareProblem(load=True, data_size=0.01)
    problem.discretise_domain(n_pts)

    # Initialize the model
    model = FeedForward(
        len(problem.input_variables), len(problem.output_variables)
    )

    return problem, model


@pytest.mark.parametrize("case", ["direct", "inverse"])
@pytest.mark.parametrize("regularized_conditions", [None, "D"])
def test_constructor(case, regularized_conditions):

    # Initialize problems and model based on the case
    if case == "direct":
        problem, model = define_direct_problem_model()
    else:
        problem, model = define_inverse_problem_model()

    # Define the solver
    solver = RBAPhysicsInformedSingleModelSolver(
        problem=problem,
        model=model,
        eta=0.001,
        gamma=0.999,
        regularized_conditions=regularized_conditions,
    )

    # Assert accepted conditions types
    assert solver.accepted_conditions_types == (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )

    # Should fail if eta is not a positive number
    with pytest.raises(ValueError):
        RBAPhysicsInformedSingleModelSolver(
            problem=problem,
            model=model,
            eta=-0.001,
            gamma=0.999,
            regularized_conditions=regularized_conditions,
        )

    # Should fail if gamma is not in the range (0, 1)
    with pytest.raises(ValueError):
        RBAPhysicsInformedSingleModelSolver(
            problem=problem,
            model=model,
            eta=0.001,
            gamma=1.0,
            regularized_conditions=regularized_conditions,
        )

    # Should fail if regularized_conditions is not among the problem conditions
    with pytest.raises(ValueError):
        RBAPhysicsInformedSingleModelSolver(
            problem=problem,
            model=model,
            eta=0.001,
            gamma=0.999,
            regularized_conditions="invalid_condition",
        )


@pytest.mark.parametrize("case", ["direct", "inverse"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("regularized_conditions", [None, "D"])
def test_solver_train(case, batch_size, regularized_conditions):

    # Initialize problems and model based on the case
    if case == "direct":
        problem, model = define_direct_problem_model()
    else:
        problem, model = define_inverse_problem_model()

    # Define the solver
    solver = RBAPhysicsInformedSingleModelSolver(
        problem=problem,
        model=model,
        regularized_conditions=regularized_conditions,
    )

    # Training procedure
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


@pytest.mark.parametrize("case", ["direct", "inverse"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("regularized_conditions", [None, "D"])
def test_solver_validation(case, batch_size, regularized_conditions):

    # Initialize problems and model based on the case
    if case == "direct":
        problem, model = define_direct_problem_model()
    else:
        problem, model = define_inverse_problem_model()

    # Define the solver
    solver = RBAPhysicsInformedSingleModelSolver(
        problem=problem,
        model=model,
        regularized_conditions=regularized_conditions,
    )

    # Training procedure
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


@pytest.mark.parametrize("case", ["direct", "inverse"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("regularized_conditions", [None, "D"])
def test_solver_test(case, batch_size, regularized_conditions):

    # Initialize problems and model based on the case
    if case == "direct":
        problem, model = define_direct_problem_model()
    else:
        problem, model = define_inverse_problem_model()

    # Define the solver
    solver = RBAPhysicsInformedSingleModelSolver(
        problem=problem,
        model=model,
        regularized_conditions=regularized_conditions,
    )

    # Training procedure
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.7,
        val_size=0.2,
        test_size=0.1,
    )
    trainer.test()


@pytest.mark.parametrize("case", ["direct", "inverse"])
def test_train_load_restore(clean_tmp_dir, case):

    # Initialize the directory to store the checkpoints
    dir = clean_tmp_dir

    # Initialize problems and model based on the case
    if case == "direct":
        problem, model = define_direct_problem_model()
    else:
        problem, model = define_inverse_problem_model()

    # Define the solver
    solver = RBAPhysicsInformedSingleModelSolver(problem=problem, model=model)

    # Training procedure
    trainer = Trainer(
        solver=solver,
        max_epochs=5,
        accelerator="cpu",
        batch_size=None,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        default_root_dir=dir,
    )
    trainer.train()

    # Restore the training from a checkpoint
    new_trainer = Trainer(
        solver=solver,
        max_epochs=5,
        accelerator="cpu",
        default_root_dir=dir,
    )
    new_trainer.train(
        ckpt_path=trainer.checkpoint_callback.best_model_path
    )

    # Load the solver from a checkpoint
    new_solver = RBAPhysicsInformedSingleModelSolver.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        problem=problem,
        model=model,
    )

    # Create input data for testing the forward pass
    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)

    # Assert the loaded solver behaves as the original one
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts), solver.forward(test_pts)
    )
