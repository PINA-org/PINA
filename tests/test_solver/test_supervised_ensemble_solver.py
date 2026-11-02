import torch
import pytest
from pina import Condition, LabelTensor, Trainer
from pina.solver import SupervisedEnsembleSolver
from pina.condition import GraphInputTargetCondition, InputTargetCondition
from pina.problem import BaseProblem
from pina.graph import KNNGraph


# Helper class for Tensor problems
class TensorProblem(BaseProblem):

    # Input and output variables
    input_variables = ["u_0", "u_1"]
    output_variables = ["u"]

    # Input and target
    input_ = torch.rand(20, 2)
    target_ = torch.rand(20, 1)

    # Condition
    conditions = {}

    def __init__(self, use_lt):
        super().__init__()

        # Add labels if use_lt is True
        if use_lt:
            self.input_ = LabelTensor(self.input_, self.input_variables)
            self.target_ = LabelTensor(self.target_, self.output_variables)

        # Initialize conditions
        self.conditions["data"] = Condition(
            input=self.input_, target=self.target_
        )


# Helper class for Graph problems
class GraphProblem(BaseProblem):

    # Input and output variables
    input_variables = ["a", "b", "c"]
    output_variables = ["u"]

    # Graph attributes and target
    x = torch.rand(20, 3)
    pos = torch.rand(20, 2)
    target_ = torch.rand(20, 1)

    # Condition
    conditions = {}

    def __init__(self, use_lt):
        super().__init__()

        # Add labels if use_lt is True
        if use_lt:
            self.x = LabelTensor(self.x, self.input_variables)
            self.pos = LabelTensor(self.pos, ["x", "y"])
            self.target_ = LabelTensor(self.target_, self.output_variables)

        # Initialize the input-output graph
        graph = KNNGraph(
            x=self.x,
            y=self.target_,
            pos=self.pos,
            neighbours=3,
            edge_attr=True,
        )

        # Initialize conditions
        self.conditions["data"] = GraphInputTargetCondition(graph=graph)


# Helper class for Graph-consistent architecture
class DummyGraphModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)

    def forward(self, batch):
        return self.layer(batch.x)


# Define the models for the tests
tensor_models = [torch.nn.Linear(2, 1) for _ in range(10)]
graph_models = [DummyGraphModel() for _ in range(10)]


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor(case, use_lt):

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = SupervisedEnsembleSolver(problem=problem, models=models)

    # Assert accepted conditions types and number of ensemble members
    assert solver.accepted_conditions_types == (
        InputTargetCondition,
        GraphInputTargetCondition,
    )
    assert solver.num_models == 10


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("use_lt", [True, False])
def test_solver_train(batch_size, case, use_lt):

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = SupervisedEnsembleSolver(
        problem=problem, models=models, use_lt=use_lt
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


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("use_lt", [True, False])
def test_solver_validation(batch_size, case, use_lt):

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = SupervisedEnsembleSolver(
        problem=problem, models=models, use_lt=use_lt
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


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("use_lt", [True, False])
def test_solver_test(batch_size, case, use_lt):

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = SupervisedEnsembleSolver(
        problem=problem, models=models, use_lt=use_lt
    )

    # Training procedure
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
    )
    trainer.test()


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("use_lt", [True, False])
def test_train_load_restore(clean_tmp_dir, case, use_lt):

    # Initialize the directory to store the checkpoints
    dir = clean_tmp_dir

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = SupervisedEnsembleSolver(
        problem=problem, models=models, use_lt=use_lt
    )

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
    ckpt_path = trainer.checkpoint_callback.best_model_path
    new_trainer = Trainer(
        solver=solver,
        max_epochs=5,
        accelerator="cpu",
        default_root_dir=dir,
    )
    new_trainer.train(ckpt_path=ckpt_path)

    # Load the solver from a checkpoint
    new_solver = SupervisedEnsembleSolver.load_from_checkpoint(
        ckpt_path,
        problem=problem,
        models=models,
    )

    # Create input data for testing the forward pass
    if case == "tensor":
        test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    else:
        test_pts = KNNGraph(
            x=LabelTensor(torch.rand(20, 3), ["a", "b", "c"]),
            pos=LabelTensor(torch.rand(20, 2), ["x", "y"]),
            neighbours=3,
            edge_attr=True,
        )

    # Assert the loaded solver behaves as the original one
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts), solver.forward(test_pts)
    )
