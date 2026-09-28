import pytest
import torch

from pina import Condition, LabelTensor, Trainer
from pina.callback import DataNormalizer
from pina.graph import RadiusGraph
from pina.model import FeedForward
from pina.problem import BaseProblem
from pina.solver import SupervisedSingleModelSolver


# Tensor-based problem
class TensorProblem(BaseProblem):
    input_variables = ["x", "y"]
    output_variables = ["u"]
    conditions = {
        "data1": Condition(input=torch.rand(20, 2), target=torch.rand(20, 1)),
        "data2": Condition(input=torch.rand(20, 2), target=torch.rand(20, 1)),
    }


# LabelTensor-based problem
class LabelTensorProblem(BaseProblem):
    input_variables = ["x", "y"]
    output_variables = ["u"]
    conditions = {
        "data1": Condition(
            input=LabelTensor(torch.rand(20, 2), ["x", "y"]),
            target=LabelTensor(torch.rand(20, 1), ["u"]),
        ),
        "data2": Condition(
            input=LabelTensor(torch.rand(20, 2), ["x", "y"]),
            target=LabelTensor(torch.rand(20, 1), ["u"]),
        ),
    }


# Graph-based problem for testing unsupported dataset case
input_graph = [RadiusGraph(radius=0.5, pos=torch.rand(10, 2)) for _ in range(5)]
target_tensor = torch.rand(5, 1)


class GraphProblem(BaseProblem):

    input_variables = ["x", "y"]
    output_variables = ["u"]
    conditions = {"data1": Condition(input=input_graph, target=target_tensor)}


# Mapping from stage to dataset names
stage_map = {
    "train": ["train_datasets"],
    "validate": ["val_datasets"],
    "test": ["test_datasets"],
    "all": ["train_datasets", "val_datasets", "test_datasets"],
}


@pytest.mark.parametrize("scale_fn", [torch.std, torch.var])
@pytest.mark.parametrize("shift_fn", [torch.mean, torch.median])
@pytest.mark.parametrize("apply_to", ["input", "target"])
@pytest.mark.parametrize("stage", ["train", "validate", "test", "all"])
def test_constructor(scale_fn, shift_fn, apply_to, stage):
    DataNormalizer(
        scale_fn=scale_fn, shift_fn=shift_fn, stage=stage, apply_to=apply_to
    )

    # Should fail if scale_fn is not Callable
    with pytest.raises(ValueError):
        DataNormalizer(scale_fn=1)

    # Should fail if shift_fn is not Callable
    with pytest.raises(ValueError):
        DataNormalizer(shift_fn=1)

    # Should fail if apply_to is invalid
    with pytest.raises(ValueError):
        DataNormalizer(apply_to="invalid")

    # Should fail if stage is invalid
    with pytest.raises(ValueError):
        DataNormalizer(stage="invalid")


@pytest.mark.parametrize("apply_to", ["input", "target"])
@pytest.mark.parametrize("stage", ["train", "validate", "test", "all"])
@pytest.mark.parametrize("scale_fn", [torch.std, torch.var])
@pytest.mark.parametrize("shift_fn", [torch.mean, torch.median])
@pytest.mark.parametrize("use_lt", [True, False])
def test_routine(apply_to, stage, scale_fn, shift_fn, use_lt):

    # Initialize problem, model and solver
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    model = FeedForward(
        len(problem.input_variables), len(problem.output_variables)
    )
    solver = SupervisedSingleModelSolver(
        problem=problem, model=model, use_lt=use_lt
    )

    # Initialize the callback
    callback = DataNormalizer(
        scale_fn=scale_fn,
        shift_fn=shift_fn,
        stage=stage,
        apply_to=apply_to,
    )

    # Initialize the trainer
    trainer = Trainer(
        solver=solver,
        callbacks=callback,
        accelerator="cpu",
        max_epochs=3,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )

    # Run the training and testing routines
    trainer.train()
    trainer.test()

    # Store datasets to check normalization
    datasets = {
        "train_datasets": trainer.datamodule.train_datasets,
        "val_datasets": trainer.datamodule.val_datasets,
        "test_datasets": trainer.datamodule.test_datasets,
    }

    # Save the expected normalized datasets for each stage
    expected_normalized_datasets = stage_map[stage]

    # Check computed normalizer exists for all input-target conditions
    for name in solver.problem.conditions.keys():
        assert name in callback.normalizer
        assert "scale" in callback.normalizer[name]
        assert "shift" in callback.normalizer[name]

    # Check normalized datasets
    for ds_name, dataset in datasets.items():
        for c_name in callback.normalizer.keys():

            # Extract the data for the current condition
            points = getattr(solver.problem.conditions[c_name], apply_to)

            # Check normalization parameters are correct for normalized datasets
            if ds_name in expected_normalized_datasets:
                expected_shift = shift_fn(points)

                # The expected shift should be close to zero after normalization
                assert torch.isclose(
                    expected_shift,
                    torch.zeros_like(expected_shift),
                    atol=1e-5,
                )

                # The expected scale should be close to one after normalization
                if scale_fn is torch.std:
                    expected_scale = scale_fn(points)

                    assert torch.isclose(
                        expected_scale,
                        torch.ones_like(expected_scale),
                        atol=1e-5,
                    )

    # Should fail if the dataset is graph-based and therefore unsupported
    with pytest.raises(NotImplementedError):

        # Initialize problem, model and solver with graph-based problem
        model = FeedForward(
            len(GraphProblem.input_variables),
            len(GraphProblem.output_variables),
        )
        solver = SupervisedSingleModelSolver(
            problem=GraphProblem(), model=model
        )

        # Initialize the callback
        callback = DataNormalizer(
            scale_fn=scale_fn,
            shift_fn=shift_fn,
            stage=stage,
            apply_to=apply_to,
        )

        # Initialize the trainer
        trainer = Trainer(
            solver=solver,
            callbacks=callback,
            accelerator="cpu",
            max_epochs=3,
        )

        # Run the training routine to trigger the error
        trainer.train()
