import torch
import pytest
from pina import Trainer, LabelTensor, Condition
from pina.solver import SupervisedSingleModelSolver
from pina.callback import DataNormalizer
from pina.problem import BaseProblem
from pina.model import FeedForward
from pina.graph import Graph
from pina.condition import GraphInputTargetCondition


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
_graph = Graph(
    x=torch.rand(10, 2),
    y=torch.rand(10, 1),
    pos=torch.rand(10, 2),
    edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
)


class GraphProblem(BaseProblem):
    input_variables = ["x", "y"]
    output_variables = ["u"]
    conditions = {"data1": GraphInputTargetCondition(graph=_graph)}


# Mapping from callback stage to split names
stage_map = {
    "train": ["train"],
    "validate": ["validate"],
    "test": ["test"],
    "all": ["train", "validate", "test"],
}


def _batch_ids_by_condition(datamodule, dataset):
    """Return {condition_name: [batch_id, ...]} for a dataset tensor."""
    ids = {}
    if dataset is None or len(dataset) == 0:
        return ids
    for condition_id, batch_id in dataset:
        name = datamodule.idx_to_condition_name[int(condition_id)]
        ids.setdefault(name, []).append(int(batch_id))
    return ids


def _collect(condition, batch_ids, field):
    """Concatenate a field across the given batch ids."""
    return torch.cat([condition.data[i][field] for i in batch_ids])


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

    # Initialize the trainer with batch_size so each condition gets multiple
    # batches and val/test splits are non-empty
    trainer = Trainer(
        solver=solver,
        callbacks=callback,
        accelerator="cpu",
        max_epochs=3,
        batch_size=5,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        shuffle=False,
    )

    # Capture original (unnormalized) condition data from the condition's
    # backing tensor before any DataModule setup transforms it.
    # We clone so later in-place mutations don't affect this snapshot.
    orig_input = {
        name: getattr(cond, f"_{apply_to}").clone().detach()
        for name, cond in solver.problem.conditions.items()
    }

    # Run the training and testing routines
    trainer.train()
    trainer.test()

    # After training, re-derive split ids from the data module
    dm = trainer.data_module
    splits_after = {
        "train": dm.train_dataset,
        "validate": dm.val_dataset,
        "test": dm.test_dataset,
    }
    split_ids_after = {
        s: _batch_ids_by_condition(dm, ds) for s, ds in splits_after.items()
    }

    # Check computed normalizer exists for all input-target conditions
    for name in solver.problem.conditions.keys():
        assert name in callback.normalizer
        assert "scale" in callback.normalizer[name]
        assert "shift" in callback.normalizer[name]

    expected_normalized = stage_map[stage]

    for split in ("train", "validate", "test"):
        for cond, ids in split_ids_after[split].items():
            if cond not in callback.normalizer:
                continue
            points = _collect(solver.problem.conditions[cond], ids, apply_to)

            # Reconstruct raw slices for these batch ids (batch_size=5, ordered)
            raw_slices = torch.cat([
                orig_input[cond][i * 5: (i + 1) * 5] for i in ids
            ])

            if split in expected_normalized:
                shift = callback.normalizer[cond]["shift"]
                scale = callback.normalizer[cond]["scale"]
                expected = (raw_slices - shift) / scale
                assert torch.allclose(
                    points.float(), expected.float(), atol=1e-5
                ), f"split={split} cond={cond} not normalized correctly"
            else:
                assert torch.allclose(
                    points.float(), raw_slices.float(), atol=1e-5
                ), f"split={split} cond={cond} should not have been normalized"


def test_graph_not_implemented():
    """DataNormalizer should raise NotImplementedError for graph conditions."""
    model = FeedForward(
        len(GraphProblem.input_variables),
        len(GraphProblem.output_variables),
    )
    solver = SupervisedSingleModelSolver(problem=GraphProblem(), model=model)

    callback = DataNormalizer(stage="all", apply_to="input")

    trainer = Trainer(
        solver=solver,
        callbacks=callback,
        accelerator="cpu",
        max_epochs=3,
    )

    with pytest.raises(NotImplementedError):
        trainer.train()
