import pytest
import torch

from pina.condition import GraphInputTargetCondition, InputTargetCondition
from pina.data import DataModule
from pina.graph import RadiusGraph
from pina.problem import BaseProblem
from pina.problem.zoo import Poisson2DSquareProblem, SupervisedProblem

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
