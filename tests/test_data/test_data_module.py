import pytest
import torch

from pina.data import DataModule
from pina.graph import RadiusGraph
from pina.problem.zoo import Poisson2DSquareProblem, SupervisedProblem

# Number of samples in the synthetic datasets
n_samples = 100


# Define helper functions to create synthetic tensor data
def _create_tensor_data(n=n_samples):
    return (torch.rand((n, 4)), torch.rand((n, 2)))


# Define helper function to create synthetic graph data
def _create_graph_data(n=n_samples):

    # Define input graphs and output tensor
    input_graphs = [
        RadiusGraph(x=torch.rand((20, 4)), pos=torch.rand((20, 2)), radius=0.2)
        for _ in range(n)
    ]
    output_tensor = torch.rand((n, 50, 2))

    return input_graphs, output_tensor


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

    # Build a tensor problem
    if problem_type == "tensor":
        input_tensor, output_tensor = _create_tensor_data()
        problem = SupervisedProblem(input_=input_tensor, output_=output_tensor)

    # Build a graph problem
    elif problem_type == "graph":
        input_graph, output_graph = _create_graph_data()
        problem = SupervisedProblem(input_=input_graph, output_=output_graph)

    # Build a pinn problem
    elif problem_type == "pinn":
        problem = Poisson2DSquareProblem()
        problem.discretise_domain(n=n_samples, mode="random")

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

    # Expected keys in the split_idxs dictionary
    expected_keys = (
        {"data"} if problem_type in ["tensor", "graph"] else {"D", "boundary"}
    )

    # Check that the split_idxs attribute has been created correctly
    assert hasattr(dm, "split_idxs")
    assert isinstance(dm.split_idxs, dict)
    assert set(dm.split_idxs.keys()) == expected_keys

    # Iterate over keys in split_idxs
    for k in dm.split_idxs.keys():

        # Assert that the value corresponding to each key is a dictionary
        assert isinstance(dm.split_idxs[k], dict)
        assert set(dm.split_idxs[k].keys()) == {"train", "val", "test"}

        # Expected lengths of splits
        expected_lengths = {
            "train": int(train_size * n_samples),
            "val": int(val_size * n_samples),
            "test": int(test_size * n_samples),
        }

        # Iterate over splits
        for split in ["train", "val", "test"]:
            assert isinstance(dm.split_idxs[k][split], list)
            assert len(dm.split_idxs[k][split]) == expected_lengths[split]


@pytest.mark.parametrize("problem_type", ["tensor", "graph", "pinn"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize(
    "train_size, val_size, test_size",
    [(0.7, 0.2, 0.1), (0.8, 0.2, 0.0), (0.0, 0.8, 0.2)],
)
def test_setup(problem_type, batch_size, train_size, val_size, test_size):

    # Build a tensor problem
    if problem_type == "tensor":
        input_tensor, output_tensor = _create_tensor_data()
        problem = SupervisedProblem(input_=input_tensor, output_=output_tensor)

    # Build a graph problem
    elif problem_type == "graph":
        input_graph, output_graph = _create_graph_data()
        problem = SupervisedProblem(input_=input_graph, output_=output_graph)

    # Build a pinn problem
    elif problem_type == "pinn":
        problem = Poisson2DSquareProblem()
        problem.discretise_domain(n=n_samples, mode="random")

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

    # Expected keys in the split_idxs dictionary
    expected_keys = (
        {"data"} if problem_type in ["tensor", "graph"] else {"D", "boundary"}
    )

    # Expected lengths of splits
    expected_lengths = {
        "train": int(train_size * n_samples),
        "val": int(val_size * n_samples),
        "test": int(test_size * n_samples),
    }

    # Iterate over datasets
    for dataset, split in [
        ("train_datasets", "train"),
        ("val_datasets", "val"),
        ("test_datasets", "test"),
    ]:

        # Assert that each dataset has been created correctly
        assert hasattr(dm, dataset)
        assert isinstance(getattr(dm, dataset), dict)

        # Assert that the keys in each dataset are correct, if not empty
        if getattr(dm, dataset):
            assert set(getattr(dm, dataset).keys()) == expected_keys

            # Iterate over keys in each dataset
            for key in expected_keys:

                # Assert that the corresponding value is a tensor of
                # sample ids with the expected length
                assert isinstance(getattr(dm, dataset)[key], torch.Tensor)
                assert getattr(dm, dataset)[key].dtype == torch.long
                assert len(getattr(dm, dataset)[key]) == expected_lengths[split]
