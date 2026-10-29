import pytest

from pina.data import MultiLoader

"""
Note: this test intentionally avoids relying on the actual DataLoader
implementation in order to keep the test focused on the loader aggregation logic
itself and independent from the behavior of external classes. The full pipeline
is tested in the DataModule tests, which ensures that the loader works correctly
when used in the intended context.
"""


# Define a dummy dataloader for testing purposes
class DummyDataloader:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


# Create dataloaders for testing
data_loaders1 = {
    "condition_1": DummyDataloader([1, 2, 3]),
    "condition_2": DummyDataloader([10, 20, 30]),
}
data_loaders2 = {
    "condition_1": DummyDataloader([1, 2]),
    "condition_2": DummyDataloader([10, 20, 30, 40, 50]),
}
data_loaders3 = {
    "condition_1": DummyDataloader([1]),
    "condition_2": DummyDataloader([10, 20, 30]),
}

# Create expected batches for testing
expected_batches1 = [
    {"condition_1": 1, "condition_2": 10},
    {"condition_1": 2, "condition_2": 20},
    {"condition_1": 3, "condition_2": 30},
]
expected_batches2 = [
    {"condition_1": 1, "condition_2": 10},
    {"condition_1": 2, "condition_2": 20},
    {"condition_1": 1, "condition_2": 30},
    {"condition_1": 2, "condition_2": 40},
    {"condition_1": 1, "condition_2": 50},
]
expected_batches3 = [
    {"condition_1": 1, "condition_2": 10},
    {"condition_1": 1, "condition_2": 20},
    {"condition_1": 1, "condition_2": 30},
]


def test_constructor():

    # Create dummy dataloaders
    dataloaders = {
        "condition_1": DummyDataloader([1, 2, 3]),
        "condition_2": DummyDataloader([10, 20]),
    }

    # Initialize the loader
    MultiLoader(dataloaders)


@pytest.mark.parametrize(
    "dataloaders, expected",
    [
        (data_loaders1, expected_batches1),
        (data_loaders2, expected_batches2),
        (data_loaders3, expected_batches3),
    ],
)
def test_len_and_iter(dataloaders, expected):

    # Initialize the loader
    loader = MultiLoader(dataloaders)

    # Check that the number of yielded batches matches len(loader)
    assert len(expected) == len(loader)

    # Check yielded batches
    assert list(loader) == expected

    # Check that the loader can be iterated multiple times
    assert list(loader) == expected
