"""
Utilities for creating and managing datasets and dataloaders.

This module defines a custom extension of the Lighting DataModule used to handle
dataset splitting, batching, and dataloader creation for PINA conditions.
"""

import warnings

import torch
from lightning.pytorch import LightningDataModule

from pina._src.data.batcher import Batcher
from pina._src.data.loader import MultiLoader


class DataModule(LightningDataModule):
    """
    An extension of the Lightning data module for managing PINA condition
    datasets.

    The data module handles train/validation/test dataset splitting, dataloader
    construction, and batching coordination across multiple conditions.

    Dataset splitting is performed independently for each condition, and each
    resulting subset is a tensor of sample ids indexing the corresponding
    condition. Dataloaders are created over these ids and aggregated into a
    :class:`MultiLoader` according to the selected batching strategy. The
    actual condition batches are materialized on device in
    :meth:`transfer_batch_to_device`.

    :Example:

        >>> import torch
        >>> from pina import LabelTensor
        >>> from pina.condition import Condition
        >>> from pina.problem import BaseProblem
        >>> class MyProblem(BaseProblem):
        ...     def __init__(self):
        ...         super().__init__()
        ...         pts = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
        ...         self.conditions = {"cond1": Condition(input=pts)}
        >>> problem = MyProblem()
        >>> dm = DataModule(problem, train_size=0.8, val_size=0.1,
        ...     test_size=0.1, batch_size=32, batching_mode="common_batch_size",
        ...     shuffle=True, num_workers=0, pin_memory=False)
        >>> dm.setup("fit")
        >>> list(dm.train_datasets.keys())
        ['cond1']
    """

    def __init__(
        self,
        problem,
        train_size=1.0,
        val_size=0.0,
        test_size=0.0,
        batch_size=None,
        batching_mode="common_batch_size",
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        dataloader_cls=None,
        collate_fn=None,
    ):
        """
        Initialization of the :class:`DataModule` class.

        :param BaseProblem problem: The problem containing the conditions and
            sampled data used to construct datasets and dataloaders.
        :param float train_size: The fraction of samples assigned to the
            training split. Must belong to the interval ``[0, 1]``.
        :param float val_size: The fraction of samples assigned to the
            validation split. Must belong to the interval ``[0, 1]``.
        :param float test_size: The fraction of samples assigned to the test
            split. Must belong to the interval ``[0, 1]``.
        :param int batch_size: The number of samples per batch. If ``None``, the
            entire dataset is processed as a single batch.
        :param str batching_mode: The strategy used to aggregate batches across
            dataloaders. Available options are ``"common_batch_size"`` for
            uniform batch sizes across conditions, and ``"proportional"`` for
            batch sizes proportional to dataset sizes.
        :param bool shuffle: Whether condition samples should be shuffled before
            splitting.
        :param int num_workers: The number of worker processes used by
            dataloaders.
        :param bool pin_memory: Whether pinned memory should be enabled during
            data loading.
        :param type dataloader_cls: The dataloader class to use for each
            condition, defaulting to :class:`torch.utils.data.DataLoader`. It
            can also be a mapping between condition names and dataloader
            classes. Custom classes must yield batches of sample ids.
        :param collate_fn: Optional callable (or mapping between condition
            names and callables) used to collapse the raw data selected by a
            condition into a batch. It receives the selected raw data and the
            ids, and returns the batch.
        :raises UserWarning: If ``num_workers`` is set to non-default value
            while ``batch_size`` is None.
        :raises UserWarning: If ``pin_memory`` is set to ``True`` while
            ``batch_size`` is None.
        """
        super().__init__()

        # Initialize the attributes -- consistency checked in trainer
        self.problem = problem
        self.batch_size = batch_size
        self.batching_mode = batching_mode
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader_cls = dataloader_cls
        self.collate_fn = collate_fn

        # If batch size is None, num_workers has no effect
        if batch_size is None and num_workers != 0:
            warnings.warn("num_workers has no effect when batch_size is None.")
            self.num_workers = 0

        # If batch size is None, pin_memory has no effect
        if batch_size is None and pin_memory:
            warnings.warn("pin_memory has no effect when batch_size is None.")
            self.pin_memory = False

        # Move domain discretisation into conditions subsets
        self.problem.move_discretisation_into_conditions()

        # If no splits are defined, use the default dataloaders
        if train_size == 0:
            self.train_dataloader = super().train_dataloader
        if val_size == 0:
            self.val_dataloader = super().val_dataloader
        if test_size == 0:
            self.test_dataloader = super().test_dataloader

        # Otherwise, create the condition splits and initialize the batcher
        self._create_condition_splits(train_size, test_size)
        self.batcher = Batcher(
            batching_mode=self.batching_mode,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            dataloader_cls=self.dataloader_cls,
        )

    def _create_condition_splits(self, train_size, test_size):
        """
        Create train/validation/test index splits for each condition.

        Samples belonging to each condition are optionally shuffled before being
        partitioned into train, validation, and test subsets according to the
        specified split fractions.

        :param float train_size: The fraction of samples assigned to the
            training split. Must belong to the interval ``[0, 1]``.
        :param float test_size: The fraction of samples assigned to the test
            split. Must belong to the interval ``[0, 1]``.
        """
        # Initialize the dictionary to store the split idx for each condition
        self.split_idxs = {}

        # Iterate through conditions and create the splits
        for condition_name, condition in self.problem.conditions.items():

            # Get the total number of samples for the current condition
            condition_length = len(condition)

            # Generate shuffled or sequential indices for the condition samples
            indices = (
                torch.randperm(condition_length).tolist()
                if self.shuffle
                else list(range(condition_length))
            )

            # Compute the split indices for train, validation, and test subsets
            train_end = int(train_size * condition_length)
            test_end = train_end + int(test_size * condition_length)

            # Store the computed split indices in the dictionary
            self.split_idxs[condition_name] = {
                "train": indices[:train_end],
                "test": indices[train_end:test_end],
                "val": indices[test_end:],
            }

    def setup(self, stage=None):
        """
        Create dataset subsets for the requested execution stage.

        Depending on the selected stage, it initializes the ``train_datasets``,
        the ``val_datasets``, or the ``test_datasets`` attributes. Each dataset
        is represented as a mapping between condition names and tensors of
        sample ids.

        :param str stage: The execution stage. Available options are ``"fit"``
            for training/validation and ``"test"`` for testing. If ``None``,
            both training/validation and testing datasets are created.
            Default is ``None``.
        :raises ValueError: If the provided stage is invalid.
        """
        # Validate the stage argument
        if stage not in ("fit", "test", None):
            raise ValueError(
                f"Invalid stage. Got {stage}, expected either 'fit' or  'test'."
            )

        # Fit stage: create training and validation datasets
        if stage in ("fit", None):

            # Train dataset
            self.train_datasets = {
                name: torch.tensor(ids, dtype=torch.long)
                for name, split in self.split_idxs.items()
                if len(split["train"]) > 0
                for ids in [split["train"]]
            }

            # Validation dataset
            self.val_datasets = {
                name: torch.tensor(ids, dtype=torch.long)
                for name, split in self.split_idxs.items()
                if len(split["val"]) > 0
                for ids in [split["val"]]
            }

        # Test stage: create testing dataset
        if stage in ("test", None):

            # Test dataset
            self.test_datasets = {
                name: torch.tensor(ids, dtype=torch.long)
                for name, split in self.split_idxs.items()
                if len(split["test"]) > 0
                for ids in [split["test"]]
            }

    def _resolve(self, name):
        """
        Resolve the per-condition collate function.

        :param str name: The condition name.
        :return: The collate function associated with the condition, or
            ``None`` if none was provided.
        :rtype: Callable | None
        """
        collate_fn = self.collate_fn
        if isinstance(collate_fn, dict):
            return collate_fn.get(name)

        return collate_fn

    def transfer_batch_to_device(self, batch, device, _):
        """
        Transfer a batch to the target device.

        The method materializes the batches of each condition by selecting the
        corresponding sample ids from the condition, and transfers them to the
        specified device.

        :param dict batch: The mapping between the condition names and the
            tensors of sample ids.
        :param torch.device device: The target device.
        :param _: Placeholder argument, not used.
        :return: A list of tuples containing condition names and transferred
            batches.
        :rtype: list[tuple[str, Any]]
        """
        return [
            (
                name,
                self.problem.conditions[name].materialize(
                    ids.tolist(),
                    device=device,
                    batch_fn=self._resolve(name),
                ),
            )
            for name, ids in batch.items()
        ]

    def _build_multi_loader(self, subsets):
        """
        Build the aggregated dataloader for the given subsets.

        :param dict[str, torch.Tensor] subsets: The mapping between condition
            names and tensors of sample ids.
        :return: The aggregated dataloader coordinating all condition
            dataloaders.
        :rtype: MultiLoader
        """
        return MultiLoader(self.batcher(subsets))

    def train_dataloader(self):
        """
        Create the aggregated train dataloader.

        :return: The aggregated dataloader coordinating all train condition
            dataloaders.
        :rtype: MultiLoader
        """
        return self._build_multi_loader(self.train_datasets)

    def val_dataloader(self):
        """
        Create the aggregated validation dataloader.

        :return: The aggregated dataloader coordinating all validation
            condition dataloaders.
        :rtype: MultiLoader
        """
        return self._build_multi_loader(self.val_datasets)

    def test_dataloader(self):
        """
        Create the aggregated test dataloader.

        :return: The aggregated dataloader coordinating all test condition
            dataloaders.
        :rtype: MultiLoader
        """
        return self._build_multi_loader(self.test_datasets)
