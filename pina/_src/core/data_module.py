"""Data module for PINA condition datasets."""

import warnings
from typing import ClassVar

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from pina import logger
from pina._src.core.utils import (
    check_consistency,
    check_positive_integer,
)


class DataModule(LightningDataModule):
    """
    Lightning data module for PINA conditions.

    The data module owns every setting used to create the datasets: the
    train/validation/test split, the batch size, the batching rule, and the
    dataloader options. Each dataset row is ``[condition_id, batch_id]``. The
    condition batch is materialized in :meth:`transfer_batch_to_device`, so the
    dataloaders yield one condition batch per step.

    The batch size can be set globally, with an ``int``, or per condition, with
    a mapping from condition names to batch sizes. A condition mapped to
    ``None`` is stored as a single batch.

    :Example:

        >>> from pina import DataModule, Trainer
        >>> # Advanced usage: build the data module explicitly and pass it
        >>> # to the trainer.
        >>> data_module = DataModule(
        ...     problem=problem,
        ...     train_size=0.8,
        ...     val_size=0.1,
        ...     test_size=0.1,
        ...     batch_size={"D": 32, "boundary": 16},
        ...     shuffle=True,
        ... )
        >>> trainer = Trainer(solver=solver, datamodule=data_module)
    """

    # Available batching modes
    _AVAIL_BATCHING_MODES: ClassVar[set[str]] = {"simple"}

    def __init__(
        self,
        problem,
        train_size=1.0,
        val_size=0.0,
        test_size=0.0,
        batch_size=None,
        shuffle=True,
        batching_mode="simple",
        num_workers=0,
        pin_memory=False,
    ):
        """
        Initialization of the :class:`DataModule` class.

        Dataset creation and condition batch storage are deferred until
        :meth:`setup`.

        :param ProblemInterface problem: The problem containing the
            conditions to batch and split.
        :param float train_size: Fraction of each condition's batches
            allocated to training, rounded down with a minimum of one
            batch when positive and batches are available. Must belong to
            the interval ``[0, 1]``. Default is ``1.0``.
        :param float val_size: Fraction of each condition's batches
            allocated to validation, rounded down with a minimum of one
            batch when positive and batches remain after training
            allocation. Must belong to the interval ``[0, 1]``.
            Default is ``0.0``.
        :param float test_size: Requested test fraction. Stored but not
            used to calculate the split: all batches remaining after
            training and validation allocation become test batches. Must
            belong to the interval ``[0, 1]``. Default is ``0.0``.
        :param batch_size: Default batch size passed to each condition's
            ``store_data`` method. It can be a positive integer applied to
            every condition, a mapping from condition names to batch sizes,
            or ``None`` to store each condition as a single batch. Mapping
            values set to ``None`` store that condition as a single batch.
            Default is ``None``.
        :type batch_size: int | dict[str, int | None] | None
        :param bool shuffle: Whether to randomly permute each condition's
            batch indices before splitting. Does not enable dataloader
            shuffling. Default is ``True``.
        :param str batching_mode: The strategy used to aggregate batches
            across conditions. Available options are ``"simple"`` for the
            one-batch-per-condition aggregation. Default is ``"simple"``.
        :param int num_workers: Number of worker processes used by the
            dataloaders. Default is ``0``.
        :param bool pin_memory: Whether dataloaders pin the identifier
            tensors in memory. Materialized condition data is not loaded
            by these dataloaders. Default is ``False``.
        :raises ValueError: If ``train_size``, ``val_size``, or ``test_size``
            is not a float in the interval ``[0, 1]``.
        :raises ValueError: If the sum of ``train_size``, ``val_size``, and
            ``test_size`` is not equal to 1.
        :raises ValueError: If ``pin_memory`` or ``shuffle`` is not a boolean.
        :raises AssertionError: If ``num_workers`` is a negative integer.
        :raises AssertionError: If ``batch_size`` is not a positive integer,
            or a mapping to positive integers or ``None``.
        :raises ValueError: If a ``batch_size`` mapping refers to unknown
            conditions.
        :raises ValueError: If ``batching_mode`` is not one of the available
            options.
        :raises UserWarning: If ``num_workers`` or ``pin_memory`` are set
            without a batch size.
        """
        super().__init__()

        # Check the split sizes
        check_consistency(train_size, float)
        check_consistency(val_size, float)
        check_consistency(test_size, float)

        for name, value in (
            ("train_size", train_size),
            ("val_size", val_size),
            ("test_size", test_size),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"`{name}` must belong to the interval [0, 1]. "
                    f"Got {value}."
                )

        # Check that train_size, test_size and val_size sum to 1
        total = train_size + val_size + test_size
        if not torch.isclose(torch.tensor(total), torch.tensor(1.0)):
            raise ValueError(
                "`train_size`, `val_size`, and `test_size` must sum to 1."
            )

        # Check the remaining options
        check_consistency(shuffle, bool)
        check_consistency(pin_memory, bool)
        check_positive_integer(num_workers, strict=False)

        # Check the batch size, which can be global or per condition
        if batch_size is not None:
            if isinstance(batch_size, dict):
                unknown = set(batch_size) - set(problem.conditions)
                if unknown:
                    raise ValueError(
                        "The `batch_size` mapping refers to unknown "
                        f"conditions: {sorted(unknown)}. Valid conditions "
                        f"are: {sorted(problem.conditions)}."
                    )
                for value in batch_size.values():
                    if value is not None:
                        check_positive_integer(value, strict=True)
            else:
                check_positive_integer(batch_size, strict=True)

        # Check the batching mode
        if batching_mode not in self._AVAIL_BATCHING_MODES:
            raise ValueError(
                f"Invalid batching mode '{batching_mode}'. "
                f"Expected one of: {sorted(self._AVAIL_BATCHING_MODES)}."
            )

        # Warn if dataloader options are set without a batch size
        if batch_size is None and (num_workers > 0 or pin_memory):
            warnings.warn(
                "`num_workers` and `pin_memory` have no effect when "
                "`batch_size` is None.",
                UserWarning,
                stacklevel=2,
            )

        self.problem = problem
        self.batch_size = batch_size
        self.batching_mode = batching_mode
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    @property
    def batching_aggregation(self):
        """
        Return the batching mode.

        Deprecated alias kept for backward compatibility with the previous
        attribute name.

        :return: The batching mode.
        :rtype: str
        """
        return self.batching_mode

    def _condition_batch_size(self, name):
        """
        Return the batch size configured for a condition.

        :param str name: The condition name.
        :return: The configured batch size, or ``None`` for a single batch.
        :rtype: int | None
        """
        if isinstance(self.batch_size, dict):
            return self.batch_size.get(name)
        return self.batch_size

    def enumerate_condition(self):
        """
        Assign integer identifiers to the problem's conditions.

        Identifiers follow the iteration order of ``problem.conditions``.
        Store the mappings in ``condition_name_to_idx`` and
        ``idx_to_condition_name``, replacing any existing mappings.

        :return: The mapping from condition names to integer identifiers.
        :rtype: dict[str, int]
        """
        self.condition_name_to_idx = {
            name: idx for idx, name in enumerate(self.problem.conditions)
        }
        self.idx_to_condition_name = {
            idx: name for name, idx in self.condition_name_to_idx.items()
        }
        return self.condition_name_to_idx

    def _simple_aggregate_condition_rows(self, condition_rows):
        non_empty_rows = [rows for rows in condition_rows if rows]
        if not non_empty_rows:
            return torch.empty((0, 0, 2), dtype=torch.long)

        # One batch per condition and step, cycling shorter conditions until
        # the condition with the most batches is exhausted.
        n_steps = max(len(rows) for rows in non_empty_rows)
        aggregated_rows = [
            [rows[step % len(rows)] for rows in non_empty_rows]
            for step in range(n_steps)
        ]
        return torch.tensor(aggregated_rows, dtype=torch.long)

    def aggregate_condition_rows(self, condition_rows):
        """
        Aggregate condition-specific rows into a single tensor.

        :param condition_rows: List of lists containing condition-specific rows.
        :type condition_rows: list[list[int]]
        :return: Aggregated tensor with one condition batch per condition and
            step.
        :rtype: torch.Tensor
        :raises ValueError: If the configured batching mode has no aggregation
            method.
        """
        aggregate_method = getattr(
            self,
            f"_{self.batching_mode}_aggregate_condition_rows",
            None,
        )

        if aggregate_method is None:
            raise ValueError(
                f"Aggregation method '{self.batching_mode}' not found."
            )

        return aggregate_method(condition_rows)

    def create_datasets(self):
        """
        Split stored condition batch identifiers into datasets.

        Optionally shuffle each condition's batch indices, then allocate
        training and validation batches using ``train_size`` and
        ``val_size``. All remaining batches become test batches;
        ``test_size`` is not used.

        Store the splits in ``train_dataset``, ``val_dataset``, and
        ``test_dataset`` as integer tensors with shape
        ``[n_steps, n_conditions, 2]``. Each row contains a condition
        identifier and a batch identifier. Empty splits have shape
        ``[0, 0, 2]``.

        Conditions must have stored their batches, and
        :meth:`enumerate_condition` must have been called.

        :return: None.
        :rtype: None
        :raises AttributeError: If condition batch storage or the condition
            identifier mapping has not been initialized.
        :raises KeyError: If a condition is absent from the identifier
            mapping.
        """
        train_rows, val_rows, test_rows = [], [], []

        for condition_name, condition in self.problem.conditions.items():
            condition_train_rows = []
            condition_val_rows = []
            condition_test_rows = []

            n_batches = len(condition)
            indices = (
                torch.randperm(n_batches).tolist()
                if self.shuffle
                else list(range(n_batches))
            )

            train_end = 0
            if self.train_size > 0 and n_batches > 0:
                train_end = max(1, int(self.train_size * n_batches))

            val_end = train_end
            if self.val_size > 0 and n_batches > train_end:
                val_end += max(1, int(self.val_size * n_batches))
                val_end = min(val_end, n_batches)

            condition_id = self.condition_name_to_idx[condition_name]
            for batch_id in indices[:train_end]:
                condition_train_rows.append([condition_id, batch_id])
            for batch_id in indices[train_end:val_end]:
                condition_val_rows.append([condition_id, batch_id])
            for batch_id in indices[val_end:]:
                condition_test_rows.append([condition_id, batch_id])

            train_rows.append(condition_train_rows)
            val_rows.append(condition_val_rows)
            test_rows.append(condition_test_rows)

        self.train_dataset = self.aggregate_condition_rows(train_rows)
        self.val_dataset = self.aggregate_condition_rows(val_rows)
        self.test_dataset = self.aggregate_condition_rows(test_rows)

        logger.debug(
            "Created datasets with %d train, %d validation, and %d test rows.",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

    def setup(self, stage=None):
        """
        Store condition batches and create all dataset splits.

        Move the problem's discretized domains into their conditions,
        then call each condition's ``store_data`` method with the batch
        size configured for that condition.

        All accepted stages perform the same setup. Calling this method
        again rebuilds condition batches and dataset splits.

        :param stage: The Lightning setup stage. Accepted values are
            ``"fit"``, ``"test"``, and ``None``. Default is ``None``.
        :type stage: str | None
        :return: None.
        :rtype: None
        :raises ValueError: If the stage is not ``"fit"``, ``"test"``,
            or ``None``.
        """
        if stage not in ("fit", "test", None):
            raise ValueError(
                f"Invalid stage. Got {stage}, expected 'fit' or 'test'."
            )

        self.problem.move_discretisation_into_conditions()

        for name, condition in self.problem.conditions.items():
            condition.store_data(
                batch_size=self._condition_batch_size(name)
            )

        self.enumerate_condition()
        self.create_datasets()

    def _dataloader(self, dataset):
        """
        Create a dataloader for condition and batch identifiers.

        Each iteration yields one step with shape ``[1, n_conditions, 2]``.
        Steps are visited in dataset order using default collation.

        :param torch.Tensor dataset: The identifier dataset with shape
            ``[n_steps, n_conditions, 2]``.
        :return: A dataloader over the identifier rows, or an empty list
            if the dataset is empty.
        :rtype: DataLoader | list
        """
        if len(dataset) == 0:
            return []
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        """
        Return the dataloader for training batch identifiers.

        :return: The training dataloader, or an empty list if the training
            dataset is empty.
        :rtype: DataLoader | list
        :raises AttributeError: If the training dataset has not been
            created.
        """
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        """
        Return the dataloader for validation batch identifiers.

        :return: The validation dataloader, or an empty list if the
            validation dataset is empty.
        :rtype: DataLoader | list
        :raises AttributeError: If the validation dataset has not been
            created.
        """
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        """
        Return the dataloader for test batch identifiers.

        :return: The test dataloader, or an empty list if the test dataset
            is empty.
        :rtype: DataLoader | list
        :raises AttributeError: If the test dataset has not been created.
        """
        return self._dataloader(self.test_dataset)

    def transfer_batch_to_device(self, batch, device, _):
        """
        Materialize the condition batches referenced by identifier rows.

        Resolve each condition identifier to its name and call the
        condition's ``materialize`` method with the batch identifier and
        requested device.

        :param torch.Tensor batch: Identifier rows with shape
            ``[n_rows, 2]``, each containing a condition identifier and a
            batch identifier. Dataloaders normally provide one row.
        :param device: The device on which condition batches are
            materialized. If ``None``, device handling is delegated to
            each condition without a target device.
        :type device: str | torch.device | None
        :param int _: The Lightning dataloader index. Unused.
        :return: Pairs containing the condition name and its materialized
            batch. The batch type depends on the condition.
        :rtype: list[tuple[str, Any]]
        :raises AttributeError: If the condition identifier mapping has
            not been initialized.
        :raises KeyError: If a condition identifier is unknown or its
            condition is absent from the problem.
        :raises IndexError: If a batch identifier is out of range for
            its condition.
        """
        result = []
        assert (
            batch.ndim == 3 and batch.shape[0] == 1
        ), "Batch must have shape [1, n_rows, 2]"

        for condition_id, batch_id in batch.squeeze(0):
            name = self.idx_to_condition_name[int(condition_id)]
            data = self.problem.conditions[name].materialize(
                int(batch_id), device=device
            )
            result.append((name, data))
        return result
