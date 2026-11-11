"""Data module for PINA condition datasets."""

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from pina import logger


class DataModule(LightningDataModule):
    """
    Lightning data module for PINA conditions.

    Each dataset row is ``[condition_id, batch_id]``. The condition batch is
    materialized in :meth:`transfer_batch_to_device`, so the dataloaders yield
    one condition batch per step.

    TODO Improve documentation for the batching and transfer process.
    """

    def __init__(
        self,
        problem,
        problem_setting=None,
        train_size=1.0,
        val_size=0.0,
        test_size=0.0,
        batch_size=None,
        batching_mode="simple",
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        dataloader_cls=None,
        collate_fn=None,
    ):
        """
        Initialization of the :class:`DataModule` class.

        Dataset creation and condition batch storage are deferred until
        :meth:`setup`.

        :param ProblemInterface problem: The problem containing the
            conditions to batch and split.
        :param problem_setting: Per-condition keyword arguments passed to
            each condition's ``store_data`` method. Keys are condition
            names and values are dictionaries of settings. These settings
            override the global ``batch_size`` for the corresponding
            condition. If ``None``, an empty dictionary is used.
        :type problem_setting: dict | None
        :param float train_size: Fraction of each condition's batches
            allocated to training, rounded down with a minimum of one
            batch when positive and batches are available. Default is
            ``1.0``.
        :param float val_size: Fraction of each condition's batches
            allocated to validation, rounded down with a minimum of one
            batch when positive and batches remain after training
            allocation. Default is ``0.0``.
        :param float test_size: Requested test fraction. Stored but not
            used to calculate the split: all batches remaining after
            training and validation allocation become test batches.
            Default is ``0.0``.
        :param batch_size: Default batch size passed to each condition's
            ``store_data`` method. Interpretation of ``None`` depends on
            the condition. Default is ``None``.
        :type batch_size: int | None
        :param str batching_mode: Batching mode identifier. Stored but not
            used by the current implementation. Default is
            ``"common_batch_size"``.
        :param bool shuffle: Whether to randomly permute each condition's
            batch indices before splitting. Does not enable dataloader
            shuffling. Default is ``True``.
        :param int num_workers: Number of worker processes used by the
            dataloaders. Default is ``0``.
        :param bool pin_memory: Whether dataloaders pin the identifier
            tensors in memory. Materialized condition data is not loaded
            by these dataloaders. Default is ``False``.
        :param dataloader_cls: Custom dataloader class or per-condition
            mapping of classes. Stored but not used: the current
            implementation always constructs a standard DataLoader.
            Default is ``None``.
        :type dataloader_cls: type | dict | None
        :param collate_fn: Custom collation function or per-condition
            mapping of functions. Stored but not used: the current
            implementation uses default collation. Default is ``None``.
        :type collate_fn: Callable | dict | None
        """
        super().__init__()
        self.problem = problem
        self.batch_size = batch_size
        self.batching_aggregation = batching_mode
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader_cls = dataloader_cls
        self.collate_fn = collate_fn
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        if problem_setting is None:
            problem_setting = {}
        self.problem_setting = problem_setting

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
        """
        aggregate_method = getattr(
            self, 
            f"_{self.batching_aggregation}_aggregate_condition_rows"
        )

        if aggregate_method is None:
            raise ValueError(f"Aggregation method '{self.batching_aggregation}' not found.")

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
                # train_rows.append([condition_id, batch_id])
                condition_train_rows.append([condition_id, batch_id])
            for batch_id in indices[train_end:val_end]:
                # val_rows.append([condition_id, batch_id])
                condition_val_rows.append([condition_id, batch_id])
            for batch_id in indices[val_end:]:
                # test_rows.append([condition_id, batch_id])
                condition_test_rows.append([condition_id, batch_id])

            train_rows.append(condition_train_rows)
            val_rows.append(condition_val_rows)
            test_rows.append(condition_test_rows)

        print("Train rows:", train_rows)
        print("Val rows:", val_rows)
        print("Test rows:", test_rows)

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
        then call each condition's ``store_data`` method with the global
        batch size and any overriding per-condition settings.

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
            settings = {
                "batch_size": self.batch_size,
                **self.problem_setting.get(name, {}),
            }
            condition.store_data(**settings)

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
        assert batch.ndim == 3 and batch.shape[0] == 1, "Batch must have shape [1, n_rows, 2]"

        for condition_id, batch_id in batch.squeeze(0):
            name = self.idx_to_condition_name[int(condition_id)]
            data = self.problem.conditions[name].materialize(
                int(batch_id), device=device
            )
            result.append((name, data))
        return result
