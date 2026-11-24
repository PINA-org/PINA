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
    train/validation/test split, the batch size, the mini-batch rules, the
    batching mode, and the dataloader options.

    Terminology:

    * **mini-batch**: a chunk of the data of a single condition, built by
      that condition's ``store_mini_batches`` method;
    * **batch**: the mini-batches of all conditions processed together in
      one optimization step, i.e. the ``batch`` received by
      ``training_step``. With the ``"simple"`` batching mode a batch holds
      one mini-batch per condition;
    * **epoch**: one pass over the batches of a split. With the
      ``"simple"`` batching mode it lasts as many batches as the condition
      with the most mini-batches, cycling the conditions that have fewer.

    Each condition builds its own mini-batches. How it does so is set per
    condition by ``mini_batch_rule``, a mapping from condition names to the
    keyword arguments of ``store_mini_batches`` (e.g. ``batch_size`` for
    tensor conditions, ``n_subgraphs`` and ``unroll_length`` for graph
    time-series conditions). A global ``batch_size`` applies to every
    condition whose rule does not set one. ``batching_mode`` then decides
    how the mini-batches of the different conditions are combined into
    batches.

    The datasets only store mini-batch identifiers: each batch is a set of
    ``[condition_id, mini_batch_id]`` rows, and the mini-batches are
    materialized in :meth:`transfer_batch_to_device`.

    :Example:

        >>> from pina import DataModule, Trainer
        >>> # Advanced usage: build the data module explicitly and pass it
        >>> # to the trainer.
        >>> data_module = DataModule(
        ...     problem=problem,
        ...     train_size=0.8,
        ...     val_size=0.1,
        ...     test_size=0.1,
        ...     batch_size=32,  # default for every condition
        ...     mini_batch_rule={
        ...         "boundary": {"batch_size": 16},  # overrides the default
        ...         "time": {"n_subgraphs": 20, "unroll_length": 3},
        ...     },
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
        mini_batch_rule=None,
    ):
        """
        Initialization of the :class:`DataModule` class.

        Dataset creation and mini-batch storage are deferred until
        :meth:`setup`.

        :param ProblemInterface problem: The problem containing the
            conditions to split into mini-batches.
        :param float train_size: Fraction of each condition's mini-batches
            allocated to training, rounded down with a minimum of one
            mini-batch when positive and mini-batches are available. Must
            belong to the interval ``[0, 1]``. Default is ``1.0``.
        :param float val_size: Fraction of each condition's mini-batches
            allocated to validation, rounded down with a minimum of one
            mini-batch when positive and mini-batches remain after training
            allocation. Must belong to the interval ``[0, 1]``.
            Default is ``0.0``.
        :param float test_size: Requested test fraction. Stored but not
            used to calculate the split: all mini-batches remaining after
            training and validation allocation become test mini-batches.
            Must belong to the interval ``[0, 1]``. Default is ``0.0``.
        :param batch_size: Default number of samples in each mini-batch,
            passed to each condition's ``store_mini_batches`` method and
            used by every condition whose ``mini_batch_rule`` does not set
            ``batch_size``. ``None`` stores each such tensor condition as a
            single mini-batch. Graph conditions ignore it, since each
            subgraph is one mini-batch. A batch, which holds the
            mini-batches of all conditions, can therefore contain more
            than ``batch_size`` samples. A mapping from condition names to
            batch sizes is deprecated: use ``mini_batch_rule`` instead.
            Default is ``None``.
        :type batch_size: int | None
        :param bool shuffle: Whether to randomly permute each condition's
            mini-batch indices before splitting. The order is drawn once in
            :meth:`setup`: it does not enable dataloader shuffling, so the
            batches are visited in the same order in every epoch. Default
            is ``True``.
        :param str batching_mode: The strategy used to combine the
            mini-batches of the conditions into batches. Available options
            are ``"simple"``, which puts one mini-batch of each condition
            in every batch. Default is ``"simple"``.
        :param int num_workers: Number of worker processes used by the
            dataloaders. Default is ``0``.
        :param bool pin_memory: Whether dataloaders pin the identifier
            tensors in memory. Materialized condition data is not loaded
            by these dataloaders. Default is ``False``.
        :param mini_batch_rule: Per-condition rules used to build the
            mini-batches. It maps each condition name to the keyword
            arguments forwarded to that condition's ``store_mini_batches``
            method, e.g.
            ``{"D": {"batch_size": 64},
            "time": {"n_subgraphs": 20, "num_hops": 30, "unroll_length": 3}}``.
            Each condition accepts only the keys it supports; conditions
            not listed use the global ``batch_size`` and their defaults.
            The rule controls how each condition splits its own data into
            mini-batches, while ``batching_mode`` controls how the
            mini-batches of different conditions are combined into batches.
            A ``batch_size`` set in a rule takes precedence over the global
            one. Default is ``None``.
        :type mini_batch_rule: dict[str, dict] | None
        :raises ValueError: If ``train_size``, ``val_size``, or ``test_size``
            is not a float in the interval ``[0, 1]``.
        :raises ValueError: If the sum of ``train_size``, ``val_size``, and
            ``test_size`` is not equal to 1.
        :raises ValueError: If ``pin_memory`` or ``shuffle`` is not a boolean.
        :raises AssertionError: If ``num_workers`` is a negative integer.
        :raises AssertionError: If ``batch_size`` is not a positive integer,
            or a mapping to positive integers or ``None``.
        :raises ValueError: If a ``batch_size`` or ``mini_batch_rule``
            mapping refers to unknown conditions.
        :raises ValueError: If a ``mini_batch_rule`` value is not a
            dictionary.
        :raises DeprecationWarning: If ``batch_size`` is a mapping.
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

        # Normalize the per-condition rules: each one must be a dictionary,
        # copied so that the caller's mapping is never modified
        rules = {}
        for name, rule in (mini_batch_rule or {}).items():
            if not isinstance(rule, dict):
                raise ValueError(
                    f"The `mini_batch_rule` of condition '{name}' must be a "
                    f"dictionary. Got {type(rule).__name__}."
                )
            rules[name] = dict(rule)

        # Backward compatibility: a per-condition batch size mapping is moved
        # into the rules; a batch size already set in a rule takes precedence
        if isinstance(batch_size, dict):
            warnings.warn(
                "Passing `batch_size` as a mapping is deprecated and will be "
                "removed in a future release. Use `batch_size` as the global "
                "default and `mini_batch_rule={name: {'batch_size': ...}}` "
                "for per-condition values instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            for name, value in batch_size.items():
                rules.setdefault(name, {}).setdefault("batch_size", value)
            batch_size = None

        # Validate the global batch size and the rules once
        unknown = set(rules) - set(problem.conditions)
        if unknown:
            raise ValueError(
                "`mini_batch_rule` (or the deprecated `batch_size` mapping) "
                f"refers to unknown conditions: {sorted(unknown)}. Valid "
                f"conditions are: {sorted(problem.conditions)}."
            )
        if batch_size is not None:
            check_positive_integer(batch_size, strict=True)
        for rule in rules.values():
            if rule.get("batch_size") is not None:
                check_positive_integer(rule["batch_size"], strict=True)

        # Check the batching mode
        if batching_mode not in self._AVAIL_BATCHING_MODES:
            raise ValueError(
                f"Invalid batching mode '{batching_mode}'. "
                f"Expected one of: {sorted(self._AVAIL_BATCHING_MODES)}."
            )

        # Warn if dataloader options are set without any batch size
        any_batch_size = batch_size is not None or any(
            rule.get("batch_size") is not None for rule in rules.values()
        )
        if not any_batch_size and (num_workers > 0 or pin_memory):
            warnings.warn(
                "`num_workers` and `pin_memory` have no effect when "
                "`batch_size` is None.",
                UserWarning,
                stacklevel=2,
            )

        self.problem = problem
        self.batch_size = batch_size
        self.mini_batch_rule = rules
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

    @property
    def has_batch_size(self):
        """
        Whether a batch size is set, globally or by any mini-batch rule.

        :return: ``True`` if any batch size is not ``None``.
        :rtype: bool
        """
        return self.batch_size is not None or any(
            rule.get("batch_size") is not None
            for rule in self.mini_batch_rule.values()
        )

    def _condition_mini_batch_rule(self, name):
        """
        Return the mini-batch rule of a condition, with the global batch size
        as default for ``batch_size``.

        :param str name: The condition name.
        :return: The keyword arguments for the condition's
            ``store_mini_batches`` method.
        :rtype: dict
        """
        rule = dict(self.mini_batch_rule.get(name, {}))
        rule.setdefault("batch_size", self.batch_size)
        return rule

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

        # One mini-batch per condition in each batch, cycling the conditions
        # with fewer mini-batches until the one with the most is exhausted.
        n_batches = max(len(rows) for rows in non_empty_rows)
        aggregated_rows = [
            [rows[batch_idx % len(rows)] for rows in non_empty_rows]
            for batch_idx in range(n_batches)
        ]
        return torch.tensor(aggregated_rows, dtype=torch.long)

    def aggregate_condition_rows(self, condition_rows):
        """
        Combine the mini-batch rows of each condition into batches.

        :param condition_rows: For each condition, the list of its
            ``[condition_id, mini_batch_id]`` rows.
        :type condition_rows: list[list[list[int]]]
        :return: Tensor with shape ``[n_batches, n_conditions, 2]``: each
            batch holds one ``[condition_id, mini_batch_id]`` row per
            condition. Empty if no condition has mini-batches.
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
        Split the stored mini-batch identifiers into datasets.

        Optionally shuffle each condition's mini-batch indices, then allocate
        training and validation mini-batches using ``train_size`` and
        ``val_size``. All remaining mini-batches become test mini-batches;
        ``test_size`` is not used.

        Store the splits in ``train_dataset``, ``val_dataset``, and
        ``test_dataset`` as integer tensors with shape
        ``[n_batches, n_conditions, 2]``. Each row contains a condition
        identifier and a mini-batch identifier. Empty splits have shape
        ``[0, 0, 2]``.

        Conditions must have stored their mini-batches, and
        :meth:`enumerate_condition` must have been called.

        :return: None.
        :rtype: None
        :raises AttributeError: If the mini-batch storage or the condition
            identifier mapping has not been initialized.
        :raises KeyError: If a condition is absent from the identifier
            mapping.
        """
        train_rows, val_rows, test_rows = [], [], []

        for condition_name, condition in self.problem.conditions.items():
            condition_train_rows = []
            condition_val_rows = []
            condition_test_rows = []

            n_mini_batches = len(condition)
            indices = (
                torch.randperm(n_mini_batches).tolist()
                if self.shuffle
                else list(range(n_mini_batches))
            )

            train_end = 0
            if self.train_size > 0 and n_mini_batches > 0:
                train_end = max(1, int(self.train_size * n_mini_batches))

            val_end = train_end
            if self.val_size > 0 and n_mini_batches > train_end:
                val_end += max(1, int(self.val_size * n_mini_batches))
                val_end = min(val_end, n_mini_batches)

            condition_id = self.condition_name_to_idx[condition_name]
            for mini_batch_id in indices[:train_end]:
                condition_train_rows.append([condition_id, mini_batch_id])
            for mini_batch_id in indices[train_end:val_end]:
                condition_val_rows.append([condition_id, mini_batch_id])
            for mini_batch_id in indices[val_end:]:
                condition_test_rows.append([condition_id, mini_batch_id])

            train_rows.append(condition_train_rows)
            val_rows.append(condition_val_rows)
            test_rows.append(condition_test_rows)

        self.train_dataset = self.aggregate_condition_rows(train_rows)
        self.val_dataset = self.aggregate_condition_rows(val_rows)
        self.test_dataset = self.aggregate_condition_rows(test_rows)

        logger.debug(
            "Created datasets with %d train, %d validation, and %d test "
            "batches.",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

    def setup(self, stage=None):
        """
        Store each condition's mini-batches and create all dataset splits.

        Move the problem's discretized domains into their conditions, then
        call each condition's ``store_mini_batches`` method with the
        ``mini_batch_rule`` of that condition, using the global
        ``batch_size`` when the rule does not set one.

        All accepted stages perform the same setup. Calling this method
        again rebuilds the mini-batches and the dataset splits.

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
            condition.store_mini_batches(
                **self._condition_mini_batch_rule(name)
            )

        self.enumerate_condition()
        self.create_datasets()

    def _dataloader(self, dataset):
        """
        Create a dataloader over the batches of a dataset.

        Each iteration yields one batch of identifiers with shape
        ``[1, n_conditions, 2]``. The dataloader ``batch_size`` is always 1
        because each dataset element is already a whole batch. Batches are
        visited in dataset order using default collation.

        :param torch.Tensor dataset: The identifier dataset with shape
            ``[n_batches, n_conditions, 2]``.
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
        Return the dataloader over the training batches.

        :return: The training dataloader, or an empty list if the training
            dataset is empty.
        :rtype: DataLoader | list
        :raises AttributeError: If the training dataset has not been
            created.
        """
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        """
        Return the dataloader over the validation batches.

        :return: The validation dataloader, or an empty list if the
            validation dataset is empty.
        :rtype: DataLoader | list
        :raises AttributeError: If the validation dataset has not been
            created.
        """
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        """
        Return the dataloader over the test batches.

        :return: The test dataloader, or an empty list if the test dataset
            is empty.
        :rtype: DataLoader | list
        :raises AttributeError: If the test dataset has not been created.
        """
        return self._dataloader(self.test_dataset)

    def transfer_batch_to_device(self, batch, device, _):
        """
        Materialize the mini-batches referenced by a batch of identifiers.

        Resolve each condition identifier to its name and call the
        condition's ``materialize`` method with the mini-batch identifier
        and requested device.

        :param torch.Tensor batch: One batch of identifiers with shape
            ``[1, n_conditions, 2]``, each row containing a condition
            identifier and a mini-batch identifier.
        :param device: The device on which the mini-batches are
            materialized. If ``None``, device handling is delegated to
            each condition without a target device.
        :type device: str | torch.device | None
        :param int _: The Lightning dataloader index. Unused.
        :return: The batch: pairs containing the condition name and its
            materialized mini-batch. The mini-batch type depends on the
            condition.
        :rtype: list[tuple[str, Any]]
        :raises AttributeError: If the condition identifier mapping has
            not been initialized.
        :raises KeyError: If a condition identifier is unknown or its
            condition is absent from the problem.
        :raises IndexError: If a mini-batch identifier is out of range for
            its condition.
        """
        result = []
        assert (
            batch.ndim == 3 and batch.shape[0] == 1
        ), "Batch must have shape [1, n_conditions, 2]"

        for condition_id, mini_batch_id in batch.squeeze(0):
            name = self.idx_to_condition_name[int(condition_id)]
            data = self.problem.conditions[name].materialize(
                int(mini_batch_id), device=device
            )
            result.append((name, data))
        return result
