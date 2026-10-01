"""Module for creating dataloaders for multiple conditions."""

import torch
from torch.utils.data.distributed import DistributedSampler


class Batcher:
    """
    Utility class for creating dataloaders associated with multiple conditions.

    Each condition dataset is a tensor of sample ids. The batcher computes the
    batch size for each condition according to the selected batching strategy
    and builds one dataloader per condition over its ids.
    """

    def __init__(
        self,
        batching_mode="common_batch_size",
        batch_size=None,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        dataloader_cls=None,
    ):
        """
        Initialization of the :class:`Batcher` class.

        :param str batching_mode: The strategy used to aggregate batches across
            dataloaders. Available options are ``"common_batch_size"`` for
            uniform batch sizes across conditions, and ``"proportional"`` for
            batch sizes proportional to dataset sizes.
        :param int batch_size: The number of samples per batch. If ``None``,
            each entire condition dataset is processed as a single batch.
        :param bool shuffle: Whether samples should be shuffled during loading.
        :param int num_workers: The number of worker processes used for data
            loading.
        :param bool pin_memory: Whether dataloaders should pin memory.
        :param type dataloader_cls: The dataloader class to use, defaulting
            to :class:`torch.utils.data.DataLoader`. It can either be a single
            class applied to all conditions, or a mapping between condition
            names and dataloader classes. Custom classes must accept the
            dataset, ``batch_size``, ``sampler``, ``num_workers``, and
            ``pin_memory`` arguments and yield batches of sample ids.
        """
        # Initialize attributes
        self.batching_mode = batching_mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader_cls = dataloader_cls

    def _resolve_dataloader_cls(self, name):
        """
        Resolve the dataloader class associated with a condition.

        :param str name: The condition name.
        :return: The dataloader class associated with the condition, defaulting
            to :class:`torch.utils.data.DataLoader`.
        :rtype: type
        """
        dataloader_cls = self.dataloader_cls
        if isinstance(dataloader_cls, dict):
            dataloader_cls = dataloader_cls.get(name)

        return dataloader_cls or torch.utils.data.DataLoader

    def __call__(self, subsets):
        """
        Create dataloaders for all provided subsets.

        Batch sizes are computed according to the selected batching mode, and a
        dedicated dataloader is created for each condition.

        :param dict[str, torch.Tensor] subsets: The mapping between condition
            names and tensors of sample ids.
        :return: The mapping between condition names and the corresponding
            dataloaders.
        :rtype: dict[str, DataLoader]
        """
        # Compute batch sizes per condition based on batching_mode
        batch_sizes = self._compute_batch_sizes(subsets)

        # Iterate through subsets and create dataloaders
        return {
            name: self._resolve_dataloader_cls(name)(
                ids,
                batch_size=batch_sizes[name],
                shuffle=False,
                sampler=self._define_sampler(ids, self.shuffle),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            for name, ids in subsets.items()
        }

    def _define_sampler(self, dataset, shuffle):
        """
        Define the sampling strategy for a dataset.

        Distributed training uses :class:`DistributedSampler`, while
        non-distributed execution uses either :class:`RandomSampler` or
        :class:`SequentialSampler` depending on ``shuffle``.

        :param torch.Tensor dataset: The dataset associated with the sampler.
        :param bool shuffle: Whether samples should be shuffled during loading.
        :return: The configured sampler instance.
        :rtype: Sampler
        """
        # Distributed training case
        if torch.distributed.is_initialized():
            return DistributedSampler(dataset, shuffle=shuffle)

        # Non-distributed training case - shuffle True
        if shuffle:
            return torch.utils.data.RandomSampler(dataset)

        # Non-distributed training case - shuffle False
        return torch.utils.data.SequentialSampler(dataset)

    def _compute_batch_sizes(self, subsets):
        """
        Compute batch sizes for each subset according to the selected batching
        mode.

        :param dict[str, torch.Tensor] subsets: The mapping between condition
            names and tensors of sample ids.
        :return: The mapping between condition names and computed batch sizes.
        :rtype: dict[str, int]
        """
        # If no batch size is given, each subset is processed as a single batch
        if self.batch_size is None:
            return {name: len(ids) for name, ids in subsets.items()}

        # Common batch size mode
        if self.batching_mode == "common_batch_size":
            return {
                name: min(self.batch_size, len(ids))
                for name, ids in subsets.items()
            }

        # Proportional batch size mode
        if self.batching_mode == "proportional":
            return self._compute_proportional_batch_sizes(subsets)

        raise ValueError(
            f"Unsupported batching mode '{self.batching_mode}'. Available "
            "options are 'common_batch_size' and 'proportional'."
        )

    def _compute_proportional_batch_sizes(self, subsets):
        """
        Compute batch sizes proportionally to subset sizes.

        Each subset receives a fraction of the total batch size proportional to
        its number of samples, while ensuring that each subset contributes at
        least one sample.

        :param dict[str, torch.Tensor] subsets: The mapping between condition
            names and tensors of sample ids.
        :return: The mapping between condition names and proportional batch
            sizes.
        :rtype: dict[str, int]
        """
        # Compute the sizes of each subset
        subset_sizes = {name: len(ids) for name, ids in subsets.items()}

        # Determine the total number of elements across all subsets
        total_size = sum(subset_sizes.values())

        # Compute the batch sizes
        batch_sizes = {
            name: max(1, int(self.batch_size * (size / total_size)))
            for name, size in subset_sizes.items()
        }

        # Compute assigned batch size and difference with the total batch size
        assigned_batch_size = sum(batch_sizes.values())
        difference = self.batch_size - assigned_batch_size

        # If difference > 0, distribute to subsets with more than 1 sample
        if difference > 0:

            # Sort subsets by size in descending order
            sorted_subsets = sorted(
                subset_sizes,
                key=lambda name: subset_sizes[name],
                reverse=True,
            )

            # Distribute to subsets with more than 1 sample
            for name in sorted_subsets:

                # Stop distribution when the difference is fully allocated
                if difference == 0:
                    break

                # Distribute to subsets with more than 1 sample
                if subset_sizes[name] > 1:
                    batch_sizes[name] += 1
                    difference -= 1

        # If difference < 0, reduce from subsets with more than 1 sample
        if difference < 0:

            # Sort batches by size in descending order
            sorted_batches = sorted(
                batch_sizes, key=lambda name: batch_sizes[name], reverse=True
            )

            # Reduce from subsets with more than 1 sample
            for name in sorted_batches:

                # Stop reduction when the difference is fully allocated
                if difference == 0:
                    break

                # Reduce from subsets with more than 1 sample
                if batch_sizes[name] > 1:
                    batch_sizes[name] -= 1
                    difference += 1

        return batch_sizes
