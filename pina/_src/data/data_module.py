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
    """

    def __init__(
        self,
        problem,
        problem_setting=None,
        train_size=0.8,
        val_size=0.2,
        test_size=0.0,
        batch_size=None,
        batching_mode="common_batch_size",
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        dataloader_cls=None,
        collate_fn=None,
    ):
        super().__init__()
        self.problem = problem
        self.batch_size = batch_size
        self.batching_mode = batching_mode
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader_cls = dataloader_cls
        self.collate_fn = collate_fn
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        if problem_setting is None:
            problem_setting = {
                "time": {"n_windows": 10, "unroll_length": 5}
            }
        self.problem_setting = problem_setting

    def enumerate_condition(self):
        """Map condition names to ids and back."""
        self.condition_name_to_idx = {
            name: idx for idx, name in enumerate(self.problem.conditions)
        }
        self.idx_to_condition_name = {
            idx: name for name, idx in self.condition_name_to_idx.items()
        }
        return self.condition_name_to_idx

    def create_datasets(self):
        """Split every condition batch id into train/validation/test rows."""
        train_rows, val_rows, test_rows = [], [], []

        for condition_name, condition in self.problem.conditions.items():
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
                train_rows.append([condition_id, batch_id])
            for batch_id in indices[train_end:val_end]:
                val_rows.append([condition_id, batch_id])
            for batch_id in indices[val_end:]:
                test_rows.append([condition_id, batch_id])

        self.train_dataset = torch.tensor(
            train_rows, dtype=torch.long
        ).reshape(-1, 2)
        self.val_dataset = torch.tensor(
            val_rows, dtype=torch.long
        ).reshape(-1, 2)
        self.test_dataset = torch.tensor(
            test_rows, dtype=torch.long
        ).reshape(-1, 2)

        logger.debug(
            "Created datasets with %d train, %d validation, and %d test rows.",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

    def setup(self, stage=None):
        """Store condition batches and create the dataset splits."""
        if stage not in ("fit", "test", None):
            raise ValueError(
                f"Invalid stage. Got {stage}, expected 'fit' or 'test'."
            )

        for name, condition in self.problem.conditions.items():
            settings = {
                "batch_size": self.batch_size,
                **self.problem_setting.get(name, {}),
            }
            condition.store_data(**settings)

        self.enumerate_condition()
        self.create_datasets()

    def _dataloader(self, dataset):
        if len(dataset) == 0:
            return []
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)

    def transfer_batch_to_device(self, batch, device, _):
        """Materialize the condition batch referenced by each dataset row."""
        result = []
        for condition_id, batch_id in batch:
            name = self.idx_to_condition_name[int(condition_id)]
            data = self.problem.conditions[name].materialize(
                int(batch_id), device=device
            )
            result.append((name, data))
        return result
