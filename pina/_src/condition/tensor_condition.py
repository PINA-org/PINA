"""Module for the Tensor Data Condition class."""

import torch

from pina._src.condition.base_condition import BaseCondition
from pina._src.core.label_tensor import LabelTensor


class TensorCondition(BaseCondition):
    """
    Base condition for tensor-based data.

    The condition splits its tensor fields into batches during
    :meth:`store_data`. ``self.data`` is a list of dictionaries, one per batch,
    and :meth:`materialize` returns the batch selected by a single id.
    """

    _avail_data_cls = (torch.Tensor, LabelTensor)

    def store_data(self, batch_size=None):
        """
        Split the tensor fields into batches stored in ``self.data``.

        Fields declared in ``__fields__`` that are tensors with the same
        number of samples as ``_input`` are sliced along the first dimension.
        Other fields, such as equations or configuration values, are stored
        unchanged in every batch.

        :param int batch_size: Number of samples per batch. If ``None``, all
            samples are stored as a single batch.
        :return: The stored batches.
        :rtype: list[dict]
        """
        n_samples = self._input.shape[0]
        if n_samples == 0:
            self.data = []
            return self.data

        batch_size = batch_size or n_samples
        self.data = []
        for start in range(0, n_samples, batch_size):
            batch = {}
            for field in self.__fields__:
                value = getattr(self, f"_{field}", None)
                if (
                    isinstance(value, self._avail_data_cls)
                    and value.shape[0] == n_samples
                ):
                    value = value[start : start + batch_size]
                batch[field] = value
            self.data.append(batch)
        return self.data

    def __len__(self):
        """
        Return the number of data points in the condition.

        :return: The number of data points.
        :rtype: int
        """
        return len(self.data)

    def materialize(self, id, device=None):
        """
        Return the batch with the given id, optionally moved to ``device``.

        :param int id: The id of the batch to retrieve.
        :param device: The target device for the batch. Default is ``None``.
        :return: The materialized batch.
        :rtype: dict
        """
        batch = self.data[int(id)]
        if device is None:
            return batch
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def __getitem__(self, id):
        """
        Materialize the data points at the specified index.

        :param int idx: The index of the data point to retrieve.
        :return: The materialized batch.
        :rtype: dict
        """
        return self.materialize(id)
