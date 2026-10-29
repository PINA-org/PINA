"""Module for the Tensor Data Condition class."""

from types import SimpleNamespace
import torch

from pina._src.condition.base_condition import BaseCondition
from pina._src.core.label_tensor import LabelTensor


def _normalize_ids(idx):
    """
    Normalize an index to a list of indices (or a slice).

    :param idx: The index to normalize.
    :type idx: int | slice | list[int] | torch.Tensor
    :return: The normalized indices.
    :rtype: list[int] | slice
    """
    if isinstance(idx, slice):
        return idx
    if isinstance(idx, torch.Tensor):
        return idx.tolist()
    if isinstance(idx, int):
        return [idx]
    return list(idx)


def _move_to_device(batch, device):
    """
    Move all compatible values of a batch to the given device.

    :param dict batch: The batch whose values should be moved.
    :param device: The target device. If ``None``, the batch is returned
        unchanged.
    :return: The batch with values moved to ``device``.
    :rtype: dict
    """
    if device is None:
        return batch

    return {
        key: (value.to(device) if hasattr(value, "to") else value)
        for key, value in batch.items()
    }


class TensorCondition(BaseCondition):
    """
    Abstract condition for tensor-based data.

    It handles storage, slicing, and batching of :class:`torch.Tensor` and
    :class:`~pina.label_tensor.LabelTensor` data. Conditions built on
    tensor data should inherit from this class.
    """

    _avail_data_cls = (torch.Tensor, LabelTensor)

    def store_data(self, **kwargs):
        """
        Store the keyword arguments as named data attributes.

        :param dict kwargs: The keyword arguments containing the data to be
            stored.
        :return: The stored data.
        :rtype: SimpleNamespace
        """
        return SimpleNamespace(**kwargs)

    def __len__(self):
        """
        Return the number of samples in the condition.

        :return: The number of samples.
        :rtype: int
        """
        return vars(self.data)[self._data_keys()[0]].shape[0]

    def _data_keys(self):
        """
        Return the names of the stored data fields.

        :return: The stored data field names.
        :rtype: list[str]
        """
        return list(vars(self.data).keys())

    def _select(self, idx):
        """
        Select the data points at the given indices.

        :param idx: The indices of the data points to select.
        :type idx: int | slice | list[int] | torch.Tensor
        :return: A mapping between data field names and selected values.
        :rtype: dict
        """
        idx = _normalize_ids(idx)
        return {
            key: value[idx] if isinstance(value, torch.Tensor) else value
            for key, value in vars(self.data).items()
        }

    def _collate(self, raw):
        """
        Collate selected data into a batch.

        Tensor slices are already stacked, so the selected data is returned
        unchanged.

        :param dict raw: The selected raw data.
        :return: The batch.
        :rtype: dict
        """
        return raw

    def materialize(self, ids, device=None, batch_fn=None):
        """
        Build a batch from the given data point ids.

        :param ids: The ids of the data points to batch.
        :type ids: int | list[int] | torch.Tensor
        :param device: The target device for the batch. Default is ``None``.
        :param batch_fn: Optional callable overriding the batch construction.
            It receives the selected raw data and the ids and returns the
            batch. Default is ``None``.
        :return: The materialized batch.
        :rtype: dict
        """
        raw = self._select(ids)
        if batch_fn is None:
            batch = self._collate(raw)
        else:
            batch = batch_fn(raw, ids)

        return _move_to_device(batch, device)

    def __getitem__(self, idx):
        """
        Materialize the data points at the specified index (or indices).

        :param idx: The index of the data point to retrieve.
        :type idx: int | slice | list[int] | torch.Tensor
        :return: The materialized batch.
        :rtype: dict
        """
        return self.materialize(idx)
