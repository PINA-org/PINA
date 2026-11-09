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

        Tensor slices share storage with the original fields. Calling this
        method replaces any existing batches.

        :param batch_size: The number of samples per batch. ``None`` or ``0``
            stores all samples in one batch. A negative value produces no
            batches. Default is ``None``.
        :type batch_size: int | None
        :raises TypeError: If a truthy ``batch_size`` is not an integer accepted
            by :func:`range`.
        :return: The stored batches, also assigned to ``self.data``. Empty
            input produces an empty list.
        :rtype: list[dict[str, Any]]
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
        Return the number of stored batches in the condition.

        :raises AttributeError: If :meth:`store_data` has not been called.
        :return: The number of stored batches.
        :rtype: int
        """
        return len(self.data)

    def materialize(self, id, device=None):
        """
        Return the batch with the given id, optionally moved to ``device``.

        Negative indices follow Python list indexing conventions. Without a
        device argument, return the stored dictionary itself. With a device,
        return a new dictionary, moving only tensor values; equations and other
        non-tensor fields are not moved.

        :param id: The index of the batch to retrieve. Tensor indices must
            contain a single integer value.
        :type id: int | torch.Tensor
        :param device: The target device for tensor fields. If ``None``, leave
            the batch on its existing device. Default is ``None``.
        :type device: str | torch.device | int | None
        :raises AttributeError: If :meth:`store_data` has not been called.
        :raises IndexError: If the batch index is out of range.
        :return: The selected batch dictionary, with fields defined by the
            concrete condition's ``__fields__``.
        :rtype: dict[str, Any]
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
        Retrieve the stored batch at the specified index.

        Negative indices follow Python list indexing conventions.

        :param id: The index of the batch to retrieve. Tensor indices must
            contain a single integer value.
        :type id: int | torch.Tensor
        :raises AttributeError: If :meth:`store_data` has not been called.
        :raises IndexError: If the batch index is out of range.
        :return: The stored batch dictionary without changing its device.
        :rtype: dict[str, Any]
        """
        return self.materialize(id)
