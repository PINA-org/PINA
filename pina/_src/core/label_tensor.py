"""Tensors with dimension names and component-label metadata."""

from copy import copy, deepcopy
import torch
from torch import Tensor


class LabelTensor(torch.Tensor):
    """
    Extend :class:`torch.Tensor` with optional labels on selected dimensions.

    Metadata maps dimension indices to records containing ``"name"`` and
    ``"dof"`` entries. A simple label list describes the last dimension.
    Component labels implement :class:`collections.abc.Hashable`, abbreviated
    as ``Hashable`` in the type fields below.
    Metadata-aware methods support label extraction, indexing, and selected
    tensor transformations.

    .. warning::

        Arbitrary PyTorch operations may return this subclass without its label
        metadata. Only the behavior of the metadata-aware methods documented
        here is guaranteed; generic reshaping does not reconstruct labels.
    """

    @staticmethod
    def __new__(cls, x, labels, *args, **kwargs):
        """
        Allocate a labeled tensor or reuse an existing labeled tensor.

        Metadata is assigned by :meth:`__init__`, not by this allocator.

        :param torch.Tensor x: The tensor to wrap. If it is already a
            :class:`LabelTensor`, return it unchanged without allocating.
        :param labels: The metadata passed to initialization; unused here.
        :type labels: str | list[Hashable] | builtins.range |
            dict[int, dict[str, Any]] | None
        :param tuple[Any, ...] args: Additional arguments forwarded to the
            PyTorch allocator only when ``x`` is not already labeled. The normal
            two-argument constructor does not support extra allocation options.
        :param dict[str, Any] kwargs: Additional PyTorch allocator options,
            subject to the same restriction as ``args``. Ignored when reusing
            an existing labeled tensor.
        :return: The allocated tensor, sharing the supplied tensor's storage, or
            ``x`` itself when it is already labeled.
        :rtype: LabelTensor
        """

        if isinstance(x, LabelTensor):
            return x
        return super().__new__(cls, x, *args, **kwargs)

    @property
    def tensor(self):
        """
        Returns the tensor part of the :class:`~pina.label_tensor.LabelTensor`
        object.

        :return: A plain tensor view sharing data storage and the autograd
            connection with this instance, without label-aware behavior.
        :rtype: torch.Tensor
        """

        return self.as_subclass(Tensor)

    def __init__(self, x, labels):
        """
        Initialization of the :class:`LabelTensor` class.

        :param torch.Tensor x: The tensor supplied to allocation; unused during
            metadata initialization. Passing an existing labeled tensor reuses
            that object and applies the supplied metadata to it.
        :param labels: The component labels. A string denotes one label; a list
            or range labels the last dimension. A dictionary maps dimension
            indices to records containing ``"name"`` and ``"dof"``. Missing
            names default to the dimension index and missing degrees of freedom
            default to a positional range. ``None`` clears stored metadata.
            Label counts must match their dimension sizes and labels within a
            dimension must be unique and hashable. Dictionary records and
            label lists are stored by reference; missing record fields are
            inserted into the caller's dictionary.
        :type labels: str | list[Hashable] | builtins.range |
            dict[int, dict[str, Any]] | None
        :raises ValueError: If the label format is unsupported, a dimension
            record is not a dictionary, labels repeat, or their count differs
            from the dimension size.
        :raises TypeError: If a component label is unhashable or a dimension
            index cannot index the tensor shape.
        :raises IndexError: If a dimension index is outside the tensor shape.

        :Example:

            >>> import torch
            >>> from pina import LabelTensor
            >>> tensor = LabelTensor(torch.ones(2, 3), ["a", "b", "c"])
            >>> named = LabelTensor(
            ...     torch.ones(2, 3),
            ...     {1: {"name": "space", "dof": ["a", "b", "c"]}},
            ... )
            >>> named.extract({"space": ["a", "c"]}).shape
            torch.Size([2, 2])
        """
        super().__init__()
        if labels is not None:
            self.labels = labels
        else:
            self._labels = {}

    @property
    def full_labels(self):
        """
        Returns the full labels of the tensor, even for the dimensions that are
        not labelled.

        :return: A new outer dictionary covering every nonnegative dimension
            index. Existing records are shared with stored metadata; unlabeled
            dimensions receive a fresh record with a positional ``range`` and
            the dimension index as its name.
        :rtype: dict[int, dict[str, Any]]
        """
        to_return_dict = {}
        shape_tensor = self.shape
        for i, value in enumerate(shape_tensor):
            if i in self._labels:
                to_return_dict[i] = self._labels[i]
            else:
                to_return_dict[i] = {"dof": range(value), "name": i}
        return to_return_dict

    @property
    def stored_labels(self):
        """
        Returns the labels stored inside the instance.

        :return: The mutable internal mapping from dimension indices to metadata
            records. Both the mapping and its records are returned by reference.
        :rtype: dict[int, dict[str, Any]]
        """
        return self._labels

    @property
    def labels(self):
        """
        Returns the labels of the last dimension of the instance.

        :return: The stored ``"dof"`` entry for the last dimension, or ``None``
            if that dimension has no record. Typically a list of labels or a
            range; dictionary-based initialization can also store a scalar
            hashable label. The value is returned by reference.
        :rtype: list[Hashable] | builtins.range | Hashable | None
        """
        if self.ndim - 1 in self._labels:
            return self._labels[self.ndim - 1]["dof"]
        return None

    @labels.setter
    def labels(self, labels):
        """
        Update stored dimension labels after validating their counts.

        Dictionary updates merge into existing metadata; other accepted formats
        replace only the last dimension's record. Input records and lists are
        retained by reference. Missing record fields are filled in place.

        :param labels: A string, list, or range for the last dimension, or a
            dictionary of dimension records containing ``"name"`` and ``"dof"``.
            Unlike initialization, this setter does not accept ``None``.
        :type labels: str | list[Hashable] | builtins.range |
            dict[int, dict[str, Any]]
        :raises ValueError: If the format is unsupported, records are not
            dictionaries, or component labels repeat or have the wrong count.
        :raises TypeError: If labels are unhashable or a dimension index cannot
            index the tensor shape.
        :raises IndexError: If a dimension index is outside the tensor shape.
        """

        if not hasattr(self, "_labels"):
            self._labels = {}
        if isinstance(labels, dict):
            self._init_labels_from_dict(labels)
        elif isinstance(labels, (list, range)):
            self._init_labels_from_list(labels)
        elif isinstance(labels, str):
            labels = [labels]
            self._init_labels_from_list(labels)
        else:
            raise ValueError("labels must be list, dict or string.")

    def _init_labels_from_dict(self, labels):
        """
        Store the internal label representation according to the values
        passed as input.

        Missing ``"name"`` and ``"dof"`` fields are inserted into the supplied
        records. Records are then stored by reference without clearing unrelated
        dimensions. Scalar degrees of freedom are wrapped only for validation,
        not in the stored record. Dimension indices are not normalized.

        :param dict[int, dict[str, Any]] labels: The dimension records to merge.
        :raises ValueError: If a record is not a dictionary, component labels
            repeat, or their count differs from the dimension size.
        :raises TypeError: If a component label is unhashable or a dimension
            index cannot index the tensor shape.
        :raises IndexError: If a dimension index is outside the tensor shape.
        """

        tensor_shape = self.shape

        def validate_dof(dof_list, dim_size):
            """
            Validate component-label uniqueness and count.

            :param dof_list: The labels to validate.
            :type dof_list: list[Hashable] | builtins.range
            :param int dim_size: The required number of component labels.
            :raises ValueError: If labels repeat or their count is incorrect.
            :raises TypeError: If a label is unhashable.
            """
            if len(dof_list) != len(set(dof_list)):
                raise ValueError("dof must be unique")
            if len(dof_list) != dim_size:
                raise ValueError(
                    f"Number of dof ({len(dof_list)}) does not match "
                    f"tensor shape ({dim_size})"
                )

        for dim, label in labels.items():
            if isinstance(label, dict):
                if "name" not in label:
                    label["name"] = dim
                if "dof" not in label:
                    label["dof"] = range(tensor_shape[dim])
                if "dof" in label and "name" in label:
                    dof = label["dof"]
                    dof_list = dof if isinstance(dof, (list, range)) else [dof]
                    if not isinstance(dof_list, (list, range)):
                        raise ValueError(
                            f"'dof' should be a list or range, not"
                            f" {type(dof_list)}"
                        )
                    validate_dof(dof_list, tensor_shape[dim])
                else:
                    raise ValueError(
                        "Labels dictionary must contain either "
                        " both 'name' and 'dof' keys"
                    )
            else:
                raise ValueError(
                    f"Invalid label format for {dim}: Expected "
                    f"list or dictionary, got {type(label)}"
                )

            # Assign validated label data to internal labels
            self._labels[dim] = label

    def _init_labels_from_list(self, labels):
        """
        Given a list of dof, this method update the internal label
        representation by assigning the dof to the last dimension.

        The supplied sequence is stored by reference in the last dimension's
        record; other dimension records are unchanged.

        :param labels: The component labels for the last dimension.
        :type labels: list[Hashable] | builtins.range
        :raises ValueError: If labels repeat or their count differs from the
            last dimension's size.
        :raises TypeError: If a component label is unhashable.
        """

        # Create a dict with labels
        last_dim_labels = {
            self.ndim - 1: {"dof": labels, "name": self.ndim - 1}
        }
        self._init_labels_from_dict(last_dim_labels)

    def extract(self, labels_to_extract):
        """
        Extract components by label while preserving the tensor's rank.

        Scalar, list, and tuple requests address the last physical dimension.
        Dictionary keys instead address stored dimension names. Selections are
        applied dimension by dimension, preserving the requested label order.
        Single-label selection uses a slice and may share data storage; multiple
        labels use advanced indexing. Untouched metadata records are shared.

        :param labels_to_extract: The requested component labels. A dictionary
            maps dimension names to scalar string/integer labels, label lists,
            or ranges. Each selection must be nonempty and unique. Sequences
            address the last physical dimension, which must have stored labels.
        :type labels_to_extract: str | int | list[Hashable] |
            tuple[Hashable, ...] |
            dict[Hashable, str | int | list[Hashable] | builtins.range]
        :raises ValueError: If the request format is unsupported, a dimension
            name or label is unknown, or selected labels repeat or do not match
            the extracted dimension size.
        :raises KeyError: If a last-dimension request has no corresponding
            stored metadata.
        :raises IndexError: If a dimension selection is empty.
        :raises AttributeError: If stored degrees of freedom do not provide an
            ``index`` method, as with a scalar integer label record.
        :return: The selected tensor with singleton dimensions retained and
            updated metadata for the extracted dimensions.
        :rtype: LabelTensor

        :Example:

            >>> import torch
            >>> from pina import LabelTensor
            >>> labels = {1: {"dof": ["a", "b", "c"], "name": "space"}}
            >>> tensor = LabelTensor(torch.ones(2, 3), labels)
            >>> tensor.extract("a").shape
            torch.Size([2, 1])
            >>> tensor.extract({"space": ["c", "a"]}).labels
            ['c', 'a']
        """

        def get_label_indices(dim_labels, labels_te):
            """Get the indices of the specified labels in the dimension labels.

            :param dim_labels: The stored component labels with an ``index``
                method.
            :type dim_labels: list[Hashable] | tuple[Hashable, ...] |
                builtins.range | str
            :param labels_te: The label(s) to find indices for.
            :type labels_te: str | int | list[Hashable] | builtins.range
            :raises ValueError: If a requested label is absent.
            :raises IndexError: If the requested sequence is empty.
            :return: The indices or slice for the label(s).
            :rtype: list[int] | slice
            """
            if isinstance(labels_te, (int, str)):
                labels_te = [labels_te]
            if len(labels_te) > 1:
                return [dim_labels.index(label) for label in labels_te]
            start = dim_labels.index(labels_te[0])
            return slice(start, start + 1)

        # Ensure labels_to_extract is a list or dict
        if isinstance(labels_to_extract, (str, int)):
            labels_to_extract = [labels_to_extract]

        labels = dict(self._labels)

        # Get the dimension names and the respective dimension index
        dim_names = {label["name"]: dim for dim, label in labels.items()}
        tensor = self.as_subclass(Tensor)
        ndim = tensor.ndim

        # Convert list/tuple to a dict for the last dimension if applicable
        if isinstance(labels_to_extract, (list, tuple)):
            last_dim = ndim - 1
            dim_name = labels[last_dim]["name"]
            labels_to_extract = {dim_name: list(labels_to_extract)}

        # Validate the labels_to_extract type
        if not isinstance(labels_to_extract, dict):
            raise ValueError(
                "labels_to_extract must be a string, list, or dictionary."
            )

        # Perform the extraction for each specified dimension
        extracted_dims = set()
        for dim_name, labels_te in labels_to_extract.items():
            if dim_name not in dim_names:
                raise ValueError(
                    f"Cannot extract labels for dimension '{dim_name}' as it is"
                    f" not present in the original labels."
                )

            idx_dim = dim_names[dim_name]
            dim_labels = labels[idx_dim]["dof"]
            indices = get_label_indices(dim_labels, labels_te)

            extractor = [slice(None)] * ndim
            extractor[idx_dim] = indices
            tensor = tensor[tuple(extractor)]

            labels[idx_dim] = {"dof": labels_te, "name": dim_name}
            extracted_dims.add(idx_dim)

        # Only the extracted dimensions changed, so validate just those (in
        # the same order the constructor would) instead of re-checking all
        shape = tensor.shape
        for dim, label in labels.items():
            if dim not in extracted_dims:
                continue
            dof = label["dof"]
            dof_list = dof if isinstance(dof, (list, range)) else [dof]
            if len(dof_list) != len(set(dof_list)):
                raise ValueError("dof must be unique")
            if len(dof_list) != shape[dim]:
                raise ValueError(
                    f"Number of dof ({len(dof_list)}) does not match "
                    f"tensor shape ({shape[dim]})"
                )

        result = LabelTensor.__new__(LabelTensor, tensor, None)
        result._labels = labels
        return result

    def __str__(self):
        """
        The string representation of the
        :class:`~pina.label_tensor.LabelTensor`.

        :return: String representation of the
            :class:`~pina.label_tensor.LabelTensor` instance.
        :rtype: str
        """

        s = ""
        for key, value in self._labels.items():
            s += f"{key}: {value}\n"
        s += "\n"
        s += self.tensor.__str__()
        return s

    @staticmethod
    def cat(tensors, dim=0):
        """
        Concatenate tensors along an existing dimension using :func:`torch.cat`.

        For multiple inputs, check label consistency on the first tensor's
        stored dimensions other than ``dim``. Degrees of freedom along a labeled
        concatenation dimension are joined without uniqueness validation.

        :param tensors: The tensors to concatenate. Shapes must agree outside
            the concatenation dimension. Component labels on that dimension
            must be stored as lists if it is labeled.
        :type tensors: list[LabelTensor] | tuple[LabelTensor, ...]
        :param int dim: The concatenation dimension. Use a nonnegative index:
            negative indices are passed to PyTorch but are not normalized for
            label handling. Default is ``0``.
        :raises RuntimeError: If tensor shapes are incompatible or checked
            metadata records differ outside the concatenation dimension.
        :raises KeyError: If an input lacks a stored dimension checked from
            the first tensor, or lacks labeled concatenation metadata.
        :raises TypeError: If labeled concatenation tries to join non-list
            degrees of freedom, such as ranges.
        :raises IndexError: If ``dim`` is out of range for multiple inputs.
        :return: The concatenated tensor. An empty input returns an empty list;
            a single input is returned unchanged, without checking ``dim``.
            For multiple inputs concatenated on an unlabeled dimension, the
            first tensor's entire metadata mapping is reused. Otherwise records
            are shallow-copied and the concatenated label list is new.
        :rtype: LabelTensor | list[LabelTensor]
        """

        if not tensors:
            return []  # Handle empty list
        if len(tensors) == 1:
            return tensors[0]  # Return single tensor as-is

            # Perform concatenation
        cat_tensor = torch.cat(tensors, dim=dim)
        tensors_labels = [tensor.stored_labels for tensor in tensors]

        # Check label consistency across tensors, excluding the
        # concatenation dimension
        for key in tensors_labels[0]:
            if key != dim:
                if any(
                    tensors_labels[i][key] != tensors_labels[0][key]
                    for i in range(len(tensors_labels))
                ):
                    raise RuntimeError(
                        f"Tensors must have the same labels along all "
                        f"dimensions except {dim}."
                    )

        # Copy and update the 'dof' for the concatenation dimension
        cat_labels = {k: copy(v) for k, v in tensors_labels[0].items()}

        # Update labels if the concatenation dimension has labels
        if dim in tensors[0].stored_labels:
            if dim in cat_labels:
                cat_dofs = [label[dim]["dof"] for label in tensors_labels]
                cat_labels[dim]["dof"] = sum(cat_dofs, [])
        else:
            cat_labels = tensors[0].stored_labels

        # Assign updated labels to the concatenated tensor
        cat_tensor._labels = cat_labels
        return cat_tensor

    @staticmethod
    def stack(tensors, dim=0):
        """
        Stack multiple tensors along a new dimension using :func:`torch.stack`.

        Shift stored dimension indices at or after the insertion point, sharing
        their metadata records with the first input. The new axis is unlabeled.
        Only dimensions stored on the first tensor are checked for matching
        records on the remaining tensors.

        :param tensors: The tensors to stack, all with the same shape and
            matching metadata on checked dimensions.
        :type tensors: list[LabelTensor] | tuple[LabelTensor, ...]
        :param int dim: The insertion dimension. Negative indices are normalized
            relative to the output rank. Default is ``0``.
        :raises IndexError: If ``dim`` is out of range for multiple inputs.
        :raises RuntimeError: If shapes or checked label records differ.
        :return: The stacked tensor for multiple inputs, an empty list for no
            inputs, or the original tensor for a single input. The single-input
            case does not add an axis or check ``dim``.
        :rtype: LabelTensor | list[LabelTensor]
        """
        if not tensors:
            return []
        if len(tensors) == 1:
            return tensors[0]

        # In PyTorch, stacking adds a new dimension, so valid dims are - (ndim + 1) <= dim <= ndim
        ndim = tensors[0].ndim
        if dim < 0:
            dim = ndim + 1 + dim
        if dim < 0 or dim > ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-(ndim + 1)}, {ndim}], but got {dim})"
            )

        # Check that all tensors have consistent labels
        tensors_labels = [tensor.stored_labels for tensor in tensors]
        for key in tensors_labels[0]:
            if any(
                tensors_labels[i].get(key) != tensors_labels[0][key]
                for i in range(len(tensors_labels))
            ):
                raise RuntimeError("All tensors must have the same labels.")

        # Perform stacking in torch
        new_tensor = torch.stack(tensors, dim=dim)

        # Shift label dimension indices at or after `dim` by +1
        labels = {}
        for key, value in tensors[0]._labels.items():
            if key >= dim:
                labels[key + 1] = value
            else:
                labels[key] = value

        new_tensor._labels = labels
        return new_tensor

    def requires_grad_(self, mode=True):
        """
        Override the :meth:`~torch.Tensor.requires_grad_` method to handle
        the labels in the new tensor.
        For more details, see :meth:`~torch.Tensor.requires_grad_`.

        :param bool mode: A boolean value indicating whether the tensor should
            track gradients. Default is ``True``.
        :raises RuntimeError: If PyTorch rejects the change, for example when
            enabling gradients on an integer tensor or changing a non-leaf's
            gradient flag.
        :return: The :class:`~pina.label_tensor.LabelTensor` itself with the
            updated ``requires_grad`` state and retained labels.
        :rtype: LabelTensor
        """

        lt = super().requires_grad_(mode)
        lt._labels = self._labels
        return lt

    @property
    def dtype(self):
        """
        Return the tensor's :class:`torch.dtype`.

        :return: The data type of the tensor.
        :rtype: torch.dtype
        """

        return super().dtype

    def to(self, *args, **kwargs):
        """
        Convert the tensor's dtype or device while sharing its label mapping.

        :param tuple[Any, ...] args: Positional options accepted by
            :meth:`torch.Tensor.to`: a dtype, a device and optional dtype, or
            another tensor whose dtype and device should be matched. PyTorch's
            positional ``non_blocking`` and ``copy`` forms are also forwarded.
        :param dict[str, Any] kwargs: Conversion options accepted by PyTorch:

            * ``device`` (str | torch.device | int | None): The destination.
              ``None`` keeps the current device. Default is ``None``.
            * ``dtype`` (torch.dtype | None): The destination type. ``None``
              keeps the current dtype. Default is ``None``.
            * ``non_blocking`` (bool): Whether to attempt asynchronous transfer.
              Default is ``False``.
            * ``copy`` (bool): Whether to force a copy without a conversion.
              Default is ``False``.
            * ``memory_format`` (torch.memory_format): The requested layout.
              Default is ``torch.preserve_format``.

            These options follow the selected PyTorch overload; not all can be
            combined with every positional form.
        :raises TypeError: If PyTorch rejects the arguments or their
            combination.
        :return: The converted tensor, or ``self`` when PyTorch needs no
            conversion and ``copy`` is ``False``. The metadata dictionary is
            shared with this instance in either case.
        :rtype: LabelTensor
        """

        lt = super().to(*args, **kwargs)
        lt._labels = self._labels
        return lt

    def clone(self, *args, **kwargs):
        """
        Clone the tensor's data and deep-copy its label metadata.

        :param tuple[Any, ...] args: Positional arguments forwarded to
            :meth:`torch.Tensor.clone`, which accepts no positional options.
        :param dict[str, torch.memory_format] kwargs: The ``memory_format``
            layout option passed to PyTorch. Default is
            ``torch.preserve_format``.
        :raises TypeError: If positional options or unsupported keywords are
            supplied to PyTorch.
        :return: A differentiable data copy with independently copied metadata.
        :rtype: LabelTensor
        """

        out = LabelTensor(
            super().clone(*args, **kwargs), deepcopy(self._labels)
        )
        return out

    def append(self, tensor, mode="std"):
        """
        Append component columns, optionally forming a Cartesian row product.

        :param LabelTensor tensor: The tensor to append. In standard mode,
            shapes must match outside the last dimension. Cross mode expects
            two-dimensional tensors of shapes ``(n1, f1)`` and ``(n2, f2)``.
        :param str mode: The append operation, either ``"std"`` for last-axis
            concatenation or ``"cross"`` for pairing every row with every other
            row. Cross mode repeats this tensor's rows for each appended row
            and retains only last-dimension labels. Default is ``"std"``.
        :raises ValueError: If ``mode`` is neither ``"std"`` nor ``"cross"``.
        :raises RuntimeError: If shapes or checked non-concatenated label
            records are incompatible, or cross-mode repetition fails.
        :raises KeyError: If an input lacks metadata needed by concatenation.
        :raises TypeError: If concatenation encounters non-list component
            labels.
        :return: The concatenated tensor, with shape ``(n1 * n2, f1 + f2)`` in
            cross mode. Standard mode follows :meth:`cat` metadata sharing.
        :rtype: LabelTensor
        """

        if mode == "std":
            # Call cat on last dimension
            new_label_tensor = LabelTensor.cat(
                [self, tensor], dim=self.ndim - 1
            )
            return new_label_tensor
        if mode == "cross":
            # Crete tensor and call cat on last dimension
            tensor1 = self
            tensor2 = tensor
            n1 = tensor1.shape[0]
            n2 = tensor2.shape[0]
            tensor1 = LabelTensor(tensor1.repeat(n2, 1), labels=tensor1.labels)
            tensor2 = LabelTensor(
                tensor2.repeat_interleave(n1, dim=0), labels=tensor2.labels
            )
            new_label_tensor = LabelTensor.cat(
                [tensor1, tensor2], dim=self.ndim - 1
            )
            return new_label_tensor
        raise ValueError('mode must be either "std" or "cross"')

    @staticmethod
    def vstack(tensors):
        """
        Concatenate tensors along dimension zero using :meth:`cat`.

        Unlike :func:`torch.vstack`, this method does not promote
        one-dimensional tensors to matrices.

        :param tensors: The tensors to concatenate, with compatible shapes and
            matching metadata on dimensions other than zero.
        :type tensors: list[LabelTensor] | tuple[LabelTensor, ...]
        :raises RuntimeError: If shapes or checked label records are
            incompatible.
        :raises KeyError: If an input lacks metadata checked from the first.
        :raises TypeError: If labeled dimension-zero concatenation encounters
            non-list component labels.
        :return: The concatenated tensor, an empty list for no inputs, or the
            original tensor for one input. Metadata sharing follows :meth:`cat`.
        :rtype: LabelTensor | list[LabelTensor]
        """

        return LabelTensor.cat(tensors, dim=0)

    # This method is used to update labels
    def _update_single_label(self, index, dim):
        """
        Update the labels of the tensor based on the index (or list of indices).

        :param index: The positional selection within the labeled dimension.
        :type index: int | slice | list[int] | torch.Tensor
        :param int dim: The dimension index in the original metadata.
        :raises NotImplementedError: If the index type is unsupported.
        :raises KeyError: If ``dim`` has no stored metadata.
        :raises IndexError: If a positional label index is out of range.
        :raises TypeError: If a tensor index cannot be iterated, including a
            boolean mask with exactly one selected position after squeezing.
        :return: The selected component labels. For list-backed labels,
            positional lists and tensors select entries. For other stored
            sequences, those indices themselves are returned. Slice selections
            preserve the stored sequence's slicing result.
        :rtype: list[Hashable] | builtins.range | Hashable | torch.Tensor
        """
        old_dof = self._labels[dim]["dof"]
        # Handle slicing
        if isinstance(index, slice):
            new_dof = old_dof[index]
        # Handle single integer index
        elif isinstance(index, int):
            new_dof = [old_dof[index]]
        # Handle lists or tensors
        elif isinstance(index, (list, torch.Tensor)):
            # Handle list of bools
            if isinstance(index, torch.Tensor) and index.dtype == torch.bool:
                index = index.nonzero().squeeze()
            new_dof = (
                [old_dof[i] for i in index]
                if isinstance(old_dof, list)
                else index
            )
        else:
            raise NotImplementedError(
                f"Unsupported index type: {type(index)}. Expected slice, int, "
                f"list, or torch.Tensor."
            )
        return new_dof

    def __getitem__(self, index):
        """
        Select tensor entries and update supported label metadata.

        A string or an all-string list/tuple delegates to :meth:`extract`.
        Other selections first use PyTorch indexing. Integer indices retain
        singleton axes on labeled dimensions but remove unlabeled dimensions.
        Untouched metadata records remain shared with the source.

        :param index: The label selection or PyTorch-style positional selection.
            Tuples combine indices for successive dimensions. Metadata updates
            support integers, slices, lists, and tensor indices, not every
            advanced indexing form accepted by PyTorch.
        :type index: int | str | slice | list[int] | list[str] | torch.Tensor |
            tuple[Any, ...]
        :raises ValueError: If a requested component label is unknown or label
            extraction rejects the selection.
        :raises KeyError: If label extraction needs missing metadata.
        :raises IndexError: If a numeric index is invalid or label extraction
            requests an empty selection.
        :raises NotImplementedError: If an indexed labeled dimension uses an
            unsupported metadata index type, such as an ellipsis.
        :raises TypeError: If metadata processing cannot iterate a tensor index,
            including a boolean mask selecting exactly one labeled component.
        :return: The selected tensor. Basic slices can share data storage;
            advanced selections follow PyTorch's copying rules. If this tensor
            has no metadata, no labels are attached to the result.
        :rtype: LabelTensor
        """

        # Handle string index
        if isinstance(index, str) or (
            isinstance(index, (tuple, list))
            and all(isinstance(i, str) for i in index)
        ):
            return self.extract(index)

        # Retrieve selected tensor and labels
        selected_tensor = super().__getitem__(index)
        if not hasattr(self, "_labels"):
            return selected_tensor

        original_labels = self._labels
        updated_labels = copy(original_labels)

        # Ensure the index is iterable
        if not isinstance(index, tuple):
            index = [index]

        # Update labels based on the index
        offset = 0
        removed = 0
        for dim, idx in enumerate(index):
            if dim in original_labels:
                if isinstance(idx, int):
                    # Compute the working dimension considering the removed
                    # dimensions due to int index on a non labled dimension
                    dim_ = dim - removed
                    selected_tensor = selected_tensor.unsqueeze(dim_)
                if idx != slice(None):
                    # Update the labels for the selected dimension
                    updated_labels[offset] = {
                        "dof": self._update_single_label(idx, dim),
                        "name": original_labels[dim]["name"],
                    }
            else:
                # Adjust label keys if dimension is reduced (case of integer
                # index on a non-labeled dimension)
                if isinstance(idx, int):
                    updated_labels = {
                        key - 1 if key > dim else key: value
                        for key, value in updated_labels.items()
                    }
                    removed += 1
                    continue
            offset += 1

        # Update the selected tensor's labels
        selected_tensor._labels = updated_labels
        return selected_tensor

    def sort_labels(self, dim=None):
        """
        Sort component labels and reorder the corresponding tensor dimension.

        :param dim: The stored dimension index to sort. ``None`` selects the
            last dimension. Explicit negative indices are not normalized for
            metadata lookup. Default is ``None``.
        :type dim: int | None
        :raises KeyError: If the selected dimension has no stored metadata and
            its size is not one.
        :raises IndexError: If the dimension index is outside the tensor shape.
        :raises TypeError: If the component labels cannot be compared for
            sorting.
        :return: The reordered tensor. If the dimension's size is one, return
            ``self`` unchanged, without looking up its metadata.
        :rtype: LabelTensor
        """

        def arg_sort(lst):
            """Return the indices that would sort the list.

            :param lst: The component labels, which must be mutually orderable.
            :type lst: list[Hashable] | builtins.range | str
            :raises TypeError: If the labels cannot be compared.
            :return: The indices that sort the list.
            :rtype: list[int]
            """
            return sorted(range(len(lst)), key=lambda x: lst[x])

        if dim is None:
            dim = self.ndim - 1
        if self.shape[dim] == 1:
            return self
        labels = self.stored_labels[dim]["dof"]
        sorted_index = arg_sort(labels)
        # Define an indexer to sort the tensor along the specified dimension
        indexer = [slice(None)] * self.ndim
        # Assigned the sorted index to the specified dimension
        indexer[dim] = sorted_index
        return self[tuple(indexer)]

    def __deepcopy__(self, memo):
        """
        Deep-copy the plain tensor data and stored label metadata.

        :param dict[int, Any] memo: The copy protocol's object-id mapping;
            unused. Nested copying does not receive this mapping.
        :raises RuntimeError: If PyTorch cannot deep-copy the underlying tensor,
            for example because it is not a leaf tensor.
        :return: A new instance of the same class with independently copied data
            and label metadata.
        :rtype: LabelTensor
        """

        cls = self.__class__
        result = cls(deepcopy(self.tensor), deepcopy(self.stored_labels))
        return result

    def permute(self, *dims):
        """
        Permute tensor dimensions and remap stored label indices.

        :param tuple[list[int] | tuple[int, ...]] dims: The permutation, passed
            as one list or tuple, such as ``tensor.permute((1, 0))``. Use
            nonnegative dimension indices. Unlike PyTorch's method, separate
            positional integers are not supported by this metadata handling.
        :raises TypeError: If dimensions are passed as separate positional
            arguments or PyTorch rejects the argument types.
        :raises RuntimeError: If the permutation has the wrong length or repeats
            dimensions.
        :raises IndexError: If a dimension index is out of range.
        :raises ValueError: If a stored dimension index is absent from the
            permutation, including a negative alias for that index.
        :return: A data view with remapped dimension indices. Metadata records
            and dimension names are shared with the original tensor.
        :rtype: LabelTensor
        """
        # Call the base class permute method
        tensor = super().permute(*dims)

        # Update lables
        labels = self._labels
        keys_list = list(*dims)
        labels = {keys_list.index(k): v for k, v in labels.items()}

        # Assign labels to the new tensor
        tensor._labels = labels
        return tensor

    def detach(self):
        """
        Detaches the tensor from the computation graph and retains the stored
        labels. For more details, see :meth:`torch.Tensor.detach`.

        :return: A tensor detached from the computation graph that shares data
            storage and, when present, the metadata mapping with this instance.
        :rtype: LabelTensor
        """

        lt = super().detach()

        # Copy the labels to the new tensor only if present
        if hasattr(self, "_labels"):
            lt._labels = self.stored_labels
        return lt

    @staticmethod
    def summation(tensors):
        """
        Sum tensor data and join positional component names with ``"+"``.

        Label equality is not checked. For multiple inputs, accumulate data
        in place into a zero tensor shaped and typed like the first input.
        Broadcasting is allowed when it fits that accumulator. Label lists are
        combined positionally using :func:`zip`, not aligned by label name.

        :param tensors: The tensors to sum. Multiple inputs require iterable
            string labels on the last dimension. Shapes and dtypes must support
            accumulation into the first tensor's shape and dtype.
        :type tensors: list[LabelTensor] | tuple[LabelTensor, ...]
        :raises ValueError: If there are no inputs, or the combined component
            labels fail the output constructor's uniqueness or size validation.
        :raises RuntimeError: If shapes or dtypes do not support accumulation.
        :raises TypeError: If last-dimension labels are missing, not iterable,
            or contain non-string values.
        :return: The elementwise sum with newly joined last-dimension labels.
            Other dimension records are shallow-copied from the first tensor.
            The last dimension's name is set to the first tensor's PyTorch
            ``name`` attribute, usually ``None``, not its stored dimension name.
            With one input, return that tensor unchanged.
        :rtype: LabelTensor
        """

        if not tensors:
            raise ValueError("The tensors list must not be empty.")

        if len(tensors) == 1:
            return tensors[0]

        # Initialize result tensor and labels
        data = torch.zeros_like(tensors[0].tensor).to(tensors[0].device)
        last_dim_labels = []

        # Accumulate tensors
        for tensor in tensors:
            data += tensor.tensor
            last_dim_labels.append(tensor.labels)

        # Construct last dimension labels
        last_dim_labels = ["+".join(items) for items in zip(*last_dim_labels)]

        # Update the labels for the resulting tensor
        labels = {k: copy(v) for k, v in tensors[0].stored_labels.items()}
        labels[tensors[0].ndim - 1] = {
            "dof": last_dim_labels,
            "name": tensors[0].name,
        }

        return LabelTensor(data, labels)

    def reshape(self, *shape):
        """
        Reshape tensor data with limited last-dimension label propagation.

        :param shape: The new shape accepted by :meth:`torch.Tensor.reshape`,
            supplied as positional integers or one list/tuple. One dimension
            may be ``-1`` for size inference.
        :type shape: tuple[int, ...] | tuple[list[int] | tuple[int, ...]]
        :raises RuntimeError: If PyTorch cannot reshape to the requested size.
        :raises ValueError: If the special propagation path encounters missing
            last-dimension labels or a label count incompatible with the output.
        :return: The reshaped tensor, sharing storage when PyTorch can use a
            view. Metadata is assigned only for the exact positional shape
            ``(-1, *self.shape[2:])`` when source metadata exists. That path
            attaches only the last-dimension labels; all other forms return
            without reconstructing metadata.
        :rtype: LabelTensor

        .. warning::

            Generic reshape results can be label-tensor instances without a
            stored label mapping. Rewrap them explicitly with valid metadata
            before using label-dependent methods.
        """

        # As for now the reshape method is used only in the context of the
        # dataset, the labels are not
        tensor = super().reshape(*shape)
        if not hasattr(self, "_labels") or shape != (-1, *self.shape[2:]):
            return tensor
        tensor.labels = self.labels
        return tensor
