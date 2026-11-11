"""Module for the Data Normalizer callback."""

from typing import Callable
import torch
from lightning.pytorch import Callback
from pina._src.core.utils import check_consistency
from pina._src.core.label_tensor import LabelTensor
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.condition.graph_condition import GraphCondition


class DataNormalizer(Callback):
    r"""
    Callback for dataset normalization on input-target conditions.

    This callback computes and applies a normalization transform to either
    input or target tensors within a dataset. The transformation is defined as:

    .. math::

        x_{\text{norm}} = \frac{x - \mu}{\sigma},

    where :math:`\mu` and :math:`\sigma` are computed using the provided
    ``shift_fn`` and ``scale_fn`` functions, respectively. Normalization
    parameters are estimated from the training dataset and then applied in-place
    to the selected datasets depending on the chosen stage.

    .. note::

        This callback ignores all conditions that are not instances of
        :class:`~pina.condition.InputTargetCondition`.

    :Example:

        >>> DataNormalizer(
        ...     scale_fn=torch.std,
        ...     shift_fn=torch.mean,
        ...     stage="all",
        ...     apply_to="input",
        ... )
    """

    # Define valid options for stage and apply_to parameters
    _VALID_STAGES = {"train", "validate", "test", "all"}
    _VALID_APPLY_TO = {"input", "target"}

    def __init__(
        self,
        scale_fn=torch.std,
        shift_fn=torch.mean,
        stage="all",
        apply_to="input",
    ):
        """
        Initialization of the :class:`DataNormalizer` class.

        :param Callable scale_fn: The function used to compute the scaling
            factor. Default is  ``torch.std``.
        :param Callable shift_fn: The function used to compute the shifting
            factor. Default is ``torch.mean``.
        :param str stage: The stage during which normalization is applied.
            Available options are ``"train"``, ``"validate"``, ``"test"``, and
            ``"all"``. Default is ``"all"``.
        :param str apply_to: Specifies whether normalization is applied to
            ``"input"`` or ``"target"`` tensors. Default is ``"input"``.
        :raises ValueError: If ``scale_fn`` is not Callable.
        :raises ValueError: If ``shift_fn`` is not Callable.
        :raises ValueError: If ``stage`` is invalid.
        :raises ValueError: If ``apply_to`` is invalid.
        """
        super().__init__()

        # Check consistency
        check_consistency(scale_fn, Callable)
        check_consistency(shift_fn, Callable)
        check_consistency(stage, str)
        check_consistency(apply_to, str)

        # Validate stage parameter
        if stage not in self._VALID_STAGES:
            raise ValueError(
                "Invalid value for 'stage'. Available options are "
                f"{self._VALID_STAGES}. Got {stage}."
            )

        # Validate apply_to parameter
        if apply_to not in self._VALID_APPLY_TO:
            raise ValueError(
                "Invalid value for 'apply_to'. Available options are "
                f"{self._VALID_APPLY_TO}. Got {apply_to}."
            )

        # Initialize attributes
        self.scale_fn = scale_fn
        self.shift_fn = shift_fn
        self.stage = stage
        self.apply_to = apply_to
        self._normalizer = {}
        self._normalized_conditions = set()

    @staticmethod
    def _batch_ids_by_condition(datamodule, dataset):
        """
        Build a mapping from condition name to list of batch ids for a dataset.

        :param DataModule datamodule: The data module containing the condition
            index mappings.
        :param torch.Tensor dataset: The dataset tensor with rows
            ``[condition_id, batch_id]``. Its shape is
            ``(n_steps, n_conditions, 2)``; leading dimensions are flattened.
        :return: Mapping from condition name to list of batch ids.
        :rtype: dict[str, list[int]]
        """
        batch_ids = {}
        if dataset is None or len(dataset) == 0:
            return batch_ids
        for condition_id, batch_id in dataset.reshape(-1, 2):
            name = datamodule.idx_to_condition_name[int(condition_id)]
            batch_ids.setdefault(name, []).append(int(batch_id))
        return batch_ids

    def setup(self, trainer, pl_module, stage):
        """
        Compute and apply normalization during the setup phase.

        :param Trainer trainer: The trainer instance managing the execution.
        :param BaseSolver pl_module: The solver module being executed.
        :param str stage: Current execution stage.
        :raises NotImplementedError: If any condition contains graph-based data
            and is therefore unsupported.
        """
        # The conditions of the underlying problem
        conditions = pl_module.problem.conditions

        # Check if any condition contains graph-based data
        if any(
            isinstance(cond, GraphCondition) for cond in conditions.values()
        ):
            raise NotImplementedError(
                "DataNormalizer is not compatible with graph-based datasets."
            )

        # Extract input-target conditions
        conditions_to_normalize = [
            name
            for name, cond in conditions.items()
            if isinstance(cond, InputTargetCondition)
        ]

        # Build train batch-id mapping and compute normalizer from training data
        datamodule = trainer.datamodule
        train_batch_ids = self._batch_ids_by_condition(
            datamodule, datamodule.train_dataset
        )

        # Compute scale and shift parameters if not already computed
        if not self.normalizer:
            for cond in conditions_to_normalize:
                ids = train_batch_ids.get(cond, [])
                if not ids:
                    continue
                pts = torch.cat(
                    [conditions[cond].data[i][self.apply_to] for i in ids]
                )
                self._normalizer[cond] = {
                    "shift": self.shift_fn(pts),
                    "scale": self.scale_fn(pts),
                }

        # Apply normalization to the appropriate splits
        if stage == "fit" and self.stage in ("train", "all"):
            self.normalize_dataset(
                conditions,
                train_batch_ids,
                "train",
            )

        if stage == "fit" and self.stage in ("validate", "all"):
            self.normalize_dataset(
                conditions,
                self._batch_ids_by_condition(
                    datamodule, datamodule.val_dataset
                ),
                "validate",
            )

        if stage == "test" and self.stage in ("test", "all"):
            self.normalize_dataset(
                conditions,
                self._batch_ids_by_condition(
                    datamodule, datamodule.test_dataset
                ),
                "test",
            )

        return super().setup(trainer, pl_module, stage)

    def normalize_dataset(self, conditions, batch_ids_by_condition, split):
        """
        Apply normalization to all matching conditions in-place for a split.

        The backing tensor (``_input`` or ``_target``) of each condition is
        updated so that normalization survives subsequent ``store_data`` calls
        that regenerate the batch dictionaries.  Each individual batch dict is
        then also refreshed from the updated backing tensor.

        :param dict conditions: Mapping from condition name to condition.
        :param dict batch_ids_by_condition: Mapping from condition name to
            list of batch ids belonging to this split.
        :param str split: Name of the split (used to track normalization state).
        """
        for cond, batch_ids in batch_ids_by_condition.items():
            if cond not in self.normalizer:
                continue
            if (cond, split) in self._normalized_conditions:
                continue

            shift = self.normalizer[cond]["shift"]
            scale = self.normalizer[cond]["scale"]
            backing = getattr(conditions[cond], f"_{self.apply_to}")

            for batch_id in batch_ids:
                # Determine the slice of the backing tensor for this batch
                batch_size = len(conditions[cond].data[batch_id][self.apply_to])
                # Reconstruct start index by summing sizes of earlier batches
                start = sum(
                    len(conditions[cond].data[i][self.apply_to])
                    for i in range(batch_id)
                )
                end = start + batch_size

                # Normalize the slice of the backing tensor in-place
                raw_slice = backing[start:end]
                scaled = (raw_slice - shift) / scale
                if isinstance(raw_slice, LabelTensor):
                    scaled = LabelTensor(scaled, raw_slice.labels)
                backing[start:end] = scaled

                # Also update the already-stored batch dict
                conditions[cond].data[batch_id][self.apply_to] = scaled

            self._normalized_conditions.add((cond, split))

    @property
    def normalizer(self):
        """
        The dictionary mapping each condition to its corresponding ``shift`` and
        ``scale`` values.

        :return: The dictionary of normalization parameters.
        :rtype: dict
        """
        return self._normalizer
