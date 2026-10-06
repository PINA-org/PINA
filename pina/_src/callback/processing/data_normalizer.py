"""Module for the Data Normalizer callback."""

from typing import Callable
import torch
from lightning.pytorch import Callback
from pina._src.core.utils import check_consistency
from pina._src.core.label_tensor import LabelTensor
from pina._src.condition.condition import InputTargetCondition


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
            hasattr(condition, "graph_key") for condition in conditions.values()
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

        # Extract the dictionary of all datasets
        dataset = trainer.datamodule.train_datasets

        # Compute scale and shift parameters if not already computed
        if not self.normalizer:

            # Iterate over conditions and compute normalization parameters
            for cond in conditions_to_normalize:
                pts = self._get_data(conditions, cond)
                shift = self.shift_fn(pts)
                scale = self.scale_fn(pts)

                self._normalizer[cond] = {
                    "shift": shift,
                    "scale": scale,
                }

        # Apply normalization to training datasets
        if stage == "fit" and self.stage in ["train", "all"]:
            self.normalize_dataset(
                conditions, trainer.datamodule.train_datasets
            )

        if stage == "fit" and self.stage in ["validate", "all"]:
            self.normalize_dataset(conditions, trainer.datamodule.val_datasets)

        if stage == "test" and self.stage in ["test", "all"]:
            self.normalize_dataset(conditions, trainer.datamodule.test_datasets)

        return super().setup(trainer, pl_module, stage)

    def normalize_dataset(self, conditions, dataset):
        """
        Apply normalization to all conditions in-place.

        Each condition is updated using precomputed normalization parameters.
        The transformation preserves tensor types.

        :param dict conditions: The mapping between condition names and their
            associated conditions.
        :param dict dataset: The mapping between condition names and their
            associated dataset subsets.
        """
        # Iterate over conditions and apply normalization
        for cond, norm_params in self.normalizer.items():
            if cond in self._normalized_conditions or cond not in dataset:
                continue

            # Extract the points to normalize, the scale and the shift
            points = getattr(conditions[cond].data, self.apply_to)
            scale = norm_params["scale"]
            shift = norm_params["shift"]

            # Apply normalization
            scaled_pts = (points - shift) / scale
            if isinstance(points, LabelTensor):
                scaled_pts = LabelTensor(scaled_pts, points.labels)

            # Update the condition data in-place
            setattr(conditions[cond].data, self.apply_to, scaled_pts)
            self._normalized_conditions.add(cond)

    def _get_data(self, conditions, cond):
        """
        Extract the selected data field for a given condition.

        :param dict conditions: The mapping between condition names and their
            associated conditions.
        :param str cond: The condition name.
        :return: The selected input or target data.
        :rtype: torch.Tensor
        """
        return getattr(conditions[cond].data, self.apply_to)

    @property
    def normalizer(self):
        """
        The dictionary mapping each condition to its corresponding ``shift`` and
        ``scale`` values.

        :return: The dictionary of normalization parameters.
        :rtype: dict
        """
        return self._normalizer
