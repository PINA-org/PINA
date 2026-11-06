"""Module for the Metric Tracker."""

import copy
import torch
from lightning.pytorch.callbacks import Callback
from pina._src.core.utils import check_consistency


class MetricTracker(Callback):
    """
    Callback for collecting selected metrics logged during training.

    Collection starts after epoch index zero. Metric values are copied at each
    subsequent epoch end, and :attr:`metrics` stacks only names present in every
    collected epoch.

    :Example:

        >>> from pina.callback import MetricTracker
        >>> tracker = MetricTracker(metrics_to_track=["train_loss"])
        >>> tracker.metrics_to_track
        ['train_loss']
    """

    def __init__(self, metrics_to_track=None):
        """
        Initialization of the :class:`MetricTracker` class.

        :param metrics_to_track: The names of the metrics to collect. If
            ``None``, setup selects ``["train_loss"]``. With a batch size, setup
            expands each name into its ``"_step"`` and ``"_epoch"`` variants,
            including the default name. Default is ``None``.
        :type metrics_to_track: str | list[str] | tuple[str, ...] | set[str] |
            None
        :raises ValueError: If any of the provided metric names are not strings.
        """
        super().__init__()

        # Check consistency
        if metrics_to_track is not None:
            check_consistency(metrics_to_track, str)

        # Convert to list if a single string is provided
        if isinstance(metrics_to_track, str):
            metrics_to_track = [metrics_to_track]

        # Initialize the collection list and store the metrics to track
        self.metrics_to_track = metrics_to_track
        self._collection = []

    def setup(self, trainer, pl_module, stage):
        """
        Configure the metrics to track before execution starts.

        When a batch size is provided (i.e. ``trainer.batch_size`` is not
        ``None``), metric names are expanded to match Lightning's logging
        convention: for each metric ``m``, both ``m_step`` and ``m_epoch`` are
        tracked. For example, ``"train_loss"`` becomes
        ``["train_loss_step", "train_loss_epoch"]``.

        This method updates ``metrics_to_track``. Calling it again
        with batching enabled appends suffixes to the already expanded names.

        :param pina.Trainer trainer: The trainer managing the execution.
        :param pina.solver.BaseSolver pl_module: The solver module
            being executed; unused by this callback.
        :param str stage: Current execution stage.
        """
        # Set default metrics to train_loss if no batch size is available
        if self.metrics_to_track is None:
            self.metrics_to_track = ["train_loss"]

        # If a batch size is provided, expand metric names to match convention
        if trainer.batch_size is not None:
            self.metrics_to_track = [
                f"{metric}_{suffix}"
                for metric in self.metrics_to_track
                for suffix in ("step", "epoch")
            ]

        return super().setup(trainer, pl_module, stage)

    def on_train_epoch_end(self, trainer, __):
        """
        Store the selected logged metrics at the end of each training epoch.

        Skip epoch index zero. Append a deep copy of the matching logged metrics
        on later epochs, even if no requested names are available.

        :param pina.Trainer trainer: The trainer managing the execution.
        :param pina.solver.BaseSolver __: The solver supplied by
            Lightning; unused.
        """
        # Only collect metrics after the first epoch to ensure they are logged
        if trainer.current_epoch > 0:

            # Collect the metrics that are being tracked
            tracked_metrics = {
                k: v
                for k, v in trainer.logged_metrics.items()
                if k in self.metrics_to_track
            }
            self._collection.append(copy.deepcopy(tracked_metrics))

    @property
    def metrics(self):
        """
        Return the collected metrics stacked over the tracked epochs.

        :return: The dictionary mapping each tracked name present in every
            collected epoch to its stacked values. The first tensor dimension
            indexes collected epochs. Returns an empty dictionary if no metrics
            have been collected or no tracked names occur in every epoch.
        :rtype: dict[str, torch.Tensor]
        """
        if not self._collection:
            return {}

        # Identify the common keys across all collected metric dictionaries
        common_keys = set(self._collection[0]).intersection(
            *self._collection[1:]
        )

        return {
            k: torch.stack([dic[k] for dic in self._collection])
            for k in common_keys
            if k in self.metrics_to_track
        }
