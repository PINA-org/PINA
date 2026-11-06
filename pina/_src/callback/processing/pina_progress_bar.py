"""Progress reporting with condition-specific and global loss filtering."""

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.progress_bar import (
    get_standard_metrics,
)
from pina._src.core.utils import check_consistency


class PINAProgressBar(TQDMProgressBar):
    """
    Custom progress bar callback for PINA training workflows.

    This callback extends the default Lightning progress bar by filtering the
    displayed metrics.

    Metrics can refer either to condition-specific losses, identified by the
    names assigned to the problem conditions, or to global losses. Global losses
    are selected using ``"train"``, ``"val"``, or ``"test"``, and are internally
    expanded to the corresponding logged loss metrics.

    :Example:

        >>> from pina.callback import PINAProgressBar
        >>> progress_bar = PINAProgressBar(metrics="val")
        >>> progress_bar._sorted_metrics
        ['val']
    """

    GLOBAL_LOSS_KEYS = ("train", "val", "test")

    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
        "{rate_noinv_fmt}{postfix}]"
    )

    def __init__(self, metrics="val", **kwargs):
        """
        Initialization of the :class:`PINAProgressBar` class.

        :param metrics: The names of the metrics to be shown in the progress
            bar. Each entry can be either a key of a condition defined in the
            problem or one of the global loss keys: ``"train"``, ``"val"``, or
            ``"test"``. These global keys are internally expanded to the
            corresponding logged loss names. Default is ``"val"``.
        :type metrics: str | list[str] | tuple[str, ...] | set[str]
        :param dict[str, int | bool] kwargs: Options forwarded to
            :class:`lightning.pytorch.callbacks.TQDMProgressBar`. Supported
            options are:

            * ``refresh_rate`` (int): The update interval in batches, with
              ``0`` disabling updates. Default is ``1``.
            * ``process_position`` (int): The display position. Default is
              ``0``.
            * ``leave`` (bool): Whether to keep finished bars visible. Default
              is ``False``.
        :raises ValueError: If ``metrics`` is not a string or a list, tuple, or
            set of strings.
        :raises TypeError: If ``kwargs`` contains an unsupported keyword.
        """
        super().__init__(**kwargs)

        # Check consistency
        check_consistency(metrics, str)

        # Convert to list if a single string is provided
        if isinstance(metrics, str):
            metrics = [metrics]

        # Store the sorted metrics for later use in get_metrics
        self._sorted_metrics = sorted(metrics)

    def get_metrics(self, trainer, __):
        """
        Retrieve and filter metrics to be displayed in the progress bar.

        Keep all standard Lightning metrics, including the logger version when
        available, and add only the selected progress bar metrics.

        :param pina.Trainer trainer: The trainer managing the training loop.
        :param pina.solver.BaseSolver __: The solver supplied by
            Lightning; unused.
        :return: Dictionary containing the metrics to display.
        :rtype: dict[str, float | int | str]

        .. note::

            This method overrides the default Lightning behaviour. It can be
            further customized by subclassing.
        """
        # Retrieve standard metrics and user-selected progress bar metrics
        standard_metrics = get_standard_metrics(trainer)
        progress_bar_metrics = trainer.progress_bar_metrics

        # Filter progress bar metrics to include only specified keys
        if progress_bar_metrics:
            progress_bar_metrics = {
                key: progress_bar_metrics[key]
                for key in progress_bar_metrics
                if key in self._sorted_metrics
            }

        return {**standard_metrics, **progress_bar_metrics}

    def setup(self, trainer, pl_module, stage):
        """
        Configure the metrics to track before execution starts.

        The requested metrics must be either names assigned to problem
        conditions or global loss keys. The accepted global loss keys are
        ``"train"``, ``"val"``, and ``"test"``.

        Replace each selected name with its logged loss name using
        ``"_loss_epoch"`` when batching is enabled, or ``"_loss"`` otherwise.
        Repeated setup calls validate the already suffixed names again.

        :param pina.Trainer trainer: The trainer managing the execution.
        :param pina.solver.BaseSolver pl_module: The solver module
            being executed; unused by this callback.
        :param str stage: Current execution stage.
        :raises KeyError: If a metric key is neither a condition key nor one of
            ``"train"``, ``"val"``, or ``"test"``.
        """
        # Get the condition keys from the problem
        condition_keys = trainer.solver.problem.conditions.keys()
        for key in self._sorted_metrics:
            if key not in condition_keys and key not in self.GLOBAL_LOSS_KEYS:
                raise KeyError(
                    f"Key '{key}' is not a valid metric. It must be either a "
                    f"problem condition key or one of {self.GLOBAL_LOSS_KEYS}."
                )

        # Add the appropriate suffix to the metric names based on batch size
        suffix = "_loss_epoch" if trainer.batch_size is not None else "_loss"
        self._sorted_metrics = [
            metric + suffix for metric in self._sorted_metrics
        ]

        return super().setup(trainer, pl_module, stage)
