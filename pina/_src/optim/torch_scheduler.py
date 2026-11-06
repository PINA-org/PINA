"""Module for wrapping PyTorch schedulers."""

from torch.optim.lr_scheduler import LRScheduler
from pina._src.core.utils import check_consistency
from pina._src.optim.optimizer_interface import OptimizerInterface
from pina._src.optim.scheduler_interface import SchedulerInterface


class TorchScheduler(SchedulerInterface):
    """
    The wrapper class for PyTorch schedulers.

    This class wraps a ``torch.optim.lr_scheduler.LRScheduler`` class and defers
    its instantiation until runtime, once the optimizer instance is available.

    :Example:

        >>> from pina.optim import TorchScheduler
        >>> import torch
        >>> scheduler = TorchScheduler(
        ...     torch.optim.lr_scheduler.StepLR, step_size=5)
        >>> scheduler.scheduler_class
        <class 'torch.optim.lr_scheduler.StepLR'>
    """

    def __init__(self, scheduler_class, **kwargs):
        """
        Initialization of the :class:`TorchScheduler` class.

        :param type[torch.optim.lr_scheduler.LRScheduler] scheduler_class: The
            scheduler class, not an initialized instance.
        :param dict[str, Any] kwargs: Options forwarded unchanged to that
            class's constructor during ``hook``, such as ``step_size`` and
            ``gamma`` for :class:`torch.optim.lr_scheduler.StepLR`. Do not
            include ``optimizer``; it is supplied by the hook. See more
            `here <https://pytorch.org/docs/stable/optim.html#algorithms>`_.
        :raises ValueError: If ``scheduler_class`` is not a subclass of
            ``torch.optim.lr_scheduler.LRScheduler``.
        """
        # Check consistency
        check_consistency(scheduler_class, LRScheduler, subclass=True)

        # Initialize attributes
        self.scheduler_class = scheduler_class
        self.kwargs = kwargs
        self._scheduler_instance = None

    def hook(self, optimizer):
        """
        Construct and store a scheduler using the optimizer wrapper's instance.

        Repeated calls replace the stored scheduler. The optimizer must already
        have been initialized through its hook.

        :param OptimizerInterface optimizer: The PINA wrapper providing an
            initialized :class:`torch.optim.Optimizer` through ``instance``.
        :raises ValueError: If ``optimizer`` is not an instance of
             :class:`OptimizerInterface`.
        :raises TypeError: If the scheduler rejects an uninitialized optimizer
            or its constructor options.
        :raises ValueError: If the scheduler rejects an option value.
        """
        # Check consistency
        check_consistency(optimizer, OptimizerInterface)

        # Initialize the scheduler instance
        self._scheduler_instance = self.scheduler_class(
            optimizer.instance, **self.kwargs
        )

    @property
    def instance(self):
        """
        The underlying scheduler object.

        :return: The stored scheduler, or ``None`` until ``hook`` succeeds.
        :rtype: torch.optim.lr_scheduler.LRScheduler | None
        """
        return self._scheduler_instance
