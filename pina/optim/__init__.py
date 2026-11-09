"""Module for the Optimizers and Schedulers."""

__all__ = [
    "OptimizerInterface",
    "SchedulerInterface",
    "TorchOptimizer",
    "TorchScheduler",
]

from pina._src.optim.optimizer_interface import OptimizerInterface
from pina._src.optim.scheduler_interface import SchedulerInterface
from pina._src.optim.torch_optimizer import TorchOptimizer
from pina._src.optim.torch_scheduler import TorchScheduler

# Back-compatibility with version 0.2, to be removed soon
import warnings

_DEPRECATED_IMPORTS = {
    "Optimizer": "OptimizerInterface",
    "Scheduler": "SchedulerInterface",
}


def __getattr__(name):
    """
    Resolve deprecated optimizer and scheduler interface names.

    :param str name: The requested legacy attribute name.
    :return: The matching interface class for ``"Optimizer"`` or
        ``"Scheduler"``, emitting ``DeprecationWarning``. Unrecognized names
        return ``None``.
    :rtype: type | None
    """
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina.optim' is deprecated; use "
            f"pina.optim.{_DEPRECATED_IMPORTS[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return globals()[_DEPRECATED_IMPORTS[name]]
