"""Module for the Scheduler Interface."""

from abc import ABCMeta, abstractmethod


class SchedulerInterface(metaclass=ABCMeta):
    """
    Abstract interface for all schedulers.
    """

    @abstractmethod
    def hook(self, optimizer):
        """
        Initialize the scheduler for an initialized optimizer wrapper.

        :param OptimizerInterface optimizer: The wrapper providing the backend
            optimizer instance associated with the scheduler.
        """

    @property
    @abstractmethod
    def instance(self):
        """
        The underlying scheduler object.

        :return: The backend scheduler, or ``None`` before initialization when
            the implementation uses deferred construction.
        :rtype: object
        """
