"""Module for the Refinement Interface."""

from abc import ABCMeta, abstractmethod


class RefinementInterface(metaclass=ABCMeta):
    """
    Abstract interface for all refinement strategies.
    """

    @abstractmethod
    def on_train_start(self, trainer, solver):
        """
        Initialize the refinement state before training begins.

        :param pina.Trainer trainer: The trainer managing the training loop.
        :param pina.solver.BaseSolver solver: The solver associated with the
            trainer.
        """

    @abstractmethod
    def on_train_epoch_end(self, trainer, solver):
        """
        Apply refinement at the end of a training epoch.

        This method is invoked after each epoch and can update the dataset based
        on the current state of the model.

        :param pina.Trainer trainer: The trainer managing the training loop.
        :param pina.solver.BaseSolver solver: The solver associated with the
            trainer.
        """

    @abstractmethod
    def sample(self, current_points, condition_name, solver):
        """
        Generate new sample points for a given condition.

        :param pina.LabelTensor current_points: The existing points, with shape
            ``(n_points, n_variables)`` and named variable columns.
        :param str condition_name: The identifier of the condition to refine.
        :param pina.solver.BaseSolver solver: The solver used for sampling
            decisions.
        :return: The replacement points with named variable columns.
        :rtype: pina.LabelTensor
        """

    @property
    @abstractmethod
    def dataset(self):
        """
        Return the training dataset state grouped by condition name.

        :return: A mapping from condition names to implementation-specific
            dataset state, or ``None`` before initialization.
        :rtype: dict[str, Any] | None
        """

    @property
    @abstractmethod
    def initial_population_size(self):
        """
        Initial size of the sampled dataset for each condition before any
        refinement is applied.

        :return: A mapping between each selected condition name and its initial
            number of training points, or ``None`` before training starts.
        :rtype: dict[str, int] | None
        """
