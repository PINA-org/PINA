"""Module for the Base Refinement class."""

from pina._src.solver.physics_informed_single_model_solver import (
    PhysicsInformedSingleModelSolver,
)
from lightning.pytorch import Callback
from pina._src.core.utils import check_consistency, check_positive_integer
from pina._src.callback.refinement.refinement_interface import (
    RefinementInterface,
)


class BaseRefinement(Callback, RefinementInterface):
    """
    Base class for all refinement strategies, implementing common functionality.

    A refinement strategy is responsible for dynamically updating the training
    dataset during optimization, typically by resampling points in the domain
    based on model behaviour (e.g., error-driven refinement).

    All specific refinement strategies should inherit from this class and
    implement its abstract methods.

    This class is not meant to be instantiated directly.
    """

    def __init__(self, sample_every, condition_to_update=None):
        """
        Initialization of the :class:`BaseRefinement` class.

        :param int sample_every: The number of epochs between successive
            refinement steps.
        :param condition_to_update: The condition(s) to be updated during
            refinement. If ``None``, all conditions associated with a domain are
            updated. Default is ``None``.
        :type condition_to_update: str | list[str] | tuple[str]
        :raises AssertionError: If ``sample_every`` is not a positive integer.
        :raises ValueError: If ``condition_to_update``, when provided, is not a
            string or an iterable of strings.
        """
        # Check consistency
        check_positive_integer(sample_every, strict=True)
        if condition_to_update is not None:
            if isinstance(condition_to_update, str):
                condition_to_update = [condition_to_update]
            check_consistency([condition_to_update], (list, tuple))
            check_consistency(condition_to_update, str)

        # Initialize attributes
        self._condition_to_update = condition_to_update
        self.sample_every = sample_every
        self._initial_population_size = None
        self._dataset = None

    def on_train_start(self, trainer, solver):
        """
        This method is called once before training begins and is typically used
        to initialize datasets, sampling conditions, or internal state.

        :param Trainer trainer: The trainer managing the training loop.
        :param BaseSolver solver: The solver associated with the trainer.
        :raise RuntimeError: If the solver is not physics-informed (i.e., does
            not implement PINNInterface).
        :raise RuntimeError: If any of the specified conditions do not exist in
            the problem.
        :raise RuntimeError: If any of the specified conditions do not have a
            'domain' attribute for sampling.
        """
        # Check solver consistency
        if not isinstance(solver, PhysicsInformedSingleModelSolver):
            raise RuntimeError(
                "Refinement strategies require a physics-informed solver. "
                f"Got '{type(solver).__name__}'."
            )

        # Initialize conditions to update if not provided
        if self._condition_to_update is None:
            self._condition_to_update = [
                name
                for name, cond in solver.problem.conditions.items()
                if hasattr(cond, "domain")
            ]

        # Validate conditions and solver
        for cond in self._condition_to_update:

            # Check if condition exists in the problem
            if cond not in solver.problem.conditions:
                raise RuntimeError(
                    f"Unknown condition '{cond}'. Available conditions: "
                    f"{list(solver.problem.conditions.keys())}."
                )

            # Check if condition has a domain to sample from
            if not hasattr(solver.problem.conditions[cond], "domain"):
                raise RuntimeError(
                    f"Condition '{cond}' has no 'domain' attribute and cannot "
                    "be used for sampling."
                )

        # Initialize dataset and compute initial population size
        self._dataset = trainer.datamodule.train_datasets
        self._initial_population_size = {
            cond: len(self.dataset[cond]) for cond in self._condition_to_update
        }

    def on_train_epoch_end(self, trainer, solver):
        """
        Apply refinement at the end of a training epoch.

        This method is invoked after each epoch and can update the dataset based
        on the current state of the model.

        :param Trainer trainer: The trainer managing the training loop.
        :param BaseSolver solver: The solver associated with the trainer.
        """
        # Store current epoch
        epoch = trainer.current_epoch

        # Sample if it's time to refine
        if epoch % self.sample_every == 0 and epoch != 0:

            # Update points for each condition to update
            for name in self._condition_to_update:

                current_points = solver.problem.conditions[name].data.input
                new_points = self.sample(current_points, name, solver)
                solver.problem.conditions[name].data.input = new_points

    @property
    def dataset(self):
        """
        The training datasets managed by the refinement strategy.

        The dataset is stored as a dictionary whose keys are condition names and
        whose values are the corresponding dataset subsets. The content of this
        dictionary can be updated dynamically during refinement.

        :return: The mapping between condition names and dataset subsets.
        :rtype: dict
        """
        return self._dataset

    @property
    def initial_population_size(self):
        """
        Initial size of the sampled dataset for each condition before any
        refinement is applied.

        :return: A mapping between each condition name and its initial number
            of sampled points.
        :rtype: dict[str, int]
        """
        return self._initial_population_size
