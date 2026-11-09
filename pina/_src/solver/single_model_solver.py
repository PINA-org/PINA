"""Module for the single-model solver class."""

from pina._src.solver.mixin.single_model_mixin import SingleModelMixin
from pina._src.solver.base_solver import BaseSolver
from pina._src.solver.mixin.condition_aggregator_mixin import (
    ConditionAggregatorMixin,
)


class SingleModelSolver(SingleModelMixin, ConditionAggregatorMixin, BaseSolver):
    """
    Base class for implementing single-model solvers.

    This class provides the standard starting point for solvers based on a
    single model. It combines the shared solver machinery from
    :class:`~pina._src.solver.base_solver.BaseSolver` with single-model handling
    and condition-wise loss aggregation.

    Subclasses can inherit from this class to implement solver-specific behaviour
    while reusing the common logic for model registration, optimizer and
    scheduler setup, loss evaluation, weighting, and aggregation across problem
    conditions.

    :Example:

        >>> from pina.solver import SingleModelSolver
        >>> from pina.problem import PoissonProblem
        >>> # Typically used as a base class.
        >>> # solver = SingleModelSolver(problem=problem, model=model)
    """

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`SingleModelSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param torch.nn.Module model: The model used by the solver.
        :param TorchOptimizer optimizer: The optimizer used by the solver.
            If ``None``, the ``torch.optim.Adam`` optimizer with a learning rate
            of ``0.001`` is used. Default is ``None``.
        :param TorchScheduler scheduler: The scheduler used by the solver.
            If ``None``, the ``torch.optim.lr_scheduler.ConstantLR`` scheduler
            with a factor of ``1.0`` is used. Default is ``None``.
        :param BaseWeighting weighting: The weighting strategy used to combine
            condition losses. If ``None``, no weighting is applied. Default is
            ``None``.
        :param loss: The loss function used to compute residual losses.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
            Default is ``True``.
        """
        BaseSolver.__init__(
            self,
            problem=problem,
            models=model,
            optimizers=optimizer,
            schedulers=scheduler,
            weighting=weighting,
            loss=loss,
            use_lt=use_lt,
        )
