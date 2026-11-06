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

    Subclasses can inherit from this class to implement solver-specific
    behaviour while reusing the common logic for model registration, optimizer,
    scheduler setup, loss evaluation, weighting, and aggregation across problem
    conditions.

    :Example:

        >>> from pina.solver import SingleModelSolver
        >>> from pina.problem.zoo import Poisson2DSquareProblem
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
        :param optimizer: The optimizer wrapper used by the solver.
            If ``None``, the ``torch.optim.Adam`` optimizer with a learning rate
            of ``0.001`` is used. Default is ``None``.
        :type optimizer: pina.optim.TorchOptimizer | None
        :param scheduler: The scheduler wrapper used by the solver.
            If ``None``, the ``torch.optim.lr_scheduler.ConstantLR`` scheduler
            with a factor of ``1.0`` is used. Default is ``None``.
        :type scheduler: pina.optim.TorchScheduler | None
        :param weighting: The weighting strategy used to combine
            condition losses. If ``None``, no weighting is applied. Default is
            ``None``.
        :type weighting: pina.weighting.BaseWeighting | None
        :param loss: The loss function used to compute residual losses.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :type loss: torch.nn.modules.loss._Loss | pina.loss.BaseDualLoss | None
        :param bool use_lt: Whether to select labeled inputs in problem-variable
            order and label forward outputs. PINA graphs support this with
            labeled ``x`` features; unlabeled inputs require ``False``.
            Default is ``True``.
        :raises ValueError: If base solver, component-count, wrapper-type, or
            loss/weighting validation fails.
        """
        # Initialize the base solver
        BaseSolver.__init__(self, problem=problem, use_lt=use_lt)

        # Initialize the components of the solver
        self._init_solver_components(
            models=model,
            optimizers=optimizer,
            schedulers=scheduler,
        )

        # Initialize the weighting scheme for the conditions and the loss
        self._init_weighting_and_loss(weighting=weighting, loss=loss)
