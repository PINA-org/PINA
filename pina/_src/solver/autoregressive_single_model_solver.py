"""Module for the autoregressive single model solver class."""

from pina._src.solver.mixin.autoregressive_mixin import AutoregressiveMixin
from pina._src.condition.time_series_condition import TimeSeriesCondition
from pina._src.solver.single_model_solver import SingleModelSolver


class AutoregressiveSingleModelSolver(AutoregressiveMixin, SingleModelSolver):
    r"""
    Single-model solver for autoregressive learning problems.

    This solver learns the time evolution of dynamical systems using a single
    model. It is intended for problems defined by time-series data and accepts
    only
    :class:`~pina._src.condition.time_series_condition.TimeSeriesCondition`.

    Given a sequence of states :math:`\{\mathbf{u}_t\}_{t=0}^{T}`, the solver
    trains a model :math:`\mathcal{M}` to predict the next state from the
    current one:

    .. math::

        \hat{\mathbf{u}}_{t+1} = \mathcal{M}(\mathbf{u}_t).

    The autoregressive training objective minimizes the discrepancy between
    the predicted states :math:`\hat{\mathbf{u}}_{t+1}` and the target states
    :math:`\mathbf{u}_{t+1}` over the sequence:

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{T} \sum_{t=0}^{T-1}
        \mathcal{L} \left( \mathbf{u}_{t+1} - \hat{\mathbf{u}}_{t+1} \right),

    where :math:`\mathcal{L}` is the selected loss function, typically the mean
    squared error.

    The solver supports adaptive weighting of autoregressive steps through the
    ``eps`` parameter. During training, each autoregressive step can contribute
    differently to the total loss depending on its accumulated difficulty. Steps
    with larger running losses are assigned larger weights, so that the solver
    focuses more on parts of the rollout where prediction errors tend to
    accumulate.     The parameter ``eps`` controls the strength of this effect:
    ``eps = 0`` disables adaptive weighting, while larger values increase the
    influence of high-loss steps on the final training objective.

    :Example:

        >>> from pina.solver import AutoregressiveSingleModelSolver
        >>> from pina.problem import PoissonProblem
        >>> # solver = AutoregressiveSingleModelSolver(
        >>> #     problem=problem, model=model, eps=0.1
        >>> # )
    """

    # Accepted conditions types for this solver
    accepted_conditions_types = (TimeSeriesCondition,)

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
        use_lt=False,
        eps=0.0,
        reset_weights_at_epoch_start=True,
    ):
        """
        Initialization of the :class:`AutoregressiveSingleModelSolver` class.

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
            Default is ``False``.
        :param eps: The hyperparameter controlling the influence of the
            cumulative loss on the adaptive weights. Higher values of eps will
            lead to more aggressive weighting of steps with higher cumulative
            loss. Default is ``0.0``.
        :type eps: float | int
        :param bool reset_weights_at_epoch_start: Whether to reset the running
            average and step count at the start of each epoch. If ``True``, the
            adaptive weights will be recalibrated at the beginning of each epoch
            based on the new training dynamics. Default is ``True``.
        """

        # Initialize the parent class
        SingleModelSolver.__init__(
            self,
            problem=problem,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            loss=loss,
            use_lt=use_lt,
        )

        # Initialize the autoregressive components
        self._init_autoregressive_components(
            eps=eps,
            reset_weights_at_epoch_start=reset_weights_at_epoch_start,
        )
