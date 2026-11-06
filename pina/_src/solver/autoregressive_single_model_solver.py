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
    ``eps`` parameter. Weights are ``exp(-eps * cumulative_loss)`` along the
    rollout-step axis, including the current step's running average loss.
    Positive ``eps`` suppresses steps with larger cumulative loss; ``eps=0``
    gives unit weights. No sign constraint is enforced on ``eps``.

    :Example:

        >>> from pina.solver import AutoregressiveSingleModelSolver
        >>> from pina.condition import TimeSeriesCondition
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
        kwargs=None,
    ):
        """
        Initialization of the :class:`AutoregressiveSingleModelSolver` class.

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
        :param bool use_lt: Whether to wrap forward with labeled input selection
            and output labeling. Requires compatible tensor labels and names.
            Default is ``False``.
        :param eps: The hyperparameter controlling the influence of the
            cumulative loss on the adaptive weights. Positive values suppress
            larger cumulative losses; zero gives unit weights.
            Default is ``0.0``.
        :type eps: float | int
        :param bool reset_weights_at_epoch_start: Whether to reset the running
            average and step count at the start of each epoch. If ``True``, the
            adaptive weights will be recalibrated at the beginning of each epoch
            based on the new training dynamics. Default is ``True``.
        :param kwargs: Options forwarded to the preprocessing/postprocessing
            hooks during condition rollouts. Base hooks ignore them. ``None``
            stores an empty mapping. Prediction takes its own keyword options.
            Default is ``None``.
        :type kwargs: dict[str, Any] | None
        :raises ValueError: If ``eps`` or the reset flag fails validation, or
            base solver/component/loss/weighting validation fails.
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
            kwargs=kwargs,
        )
