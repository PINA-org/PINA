"""Module for the supervised single-model solver class."""

from pina._src.condition.graph_input_target_condition import (
    GraphInputTargetCondition,
)
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.solver.single_model_solver import SingleModelSolver


class SupervisedSingleModelSolver(SingleModelSolver):
    r"""
    Single-model solver for supervised learning problems.

    This solver is designed for problems defined by input-target pairs and uses
    a single model to approximate the mapping from input variables to target
    variables. It supports tensor
    :class:`~pina._src.condition.input_target_condition.InputTargetCondition`
    and :class:`~pina.condition.GraphInputTargetCondition` conditions.

    Given a model :math:`\mathcal{M}`, the solver minimizes the discrepancy
    between the target values :math:`\mathbf{u}_i` and the model predictions
    :math:`\mathcal{M}(\mathbf{s}_i)` evaluated at the input data
    :math:`\mathbf{s}_i`.

    The supervised loss minimized during training is

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{N} \sum_{i=1}^{N}
        \mathcal{L} \left( \mathbf{u}_i - \mathcal{M}(\mathbf{s}_i) \right),

    where :math:`\mathcal{L}` is the selected loss function, typically the mean
    squared error.

    :Example:

        >>> from pina.solver import SupervisedSingleModelSolver
        >>> from pina.problem.zoo import SupervisedProblem
        >>> # solver = SupervisedSingleModelSolver(
        >>> #     problem=problem, model=model
        >>> # )
        >>> # Train with pina.Trainer(solver=solver, ...).train().
    """

    # Accepted conditions types for this solver
    accepted_conditions_types = (
        InputTargetCondition,
        GraphInputTargetCondition,
    )

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
        Initialization of the :class:`SupervisedSingleModelSolver` class.

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
        :raises ValueError: If the problem, conditions, model, wrappers, loss,
            or weighting fail base-solver validation.
        """
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
