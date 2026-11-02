"""Module for the supervised ensemble-model solver class."""

from pina._src.condition.graph_input_target_condition import (
    GraphInputTargetCondition,
)
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.solver.ensemble_solver import EnsembleSolver


class SupervisedEnsembleSolver(EnsembleSolver):
    r"""
    Ensemble-model solver for supervised learning problems.

    This solver approximates the mapping between input data and target data
    using an ensemble of models. It is intended for problems whose conditions
    are defined by input-target pairs and accepts only
    :class:`~pina._src.condition.input_target_condition.InputTargetCondition`.

    Given input samples :math:`\mathbf{s}_i`, target values
    :math:`\mathbf{u}_i`, and an ensemble of models
    :math:`\{\mathcal{M}_j\}_{j=1}^{M}`, the prediction of each model is

    .. math::

        \hat{\mathbf{u}}_{i}^{(j)} = \mathcal{M}_j(\mathbf{s}_i),
        \qquad j = 1, \ldots, M.

    The supervised training objective minimizes the discrepancy between the
    target values and the ensemble predictions:

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{M} \sum_{j=1}^{M}
        \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}
        \left( \mathbf{u}_i - \hat{\mathbf{u}}_{i}^{(j)} \right),

    where :math:`\mathcal{L}` is the selected loss function, typically the
    mean squared error.

    :Example:

        >>> from pina.solver import SupervisedEnsembleSolver
        >>> from pina.problem import SupervisedProblem
        >>> # solver = SupervisedEnsembleSolver(
        >>> #     problem=problem, models=[model1, model2, model3]
        >>> # )
        >>> # solver.train()
    """

    # Accepted conditions types for this solver
    accepted_conditions_types = (
        InputTargetCondition,
        GraphInputTargetCondition,
    )

    def __init__(
        self,
        problem,
        models,
        optimizers=None,
        schedulers=None,
        weighting=None,
        loss=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`SupervisedEnsembleSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param models: The model or list of models used by the solver.
        :type models: torch.nn.Module | list[torch.nn.Module]
        :param optimizers: The optimizer or list of optimizers used by the
            solver. If ``None``, the ``torch.optim.Adam`` optimizer with a
            learning rate of ``0.001`` is used for each model.
            Default is ``None``.
        :type optimizers: TorchOptimizer | list[TorchOptimizer]
        :param schedulers: The scheduler or list of schedulers used by the
            solver. If ``None``, the ``torch.optim.lr_scheduler.ConstantLR``
            scheduler with a factor of ``1.0`` is used for each model.
            Default is ``None``.
        :type schedulers: TorchScheduler | list[TorchScheduler]
        :param BaseWeighting weighting: The weighting strategy used to combine
            condition losses. If ``None``, no weighting is applied. Default is
            ``None``.
        :param loss: The loss function used to compute residual losses.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
            Default is ``True``.
        """
        EnsembleSolver.__init__(
            self,
            problem=problem,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            weighting=weighting,
            loss=loss,
            use_lt=use_lt,
        )
