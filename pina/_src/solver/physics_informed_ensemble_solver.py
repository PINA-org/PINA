"""Module for the physics-informed ensemble solver class."""

from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.solver.ensemble_solver import EnsembleSolver
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class PhysicsInformedEnsembleSolver(PhysicsInformedMixin, EnsembleSolver):
    r"""
    Ensemble-model solver for physics-informed learning problems.

    This solver approximates the solution of a differential problem using an
    ensemble of models. It is intended for problems whose conditions may include
    supervised data, equation residuals evaluated on input points, and equation
    residuals sampled from domains.

    Given an ensemble of models :math:`\{\mathcal{M}_j\}_{j=1}^{M}`, the
    predicted solution of each model is

    .. math::

        \hat{\mathbf{u}}^{(j)}(\mathbf{x}) = \mathcal{M}_j(\mathbf{x}),
        \qquad j = 1, \ldots, M.

    The solver minimizes the residuals of the differential operators defining
    the problem for each model in the ensemble. For a problem with governing
    equation operator :math:`\mathcal{A}` in the domain :math:`\Omega` and
    boundary operator :math:`\mathcal{B}` on the boundary
    :math:`\partial\Omega`, the objective can be written as

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{M} \sum_{j=1}^{M}
        \left[ \frac{1}{N_{\Omega}} \sum_{i=1}^{N_{\Omega}} \mathcal{L}
        \left( \mathcal{A}[\hat{\mathbf{u}}^{(j)}](\mathbf{x}_i) \right)
        + \frac{1}{N_{\partial\Omega}} \sum_{i=1}^{N_{\partial\Omega}}
        \mathcal{L}
        \left( \mathcal{B}[\hat{\mathbf{u}}^{(j)}](\mathbf{x}_i) \right)
        \right],

    where :math:`\mathcal{L}` is the selected loss function, typically the
    mean squared error.

    :Example:

        >>> from pina.solver import PhysicsInformedEnsembleSolver
        >>> from pina.problem.zoo import Poisson2DSquareProblem
        >>> # solver = PhysicsInformedEnsembleSolver(
        >>> #     problem=problem, models=[model1, model2, model3]
        >>> # )
        >>> # Train with pina.Trainer(solver=solver, ...).train().
    """

    # Accepted conditions types for this solver
    accepted_conditions_types = (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )

    def __init__(
        self,
        problem,
        models,
        optimizers=None,
        schedulers=None,
        weighting=None,
        loss=None,
    ):
        """
        Initialization of the :class:`PhysicsInformedEnsembleSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param models: The model or list of models used by the solver.
        :type models: torch.nn.Module | list[torch.nn.Module] |
            tuple[torch.nn.Module, ...]
        :param optimizers: The optimizer or list of optimizers used by the
            solver. If ``None``, the ``torch.optim.Adam`` optimizer with a
            learning rate of ``0.001`` is used for each model.
            Default is ``None``.
        :type optimizers: pina.optim.TorchOptimizer |
            list[pina.optim.TorchOptimizer] |
            tuple[pina.optim.TorchOptimizer, ...] | None
        :param schedulers: The scheduler or list of schedulers used by the
            solver. If ``None``, the ``torch.optim.lr_scheduler.ConstantLR``
            scheduler with a factor of ``1.0`` is used for each model.
            Default is ``None``.
        :type schedulers: pina.optim.TorchScheduler |
            list[pina.optim.TorchScheduler] |
            tuple[pina.optim.TorchScheduler, ...] | None
        :param weighting: The weighting strategy used to combine
            condition losses. If ``None``, no weighting is applied. Default is
            ``None``.
        :type weighting: pina.weighting.BaseWeighting | None
        :param loss: The loss function used to compute residual losses.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :type loss: torch.nn.modules.loss._Loss | pina.loss.BaseDualLoss | None
        :raises ValueError: If base solver, component-count, wrapper-type, or
            loss/weighting validation fails. Single wrappers are not broadcast.
        """
        EnsembleSolver.__init__(
            self,
            problem=problem,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            weighting=weighting,
            loss=loss,
            use_lt=True,
        )
