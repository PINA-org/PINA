"""Module for the physics-informed single-model solver class."""

from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.solver.single_model_solver import SingleModelSolver
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class PhysicsInformedSingleModelSolver(PhysicsInformedMixin, SingleModelSolver):
    r"""
    Single-model solver for physics-informed learning problems.

    This solver approximates the solution of a differential problem using a
    single model. It is intended for problems whose conditions may include
    supervised data, equation residuals evaluated on input points, and equation
    residuals sampled from domains.

    Given a model :math:`\mathcal{M}`, the predicted solution is

    .. math::

        \hat{\mathbf{u}}(\mathbf{x}) = \mathcal{M}(\mathbf{x}).

    The solver minimizes the residuals of the differential operators defining
    the problem. For a problem with governing equation operator
    :math:`\mathcal{A}` in the domain :math:`\Omega` and boundary operator
    :math:`\mathcal{B}` on the boundary :math:`\partial\Omega`, the objective
    can be written as

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{N_{\Omega}}
        \sum_{i=1}^{N_{\Omega}} \mathcal{L}
        \left( \mathcal{A}[\hat{\mathbf{u}}](\mathbf{x}_i) \right)
        + \frac{1}{N_{\partial\Omega}} \sum_{i=1}^{N_{\partial\Omega}}
        \mathcal{L} \left( \mathcal{B}[\hat{\mathbf{u}}](\mathbf{x}_i) \right),

    where :math:`\mathcal{L}` is the selected loss function, typically the
    mean squared error.

    .. seealso::

        **Original reference**: Karniadakis, G. E., Kevrekidis, I. G., Lu, L.,
        Perdikaris, P., Wang, S., & Yang, L. (2021).
        *Physics-informed machine learning.*
        Nature Reviews Physics, 3, 422-440.
        DOI: `10.1038/s42254-021-00314-5
        <https://doi.org/10.1038/s42254-021-00314-5>`_.

    :Example:

        >>> import torch
        >>> from pina.solver import PhysicsInformedSingleModelSolver
        >>> from pina.problem.zoo import Poisson2DSquareProblem
        >>> problem = Poisson2DSquareProblem()
        >>> solver = PhysicsInformedSingleModelSolver(
        ...     problem=problem, model=torch.nn.Linear(2, 1)
        ... )
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
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
    ):
        """
        Initialization of the :class:`PhysicsInformedSingleModelSolver` class.

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
            use_lt=True,
        )
