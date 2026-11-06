"""Module for the gradient physics-informed single-model solver class."""

from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.solver.single_model_solver import SingleModelSolver
from pina._src.solver.mixin.gradient_enhanced_mixin import (
    GradientEnhancedMixin,
)
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class GradientPhysicsInformedSingleModelSolver(
    PhysicsInformedMixin, GradientEnhancedMixin, SingleModelSolver
):
    r"""
    Single-model solver for gradient-enhanced physics-informed learning
    problems.

    This solver approximates the solution of a differential problem using a
    single model and augments the standard physics-informed objective with
    gradient-enhanced residual terms. It can be used for both forward and
    inverse problems.

    Given a model :math:`\mathcal{M}`, the predicted solution is

    .. math::

        \hat{\mathbf{u}}(\mathbf{x}) = \mathcal{M}(\mathbf{x}).

    The solver minimizes both the residuals of the differential operators
    defining the problem and the gradients of those residuals with respect to
    the input variables. For a problem with governing equation operator
    :math:`\mathcal{A}` in the domain :math:`\Omega` and boundary operator
    :math:`\mathcal{B}` on the boundary :math:`\partial\Omega`, the objective
    can be written as

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{N_{\Omega}}
        \sum_{i=1}^{N_{\Omega}} \mathcal{L}
        \left( \mathcal{A}[\hat{\mathbf{u}}](\mathbf{x}_i) \right)
        + \frac{1}{N_{\partial\Omega}} \sum_{i=1}^{N_{\partial\Omega}}
        \mathcal{L} \left( \mathcal{B}[\hat{\mathbf{u}}](\mathbf{x}_i) \right)
        + \frac{\lambda}{N_{\Omega}} \sum_{i=1}^{N_{\Omega}} \mathcal{L}
        \left( \nabla_{\mathbf{x}} \mathcal{A}[\hat{\mathbf{u}}](\mathbf{x}_i)
        \right) + \frac{\lambda}{N_{\partial\Omega}}
        \sum_{i=1}^{N_{\partial\Omega}}
        \mathcal{L} \left( \nabla_{\mathbf{x}} \mathcal{B}[\hat{\mathbf{u}}]
        (\mathbf{x}_i) \right),

    where :math:`\mathcal{L}` is the selected loss function, typically the mean
    squared error, and :math:`\lambda` is ``regularization_weight``. Gradient
    terms apply only to selected conditions and use spatial coordinates.

    .. seealso::

        **Original reference**: Yu, J., Lu, L., Meng, X., & Karniadakis, G. E.
        (2022). *Gradient-enhanced physics-informed neural networks for forward
        and inverse PDE problems.* Computer Methods in Applied Mechanics and
        Engineering, 393, 114823.
        DOI: `10.1016/j.cma.2022.114823
        <https://doi.org/10.1016/j.cma.2022.114823>`_.

    :Example:

        >>> from pina.solver import GradientPhysicsInformedSingleModelSolver
        >>> from pina.problem.zoo import Poisson2DSquareProblem
        >>> # solver = GradientPhysicsInformedSingleModelSolver(
        >>> #     problem=problem, model=model, regularization_weight=0.1
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
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
        regularization_weight=1.0,
        regularized_conditions=None,
    ):
        """
        Initialization of the :class:`GradientPhysicsInformedSingleModelSolver`
        class.

        :param pina.problem.SpatialProblem problem: The spatial problem.
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
        :param regularization_weight: The weight of the gradient regularization
            term. Default is ``1.0``.
        :type regularization_weight: float | int
        :param regularized_conditions: The names of the conditions that should
            receive gradient regularization. If ``None``, all conditions are
            regularized. Default is ``None``.
        :type regularized_conditions: str | list[str] | tuple[str, ...] | None
        :raises ValueError: If the problem is not spatial, conditions are
            unknown, the weight is not numeric, or base solver validation fails.
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
            use_lt=True,
        )

        # Initialize the gradient-enhanced components
        self._init_gradient_enhanced_components(
            regularization_weight=regularization_weight,
            regularized_conditions=regularized_conditions,
        )
