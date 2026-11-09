"""
Module for the residual-based attention physics-informed single-model solver
class.
"""

from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.solver.single_model_solver import SingleModelSolver
from pina._src.solver.mixin.residual_based_attention_mixin import (
    ResidualBasedAttentionMixin,
)
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class RBAPhysicsInformedSingleModelSolver(
    PhysicsInformedMixin, ResidualBasedAttentionMixin, SingleModelSolver
):
    r"""
    Residual-based attention solver for physics-informed learning problems.

    This solver approximates the solution of a differential problem using a
    single model equipped with residual-based attention weights. It can be used
    for both forward and inverse problems.

    Given a model :math:`\mathcal{M}`, the predicted solution is

    .. math::

        \hat{\mathbf{u}}(\mathbf{x}) = \mathcal{M}(\mathbf{x}).

    The solver minimizes a weighted objective in which each residual
    contribution is scaled by an attention weight. For a problem with governing
    equation operator :math:`\mathcal{A}` in the domain :math:`\Omega` and
    boundary operator :math:`\mathcal{B}` on the boundary
    :math:`\partial\Omega`, the objective can be written as

    .. math::

        \mathcal{L}_{\mathrm{problem}} =
        \frac{1}{N_{\Omega}} \sum_{i=1}^{N_{\Omega}}
        \lambda_{\Omega}^{i} \mathcal{L}
        \left( \mathcal{A}[\hat{\mathbf{u}}](\mathbf{x}_i) \right)
        + \frac{1}{N_{\partial\Omega}} \sum_{i=1}^{N_{\partial\Omega}}
        \lambda_{\partial\Omega}^{i} \mathcal{L}
        \left( \mathcal{B}[\hat{\mathbf{u}}](\mathbf{x}_i) \right),

    where :math:`\mathcal{L}` is the selected loss function, typically the
    mean squared error, and :math:`\lambda_{\Omega}^{i}` and
    :math:`\lambda_{\partial\Omega}^{i}` are the attention weights associated
    with the domain and boundary residuals, respectively.

    On each selected condition evaluation, including validation and testing,
    attention weights are updated using the residual vector's rowwise norm:

    .. math::

        \lambda_i^{k+1} = \gamma \lambda_i^k
            + \eta \frac{\|r_i\|_2}{\max_j \|r_j\|_2 + 10^{-12}},

    where :math:`r_i` is the residual at point :math:`i`, :math:`\gamma` is the
    decay rate, and :math:`\eta` is the learning rate used for the attention
    weight update.

    .. seealso::

        **Original reference**: Anagnostopoulos, S. J., Toscano, J. D.,
        Stergiopulos, N., & Karniadakis, G. E. (2024).
        *Residual-based attention and connection to information
        bottleneck theory in PINNs.*
        Computer Methods in Applied Mechanics and Engineering, 421, 116805.
        DOI: `10.1016/j.cma.2024.116805
        <https://doi.org/10.1016/j.cma.2024.116805>`_.

    :Example:

        >>> from pina.solver import RBAPhysicsInformedSingleModelSolver
        >>> from pina.problem.zoo import Poisson2DSquareProblem
        >>> # solver = RBAPhysicsInformedSingleModelSolver(
        >>> #     problem=problem, model=model, eta=0.001, gamma=0.999
        >>> # )
        >>> # Discretise problem domains before constructing this solver.
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
        eta=0.001,
        gamma=0.999,
        regularized_conditions=None,
    ):
        """
        Initialization of the :class:`RBAPhysicsInformedSingleModelSolver`
        class.

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
        :param eta: The learning rate for the residual-based attention weights
            update. Default is ``0.001``.
        :type eta: float | int
        :param float gamma: The decay factor for the residual-based attention
            mechanism. Default is ``0.999``.
        :param regularized_conditions: The names of the conditions that should
            receive attention regularization. If ``None``, all conditions are
            regularized. Default is ``None``.
        :type regularized_conditions: str | list[str] | tuple[str, ...] | None
        :raises ValueError: If attention parameters, condition names, or base
            solver components fail validation.
        :raises KeyError: If a selected domain-equation condition lacks sampled
            points under its condition name. Discretise before construction.
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

        # Initialize the residual-based attention components
        self._init_residual_attention_components(
            eta=eta,
            gamma=gamma,
            regularized_conditions=regularized_conditions,
        )
