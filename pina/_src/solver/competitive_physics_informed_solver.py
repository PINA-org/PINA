"""Module for the competitive physics-informed multi-model solver."""

import copy
from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.solver.multi_model_solver import MultiModelSolver
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class CompetitivePhysicsInformedSolver(PhysicsInformedMixin, MultiModelSolver):
    r"""
    Multi-model solver for competitive physics-informed learning problems.

    This solver approximates the solution of a differential problem using a
    trainable model together with a discriminator network. It is intended for
    problems whose conditions may include supervised data, equation residuals
    evaluated on input points, and equation residuals sampled from domains.

    Given a model :math:`\mathcal{M}`, the predicted solution is

    .. math::

        \hat{\mathbf{u}}(\mathbf{x}) = \mathcal{M}(\mathbf{x}).

    The discriminator :math:`D` assigns pointwise weights to the residuals,
    encouraging the model to focus on regions where the approximation performs
    poorly. The model parameters are optimized by minimizing the loss, while the
    discriminator parameters are optimized by maximizing it.

    For a problem with governing equation operator :math:`\mathcal{A}` in the
    domain :math:`\Omega` and boundary operator :math:`\mathcal{B}` on the
    boundary :math:`\partial\Omega`, the competitive objective can be written as

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{N_{\Omega}}
        \sum_{i=1}^{N_{\Omega}} \mathcal{L}
        \left(D(\mathbf{x}_i)\mathcal{A}[\hat{\mathbf{u}}](\mathbf{x}_i)\right)
        +\frac{1}{N_{\partial\Omega}} \sum_{i=1}^{N_{\partial\Omega}}
        \mathcal{L}
        \left(D(\mathbf{x}_i)\mathcal{B}[\hat{\mathbf{u}}](\mathbf{x}_i)\right),

    where :math:`D` is the discriminator network and :math:`\mathcal{L}` is the
    selected loss function, typically the mean squared error.

    The model and discriminator are trained through a min-max problem:

    .. math::

        \min_{\theta} \max_{\phi} \mathcal{L}_{\mathrm{problem}},

    where :math:`\theta` denotes the model parameters and :math:`\phi` denotes
    the discriminator parameters.

    .. seealso::

        **Original reference**: Zeng, Q., Kothari, P., Chou, E., & Masi, G.
        (2022).
        *Competitive physics informed networks.*
        International Conference on Learning Representations, ICLR 2022.
        `OpenReview Preprint <https://openreview.net/forum?id=z9SIj-IM7tn>`_.

    :Example:

        >>> from pina.solver import CompetitivePhysicsInformedSolver
        >>> from pina.problem import PoissonProblem
        >>> # solver = CompetitivePhysicsInformedSolver(
        >>> #     problem=problem, model=model
        >>> # )
        >>> # solver.train()
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
        discriminator=None,
        optimizer_model=None,
        optimizer_discriminator=None,
        scheduler_model=None,
        scheduler_discriminator=None,
        weighting=None,
        loss=None,
    ):
        """
        Initialization of the :class:`CompetitivePhysicsInformedSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param torch.nn.Module model: The model used by the solver.
        :param torch.nn.Module discriminator: The discriminator used by the
            solver. If ``None``, a deep copy of the model is used as
            discriminator. Default is ``None``.
        :param TorchOptimizer optimizer_model: The optimizer of the main model.
            If ``None``, the ``torch.optim.Adam`` optimizer with a learning rate
            of ``0.001`` is used. Default is ``None``.
        :param TorchOptimizer optimizer_discriminator: The optimizer of the
            discriminator. If ``None``, the ``torch.optim.Adam`` optimizer with
            a learning rate of ``0.001`` is used. Default is ``None``.
        :param TorchScheduler scheduler_model: The scheduler of the main model.
            If ``None``, the ``torch.optim.lr_scheduler.ConstantLR`` scheduler
            with a factor of ``1.0`` is used. Default is ``None``.
        :param TorchScheduler scheduler_discriminator: The scheduler of the
            discriminator.
            If ``None``, the ``torch.optim.lr_scheduler.ConstantLR`` scheduler
            with a factor of ``1.0`` is used. Default is ``None``.
        :param BaseWeighting weighting: The weighting strategy used to combine
            condition losses. If ``None``, no weighting is applied. Default is
            ``None``.
        :param loss: The loss function used to compute residual losses.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :raises ValueError: If ``weight_function`` is not a ``torch.nn.Module``.
        :raises ValueError: If not all domains have been discretised.
        """
        # Initialize the discriminator if not provided
        if discriminator is None:
            discriminator = copy.deepcopy(model)

        # Prepare optimizers
        optimizers = (
            [optimizer_model, optimizer_discriminator]
            if any(
                o is not None
                for o in (optimizer_model, optimizer_discriminator)
            )
            else None
        )

        # Prepare schedulers
        schedulers = (
            [scheduler_model, scheduler_discriminator]
            if any(
                s is not None
                for s in (scheduler_model, scheduler_discriminator)
            )
            else None
        )

        # Initialize the base solver
        MultiModelSolver.__init__(
            self,
            problem=problem,
            models=[model, discriminator],
            optimizers=optimizers,
            schedulers=schedulers,
            weighting=weighting,
            loss=loss,
            use_lt=True,
        )

    def training_step(self, batch, batch_idx):
        """
        Solver training step.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        # Zero the gradients of the model optimizer and compute the loss
        self.optimizer_model.instance.zero_grad()
        loss = self.batch_evaluation_step(batch, batch_idx)

        # Perform the backward pass and complete a step for the model
        self.manual_backward(loss)
        self.optimizer_model.instance.step()
        self.scheduler_model.instance.step()

        # Zero the gradients of the discriminator optimizer and compute the loss
        self.optimizer_discriminator.instance.zero_grad()
        loss = self.batch_evaluation_step(batch, batch_idx)

        # Perform the backward pass and complete a step for the discriminator
        self.manual_backward(-loss)
        self.optimizer_discriminator.instance.step()
        self.scheduler_discriminator.instance.step()

        # Log the training loss
        self.log(
            name="train_loss",
            value=loss.item(),
            batch_size=self.get_batch_size(batch),
            **self.trainer.logging_kwargs,
        )

        return loss

    def _compute_condition_loss(self, condition, data, batch_idx):
        """
        Compute the scalar loss for a given condition and its data.

        :param BaseCondition condition: The condition for which to compute the
            loss.
        :param dict data: The data corresponding to the condition.
        :param int batch_idx: The index of the current batch.
        :return: The scalar loss for the condition.
        :rtype: torch.Tensor
        """
        # Clone the input tensor if it exists to avoid in-place modifications
        if "input" in data and hasattr(data["input"], "clone"):
            data = dict(data)
            data["input"] = data["input"].clone()

        # Prepare condition data, e.g. by enabling gradient for regularizations
        data = self._prepare_condition_data(data=data)

        # Compute and store the residual tensor for the condition
        self.residual_tensor = condition.evaluate(data, self)

        # Compute the discriminator bets for the current condition
        discriminator_input = data["input"][self.problem.input_variables]
        discriminator_bets = self.discriminator(discriminator_input)

        # Weight the residual tensor using the discriminator bets
        self.residual_tensor = self.residual_tensor * discriminator_bets

        # Retrieve condition name for more complex weighting schemes
        condition_name = condition.name if hasattr(condition, "name") else None

        # Compute the tensor loss from the residual tensor
        condition_tensor_loss = self._loss_from_residual(condition_name)

        # Optional regularization hook, e.g gradient-enhanced or residual-based
        condition_tensor_loss = self._regularize_condition_loss(
            condition_tensor_loss=condition_tensor_loss,
            condition_name=condition_name,
            data=data,
            batch_idx=batch_idx,
        )

        # Compute the scalar loss from the tensor loss and return it
        condition_scalar_loss = self._apply_reduction(condition_tensor_loss)

        return condition_scalar_loss

    @property
    def model(self):
        """
        The single model used by the solver.

        :return: The single model used by the solver.
        :rtype: torch.nn.Module
        """
        return self._pina_models[0]

    @property
    def discriminator(self):
        """
        The discriminator used by the solver.

        :return: The discriminator used by the solver.
        :rtype: torch.nn.Module
        """
        return self._pina_models[1]

    @property
    def optimizer_model(self):
        """
        The optimizer for the model used by the solver.

        :return: The optimizer for the model used by the solver.
        :rtype: TorchOptimizer
        """
        return self.optimizers[0]

    @property
    def optimizer_discriminator(self):
        """
        The optimizer for the discriminator used by the solver.

        :return: The optimizer for the discriminator used by the solver.
        :rtype: TorchOptimizer
        """
        return self.optimizers[1]

    @property
    def scheduler_model(self):
        """
        The scheduler for the model used by the solver.

        :return: The scheduler for the model used by the solver.
        :rtype: TorchScheduler
        """
        return self.schedulers[0]

    @property
    def scheduler_discriminator(self):
        """
        The scheduler for the discriminator used by the solver.

        :return: The scheduler for the discriminator used by the solver.
        :rtype: TorchScheduler
        """
        return self.schedulers[1]
