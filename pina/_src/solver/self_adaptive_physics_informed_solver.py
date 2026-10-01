"""Module for the self-adaptive physics-informed multi-model solver."""

import torch
from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.solver.multi_model_solver import MultiModelSolver
from pina._src.core.utils import check_consistency
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class SelfAdaptivePhysicsInformedSolver(PhysicsInformedMixin, MultiModelSolver):
    r"""
    Multi-model solver for self-adaptive physics-informed learning problems.

    This solver approximates the solution of a differential problem using a
    trainable model together with condition-wise self-adaptive weights. It is
    intended for problems whose conditions may include supervised data, equation
    residuals evaluated on input points, and equation residuals sampled from
    domains.

    Given a model :math:`\mathcal{M}`, the predicted solution is

    .. math::

        \hat{\mathbf{u}}(\mathbf{x}) = \mathcal{M}(\mathbf{x}).

    For each condition, the solver introduces trainable pointwise weights. These
    weights are passed through a user-defined weight function :math:`m`,
    typically chosen to keep the effective weights bounded or positive. The
    resulting weighted objective encourages the model to focus on regions where
    the residual is larger.

    For a problem with governing equation operator :math:`\mathcal{A}` in the
    domain :math:`\Omega` and boundary operator :math:`\mathcal{B}` on the
    boundary :math:`\partial\Omega`, the objective can be written as

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{N_{\Omega}}
        \sum_{i=1}^{N_{\Omega}} m(\lambda_{\Omega}^{i}) \mathcal{L}
        \left( \mathcal{A}[\hat{\mathbf{u}}](\mathbf{x}_i) \right)
        + \frac{1}{N_{\partial\Omega}} \sum_{i=1}^{N_{\partial\Omega}}
        m(\lambda_{\partial\Omega}^{i})
        \mathcal{L} \left( \mathcal{B}[\hat{\mathbf{u}}](\mathbf{x}_i) \right),

    where :math:`\lambda_{\Omega}^{i}` and :math:`\lambda_{\partial\Omega}^{i}`
    are the self-adaptive weights associated with points in :math:`\Omega` and
    :math:`\partial\Omega`, respectively, and :math:`\mathcal{L}` is the
    selected loss function, typically the mean squared error.

    The model parameters and the self-adaptive weights are optimized through a
    min-max problem:

    .. math::

        \min_{\theta} \max_{\lambda} \mathcal{L}_{\mathrm{problem}},

    where :math:`\theta` denotes the model parameters and :math:`\lambda`
    denotes the collection of self-adaptive weights.

    .. seealso::

        **Original reference**: McClenny, L. D., & Braga-Neto, U. M. (2023).
        *Self-adaptive physics-informed neural networks.*
        Journal of Computational Physics, 474, 111722.
        DOI: `10.1016/j.jcp.2022.111722
        <https://doi.org/10.1016/j.jcp.2022.111722>`_.

    :Example:

        >>> from pina.solver import SelfAdaptivePhysicsInformedSolver
        >>> from pina.problem import PoissonProblem
        >>> # solver = SelfAdaptivePhysicsInformedSolver(
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
        weight_function=torch.nn.Sigmoid(),
        optimizer_model=None,
        optimizer_weights=None,
        scheduler_model=None,
        scheduler_weights=None,
        weighting=None,
        loss=None,
    ):
        """
        Initialization of the :class:`SelfAdaptivePhysicsInformedSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param torch.nn.Module model: The model used by the solver.
        :param torch.nn.Module weight_function: The weight function used to
            compute self-adaptive weights. Default is ``torch.nn.Sigmoid()``.
        :param TorchOptimizer optimizer_model: The optimizer of the main model.
            If ``None``, the ``torch.optim.Adam`` optimizer with a learning rate
            of ``0.001`` is used. Default is ``None``.
        :param TorchOptimizer optimizer_weights: The optimizer of the
            self-adaptive weights. If ``None``, the ``torch.optim.Adam``
            optimizer with a learning rate of ``0.001`` is used.
            Default is ``None``.
        :param TorchScheduler scheduler_model: The scheduler of the main model.
            If ``None``, the ``torch.optim.lr_scheduler.ConstantLR`` scheduler
            with a factor of ``1.0`` is used. Default is ``None``.
        :param TorchScheduler scheduler_weights: The scheduler of the
            self-adaptive weights. If ``None``, the
            ``torch.optim.lr_scheduler.ConstantLR`` scheduler with a factor of
            ``1.0`` is used. Default is ``None``.
        :param BaseWeighting weighting: The weighting strategy used to combine
            condition losses. If ``None``, no weighting is applied. Default is
            ``None``.
        :param loss: The loss function used to compute residual losses.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :raises ValueError: If ``weight_function`` is not a ``torch.nn.Module``.
        :raises ValueError: If not all domains have been discretised.
        """
        # Check consistency
        check_consistency(weight_function, torch.nn.Module)

        # Check that all domains have been discretised
        if not problem.are_all_domains_discretised:
            raise ValueError(
                "All domains must be discretised before initializing the "
                "solver."
            )

        # Compute the number of points for each condition
        num_points = {
            cond: (
                problem._discretised_domains[cond].shape[0]
                if isinstance(problem.conditions[cond], DomainEquationCondition)
                else problem.conditions[cond].data.input.shape[0]
            )
            for cond in problem.conditions
        }

        # Initialize weights container and per-condition parameters
        weights = torch.nn.Module()

        # Attach the weight function as a submodule
        weights.func = weight_function

        # Register a torch.nn.Parameter for each condition to store the weights
        for cond in problem.conditions:
            p = torch.nn.Parameter(torch.zeros(num_points[cond], 1))
            setattr(weights, cond, p)

        # Prepare optimizers
        optimizers = (
            [optimizer_model, optimizer_weights]
            if any(o is not None for o in (optimizer_model, optimizer_weights))
            else None
        )

        # Prepare schedulers
        schedulers = (
            [scheduler_model, scheduler_weights]
            if any(s is not None for s in (scheduler_model, scheduler_weights))
            else None
        )

        # Initialize the base solver
        MultiModelSolver.__init__(
            self,
            problem=problem,
            models=[model, weights],
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
        # Zero the gradients of weights optimizer and compute the loss
        self.optimizer_weights.instance.zero_grad()
        loss = self.batch_evaluation_step(batch, batch_idx)

        # Perform the backward pass and complete a step for the weights
        self.manual_backward(-loss)
        self.optimizer_weights.instance.step()
        self.scheduler_weights.instance.step()

        # Zero the gradients of model optimizer and compute the loss again
        self.optimizer_model.instance.zero_grad()
        loss = self.batch_evaluation_step(batch, batch_idx)

        # Perform the backward pass and complete a step for the model
        self.manual_backward(loss)
        self.optimizer_model.instance.step()
        self.scheduler_model.instance.step()

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

        # Retrieve condition name for more complex weighting schemes
        condition_name = condition.name

        # Apply the activation function to the condition-specific weights
        weight_param = getattr(self.weights, condition_name)
        weight_tensor = self.weights.func(weight_param)

        # Compute the tensor loss from the residual tensor
        condition_tensor_loss = self._loss_from_residual(condition_name)

        # Optional regularization hook, e.g gradient-enhanced or residual-based
        condition_tensor_loss = self._regularize_condition_loss(
            condition_tensor_loss=condition_tensor_loss,
            condition_name=condition_name,
            data=data,
            batch_idx=batch_idx,
        )

        # Get the correct indices to retrieve the weights for the current batch
        len_residuals = self.residual_tensor.shape[0]

        # Get the total number of points, together with the start / end indices
        total_points = weight_param.shape[0]
        start = (batch_idx * len_residuals) % total_points
        end = start + len_residuals

        # Retrieve the weights for the current batch using modular indexing
        idx = torch.arange(start, end, device=self.residual_tensor.device)
        idx = idx % total_points

        # Compute the scalar loss from the tensor loss and return it
        condition_scalar_loss = self._apply_reduction(
            condition_tensor_loss * weight_tensor[idx]
        )

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
    def weights(self):
        """
        The self-adaptive weights used by the solver.

        :return: The self-adaptive weights used by the solver.
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
    def optimizer_weights(self):
        """
        The optimizer for the weights used by the solver.

        :return: The optimizer for the weights used by the solver.
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
    def scheduler_weights(self):
        """
        The scheduler for the weights used by the solver.

        :return: The scheduler for the weights used by the solver.
        :rtype: TorchScheduler
        """
        return self.schedulers[1]
