"""Module for the base solver class."""

from abc import ABCMeta
import lightning
import torch
from pina._src.core.utils import labelize_forward, check_consistency
from pina._src.solver.solver_interface import SolverInterface
from pina._src.weighting.base_weighting import BaseWeighting
from pina._src.problem.inverse_problem import InverseProblem
from pina._src.optim.torch_optimizer import TorchOptimizer
from pina._src.optim.torch_scheduler import TorchScheduler
from pina._src.weighting.no_weighting import _NoWeighting
from pina._src.problem.base_problem import BaseProblem
from pina._src.loss.base_dual_loss import BaseDualLoss


class BaseSolver(SolverInterface, metaclass=ABCMeta):
    """
    Base class for all solvers, implementing common functionality.

    All solvers must inherit from this class and implement abstract methods
    defined in :class:`~pina.solver.solver_interface.SolverInterface`.

    This class is not meant to be instantiated directly."""

    # Define the available reductions for loss computation
    _AVAILABLE_REDUCTIONS = {
        "none": lambda x: x,
        "mean": lambda x: x.mean(),
        "sum": lambda x: x.sum(),
    }

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
        Initialization of the :class:`BaseSolver` class.

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
        :type loss: torch.nn.Module | BaseDualLoss
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
            Default is ``True``.
        :raises ValueError: If ``use_lt`` is not a boolean.
        :raises ValueError: If ``problem`` is not an instance of
            :class:`~pina.problem.base_problem.BaseProblem`.
        :raises ValueError: If one or more problem conditions are not supported
            by the solver.
        """
        # Reset the solver state
        self.reset()

        # Call the parent class initializer
        lightning.pytorch.LightningModule.__init__(self)

        # Check consistency
        check_consistency(use_lt, bool)
        check_consistency(problem, BaseProblem)
        for condition in problem.conditions.values():
            check_consistency(condition, self.accepted_conditions_types)

        # Store problem and use_lt
        self._pina_problem = problem
        self._use_lt = use_lt

        # Manage InverseProblem parameters if needed
        if isinstance(self.problem, InverseProblem):
            self._params = self.problem.unknown_parameters
            self._clamp_params = self._clamp_inverse_problem_params
        else:
            self._params = None
            self._clamp_params = lambda: None

        # Labelize the forward method if using LabelTensors
        if self.use_lt:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=problem.input_variables,
                output_variables=problem.output_variables,
            )

        # Initialize solver components (models, optimizers, schedulers)
        self._init_solver_components(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
        )

        # Initialize the weighting scheme and loss function
        self._init_weighting_and_loss(weighting=weighting, loss=loss)

    def forward(self, x):
        """
        Default forward pass: evaluates the first model on the input.

        :param x: The input data.
        :type x: torch.Tensor | LabelTensor | Data | Graph
        :return: The output of the first model.
        :rtype: torch.Tensor | LabelTensor | Data | Graph
        """
        return self._pina_models[0](x)

    def reset(self):
        """
        Reset the internal solver state, clearing the stored problem, models,
        optimizers and schedulers.
        """
        self._pina_problem = None
        self._pina_models = None
        self._pina_optimizers = None
        self._pina_schedulers = None

    def _clamp_inverse_problem_params(self):
        """
        Clamp the unknown parameters of an inverse problem. Each unknown
        parameter is constrained to lie within the corresponding bounds defined
        by the inverse problem parameter domain.
        """
        for v in self._params:
            self._params[v].data.clamp_(
                self.problem.unknown_parameter_domain.range[v][0],
                self.problem.unknown_parameter_domain.range[v][1],
            )

    def _init_weighting_and_loss(self, weighting=None, loss=None):
        """
        Initialize the weighting strategy and loss function.

        :param BaseWeighting weighting: The weighting strategy used to combine
            condition losses. If ``None``, no weighting is applied. Default is
            ``None``.
        :param loss: The loss function used to compute residual losses.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :type loss: torch.nn.Module | BaseDualLoss
        :raises ValueError: If ``weighting`` is not an instance of
            :class:`~pina.weighting.base_weighting.BaseWeighting`.
        :raises ValueError: If ``loss`` is not a valid PyTorch loss or
            :class:`~pina.loss.base_dual_loss.BaseDualLoss`.
        """
        # If no weighting schema is provided, use a default no-weighting schema
        if weighting is None:
            weighting = _NoWeighting()

        # Set default loss function to MSE if not provided
        if loss is None:
            loss = torch.nn.MSELoss()

        # Check consistency
        check_consistency(weighting, BaseWeighting)
        check_consistency(loss, (BaseDualLoss, torch.nn.modules.loss._Loss))

        # Store the weighting and loss function for use in the solver
        self._pina_weighting = weighting
        weighting._solver = self
        self._loss_fn = loss
        self._reduction = getattr(loss, "reduction", "mean")
        if hasattr(self._loss_fn, "reduction"):
            self._loss_fn.reduction = "none"

    def _init_solver_components(
        self,
        models,
        optimizers=None,
        schedulers=None,
    ):
        """
        Initialize the solver models, optimizers and schedulers.

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
        :raises ValueError: If ``models`` are not instances of
            :class:`torch.nn.Module`.
        :raises ValueError: If ``optimizers`` are not instances of
            :class:`~pina.optim.torch_optimizer.TorchOptimizer`.
        :raises ValueError: If ``schedulers`` are not instances of
            :class:`~pina.optim.torch_scheduler.TorchScheduler`.
        :raises ValueError: If the number of optimizers does not match that of
            models.
        :raises ValueError: If the number of schedulers does not match that of
            models.
        """

        # Helper function to map single items to lists if needed
        _to_list = lambda x: [x] if not isinstance(x, (list, tuple)) else x

        # Map models to list if a single model is provided
        models = _to_list(models)

        # Set default optimizers to Adam if not provided
        if optimizers is None:
            optimizers = [
                TorchOptimizer(torch.optim.Adam, lr=0.001)
                for _ in range(len(models))
            ]

        # Set default schedulers to ConstantLR if not provided
        if schedulers is None:
            schedulers = [
                TorchScheduler(torch.optim.lr_scheduler.ConstantLR, factor=1.0)
                for _ in range(len(models))
            ]

        # Map optimizers and schedulers to lists if single items are provided
        optimizers = _to_list(optimizers)
        schedulers = _to_list(schedulers)

        # Check consistency
        check_consistency(optimizers, TorchOptimizer)
        check_consistency(schedulers, TorchScheduler)
        check_consistency(models, torch.nn.Module)

        # Check that the number of optimizers matches the number of models
        if len(optimizers) != len(models):
            raise ValueError(
                "You must define one optimizer for each model."
                f"Got {len(models)} models, and {len(optimizers)} optimizers."
            )

        # Check that the number of schedulers matches the number of models
        if len(schedulers) != len(models):
            raise ValueError(
                "You must define one scheduler for each model."
                f"Got {len(models)} models, and {len(schedulers)} schedulers."
            )

        # Initialize the solver components
        self._pina_models = torch.nn.ModuleList(models)
        self._pina_optimizers = optimizers
        self._pina_schedulers = schedulers

    def training_step(self, batch, batch_idx):
        """
        Solver training step.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        loss = self.batch_evaluation_step(batch=batch, batch_idx=batch_idx)
        self.log(
            name="train_loss",
            value=loss.item(),
            batch_size=self.get_batch_size(batch),
            **self.trainer.logging_kwargs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Solver validation step.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        loss = self.batch_evaluation_step(batch=batch, batch_idx=batch_idx)
        self.log(
            name="val_loss",
            value=loss.item(),
            batch_size=self.get_batch_size(batch),
            **self.trainer.logging_kwargs,
        )
        return loss

    def test_step(self, batch, batch_idx):
        """
        Solver test step.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        loss = self.batch_evaluation_step(batch=batch, batch_idx=batch_idx)
        self.log(
            name="test_loss",
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
        condition_name = condition.name if hasattr(condition, "name") else None

        # Compute the tensor loss from the residual tensor
        condition_tensor_loss = self._loss_from_residual(condition_name)

        # Additive regularization hook, e.g. gradient-enhanced penalty
        condition_tensor_loss = self._regularize_condition_loss(
            condition_tensor_loss=condition_tensor_loss,
            condition_name=condition_name,
            data=data,
            batch_idx=batch_idx,
        )

        # Multiplicative weighting hook, e.g. residual-based attention
        condition_tensor_loss = self._weight_condition_loss(
            condition_tensor_loss=condition_tensor_loss,
            condition_name=condition_name,
            data=data,
            batch_idx=batch_idx,
        )

        # Compute the scalar loss from the tensor loss and return it
        condition_scalar_loss = self._apply_reduction(condition_tensor_loss)

        return condition_scalar_loss

    def _prepare_condition_data(self, data):
        """
        Prepare the condition data for loss computation. This method can be
        overridden by mixins to implement specific data preparation steps, such
        as enabling gradient tracking for inputs in gradient-enhanced solvers.

        :param dict data: The original condition data.
        :return: The prepared condition data.
        :rtype: dict
        """
        return data

    def _regularize_condition_loss(
        self,
        condition_tensor_loss,
        condition_name,
        data,
        batch_idx,
    ):
        """
        Additive regularization hook for the condition loss. Override this to
        implement additive regularization strategies such as gradient-enhanced
        penalties.

        :param condition_tensor_loss: The original tensor loss for the
            condition.
        :type condition_tensor_loss: torch.Tensor | LabelTensor
        :param str condition_name: The name of the condition.
        :param dict data: The data corresponding to the condition.
        :param int batch_idx: The index of the current batch.
        :return: The regularized tensor loss for the condition.
        :rtype: torch.Tensor | LabelTensor
        """
        return condition_tensor_loss

    def _weight_condition_loss(
        self,
        condition_tensor_loss,
        condition_name,
        data,
        batch_idx,
    ):
        """
        Multiplicative weighting hook for the condition loss. Override this to
        implement multiplicative weighting strategies such as residual-based
        attention.

        :param condition_tensor_loss: The tensor loss for the condition.
        :type condition_tensor_loss: torch.Tensor | LabelTensor
        :param str condition_name: The name of the condition.
        :param dict data: The data corresponding to the condition.
        :param int batch_idx: The index of the current batch.
        :return: The weighted tensor loss for the condition.
        :rtype: torch.Tensor | LabelTensor
        """
        return condition_tensor_loss

    def _loss_from_residual(self, condition_name=None):
        """
        Compute the tensor loss from the residual tensor.

        :param str condition_name: The name of the condition.
        :return: The tensor loss computed from the residual tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        # Compute the loss tensor and appply reduction
        return self._loss_fn(
            self.residual_tensor, torch.zeros_like(self.residual_tensor)
        )

    def _apply_reduction(self, value):
        """
        Apply the specified reduction to the loss tensor.

        :param value: The loss tensor to reduce.
        :type value: torch.Tensor | LabelTensor
        :return: The reduced loss.
        :rtype: torch.Tensor | LabelTensor
        """
        # Get the reduction function based on the specified reduction type
        reduction_fn = self._AVAILABLE_REDUCTIONS.get(self._reduction)

        # If the reduction type is not supported, raise an error
        if reduction_fn is None:
            raise ValueError(
                f"Unsupported reduction '{self._reduction}'. "
                f"Available options include {self._AVAILABLE_REDUCTIONS.keys()}"
            )

        return reduction_fn(value)

    @staticmethod
    def get_batch_size(batch):
        """
        Get the batch size.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The size of the batch.
        :rtype: int
        """
        return sum(len(data[1]["input"]) for data in batch)

    @property
    def problem(self):
        """
        The problem instance.

        :return: The problem instance.
        :rtype: :class:`~pina.problem.base_problem.BaseProblem`
        """
        return self._pina_problem

    @property
    def use_lt(self):
        """
        Using LabelTensors as input during training.

        :return: The use_lt attribute.
        :rtype: bool
        """
        return self._use_lt

    @property
    def weighting(self):
        """
        The weighting schema used by the solver.

        :return: The weighting schema used by the solver.
        :rtype: :class:`~pina.weighting.base_weighting.BaseWeighting`
        """
        return self._pina_weighting

    @property
    def loss(self):
        """
        The element-wise loss module used by the solver.

        :return: The element-wise loss module used by the solver.
        :rtype: torch.nn.Module
        """
        return self._loss_fn
