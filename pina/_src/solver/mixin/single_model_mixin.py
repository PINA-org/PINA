"""Module for the single-model mixin class."""

from pina._src.problem.inverse_problem import InverseProblem


class SingleModelMixin:
    """
    Mixin that defines the forward pass and optimizer configuration for solvers
    backed by exactly one model. Provides properties to access the single model,
    optimizer, and scheduler.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def forward(self, x):
        """
        The forward pass implementation for the single model, which simply
        evaluates the model on the input.

        :param x: The input data.
        :type x: torch.Tensor | LabelTensor | torch_geometric.data.Data |
            pina.graph.Graph
        :return: The output of the single model.
        :rtype: Any
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Configure the optimizer and scheduler for the single model.

        Reconstructs wrapper instances through their hooks. For inverse
        problems, adds the shared unknown parameters as an extra optimizer
        group before initializing the scheduler.

        :raises ValueError: If optimizer parameter-group validation fails.
        :raises TypeError: If a wrapped constructor rejects its options.
        :return: One-element lists of backend PyTorch instances, not wrappers.
        :rtype: tuple[list[torch.optim.Optimizer],
            list[torch.optim.lr_scheduler.LRScheduler]]
        """
        # Hook the optimizer to the model parameters
        self.optimizer.hook(self.model.parameters())

        # Add parameter group for inverse problems if needed
        if isinstance(self.problem, InverseProblem):
            self.optimizer.instance.add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )

        # Hook the scheduler to the optimizer
        self.scheduler.hook(self.optimizer)

        return ([self.optimizer.instance], [self.scheduler.instance])

    @property
    def model(self):
        """
        The single model used by the solver.

        :return: The single model used by the solver.
        :rtype: torch.nn.Module
        """
        return self._pina_models[0]

    @property
    def optimizer(self):
        """
        The optimizer used by the solver.

        :return: The optimizer used by the solver.
        :rtype: TorchOptimizer
        """
        return self._pina_optimizers[0]

    @property
    def scheduler(self):
        """
        The scheduler used by the solver.

        :return: The scheduler used by the solver.
        :rtype: TorchScheduler
        """
        return self._pina_schedulers[0]
