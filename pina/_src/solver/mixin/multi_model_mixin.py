"""Module for the multi-model mixin class."""

import torch
from pina._src.problem.inverse_problem import InverseProblem


class MultiModelMixin:
    """
    Mixin that defines the forward pass and optimizer configuration for solvers
    backed by multiple models. Provides properties to access the models,
    optimizers, and schedulers.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def forward(self, x):
        """
        The forward pass implementation that evaluates all models and returns a
        stacked tensor of their outputs.

        :param x: The input data.
        :type x: torch.Tensor | LabelTensor | torch_geometric.data.Data |
            pina.graph.Graph
        :raises RuntimeError: If model output shapes differ or no models exist.
        :raises TypeError: If a model returns a non-tensor object.
        :return: Tensor outputs stacked along a new leading model axis, with
            shape ``(n_models, *model_output.shape)``. No averaging is applied.
        :rtype: torch.Tensor | LabelTensor
        """
        return torch.stack(
            [self.models[idx](x) for idx in range(self.num_models)]
        )

    def configure_optimizers(self):
        """
        Configure the optimizers and schedulers for all models.

        Reconstructs all wrapper instances through their hooks. For inverse
        problems, adds the same unknown parameter objects as an extra group
        to every optimizer.

        :raises ValueError: If optimizer parameter-group validation fails.
        :raises TypeError: If a wrapped constructor rejects its options.
        :return: Lists of backend instances in model order, not wrappers.
        :rtype: tuple[list[torch.optim.Optimizer],
            list[torch.optim.lr_scheduler.LRScheduler]]
        """
        # Iterate over models, optimizers, and schedulers to hook them together
        for optimizer, scheduler, model in zip(
            self.optimizers, self.schedulers, self.models
        ):

            # Hook the optimizer to the model parameters
            optimizer.hook(model.parameters())

            # Add parameter group for inverse problems if needed
            if isinstance(self.problem, InverseProblem):
                optimizer.instance.add_param_group(
                    {
                        "params": [
                            self._params[var]
                            for var in self.problem.unknown_variables
                        ]
                    }
                )

            # Hook the scheduler to the optimizer
            scheduler.hook(optimizer)

        return (
            [optimizer.instance for optimizer in self.optimizers],
            [scheduler.instance for scheduler in self.schedulers],
        )

    @property
    def models(self):
        """
        The models used by the solver.

        :return: The models used by the solver.
        :rtype: torch.nn.ModuleList
        """
        return self._pina_models

    @property
    def optimizers(self):
        """
        The optimizers used by the solver.

        :return: The optimizers used by the solver.
        :rtype: list[pina.optim.TorchOptimizer] |
            tuple[pina.optim.TorchOptimizer, ...]
        """
        return self._pina_optimizers

    @property
    def schedulers(self):
        """
        The schedulers used by the solver.

        :return: The schedulers used by the solver.
        :rtype: list[pina.optim.TorchScheduler] |
            tuple[pina.optim.TorchScheduler, ...]
        """
        return self._pina_schedulers

    @property
    def num_models(self):
        """
        The number of models used by the solver.

        :return: The number of models used by the solver.
        :rtype: int
        """
        return len(self.models)
