"""Module for the multi-model mixin class."""

from pina._src.problem.inverse_problem import InverseProblem


class MultiModelMixin:
    """
    Mixin that defines optimizer configuration for solvers backed by multiple
    models. Provides properties to access the models, optimizers, and
    schedulers.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def configure_optimizers(self):
        """
        Configure the optimizers and schedulers for all models.

        :return: The optimizer and the scheduler
        :rtype: tuple[list[TorchOptimizer], list[TorchScheduler]]
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
        :rtype: list[torch.nn.Module]
        """
        return self._pina_models

    @property
    def optimizers(self):
        """
        The optimizers used by the solver.

        :return: The optimizers used by the solver.
        :rtype: list[TorchOptimizer]
        """
        return self._pina_optimizers

    @property
    def schedulers(self):
        """
        The schedulers used by the solver.

        :return: The schedulers used by the solver.
        :rtype: list[TorchScheduler]
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
