"""Module for the manual optimization mixin class."""


class ManualOptimizationMixin:
    """
    Mixin that handles Lightning manual optimization loops, useful for solvers
    that require explicit control over optimization steps, such as those with
    multiple optimizers or custom training loops.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def _init_manual_optimization(self):
        """
        Disable Lightning's automatic optimization.
        """
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        """
        Backpropagate once and step all optimizers and schedulers per batch.

        Zeroes every optimizer before evaluation. Each scheduler receives
        ``step()`` without a metric or epoch, so metric-requiring schedulers and
        closure-requiring optimizers need a different training implementation.

        :param batch: The condition-name/data pairs supplied by the data module.
        :type batch: list[tuple[str, dict[str, Any] | pina.graph.Graph |
            torch_geometric.data.Data]]
        :param int batch_idx: The index of the current batch.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        # Zero the gradients of all optimizers
        for opt in self.optimizers:
            opt.instance.zero_grad()

        # Perform the forward pass and compute the loss
        loss = super().training_step(batch, batch_idx)

        # Perform the backward pass
        self.manual_backward(loss)

        # Step the optimizers and schedulers
        for opt, sched in zip(self.optimizers, self.schedulers):
            opt.instance.step()
            sched.instance.step()

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Keep Lightning's manual optimization progress counters in sync.

        This hook increments the completed optimization-step counter used by
        Lightning's manual optimization loop, then delegates to the parent
        implementation.

        :param torch.Tensor outputs: The loss of the training step.
        :param batch: The condition-name/data pairs supplied by the data module.
        :type batch: list[tuple[str, dict[str, Any] | pina.graph.Graph |
            torch_geometric.data.Data]]
        :param int batch_idx: The index of the current batch.
        """
        # Sync the manual optimization progress counters in Lightning's loop
        epoch_loop = self.trainer.fit_loop.epoch_loop
        epoch_loop.manual_optimization.optim_step_progress.total.completed += 1

        return super().on_train_batch_end(outputs, batch, batch_idx)
