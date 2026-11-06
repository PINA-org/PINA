"""Module for the physics-informed mixin class."""

import torch


class PhysicsInformedMixin:
    """
    Mixin that enables physics-informed training by ensuring gradients are
    enabled during validation and testing, which is necessary for computing
    physics residuals.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        """
        Solver validation step.

        :param batch: The condition-name/data pairs supplied by the data module.
        :type batch: list[tuple[str, dict[str, Any] | pina.graph.Graph |
            torch_geometric.data.Data]]
        :param int batch_idx: The index of the current batch.
        :return: The validation loss evaluated with autograd enabled. This does
            not override PyTorch inference mode or enable input tracking itself.
        :rtype: torch.Tensor
        """
        return super().validation_step(batch, batch_idx)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        """
        Solver test step.

        :param batch: The condition-name/data pairs supplied by the data module.
        :type batch: list[tuple[str, dict[str, Any] | pina.graph.Graph |
            torch_geometric.data.Data]]
        :param int batch_idx: The index of the current batch.
        :return: The test loss evaluated with autograd enabled. This does not
            override PyTorch inference mode or enable input tracking itself.
        :rtype: torch.Tensor
        """
        return super().test_step(batch, batch_idx)
