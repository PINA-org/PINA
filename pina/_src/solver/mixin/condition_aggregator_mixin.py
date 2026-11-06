"""Module for the condition aggregator mixin class."""

import torch

from pina import logger


class ConditionAggregatorMixin:
    """
    Mixin that logs per-condition scalar losses, weights them following the
    provided weighting scheme, and aggregates them into the total loss.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def batch_evaluation_step(self, batch, batch_idx):
        """
        Evaluate and aggregate the losses for all conditions in a batch.

        For each condition in the batch, this method computes the corresponding
        scalar loss, logs it using the condition name, and combines all
        condition losses into a single scalar loss through the configured
        weighting scheme.

        :param batch: The condition-name/data pairs supplied by the data module.
        :type batch: list[tuple[str, dict[str, Any] | pina.graph.Graph |
            torch_geometric.data.Data]]
        :param int batch_idx: The index of the current batch.
        :raises KeyError: If a condition name or saved weight is absent.
        :raises RuntimeError: If an empty batch cannot be stacked or a loss is
            not scalar enough for logging with ``item()``.
        :return: The weighted aggregate cast to a plain PyTorch tensor. Inverse
            parameters are clamped after loss computation and before weighting;
            repeated condition names overwrite earlier losses in this batch.
        :rtype: torch.Tensor
        """
        # Initialize a dictionary to hold the scalar losses for each condition
        condition_losses = {}
        logger.debug("Batch: %s", batch)

        # Loop through each condition in the batch and compute its scalar loss
        for condition_name, data in batch:

            # Compute the scalar loss for the current condition
            condition_losses[condition_name] = self._compute_condition_loss(
                condition=self.problem.conditions[condition_name],
                data=data,
                batch_idx=batch_idx,
            )

        # Clamp parameters - null operation if problem is not InverseProblem
        self._clamp_params()

        # Log the individual condition losses
        for name, value in condition_losses.items():
            self.log(
                name=f"{name}_loss",
                value=value.item(),
                **self.trainer.logging_kwargs,
            )

        # Aggregate into the total loss using the weighting scheme
        aggregated_loss = self.weighting.aggregate(condition_losses)

        return aggregated_loss.as_subclass(torch.Tensor)
