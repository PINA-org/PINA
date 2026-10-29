"""Module for the R3Refinement callback."""

import torch
from pina._src.core.utils import check_consistency
from pina._src.core.label_tensor import LabelTensor
from pina._src.loss.dual_loss_interface import DualLossInterface
from pina._src.callback.refinement.base_refinement import BaseRefinement


class R3Refinement(BaseRefinement):
    """
    Refinement strategy based on the R3 (Retain-Resample-Release) algorithm.

    This method adaptively updates collocation points by retaining points with
    high residuals, resampling new points in the domain, releasing points with
    low residuals.

    The objective is to concentrate sampling in regions where the PDE residual
    is large, improving training efficiency and solution accuracy.

    .. seealso::

        **Original Reference**: Daw, Arka, et al. (2023).
        *Mitigating Propagation Failures in Physics-informed Neural Networks
        using Retain-Resample-Release (R3) Sampling*.
        DOI: `10.48550/arXiv.2207.02338
        <https://doi.org/10.48550/arXiv.2207.02338>`_

    :Example:

        >>> r3 = R3Refinement(sample_every=5)
    """

    def __init__(
        self,
        sample_every,
        residual_loss=torch.nn.L1Loss,
        condition_to_update=None,
    ):
        """
        Initialization of the :class:`R3Refinement` class.

        :param int sample_every: The number of epochs between successive
            refinement steps.
        :param residual_loss: The loss used to evaluate residual magnitude. Must
            be a subclass of :class:`torch.nn.Module` or
            :class:`pina.loss.DualLossInterface`.
            Default is :class:`torch.nn.L1Loss`.
        :type residual_loss: DualLossInterface | torch.nn.modules.loss._Loss
        :param condition_to_update: The condition(s) to be updated during
            refinement. If ``None``, all conditions associated with a domain are
            updated. Default is ``None``.
        :type condition_to_update: str | list[str] | tuple[str]
        :raises ValueError: If the condition_to_update is neither a string nor
            an iterable of strings.
        :raises ValueError: If the residual_loss is not a valid loss class.
        """
        super().__init__(sample_every, condition_to_update)

        # Check consistency
        check_consistency(
            residual_loss,
            (DualLossInterface, torch.nn.modules.loss._Loss),
            subclass=True,
        )

        # Store the loss function for computing residuals during sampling
        self.loss_fn = residual_loss(reduction="none")

    def sample(self, current_points, condition_name, solver):
        """
        Generate new sample points for a given condition.

        :param LabelTensor current_points: The existing points in the domain.
        :param str condition_name: The identifier of the condition to refine.
        :param BaseSolver solver: The solver used for sampling decisions.
        :return: Newly sampled points.
        :rtype: LabelTensor
        """
        # Retrieve condition and current points
        device = solver.trainer.strategy.root_device
        condition = solver.problem.conditions[condition_name]
        current_points = current_points.to(device).requires_grad_(True)

        # Compute residuals for the given condition
        target = condition.evaluate({"input": current_points}, solver)
        residuals = self.loss_fn(target, torch.zeros_like(target)).mean(
            dim=tuple(range(1, target.ndim))
        )

        # Retrieve domain and initial population size
        domain_name = solver.problem.conditions[condition_name].domain
        domain = solver.problem.domains[domain_name]
        num_old_points = self.initial_population_size[condition_name]

        # Select points with residual above the mean
        mask = (residuals >= residuals.mean()).flatten()
        high_residual_pts = current_points[mask].detach()
        high_residual_pts.labels = current_points.labels

        # Sample new points to maintain the initial population size
        num_new_pts = max(num_old_points - len(high_residual_pts), 0)
        samples = domain.sample(num_new_pts, "random").to(device)

        return LabelTensor.cat([high_residual_pts, samples])
