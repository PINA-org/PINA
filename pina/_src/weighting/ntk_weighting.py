"""Module for Neural Tangent Kernel Class"""

import torch
from pina._src.weighting.base_weighting import BaseWeighting
from pina._src.core.utils import check_consistency, in_range


class NeuralTangentKernelWeighting(BaseWeighting):
    """
    The Neural Tangent Kernel (NTK) weighting strategy.

    This weighting scheme dynamically adjusts the contribution of each loss term
    during training by leveraging gradient information with respect to the model
    parameters. For each loss component, the norm of its gradient is computed
    and used to derive relative importance weights. The resulting weights are
    smoothed over time using an exponential moving average controlled by the
    parameter ``alpha``.

    .. seealso::

        **Original reference**: Wang, Sifan, Xinling Yu, and
        Paris Perdikaris. *When and why PINNs fail to train:
        A neural tangent kernel perspective*. Journal of
        Computational Physics 449 (2022): 110768.
        DOI: `10.1016 <https://doi.org/10.1016/j.jcp.2021.110768>`_.

    :Example:

        >>> from pina.weighting import NeuralTangentKernelWeighting
        >>> weighting = NeuralTangentKernelWeighting(alpha=0.5)
        >>> # Typically used within a PINA solver:
        >>> # solver = PINN(problem=problem, weighting=weighting)
    """

    def __init__(self, update_every_n_epochs=1, alpha=0.5):
        """
        Initialization of the :class:`NeuralTangentKernelWeighting` class.

        :param int update_every_n_epochs: The number of training epochs between
            eligible epochs. Every aggregation call in an eligible epoch
            recomputes weights, including epoch zero.
            Default is ``1``.
        :param float alpha: The parameter controlling the exponential moving
            average of the weights. It must be in the range [0, 1], where a
            value of ``0.0`` uses each gradient norm divided by their sum, and
            ``1.0`` preserves saved weights, falling back to ``1`` for unseen
            conditions. Gradients are still computed at either endpoint.
            Default is ``0.5``.
        :raises AssertionError: If the update interval is not a positive
            integer.
        :raises ValueError: If ``alpha`` is not a float.
        :raises ValueError: If ``alpha`` is not between 0.0 and 1.0 (inclusive).
        """
        super().__init__(
            update_every_n_epochs=update_every_n_epochs, aggregator="sum"
        )

        # Check consistency
        check_consistency(alpha, float)
        if not in_range(alpha, [0.0, 1.0], strict=False):
            raise ValueError(
                "The alpha parameter must be between 0.0 and 1.0 (inclusive)."
                f" Got {alpha}."
            )

        # Initialization
        self.alpha = alpha
        self.weights = {}

    def update_weights(self, losses):
        """
        Smooth normalized model-parameter gradient norms with saved weights.

        :param dict[str, torch.Tensor] losses: Differentiable scalar or
            one-element losses connected to trainable parameters of the attached
            solver's ``model``. Unused parameter gradients are omitted.
        :raises AttributeError: If no solver is attached or it has no ``model``.
        :raises RuntimeError: If gradients cannot be computed or no trainable
            parameters exist.
        :raises ValueError: If a loss has no usable model-parameter gradients.
        :return: A new mapping with
            ``alpha * saved + (1 - alpha) * norm / total`` for each condition;
            unseen saved values default to ``1``. Norms use
            the supplied condition subset and a retained autograd graph without
            higher-order tracking. Does not save the returned weights. A zero
            total norm is unguarded and can produce NaN, even at ``alpha=1.0``.
        :rtype: dict[str, torch.Tensor]
        """
        # Get model parameters and define a dictionary to store the norms
        params = [p for p in self.solver.model.parameters() if p.requires_grad]
        norms = {}

        # Iterate over conditions
        for condition, loss in losses.items():

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )

            # Compute norms
            norms[condition] = torch.cat(
                [g.flatten() for g in grads if g is not None]
            ).norm()

        return {
            condition: self.alpha * self.last_saved_weights().get(condition, 1)
            + (1 - self.alpha) * norms[condition] / sum(norms.values())
            for condition in losses
        }
