"""Module for the Scalar Weighting."""

from pina._src.weighting.base_weighting import BaseWeighting
from pina._src.core.utils import check_consistency


class ScalarWeighting(BaseWeighting):
    """
    Weighting strategy based on fixed scalar coefficients.

    This scheme assigns a constant multiplicative weight to each loss term,
    without adapting over time. The same weight can be applied to all terms,
    or distinct weights can be specified for individual conditions.

    :Example:

        >>> import torch
        >>> from pina.weighting import ScalarWeighting
        >>> # Uniform weighting
        >>> weighting = ScalarWeighting(weights=1.0)
        >>> losses = {"residual": torch.tensor(0.1), "data": torch.tensor(0.2)}
        >>> weights = weighting.update_weights(losses)
        >>> sum(weights[key] * value for key, value in losses.items())
        tensor(0.3000)
        >>> # Per-condition weighting
        >>> weighting = ScalarWeighting(weights={"residual": 0.5, "data": 2.0})
        >>> weights = weighting.update_weights(losses)
        >>> sum(weights[key] * value for key, value in losses.items())
        tensor(0.4500)
    """

    def __init__(self, weights):
        """
        Initialization of the :class:`ScalarWeighting` class.

        :param weights: The scalar weights associated with each loss term. It
            can be provided either as a single numeric value or as a dictionary.
            If a scalar is given, the same weight is applied to all loss terms.
            If a dictionary is provided, its keys represent the loss identifiers
            (e.g., conditions) and its values specify the corresponding weights.
            Loss terms not explicitly defined in the dictionary are assigned a
            default weight of ``1``. A supplied dictionary is retained by
            reference, and its keys and values are not validated.
        :type weights: float | int | dict[str, int | float | torch.Tensor]
        :raises ValueError: If the input weights are neither numeric nor a
            dictionary.
        """
        super().__init__(update_every_n_epochs=1, aggregator="sum")

        # Check consistency
        check_consistency([weights], (float, dict, int))

        # Initialization
        if isinstance(weights, dict):
            self.values = weights
            self.default_value_weights = 1
        else:
            self.values = {}
            self.default_value_weights = weights

    def update_weights(self, losses):
        """
        Look up constant weights for the supplied condition names.

        :param dict[str, torch.Tensor] losses: The condition losses. Only their
            names are used; values do not affect the weights.
        :return: A new mapping with configured or default weights. Does not
            modify the saved weight mapping when called directly.
        :rtype: dict[str, int | float | torch.Tensor]
        """
        return {
            condition: self.values.get(condition, self.default_value_weights)
            for condition in losses.keys()
        }
