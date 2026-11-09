"""Module for the SinkhornLoss class."""

import torch
from pina._src.loss.base_dual_loss import BaseDualLoss
from pina._src.core.utils import check_consistency, check_positive_integer


class SinkhornLoss(BaseDualLoss):
    r"""
    Implementation of the Sinkhorn loss measuring the entropy-regularized
    optimal transport distance between two empirical distributions.

    Given an input tensor :math:`x` with :math:`N` samples and a target tensor
    :math:`y` with :math:`M` samples, both in :math:`\mathbb{R}^D`, the loss is
    defined through the entropy-regularized optimal transport problem:

    .. math::

        W_\varepsilon(\mu, \nu) = \min_{\pi \in \Pi(\mu, \nu)}
        \langle C, \pi \rangle - \varepsilon H(\pi)

    where :math:`\mu` and :math:`\nu` are the empirical distributions associated
    with :math:`x` and :math:`y`, :math:`\pi` is a transport plan, and
    :math:`\Pi(\mu, \nu)` is the set of admissible transport plans with
    marginals :math:`\mu` and :math:`\nu`.

    The cost matrix is defined as:

    .. math::

        C_{ij} = \left\| x_i - y_j \right\|_p^p
               = \sum_{k=1}^{D} |x_i^k - y_j^k|^p

    and the entropy term is:

    .. math::

        H(\pi) = - \sum_{i,j} \pi_{ij} \log \pi_{ij}

    where :math:`\varepsilon > 0` controls the strength of the entropic
    regularization.

    A fixed number of Sinkhorn iterations approximates dual potentials
    :math:`f` and :math:`g` in log space. The returned estimate is:

    .. math::

        \widehat{W}_\varepsilon = \langle a, f \rangle + \langle b, g \rangle

    where :math:`a` and :math:`b` are uniform probability weights over the
    :math:`N` input samples and :math:`M` target samples, respectively.

    Unlike pointwise losses, the Sinkhorn loss compares whole empirical
    distributions. The output contains one value with shape ``(1,)``, not a
    zero-dimensional scalar. No convergence check or self-transport debiasing
    is performed; the entropy-regularized value can be negative and need not
    vanish when the input and target distributions coincide.

    Smaller values of ``eps`` provide a closer approximation to the true
    Wasserstein distance, but may require more Sinkhorn iterations to converge.

    .. seealso::

        **Original reference:** Patrini, G., Carioni, M., Forr'e, P., Bhargav,
        S., Welling, M., Van den Berg, R., Genewein, T., and Nielsen, F. (2019).
        *Sinkhorn AutoEncoders*.
        In Proceedings of the 35th Conference on Uncertainty in Artificial
        Intelligence.
        URL: `<https://openreview.net/forum?id=BygNqoR9tm>`_.

    :Example:

        >>> import torch
        >>> from pina.loss import SinkhornLoss
        >>> loss = SinkhornLoss(p=2, eps=0.1, iterations=50)
        >>> input = torch.tensor([[0., 0.]])
        >>> target = torch.tensor([[3., 4.]])
        >>> loss(input, target)
        tensor([25.])
    """

    def __init__(self, p=2, eps=0.1, iterations=100):
        """
        Initialization of the :class:`SinkhornLoss` class.

        :param int p: The positive integer distance order and cost exponent:
            ``torch.cdist(input, target, p=p) ** p``. Default is ``2``.
        :param eps: The entropy regularization strength. Smaller values provide
            a closer approximation to the unregularized Wasserstein distance,
            but may require more iterations for convergence. Default is ``0.1``.
        :type eps: int | float
        :param int iterations: The fixed positive number of Sinkhorn iterations;
            there is no early stopping based on convergence.
            Default is ``100``.
        :raises AssertionError: If ``iterations`` is not a positive integer.
        :raises AssertionError: If ``p`` is not a positive integer.
        :raises ValueError: If ``eps`` is not a positive numeric value.
        """
        # Initialize the base class with mean reduction
        super().__init__(reduction="mean")

        # Check consistency
        check_positive_integer(iterations, strict=True)
        check_positive_integer(p, strict=True)
        check_consistency(eps, (int, float))
        if eps <= 0:
            raise ValueError(
                f"Expected 'eps' to be strictly positive, but got {eps}."
            )

        # Initialize parameters
        self.iterations = iterations
        self.eps = eps
        self.p = p

    def forward(self, input, target):
        """
        Forward method of the loss function.

        :param torch.Tensor input: The nonempty two-dimensional point cloud of
            shape ``(n, d)``, with a floating dtype supported by
            :func:`torch.cdist`. Samples have uniform probability weights.
        :param torch.Tensor target: The nonempty cloud of shape ``(m, d)``, on
            the same device and with the same dtype as ``input``. Sample counts
            may differ; feature counts must match.
        :raises RuntimeError: If pairwise distances or potential updates reject
            the point-cloud shapes, dtypes, or devices.
        :return: The final mean input potential plus mean target potential,
            wrapped as a one-element tensor of shape ``(1,)``. Empty clouds are
            not explicitly rejected and can produce NaN instead of a loss.
        :rtype: torch.Tensor
        """
        # Extract the number of samples in input and target
        n, m = input.shape[0], target.shape[0]

        # Initialize log-uniform weights for the empirical distributions
        log_a = -input.new_tensor(n).log().expand(n)
        log_b = -target.new_tensor(m).log().expand(m)

        # Initialize dual potentials f and g
        f = torch.zeros(n, dtype=input.dtype, device=input.device)
        g = torch.zeros(m, dtype=target.dtype, device=target.device)

        # Define the cost matrix, shape (n, m)
        C = torch.cdist(input, target, p=self.p) ** self.p

        # Perform Sinkhorn iterations in log space for numerical stability
        for _ in range(self.iterations):

            # Update dual potential f with the softmin operation in log space
            softmin_f = torch.logsumexp((g.unsqueeze(0) - C) / self.eps, dim=1)
            f = self.eps * (log_a - softmin_f)

            # Update dual potential g with the softmin operation in log space
            softmin_g = torch.logsumexp((f.unsqueeze(1) - C) / self.eps, dim=0)
            g = self.eps * (log_b - softmin_g)

        # Compute the Sinkhorn loss as the sum of the means of f and g
        loss = f.mean() + g.mean()

        return self._reduction(loss.unsqueeze(0))
