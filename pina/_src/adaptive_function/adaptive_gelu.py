"""Module for the Adaptive GELU activation function."""

import torch
from pina._src.adaptive_function.base_adaptive_function import (
    BaseAdaptiveFunction,
)


class AdaptiveGELU(BaseAdaptiveFunction):
    r"""
    Adaptive, trainable variant of the :class:`~torch.nn.GELU` activation.

    This module extends the standard GELU by introducing learnable scaling
    and shifting parameters applied to both the input and the output.

    Given the function :math:`\text{GELU}:\mathbb{R}^n\rightarrow\mathbb{R}^n`,
    the corresponding adaptive activation
    :math:`\text{GELU}_{\text{adaptive}}:\mathbb{R}^n\rightarrow\mathbb{R}^n` is
    defined as:

    .. math::
        \text{GELU}_{\text{adaptive}}({x})=\alpha\,\text{GELU}(\beta{x}+\gamma),

    where :math:`\alpha`, :math:`\beta`, and :math:`\gamma` are trainable
    parameters controlling output scaling, input scaling, and input shifting,
    respectively.

    The GELU function uses the exact, non-approximate PyTorch default and is
    defined elementwise as:

    .. math::
        \text{GELU}(x) = \frac{x}{2}
        \left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)

    .. seealso::

        **Original reference**: Godfrey, L. B., Gashler, M. S. (2015).
        *A continuum among logarithmic, linear, and exponential functions,
        and its potential to improve generalization in neural networks.*
        7th international joint conference on knowledge discovery, knowledge
        engineering and knowledge management (IC3K), Vol. 1.
        DOI: `arXiv preprint arXiv:1602.01321.
        <https://arxiv.org/abs/1602.01321>`_.

        **Original reference**: Jagtap, A. D., Karniadakis, G. E. (2020).
        *Adaptive activation functions accelerate convergence in deep and
        physics-informed neural networks*.
        Journal of Computational Physics, 404.
        DOI: `JCP 10.1016 <https://doi.org/10.1016/j.jcp.2019.109136>`_.

    :Example:

        >>> import torch
        >>> from pina.adaptive_function import AdaptiveGELU
        >>> act = AdaptiveGELU()
        >>> x = torch.randn(10, 3)
        >>> out = act(x)
        >>> out.shape
        torch.Size([10, 3])
    """

    def __init__(self, alpha=None, beta=None, gamma=None, fixed=None):
        """
        Initialization of the :class:`AdaptiveGELU` class.

        :param alpha: The output scaling parameter of the adaptive function.
            If ``None``, it is initialized to ``1``. Default is ``None``.
        :type alpha: int | float | None
        :param beta: The input scaling parameter of the adaptive function.
            If ``None``, it is initialized to ``1``. Default is ``None``.
        :type beta: int | float | None
        :param gamma: The input shifting parameter of the adaptive function.
            If ``None``, it is initialized to ``0``. Default is ``None``.
        :type gamma: int | float | None
        :param fixed: The names of parameters to keep fixed during training.
            These parameters will not be optimized and will have
            ``requires_grad=False``. Available options are ``"alpha"``,
            ``"beta"``, and ``"gamma"``. If ``None``, all parameters are
            trainable. Default is ``None``.
        :type fixed: str | list[str] | tuple[str, ...] | set[str] | None
        :raises ValueError: If ``alpha``, ``beta``, or ``gamma`` fails numeric
            type validation.
        :raises ValueError: If ``fixed`` is not a string or a list, tuple, or
            set of strings, or contains invalid parameter names.
        """
        super().__init__(alpha, beta, gamma, fixed)
        self._func = torch.nn.GELU()
