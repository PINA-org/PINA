"""Module for the Kernel Neural Operator model class."""

import torch
from pina._src.core.utils import check_consistency


class KernelNeuralOperator(torch.nn.Module):
    r"""
    Base class for Neural Operators with integral kernels.

    This class serves as a foundation for building Neural Operators that
    incorporate integral kernels. Several neural operators in PINA inherit
    from this class. The design follows the framework proposed
    by Kovachki et al., as illustrated in Figure 2 of their work.

    Neural Operators derived from this class can be expressed as:

    .. math::
        G_\theta  := P \circ K_m \circ \cdot \circ K_1 \circ L

    where:

    *   :math:`G_\theta: \mathcal{A}\subset \mathbb{R}^{\rm{in}} \rightarrow
        \mathcal{D}\subset \mathbb{R}^{\rm{out}}` is the neural operator
        approximation of the unknown real operator :math:`G`, that is
        :math:`G \approx G_\theta`
    *   :math:`L: \mathcal{A}\subset \mathbb{R}^{\rm{in}} \rightarrow
        \mathbb{R}^{\rm{emb}}` is a lifting operator mapping the input
        from its domain :math:`\mathcal{A}\subset \mathbb{R}^{\rm{in}}`
        to its embedding dimension :math:`\mathbb{R}^{\rm{emb}}`
    *   :math:`\{K_i : \mathbb{R}^{\rm{emb}} \rightarrow
        \mathbb{R}^{\rm{emb}} \}_{i=1}^m` are :math:`m` integral kernels
        mapping each hidden representation to the next one.
    *   :math:`P : \mathbb{R}^{\rm{emb}} \rightarrow  \mathcal{D}\subset
        \mathbb{R}^{\rm{out}}` is a projection operator mapping the hidden
        representation to the output function.

    .. seealso::

        **Original reference**: Kovachki, N., Li, Z., Liu, B.,
        Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A.
        (2023).
        *Neural operator: Learning maps between function spaces with
        applications to PDEs*.
        Journal of Machine Learning Research, 24(89), 1-97.

    :Example:

        >>> import torch
        >>> from torch import nn
        >>> from pina.model import KernelNeuralOperator
        >>> lifting_net = nn.Linear(3, 20)
        >>> integral_net = nn.Identity()
        >>> projecting_net = nn.Linear(20, 1)
        >>> model = KernelNeuralOperator(
        ...     lifting_operator=lifting_net,
        ...     integral_kernels=integral_net,
        ...     projection_operator=projecting_net
        ... )
        >>> x = torch.randn(10, 3)
        >>> out = model(x)
    """

    def __init__(self, lifting_operator, integral_kernels, projection_operator):
        """
        Initialization of the :class:`KernelNeuralOperator` class.

        :param torch.nn.Module lifting_operator: The lifting operator mapping
            the input to its hidden dimension.
        :param torch.nn.Module integral_kernels: A callable kernel module,
            possibly :class:`torch.nn.Sequential`, mapping hidden features.
            A plain list or ModuleList cannot be called by this forward method.
        :param torch.nn.Module projection_operator: The projection operator
            mapping the hidden representation to the output function.
        """

        super().__init__()

        self._lifting_operator = lifting_operator
        self._integral_kernels = integral_kernels
        self._projection_operator = projection_operator

    @property
    def lifting_operator(self):
        """
        The lifting operator module.

        :return: The lifting operator module.
        :rtype: torch.nn.Module
        """
        return self._lifting_operator

    @lifting_operator.setter
    def lifting_operator(self, value):
        """
        Set the lifting operator module.

        :param torch.nn.Module value: The lifting operator module.
        :raises ValueError: If ``value`` is not a module instance.
        """
        check_consistency(value, torch.nn.Module)
        self._lifting_operator = value

    @property
    def projection_operator(self):
        """
        The projection operator module.

        :return: The projection operator module.
        :rtype: torch.nn.Module
        """
        return self._projection_operator

    @projection_operator.setter
    def projection_operator(self, value):
        """
        Set the projection operator module.

        :param torch.nn.Module value: The projection operator module.
        :raises ValueError: If ``value`` is not a module instance.
        """
        check_consistency(value, torch.nn.Module)
        self._projection_operator = value

    @property
    def integral_kernels(self):
        """
        The integral kernels operator module.

        :return: The integral kernels operator module.
        :rtype: torch.nn.Module
        """
        return self._integral_kernels

    @integral_kernels.setter
    def integral_kernels(self, value):
        """
        Set the integral kernels operator module.

        :param torch.nn.Module value: The integral kernels operator module.
        :raises ValueError: If ``value`` is not a module instance.
        """
        check_consistency(value, torch.nn.Module)
        self._integral_kernels = value

    def forward(self, x):
        r"""
        Forward pass for the :class:`KernelNeuralOperator` model.

        The ``lifting_operator`` maps the input to the hidden dimension.
        The ``integral_kernels`` apply the integral kernels to the hidden
        representation. The ``projection_operator`` maps the hidden
        representation to the output function.

        :param x: The input tensor, with shape determined by the three supplied
            modules. A common convention is
            a tensor :math:`B \times N \times D`, where :math:`B` is the
            batch_size, :math:`N` the number of points in the mesh, and
            :math:`D` the dimension of the problem. In particular, :math:`D`
            is the number of spatial, parametric, and/or temporal variables
            plus the field variables. For instance, for 2D problems with 2
            output variables, :math:`D=4`.
        :type x: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        x = self.lifting_operator(x)
        x = self.integral_kernels(x)
        x = self.projection_operator(x)
        return x
