"""Module for the Low Rank Neural Operator model class."""

import torch
from torch import nn

from pina._src.core.utils import check_consistency

from pina._src.model.kernel_neural_operator import KernelNeuralOperator
from pina._src.model.block.low_rank_block import LowRankBlock


class LowRankNeuralOperator(KernelNeuralOperator):
    """
    Low Rank Neural Operator model class.

    The Low Rank Neural Operator is a general architecture for learning
    operators, which map functions to functions. It can be trained both with
    Supervised and Physics-Informed learning strategies. The Low Rank Neural
    Operator performs convolution by means of a low rank approximation.

    .. seealso::

        **Original reference**: Kovachki, N., Li, Z., Liu, B., Azizzadenesheli,
        K., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2023).
        *Neural operator: Learning maps between function spaces with
        applications to PDEs*.
        Journal of Machine Learning Research, 24(89), 1-97.

    :Example:

        >>> import torch
        >>> from pina import LabelTensor
        >>> from pina.model import LowRankNeuralOperator, FeedForward
        >>> lifting_net = FeedForward(input_dimensions=4, output_dimensions=20)
        >>> projecting_net = FeedForward(23, 1)
        >>> model = LowRankNeuralOperator(
        ...     lifting_net, projecting_net,
        ...     field_indices=["f"], coordinates_indices=["x", "y", "z"],
        ...     n_kernel_layers=2, rank=10
        ... )
        >>> x = LabelTensor(torch.randn(10, 5, 4), ["f", "x", "y", "z"])
        >>> out = model(x)
    """

    def __init__(
        self,
        lifting_net,
        projecting_net,
        field_indices,
        coordinates_indices,
        n_kernel_layers,
        rank,
        inner_size=20,
        n_layers=2,
        func=torch.nn.Tanh,
        bias=True,
    ):
        """
        Initialization of the :class:`LowRankNeuralOperator` class.

        :param torch.nn.Module lifting_net: The lifting neural network mapping
            the input to its hidden dimension. It must take as input the input
            field and the coordinates at which the input field is evaluated.
        :param torch.nn.Module projecting_net: The projection neural network
            mapping the hidden representation to the output function. It must
            take as input the embedding dimension plus the dimension of the
            coordinates.
        :param list[str] field_indices: The labels of the fields in the input
            tensor.
        :param list[str] coordinates_indices: The labels of the coordinates in
            the input tensor.
        :param int n_kernel_layers: The number of hidden kernel layers.
        :param int rank: The rank of the low rank approximation.
        :param int inner_size: The number of neurons for each hidden layer in
            the basis function neural network. Default is ``20``.
        :param int n_layers: The number of hidden layers in the basis function
            neural network. Default is ``2``.
        :param type[torch.nn.Module] func: Activation class for basis hidden
            layers and each block's output. Lists accepted by FeedForward
            cannot be used by the block's output activation.
            Default is :class:`torch.nn.Tanh`.
        :param bool bias: If ``True`` bias is considered for the basis function
            neural network. Default is ``True``.
        :raises ValueError: If the input dimension does not match with the
            labels of the fields and coordinates.
        :raises ValueError: If the input dimension of the projecting network
            does not match with the hidden dimension of the lifting network.

        .. note::

            Width inference requires network parameters and probes the lifting
            network with random CPU input in training mode. The kernel contracts
            features at each point; it does not integrate over mesh nodes.
        """

        # check consistency
        check_consistency(field_indices, str)
        check_consistency(coordinates_indices, str)
        check_consistency(n_kernel_layers, int)

        # check hidden dimensions match
        input_lifting_net = next(lifting_net.parameters()).size()[-1]
        output_lifting_net = lifting_net(
            torch.rand(size=next(lifting_net.parameters()).size())
        ).shape[-1]
        projecting_net_input = next(projecting_net.parameters()).size()[-1]

        if len(field_indices) + len(coordinates_indices) != input_lifting_net:
            raise ValueError(
                "The lifting_net must take as input the "
                "coordinates vector and the field vector."
            )

        if (
            output_lifting_net + len(coordinates_indices)
            != projecting_net_input
        ):
            raise ValueError(
                "The projecting_net input must be equal to "
                "the embedding dimension (which is the output) "
                "of the lifting_net plus the dimension of the "
                "coordinates, i.e. len(coordinates_indices)."
            )

        # assign
        self.coordinates_indices = coordinates_indices
        self.field_indices = field_indices
        integral_net = nn.Sequential(
            *[
                LowRankBlock(
                    input_dimensions=len(coordinates_indices),
                    embedding_dimenion=output_lifting_net,
                    rank=rank,
                    inner_size=inner_size,
                    n_layers=n_layers,
                    func=func,
                    bias=bias,
                )
                for _ in range(n_kernel_layers)
            ]
        )
        super().__init__(lifting_net, integral_net, projecting_net)

    def forward(self, x):
        r"""
        Forward pass for the :class:`LowRankNeuralOperator` model.

        The ``lifting_net`` maps the input to the hidden dimension.
        Then, several layers of
        :class:`~pina.model.block.LowRankBlock` are
        applied. Finally, the ``projecting_net`` maps the hidden representation
        to the output function.

        :param LabelTensor x: The input tensor for performing the computation.
            It expects a tensor :math:`B \times N \times D`, where :math:`B` is
            the batch_size, :math:`N` the number of points in the mesh,
            :math:`D` the dimension of the problem, i.e. the sum
            of ``len(coordinates_indices)`` and ``len(field_indices)``.
        :return: Predictions with shape ``(B, N, output_width)``. Input column
            order is retained for lifting; coordinates are appended before
            projection.
        :rtype: torch.Tensor | LabelTensor
        """
        # extract points
        coords = x.extract(self.coordinates_indices)
        # lifting
        x = self._lifting_operator(x)
        # kernel
        for module in self._integral_kernels:
            x = module(x, coords)
        # projecting
        return self._projection_operator(torch.cat((x, coords), dim=-1))
