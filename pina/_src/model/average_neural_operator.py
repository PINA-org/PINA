"""Module for the Averaging Neural Operator model class."""

import torch
from torch import nn
from pina._src.model.block.average_neural_operator_block import AVNOBlock
from pina._src.model.kernel_neural_operator import KernelNeuralOperator
from pina._src.core.utils import check_consistency


class AveragingNeuralOperator(KernelNeuralOperator):
    """
    Averaging Neural Operator model class.

    The Averaging Neural Operator is a general architecture for learning
    operators, which map functions to functions. It can be trained both with
    Supervised and Physics-Informed learning strategies. The Averaging Neural
    Operator performs convolution by means of a field average.

    .. seealso::

        **Original reference**: Lanthaler S., Li, Z., Stuart, A. (2020).
        *The Nonlocal Neural Operator: Universal Approximation*.
        DOI: `arXiv preprint arXiv:2304.13221.
        <https://arxiv.org/abs/2304.13221>`_

    :Example:

        >>> import torch
        >>> from pina import LabelTensor
        >>> from pina.model import AveragingNeuralOperator, FeedForward
        >>> lifting_net = FeedForward(input_dimensions=4, output_dimensions=20)
        >>> projecting_net = FeedForward(23, 1)
        >>> model = AveragingNeuralOperator(
        ...     lifting_net, projecting_net,
        ...     field_indices=["f"], coordinates_indices=["x", "y", "z"]
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
        n_layers=4,
        func=nn.GELU,
    ):
        """
        Initialization of the :class:`AveragingNeuralOperator` class.

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
        :param int n_layers: The number of hidden layers. Default is ``4``.
        :param type[torch.nn.Module] func: The activation class to instantiate.
            Default is :class:`torch.nn.GELU`.
        :raises ValueError: If the input dimension does not match with the
            labels of the fields and coordinates.
        :raises ValueError: If the input dimension of the projecting network
            does not match with the hidden dimension of the lifting network.

        .. note::

            Network widths are inferred from their first parameters and a
            random CPU lifting call in training mode. Both networks must have
            parameters; the probe can update the lifting network's state.
        """

        # check consistency
        check_consistency(field_indices, str)
        check_consistency(coordinates_indices, str)
        check_consistency(n_layers, int)
        check_consistency(func, nn.Module, subclass=True)

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
                "The projecting_net input must be equal to"
                "the embedding dimension (which is the output) "
                "of the lifting_net plus the dimension of the "
                "coordinates, i.e. len(coordinates_indices)."
            )

        # assign
        self.coordinates_indices = coordinates_indices
        self.field_indices = field_indices
        integral_net = nn.Sequential(
            *[AVNOBlock(output_lifting_net, func) for _ in range(n_layers)]
        )
        super().__init__(lifting_net, integral_net, projecting_net)

    def forward(self, x):
        r"""
        Forward pass for the :class:`AveragingNeuralOperator` model.

        The ``lifting_net`` maps the input to the hidden dimension.
        Then, several layers of
        :class:`~pina.model.block.AVNOBlock` are
        applied. Finally, the ``projection_net`` maps the hidden representation
        to the output function.

        :param LabelTensor x: The input tensor for performing the computation.
            It expects a tensor :math:`B \times N \times D`, where :math:`B` is
            the batch_size, :math:`N` the number of points in the mesh,
            :math:`D` the dimension of the problem, i.e. the sum
            of ``len(coordinates_indices)`` and ``len(field_indices)``.
        :return: Predictions with shape ``(B, N, output_width)``. Lifting input
            is reordered to fields followed by coordinates.
        :rtype: torch.Tensor | LabelTensor
        """
        points_tmp = x.extract(self.coordinates_indices)
        new_batch = x.extract(self.field_indices)
        new_batch = torch.cat((new_batch, points_tmp), dim=-1)
        new_batch = self._lifting_operator(new_batch)
        new_batch = self._integral_kernels(new_batch)
        new_batch = torch.cat((new_batch, points_tmp), dim=-1)
        new_batch = self._projection_operator(new_batch)
        return new_batch
