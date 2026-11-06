"""Module for the Low Rank Neural Operator Block class."""

import torch

from pina._src.core.utils import check_consistency


class LowRankBlock(torch.nn.Module):
    """
    The inner block of the Low Rank Neural Operator.

    .. seealso::

        **Original reference**: Kovachki, N., Li, Z., Liu, B.,
        Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A.
        (2023). *Neural operator: Learning maps between function
        spaces with applications to PDEs*. Journal of Machine Learning
        Research, 24(89), 1-97.

    :Example:

        >>> import torch
        >>> from pina.model.block import LowRankBlock
        >>> block = LowRankBlock(
        ...     input_dimensions=3, embedding_dimenion=20, rank=10
        ... )
        >>> x = torch.randn(10, 20)
        >>> coords = torch.randn(10, 3)
        >>> out = block(x, coords)
    """

    def __init__(
        self,
        input_dimensions,
        embedding_dimenion,
        rank,
        inner_size=20,
        n_layers=2,
        func=torch.nn.Tanh,
        bias=True,
    ):
        r"""
        Initialization of the :class:`LowRankBlock` class.

        :param int input_dimensions: The number of coordinate components.
        :param int embedding_dimenion: The embedding dimension of the field.
        :param int rank: Number of terms in each of the two learned bases.
        :param int inner_size: The number of neurons for each hidden layer in
            the basis function neural network. Default is ``20``.
        :param int n_layers: The number of hidden layers in the basis function
            neural network. Default is ``2``.
        :param type[torch.nn.Module] func: Activation class in the basis network
            and on the final result. Lists accepted by FeedForward fail when
            invoked as a class.
            Default is :class:`torch.nn.Tanh`.
        :param bool bias: If ``True`` bias is considered for the basis function
            neural network. Default is ``True``.
        :raises ValueError: If dimensions or rank fail integer validation.
        :raises TypeError: If ``func`` is not callable without arguments.
        """
        super().__init__()
        from ..feed_forward import FeedForward

        # Assignment (check consistency inside FeedForward)
        self._basis = FeedForward(
            input_dimensions=input_dimensions,
            output_dimensions=2 * rank * embedding_dimenion,
            inner_size=inner_size,
            n_layers=n_layers,
            func=func,
            bias=bias,
        )
        self._nn = torch.nn.Linear(embedding_dimenion, embedding_dimenion)

        check_consistency(rank, int)
        self._rank = rank
        self._func = func()

    def forward(self, x, coords):
        r"""
        Forward pass of the block. It performs an affine transformation of the
        field, followed by a low rank approximation. The latter is performed by
        means of a dot product of the basis :math:`\psi^{(i)}` with the vector
        field :math:`v` to compute coefficients used to expand
        :math:`\phi^{(i)}`, evaluated in the spatial input :math:`x`.

        :param torch.Tensor x: Features shaped ``(..., embedding_dimenion)``.
        :param torch.Tensor coords: Corresponding coordinates shaped
            ``(..., input_dimensions)``; labels are stripped before evaluation.
        :return: Same-shaped field update. Contractions sum over features and
            rank at each point independently, not over mesh nodes.
        :rtype: torch.Tensor
        """
        # extract basis
        coords = coords.as_subclass(torch.Tensor)
        basis = self._basis(coords)
        # reshape [B, N, D, 2*rank]
        shape = list(basis.shape[:-1]) + [-1, 2 * self.rank]
        basis = basis.reshape(shape)
        # divide
        psi = basis[..., : self.rank]
        phi = basis[..., self.rank :]
        # compute dot product
        coeff = torch.einsum("...dr,...d->...r", psi, x)
        # expand the basis
        expansion = torch.einsum("...r,...dr->...d", coeff, phi)
        # apply linear layer and return
        return self._func(self._nn(x) + expansion)

    @property
    def rank(self):
        """
        The basis rank.

        :return: The basis rank.
        :rtype: int
        """
        return self._rank
