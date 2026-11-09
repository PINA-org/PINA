"""Module for the Orthogonal Block class."""

import torch
from pina._src.core.utils import check_consistency


class OrthogonalBlock(torch.nn.Module):
    """
    Orthogonal Block.

    This block transforms an input tensor of shape :math:`[N, M]` into a tensor
    of the same shape whose columns (``dim=-1``) or rows (``dim=0``) are
    orthonormal for linearly independent nonzero inputs. The block performs the
    Gram Schmidt orthogonalization, see
    `here <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>` for
    details.

    :Example:

        >>> import torch
        >>> from pina.model.block import OrthogonalBlock
        >>> block = OrthogonalBlock(dim=-1)
        >>> x = torch.randn(5, 3)
        >>> out = block(x)
    """

    def __init__(self, dim=-1, requires_grad=True):
        """
        Initialization of the :class:`OrthogonalBlock` class.

        :param int dim: The dimension on which orthogonalization is performed.
            If ``-1``, the orthogonalization is performed on the last dimension.
            Default is ``-1``.
        :param bool requires_grad: Whether the initial output tensor requires
            gradients. ``False`` does not detach inputs or disable autograd.
            Default is ``True``.
        :raises ValueError: If ``dim`` or ``requires_grad`` types are invalid.
        :raises IndexError: If ``dim`` is not ``-1``, ``0``, or ``1``.
        """
        super().__init__()
        # store dim
        self.dim = dim
        # store requires_grad
        check_consistency(requires_grad, bool)
        self._requires_grad = requires_grad

    def forward(self, X):
        """
        Forward pass.

        :param torch.Tensor X: Two-dimensional tensor to orthogonalize. The
            chosen dimension must be no larger than the other dimension.
        :raises Warning: If the chosen dimension is greater than the other
            dimensions in the input.
        :return: Same-shaped orthonormal tensor. Zero or dependent vectors can
            produce nonfinite values; no rank or norm safeguard is applied.
        :rtype: torch.Tensor
        """
        # check dim is less than all the other dimensions
        if X.shape[self.dim] > min(X.shape):
            raise Warning(
                "The dimension where to orthogonalize is greater"
                " than the other dimensions"
            )

        result = torch.zeros_like(X, requires_grad=self._requires_grad)
        X_0 = torch.select(X, self.dim, 0).clone()
        result_0 = X_0 / torch.linalg.norm(X_0)
        result = self._differentiable_copy(result, 0, result_0)

        # iterate over the rest of the basis with Gram-Schmidt
        for i in range(1, X.shape[self.dim]):
            v = torch.select(X, self.dim, i).clone()
            for j in range(i):
                vj = torch.select(result, self.dim, j).clone()
                v = v - torch.sum(v * vj, dim=self.dim, keepdim=True) * vj
            # result_i = torch.select(result, self.dim, i)
            result_i = v / torch.linalg.norm(v)
            result = self._differentiable_copy(result, i, result_i)
        return result

    def _differentiable_copy(self, result, idx, value):
        """
        Perform a differentiable copy operation.

        :param torch.Tensor result: The tensor where values are be copied to.
        :param int idx: The index along the specified dimension where the
            values are copied.
        :param torch.Tensor value: The tensor value to copy into ``result``.
        :return: A new tensor with the copied values.
        :rtype: torch.Tensor
        """
        return result.index_copy(
            self.dim, torch.tensor([idx]), value.unsqueeze(self.dim)
        )

    @property
    def dim(self):
        """
        The dimension along which operations are performed.

        :return: The current dimension value.
        :rtype: int
        """
        return self._dim

    @dim.setter
    def dim(self, value):
        """
        Set the dimension along which operations are performed.

        :param int value: The dimension to be set. Must be ``0``, ``1``, or
            ``-1``.
        :raises ValueError: If ``value`` is not an integer.
        :raises IndexError: If the provided dimension is not ``0``, ``1``, or
            ``-1``.
        """
        # check consistency
        check_consistency(value, int)
        if value not in [0, 1, -1]:
            raise IndexError(
                "Dimension out of range (expected to be in "
                f"range of [-1, 1], but got {value})"
            )
        # assign value
        self._dim = value

    @property
    def requires_grad(self):
        """
        Indicates whether gradient computation is required for operations
        on the tensors.

        :return: ``True`` if gradients are required, ``False`` otherwise.
        :rtype: bool
        """
        return self._requires_grad
