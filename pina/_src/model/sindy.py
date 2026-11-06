"""Module for the SINDy model class."""

from typing import Callable
import torch
from pina._src.core.utils import check_consistency, check_positive_integer


class SINDy(torch.nn.Module):
    r"""
    SINDy model class.

    The Sparse Identification of Nonlinear Dynamics (SINDy) model identifies the
    governing equations of a dynamical system from data by learning a sparse
    linear combination of non-linear candidate functions. This module itself
    imposes no sparsity penalty or thresholding on the learned coefficients.

    The output of the model is expressed as product of a library matrix and a
    coefficient matrix:

    .. math::

        \dot{X} = \Theta(X) \Xi

    where:
      - :math:`X \in \mathbb{R}^{B \times D}` is the input snapshots of the
        system state. Here, :math:`B` is the batch size and :math:`D` is the
        number of state variables.
      - :math:`\Theta(X) \in \mathbb{R}^{B \times L}` is the library matrix
        obtained by summing each candidate's final feature axis.
        Here, :math:`L` is the number of candidate functions in the library.
      - :math:`\Xi \in \mathbb{R}^{L \times D}` is the learned coefficient
        matrix that defines the sparse model.

    .. seealso::

        **Original reference**:
        Brunton, S.L., Proctor, J.L., and Kutz, J.N. (2016).
        *Discovering governing equations from data: Sparse identification of
        non-linear dynamical systems.*
        Proceedings of the National Academy of Sciences, 113(15), 3932-3937.
        DOI: `10.1073/pnas.1517384113
        <https://doi.org/10.1073/pnas.1517384113>`_

    :Example:

        >>> import torch
        >>> from pina.model import SINDy
        >>> library = [lambda x: x, lambda x: x**2]
        >>> model = SINDy(library=library, output_dimension=2)
        >>> x = torch.randn(10, 2)
        >>> out = model(x)
    """

    def __init__(self, library, output_dimension):
        """
        Initialization of the :class:`SINDy` class.

        :param list[Callable] library: The collection of candidate functions
            used to construct the library matrix. Each function must accept an
            input tensor of shape ``[..., D]`` and return a tensor of shape
            ``[..., q]`` with a final feature axis of any common size ``q``.
            The implementation sums that axis in the final contraction; vector
            outputs do not create separate library entries.
        :param int output_dimension: The number of output variables, typically
            the number of state derivatives. It determines the number of columns
            in the coefficient matrix.
        :raises ValueError: If ``library`` is not a list of callables.
        :raises AssertionError: If ``output_dimension`` is not a positive
            integer.
        """
        super().__init__()

        # Check consistency
        check_positive_integer(output_dimension, strict=True)
        check_consistency(library, Callable)
        if not isinstance(library, list):
            raise ValueError("`library` must be a list of callables.")

        # Initialization
        self._library = library
        self._coefficients = torch.nn.Parameter(
            torch.zeros(len(library), output_dimension)
        )

    def forward(self, x):
        """
        Forward pass of the :class:`SINDy` model.

        :param torch.Tensor x: State variables of shape ``(..., d)``.
        :raises RuntimeError: If library results cannot be stacked or
            contracted.
        :return: Predictions of shape ``(..., output_dimension)``. An empty
            library fails during stacking rather than construction.
        :rtype: torch.Tensor
        """
        theta = torch.stack([f(x) for f in self.library], dim=-2)
        return torch.einsum("...li , lo -> ...o", theta, self.coefficients)

    @property
    def library(self):
        """
        The library of candidate functions.

        :return: The library.
        :rtype: list[Callable]
        """
        return self._library

    @property
    def coefficients(self):
        """
        The coefficients of the model.

        :return: Trainable matrix shaped ``(len(library), output_dimension)``,
            initialized to zero.
        :rtype: torch.nn.Parameter
        """
        return self._coefficients
