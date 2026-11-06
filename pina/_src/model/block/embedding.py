"""Modules for the the Embedding blocks."""

import torch
from pina._src.core.utils import check_consistency


class PeriodicBoundaryEmbedding(torch.nn.Module):
    r"""
    Enforcing hard-constrained periodic boundary conditions by embedding the
    input.

    A function :math:`u:\mathbb{R}^{\rm{in}} \rightarrow\mathbb{R}^{\rm{out}}`
    is periodic with respect to the spatial coordinates :math:`\mathbf{x}`
    with period :math:`\mathbf{L}` if:

    .. math::
        u(\mathbf{x})  = u(\mathbf{x} + n \mathbf{L})\;\;
        \forall n\in\mathbb{N}.

    The :class:`PeriodicBoundaryEmbedding` augments the input as follows:

    .. math::
        \tilde{\mathbf{x}} = [\mathbf{1},
        \cos(2\pi\mathbf{x}/\mathbf{L}),
        \sin(2\pi\mathbf{x}/\mathbf{L})],

    where :math:`\text{dim}(\tilde{\mathbf{x}}) = 3\text{dim}(\mathbf{x})`.

    .. seealso::
        **Original reference**:
            1.  Dong, Suchuan, and Naxian Ni (2021).
                *A method for representing periodic functions and enforcing
                exactly periodic boundary conditions with deep neural networks*.
                Journal of Computational Physics 435, 110242.
                DOI: `10.1016/j.jcp.2021.110242.
                <https://doi.org/10.1016/j.jcp.2021.110242>`_
            2.  Wang, S., Sankaran, S., Wang, H., & Perdikaris, P. (2023).
                *An expert's guide to training physics-informed neural
                networks*.
                DOI: `arXiv preprint arXiv:2308.0846.
                <https://arxiv.org/abs/2308.08468>`_

    .. note::

        Sine and cosine features are periodic in the selected coordinates.
        Smooth downstream compositions have periodic derivatives wherever those
        derivatives exist, up to floating-point error. Unselected coordinates
        are dropped, not passed through unchanged.

    :Example:

        >>> import torch
        >>> from pina.model.block import PeriodicBoundaryEmbedding
        >>> emb = PeriodicBoundaryEmbedding(
        ...     input_dimension=2, periods={"x": 1.0, "y": 2.0}
        ... )
        >>> from pina import LabelTensor
        >>> x = LabelTensor(torch.randn(10, 2), ["x", "y"])
        >>> out = emb(x)
    """

    def __init__(self, input_dimension, periods, output_dimension=None):
        """
        Initialization of the :class:`PeriodicBoundaryEmbedding` class.

        :param int input_dimension: The dimension of the input tensor.
        :param periods: The periodicity with respect to each dimension for the
            input data. If ``float`` or ``int`` is passed, the period is assumed
            to be constant over all the dimensions of the data. If a ``dict`` is
            passed the dictionary values represent periods, while its keys
            represent the dimension where the periodicity is enforced.
            Keys can be ``int`` if working with
            :class:`torch.Tensor`, or ``str`` if working with
            :class:`pina.label_tensor.LabelTensor`.
        :type periods: float | int | dict[int | str, int | float]
        :param output_dimension: The dimension of the output after the
            fourier embedding. If not ``None``, a :class:`torch.nn.Linear` layer
            is applied to the fourier embedding output to match the desired
            dimensionality. Default is ``None``.
        :type output_dimension: int | None
        :raises TypeError: If the periods dict is not consistent.
        :raises ValueError: If dimension or period types fail validation.

        .. note::

            Period values are not checked for zero or sign. Dictionary order
            determines feature order. A projection expects three times
            ``input_dimension`` features, even with fewer selected coordinates.
        """
        super().__init__()

        # check input consistency
        check_consistency(periods, (float, int, dict))
        check_consistency(input_dimension, int)
        if output_dimension is not None:
            check_consistency(output_dimension, int)
            self._layer = torch.nn.Linear(input_dimension * 3, output_dimension)
        else:
            self._layer = torch.nn.Identity()

        # checks on the periods
        if isinstance(periods, dict):
            if not all(
                isinstance(dim, (str, int)) and isinstance(period, (float, int))
                for dim, period in periods.items()
            ):
                raise TypeError(
                    "In dictionary periods, keys must be integers"
                    " or strings, and values must be float or int."
                )
            self._period = periods
        else:
            self._period = {k: periods for k in range(input_dimension)}

    def forward(self, x):
        """
        Forward pass.

        :param x: Input of shape ``(..., input_dimension)``. String selections
            require a :class:`pina.LabelTensor`.
        :type x: torch.Tensor | LabelTensor
        :raises IndexError: If the period dictionary is empty.
        :raises RuntimeError: If string extraction is unavailable or the
            projection width is incompatible with the selected coordinates.
        :return: Ones, cosine, and sine groups in that order, with final width
            ``3 * n_selected`` or ``output_dimension`` if projected.
        :rtype: torch.Tensor | LabelTensor
        """
        omega = torch.stack(
            [
                torch.pi * 2.0 / torch.tensor([val], device=x.device)
                for val in self._period.values()
            ],
            dim=-1,
        )
        x = self._get_vars(x, list(self._period.keys()))
        return self._layer(
            torch.cat(
                [
                    torch.ones_like(x),
                    torch.cos(omega * x),
                    torch.sin(omega * x),
                ],
                dim=-1,
            )
        )

    def _get_vars(self, x, indeces):
        """
        Get the variables from input tensor ordered by specific indeces.

        :param x: The input tensor from which to extract.
        :type x: torch.Tensor | LabelTensor
        :param indeces: The indeces to extract.
        :type indeces: list[int] | list[str]
        :raises RuntimeError: If the indeces are not consistent.
        :raises RuntimeError: If the extraction is not possible.
        :return: The extracted tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        if isinstance(indeces[0], str):
            try:
                return x.extract(indeces)
            except AttributeError as e:
                raise RuntimeError(
                    "Not possible to extract input variables from tensor."
                    " Ensure that the passed tensor is a LabelTensor or"
                    " pass list of integers to extract variables. For"
                    " more information refer to warning in the documentation."
                ) from e
        elif isinstance(indeces[0], int):
            return x[..., indeces]
        else:
            raise RuntimeError(
                "Not able to extract correct indeces for tensor."
                " For more information refer to warning in the documentation."
            )

    @property
    def period(self):
        """
        The period of the function.

        :return: The stored coordinate-to-period mapping, retained by reference
            for dictionary input and constructed for scalar input.
        :rtype: dict[int | str, int | float]
        """
        return self._period


class FourierFeatureEmbedding(torch.nn.Module):
    r"""
    Fourier Feature Embedding class to encode the input features using random
    Fourier features.

    This class applies a Fourier transformation to the input features, which can
    help in learning high-frequency variations in data. Each instance uses one
    scalar ``sigma`` scale; sequence scales are not implemented.

    The Fourier Feature Embedding augments the input features as follows
    (3.10 of original paper):

    .. math::
        \mathbf{x} \rightarrow \tilde{\mathbf{x}} = \left[
        \cos\left( 2\pi\mathbf{x}\mathbf{B} \right),
        \sin\left( 2\pi\mathbf{x}\mathbf{B} \right)\right],

    where :math:`\mathbf{B}_{ij} = \sigma U_{ij}` and
    :math:`U_{ij} \sim \mathcal{U}[0,1)`.

    Separate embedding modules can be used for multiple scales. For example,
    their concatenation would have the form:

    .. math::
        \mathbf{x} \rightarrow \tilde{\mathbf{x}} = \left[
        \cos\left( 2\pi\mathbf{x}\mathbf{B}^1 \right),
        \sin\left( 2\pi\mathbf{x}\mathbf{B}^1 \right),
        \cos\left( 2\pi\mathbf{x}\mathbf{B}^2 \right),
        \sin\left( 2\pi\mathbf{x}\mathbf{B}^2 \right),
        \dots,
        \cos\left( 2\pi\mathbf{x}\mathbf{B}^M \right),
        \sin\left( 2\pi\mathbf{x}\mathbf{B}^M \right)\right],

    with the same ``2 * pi`` frequency factor as the single-scale embedding.
    The random matrix is a plain tensor attribute, not a parameter or buffer:
    it is omitted from ``state_dict`` and cast to input device/dtype per call.

    .. seealso::
        **Original reference**:
        Wang, S., Wang, H., and Perdikaris, P. (2021).
        *On the eigenvector bias of Fourier feature networks: From regression to
        solving multi-scale PDEs with physics-informed neural networks.*
        Computer Methods in Applied Mechanics and Engineering 384 (2021):
        113938.
        DOI: `10.1016/j.cma.2021.113938.
        <https://doi.org/10.1016/j.cma.2021.113938>`_

    :Example:

        >>> import torch
        >>> from pina.model.block import FourierFeatureEmbedding
        >>> emb = FourierFeatureEmbedding(
        ...     input_dimension=2, output_dimension=16, sigma=2.0
        ... )
        >>> x = torch.randn(10, 2)
        >>> out = emb(x)
    """

    def __init__(self, input_dimension, output_dimension, sigma):
        """
        Initialization of the :class:`FourierFeatureEmbedding` class.

        :param int input_dimension: The dimension of the input tensor.
        :param int output_dimension: The dimension of the output tensor. The
            output is obtained as a concatenation of cosine and sine embeddings.
        :param sigma: The scalar multiplier for uniform random frequencies;
            not their Gaussian standard deviation. No sign check is applied.
        :type sigma: float | int
        :raises RuntimeError: If the output dimension is not an even number.
        :raises ValueError: If dimension or scale types fail validation.
        """
        super().__init__()

        # check consistency
        check_consistency(sigma, (int, float))
        check_consistency(output_dimension, int)
        check_consistency(input_dimension, int)
        if output_dimension % 2:
            raise RuntimeError(
                "Expected output_dimension to be a even number, "
                f"got {output_dimension}."
            )

        # assign sigma
        self._sigma = sigma

        # create non-trainable matrices
        self._matrix = (
            torch.rand(
                size=(input_dimension, output_dimension // 2),
                requires_grad=False,
            )
            * self.sigma
        )

    def forward(self, x):
        """
        Forward pass.

        :param x: A two-dimensional tensor of shape ``(n, input_dimension)``.
        :type x: torch.Tensor | LabelTensor
        :raises RuntimeError: If matrix multiplication rejects the shape/dtype.
        :return: Cosine then sine features of shape ``(n, output_dimension)``.
        :rtype: torch.Tensor | LabelTensor
        """
        # compute random matrix multiplication
        out = torch.mm(x, self._matrix.to(device=x.device, dtype=x.dtype))
        # return embedding
        return torch.cat(
            [torch.cos(2 * torch.pi * out), torch.sin(2 * torch.pi * out)],
            dim=-1,
        )

    @property
    def sigma(self):
        """
        The scale multiplier applied to uniform frequency samples.

        :return: The configured scale multiplier.
        :rtype: float | int
        """
        return self._sigma
