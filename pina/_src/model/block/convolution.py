"""Module for the Base Continuous Convolution class."""

from abc import ABCMeta, abstractmethod
import torch
from pina._src.model.block.stride import Stride
from pina._src.model.block.utils_convolution import optimizing


class BaseContinuousConv(torch.nn.Module, metaclass=ABCMeta):
    r"""
    Base Class for Continuous Convolution.

    The class expects the input to be in the form:
    :math:`[B \times N_{in} \times N \times D]`, where :math:`B` is the
    batch_size, :math:`N_{in}` is the number of input fields, :math:`N`
    the number of points in the mesh, :math:`D` the dimension of the problem.
    In particular:

    *   :math:`D` is the number of spatial variables + 1. The last column must
        contain the field value.
    *   :math:`N_{in}` represents the number of function components.
        For instance, a vectorial function :math:`f = [f_1, f_2]` has
        :math:`N_{in}=2`.

    .. note::

        A 2-dimensional vector-valued function defined on a 3-dimensional input
        evaluated on a 100 points input mesh and batch size of 8 is represented
        as a tensor of shape ``[8, 2, 100, 4]``, where the columns
        ``[:, 0, :, -1]`` and ``[:, 1, :, -1]`` represent the first and second,
        components of the function, respectively.

    The algorithm returns a tensor of shape:
    :math:`[B \times N_{out} \times S \times D]`, where :math:`S` is the
    number of filter positions rather than the number of input mesh points.

    :Example:

        >>> import torch
        >>> from pina._src.model.block.convolution import BaseContinuousConv
        >>> class MyConv(BaseContinuousConv):
        ...     def forward(self, X): return X
        ...     def transpose_overlap(self, X): return X
        ...     def transpose_no_overlap(self, X): return X
        ...     def _initialize_convolution(self, X, type_): pass
        >>> stride = {
        ...     "domain": [10, 10],
        ...     "start": [0, 0],
        ...     "jump": [3, 3],
        ...     "direction": [1, 1],
        ... }
        >>> conv = MyConv(1, 2, [3, 3], stride)
    """

    def __init__(
        self,
        input_numb_field,
        output_numb_field,
        filter_dim,
        stride,
        model=None,
        optimize=False,
        no_overlap=False,
    ):
        """
        Initialization of the :class:`BaseContinuousConv` class.

        :param int input_numb_field: The number of input fields.
        :param int output_numb_field: The number of output fields.
        :param filter_dim: The shape of the filter.
        :type filter_dim: list[int | float] | tuple[int | float, ...]
        :param dict[str, list[int | float]] stride: Filter grid specifications,
            inserted in order: ``"domain"``, ``"start"``, ``"jump"``,
            ``"direction"``. Zero jumps/directions are mutated by Stride.
        :param model: Kernel factory retained for subclasses. ``None`` lets the
            concrete block create default kernels. Default is ``None``.
        :type model: Callable[[], torch.nn.Module] | None
        :param bool optimize: Whether to wrap initialization with call-tracking
            flags. The wrapper still initializes on every call; evaluation
            bypasses it without resetting this flag. Default is ``False``.
        :param bool no_overlap: Unsupported transpose optimization flag;
            ``True`` raises an error. Default is ``False``.
        :raises ValueError: If ``input_numb_field`` is not an integer.
        :raises ValueError: If ``output_numb_field`` is not an integer.
        :raises ValueError: If ``filter_dim`` is not a list or tuple.
        :raises ValueError: If ``stride`` is not a dictionary.
        :raises ValueError: If ``optimize`` is not a boolean.
        :raises ValueError: If ``no_overlap`` is not a boolean.
        :raises NotImplementedError: If ``no_overlap`` is ``True``.
        """
        super().__init__()

        if not isinstance(input_numb_field, int):
            raise ValueError("input_numb_field must be int.")
        self._input_numb_field = input_numb_field

        if not isinstance(output_numb_field, int):
            raise ValueError("input_numb_field must be int.")
        self._output_numb_field = output_numb_field

        if not isinstance(filter_dim, (tuple, list)):
            raise ValueError("filter_dim must be tuple or list.")
        vect = filter_dim
        vect = torch.tensor(vect)
        self.register_buffer("_dim", vect, persistent=False)

        if not isinstance(stride, dict):
            raise ValueError("stride must be dictionary.")
        self._stride = Stride(stride)

        self._net = model

        if not isinstance(optimize, bool):
            raise ValueError("optimize must be bool.")
        self._optimize = optimize

        # choosing how to initialize based on optimization
        if self._optimize:
            # optimizing decorator ensure the function is called
            # just once
            self._choose_initialization = optimizing(
                self._initialize_convolution
            )
        else:
            self._choose_initialization = self._initialize_convolution

        if not isinstance(no_overlap, bool):
            raise ValueError("no_overlap must be bool.")

        if no_overlap:
            raise NotImplementedError

        self.transpose = self.transpose_overlap

    class DefaultKernel(torch.nn.Module):
        """
        The default kernel.

        :Example:

            >>> import torch
            >>> from pina._src.model.block.convolution import BaseContinuousConv
            >>> kernel = BaseContinuousConv.DefaultKernel(2, 4)
            >>> x = torch.randn(10, 2)
            >>> out = kernel(x)
        """

        def __init__(self, input_dim, output_dim):
            """
            Initialization of the :class:`DefaultKernel` class.

            :param int input_dim: The input dimension.
            :param int output_dim: The output dimension.
            :raises AssertionError: If ``input_dim`` is not an integer.
            :raises AssertionError: If ``output_dim`` is not an integer.
            """
            super().__init__()
            assert isinstance(input_dim, int)
            assert isinstance(output_dim, int)
            self._model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, output_dim),
            )

        def forward(self, x):
            """
            Forward pass.

            :param torch.Tensor x: The input data.
            :return: The output data.
            :rtype: torch.Tensor
            """
            return self._model(x)

    @property
    def net(self):
        """
        The neural network for inner parametrization.

        :return: The retained factory or, for a concrete block, spawned kernels.
        :rtype: Callable[[], torch.nn.Module] | torch.nn.ModuleList | None
        """
        return self._net

    @property
    def stride(self):
        """
        The stride of the filter.

        :return: The stride specification, replaced by a position tensor in
            :class:`pina.model.block.ContinuousConvBlock`.
        :rtype: Stride | torch.Tensor
        """
        return self._stride

    @property
    def filter_dim(self):
        """
        The shape of the filter.

        :return: The shape of the filter.
        :rtype: torch.Tensor
        """
        return self._dim

    @property
    def input_numb_field(self):
        """
        The number of input fields.

        :return: The number of input fields.
        :rtype: int
        """
        return self._input_numb_field

    @property
    def output_numb_field(self):
        """
        The number of output fields.

        :return: The number of output fields.
        :rtype: int
        """
        return self._output_numb_field

    @abstractmethod
    def forward(self, X):
        """
        Forward pass.

        :param torch.Tensor X: The input data.
        """

    @abstractmethod
    def transpose_overlap(self, X):
        """
        Transpose the convolution with overlap.

        :param torch.Tensor X: The input data.
        """

    @abstractmethod
    def transpose_no_overlap(self, X):
        """
        Transpose the convolution without overlap.

        :param torch.Tensor X: The input data.
        """

    @abstractmethod
    def _initialize_convolution(self, X, type_):
        """
        Initialize the convolution.

        :param torch.Tensor X: The input data.
        :param str type_: The type of initialization.
        """
