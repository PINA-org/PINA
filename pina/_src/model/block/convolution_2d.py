"""Module for the Continuous 2D Convolution class."""

import torch
from pina._src.model.block.convolution import BaseContinuousConv
from pina._src.model.block.utils_convolution import check_point, map_points_
from pina._src.model.block.integral import Integral


class ContinuousConvBlock(BaseContinuousConv):
    r"""
    Continuous Convolutional block.

    The class expects the input to be in the form:
    :math:`[B \times N_{in} \times N \times D]`, where :math:`B` is the
    batch_size, :math:`N_{in}` is the number of input fields, :math:`N`
    the number of points in the mesh, :math:`D` the dimension of the problem.
    In particular:

    *   :math:`D` is the number of spatial variables + 1. The last column must
        contain the field value. For example for 2D problems :math:`D=3` and
        the tensor will be something like ``[first coordinate, second
        coordinate, field value]``.
    *   :math:`N_{in}` represents the number of vectorial function presented.
        For example a vectorial function :math:`f = [f_1, f_2]` will have
        :math:`N_{in}=2`.

    .. seealso::

        **Original reference**:
        Coscia, D., Meneghetti, L., Demo, N. et al.
        *A continuous convolutional trainable filter for modelling unstructured
        data*. Comput Mech 72, 253-265 (2023).
        DOI `<https://doi.org/10.1007/s00466-023-02291-1>`_

    :Example:

        >>> import torch
        >>> from pina.model.block import ContinuousConvBlock
        >>> class MLP(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.model = torch.nn.Sequential(
        ...             torch.nn.Linear(2, 8), torch.nn.ReLU(),
        ...             torch.nn.Linear(8, 8), torch.nn.ReLU(),
        ...             torch.nn.Linear(8, 1)
        ...         )
        ...     def forward(self, x):
        ...         return self.model(x)
        >>> stride = {
        ...     "domain": [10, 10],
        ...     "start": [0, 0],
        ...     "jump": [3, 3],
        ...     "direction": [1, 1],
        ... }
        >>> conv = ContinuousConvBlock(1, 2, [3, 3], stride, MLP)
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
        Initialization of the :class:`ContinuousConvBlock` class.

        :param int input_numb_field: The number of input fields.
        :param int output_numb_field: The number of output fields.
        :param filter_dim: The shape of the filter.
        :type filter_dim: list[int | float] | tuple[int | float, ...]
        :param dict[str, list[int | float]] stride: Grid specification inserted
            in order: ``"domain"``, ``"start"``, ``"jump"``, ``"direction"``.
            Zero jump/direction entries are mutated during construction.
        :param model: No-argument kernel factory. ``None`` creates independent
            default MLPs from coordinates to one scalar. One kernel is created
            per input/output field pair. Default is ``None``.
        :type model: Callable[[], torch.nn.Module] | None
        :param bool optimize: Whether to track initialization calls using the
            wrapper. It does not skip initialization. Default is ``False``.
        :param bool no_overlap: Unsupported optimization flag. ``True`` raises
            an error. Default is ``False``.
        :raises ValueError: If field counts, filter/stride containers, flags,
            or spawned kernel types fail validation.
        :raises NotImplementedError: If ``no_overlap`` is ``True``.

        .. note::

            Initialization recomputes masks and grids on each call, including
            when ``optimize=True``. Both forward and transpose are available.

        :Example:
            >>> import torch
            >>> from pina.model.block import ContinuousConvBlock
            >>> class MLP(torch.nn.Module):
            ...     def __init__(self) -> None:
            ...         super().__init__()
            ...         self. model = torch.nn.Sequential(
            ...             torch.nn.Linear(2, 8),
            ...             torch.nn.ReLU(),
            ...             torch.nn.Linear(8, 8),
            ...             torch.nn.ReLU(),
            ...             torch.nn.Linear(8, 1)
            ...         )
            ...     def forward(self, x):
            ...         return self.model(x)
            >>> dim = [3, 3]
            >>> stride = {
            ...     "domain": [10, 10],
            ...     "start": [0, 0],
            ...     "jump": [3, 3],
            ...     "direction": [1, 1.]
            ... }
            >>> conv = ContinuousConvBlock(1, 2, dim, stride, MLP)
        """
        super().__init__(
            input_numb_field=input_numb_field,
            output_numb_field=output_numb_field,
            filter_dim=filter_dim,
            stride=stride,
            model=model,
            optimize=optimize,
            no_overlap=no_overlap,
        )

        # integral routine
        self._integral = Integral("discrete")

        # create the network
        self._net = self._spawn_networks(model)

        # stride for continuous convolution overridden
        self._stride = self._stride._stride_discrete

        # Define variables
        self._index = None
        self._grid = None
        self._grid_transpose = None

    def _spawn_networks(self, model):
        """
        Create a collection of kernels

        :param model: No-argument kernel factory; ``None`` selects DefaultKernel
            when the retained ``net`` attribute is also ``None``.
        :type model: Callable[[], torch.nn.Module] | None
        :raises ValueError: If a factory result is not a module instance.
        :return: A list of models.
        :rtype: torch.nn.ModuleList
        """
        nets = []
        if self._net is None:
            for _ in range(self._input_numb_field * self._output_numb_field):
                tmp = ContinuousConvBlock.DefaultKernel(len(self._dim), 1)
                nets.append(tmp)
        else:
            if not isinstance(model, object):
                raise ValueError(
                    "Expected a python class inheriting from torch.nn.Module"
                )

            for _ in range(self._input_numb_field * self._output_numb_field):
                tmp = model()
                if not isinstance(tmp, torch.nn.Module):
                    raise ValueError(
                        "The python class must be inherited from"
                        " torch.nn.Module. See the docstring for"
                        " an example."
                    )
                nets.append(tmp)

        return torch.nn.ModuleList(nets)

    def _extract_mapped_points(self, batch_idx, index, x):
        """
        Extract mapped points in the filter.

        :param int batch_idx: Batch position used to select the masks.
        :param list[torch.Tensor] index: Per-stride Boolean masks with shape
            ``(B, channels, N)``.
        :param torch.Tensor x: Input of shape ``(channels, N, d + 1)``.
        :return: Per-channel concatenated mapped points and per-channel tuples
            of point counts for each stride. The original input is unchanged.
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[tuple[int, ...], ...]]
        """
        mapped_points = []
        indeces_channels = []

        for stride_idx, current_stride in enumerate(self._stride):

            # indeces of points falling into filter range
            indeces = index[stride_idx][batch_idx]

            # how many points for each channel fall into the filter?
            numb_points_insiede = torch.sum(indeces, dim=-1).tolist()

            # extracting points for each channel
            # shape: [sum(numb_points_insiede), filter_dim + 1]
            point_stride = x[indeces]

            # mapping points in filter domain
            map_points_(point_stride[..., :-1], current_stride)

            # extracting points for each channel
            point_stride_channel = point_stride.split(numb_points_insiede)

            # appending in list for later use
            mapped_points.append(point_stride_channel)
            indeces_channels.append(numb_points_insiede)

        # stacking input for passing to neural net
        mapping = map(torch.cat, zip(*mapped_points))
        stacked_input = tuple(mapping)
        indeces_channels = tuple(zip(*indeces_channels))

        return stacked_input, indeces_channels

    def _find_index(self, X):
        """
        Extract indeces for convolution.

        :param torch.Tensor X: The input tensor.
        """
        # append the index for each stride
        index = []
        for _, current_stride in enumerate(self._stride):

            tmp = check_point(X, current_stride, self._dim)
            index.append(tmp)

        # storing the index
        self._index = index

    def _make_grid_forward(self, X):
        """
        Create forward convolution grid.

        :param torch.Tensor X: The input tensor.
        """
        # filter dimension + number of points in output grid
        filter_dim = len(self._dim)
        number_points = len(self._stride)

        # initialize the grid
        grid = torch.zeros(
            size=(
                X.shape[0],
                self._output_numb_field,
                number_points,
                filter_dim + 1,
            ),
            device=X.device,
            dtype=X.dtype,
        )
        grid[..., :-1] = self._stride + self._dim * 0.5

        # saving the grid
        self._grid = grid.detach()

    def _make_grid_transpose(self, X):
        """
        Create transpose convolution grid.

        :param torch.Tensor X: The input tensor.
        """
        # initialize to all zeros
        tmp = torch.zeros_like(X).as_subclass(torch.Tensor)
        tmp[..., :-1] = X[..., :-1]

        # save on tmp
        self._grid_transpose = tmp

    def _make_grid(self, X, type_):
        """
        Create convolution grid.

        :param torch.Tensor X: The input tensor.
        :param str type_: The type of convolution.
            Available options are: ``forward`` and ``inverse``.
        :raises TypeError: If the type is not in the available options.
        """
        # choose the type of convolution
        if type_ == "forward":
            self._make_grid_forward(X)
            return
        if type_ == "inverse":
            self._make_grid_transpose(X)
            return
        raise TypeError

    def _initialize_convolution(self, X, type_="forward"):
        """
        Initialize the convolution by setting a grid and computing the index to
        find the points inside the filter.

        :param torch.Tensor X: The input tensor.
        :param str type_: The type of convolution. Available options are:
            ``forward`` and ``inverse``. Default is ``forward``.
        """

        # variable for the convolution
        self._make_grid(X, type_)

        # calculate the index
        self._find_index(X)

    def forward(self, X):
        """
        Forward pass.

        :param torch.Tensor X: Input shaped ``(B, input_numb_field, N, d + 1)``
            with coordinates followed by the scalar field value.
        :return: Tensor shaped ``(B, output_numb_field, S, d + 1)``, containing
            filter centers and unweighted kernel/field sums at ``S`` strides.
            The input is not modified; masks and the internal grid are replaced.
        :rtype: torch.Tensor
        """

        # initialize convolution
        if self.training:  # we choose what to do based on optimization
            self._choose_initialization(X, type_="forward")

        else:  # we always initialize on testing
            self._initialize_convolution(X, "forward")

        # create convolutional array
        conv = self._grid.clone().detach()

        # total number of fields
        tot_dim = self._output_numb_field * self._input_numb_field

        for batch_idx, x in enumerate(X):

            # extract mapped points
            stacked_input, indeces_channels = self._extract_mapped_points(
                batch_idx, self._index, x
            )

            # compute the convolution

            # storing intermidiate results for each channel convolution
            res_tmp = []
            # for each field
            for idx_conv in range(tot_dim):
                # index for each input field
                idx = idx_conv % self._input_numb_field
                # extract input for each channel
                single_channel_input = stacked_input[idx]
                # extract filter
                net = self._net[idx_conv]
                # calculate filter value
                staked_output = net(single_channel_input[..., :-1])
                # perform integral for all strides in one field
                integral = self._integral(
                    staked_output,
                    single_channel_input[..., -1],
                    indeces_channels[idx],
                )
                res_tmp.append(integral)

            # stacking integral results
            res_tmp = torch.stack(res_tmp)

            # sum filters (for each input fields) in groups
            # for different ouput fields
            conv[batch_idx, ..., -1] = res_tmp.reshape(
                self._output_numb_field, self._input_numb_field, -1
            ).sum(1)
        return conv

    def transpose_no_overlap(self, integrals, X):
        """
        Transpose pass for non-overlapping filters.

        :param torch.Tensor integrals: Weights of shape ``(B, input_numb_field,
            S)`` at the stride positions.
        :param torch.Tensor X: Target coordinates and placeholder values of
            shape ``(B, output_numb_field, M, d + 1)``.
        :return: Target grid with transposed values, of the same shape as ``X``.
        :rtype: torch.Tensor

        .. note::

            This method can be called directly, but the constructor rejects
            ``no_overlap=True`` and never selects it as ``transpose``.
        """

        # initialize convolution
        if self.training:  # we choose what to do based on optimization
            self._choose_initialization(X, type_="inverse")

        else:  # we always initialize on testing
            self._initialize_convolution(X, "inverse")

        # initialize grid
        X = self._grid_transpose.clone().detach()
        conv_transposed = self._grid_transpose.clone().detach()

        # total number of dim
        tot_dim = self._input_numb_field * self._output_numb_field

        for batch_idx, x in enumerate(X):

            # extract mapped points
            stacked_input, indeces_channels = self._extract_mapped_points(
                batch_idx, self._index, x
            )

            # compute the transpose convolution

            # total number of fields
            res_tmp = []

            # for each field
            for idx_conv in range(tot_dim):
                # index for each output field
                idx = idx_conv % self._output_numb_field
                # index for each input field
                idx_in = idx_conv % self._input_numb_field
                # extract input for each field
                single_channel_input = stacked_input[idx]
                rep_idx = torch.tensor(indeces_channels[idx])
                integral = integrals[batch_idx, idx_in, :].repeat_interleave(
                    rep_idx
                )
                # extract filter
                net = self._net[idx_conv]
                # perform transpose convolution for all strides in one field
                staked_output = net(single_channel_input[..., :-1]).flatten()
                integral = staked_output * integral
                res_tmp.append(integral)

            # stacking integral results and sum
            # filters (for each input fields) in groups
            # for different output fields
            res_tmp = (
                torch.stack(res_tmp)
                .reshape(self._input_numb_field, self._output_numb_field, -1)
                .sum(0)
            )
            conv_transposed[batch_idx, ..., -1] = res_tmp

        return conv_transposed

    def transpose_overlap(self, integrals, X):
        """
        Transpose pass in the layer for overlapping filters.

        :param torch.Tensor integrals: Weights of shape ``(B, input_numb_field,
            S)`` at the stride positions.
        :param torch.Tensor X: Target coordinates and placeholder values of
            shape ``(B, output_numb_field, M, d + 1)``.
        :return: Target grid with transposed values, of the same shape as ``X``.
            Overlapping filter contributions are summed; input is unchanged.
        :rtype: torch.Tensor

        .. note::

            This is the method selected by the ``transpose`` attribute.
        """

        # initialize convolution
        if self.training:  # we choose what to do based on optimization
            self._choose_initialization(X, type_="inverse")

        else:  # we always initialize on testing
            self._initialize_convolution(X, "inverse")

        # initialize grid
        X = self._grid_transpose.clone().detach()
        conv_transposed = self._grid_transpose.clone().detach()

        # list to iterate for calculating nn output
        tmp = list(range(self._output_numb_field))
        iterate_conv = [
            item for item in tmp for _ in range(self._input_numb_field)
        ]

        for batch_idx, x in enumerate(X):

            # accumulator for the convolution on different batches
            accumulator_batch = torch.zeros(
                size=(
                    self._grid_transpose.shape[1],
                    self._grid_transpose.shape[2],
                ),
                requires_grad=True,
                device=X.device,
                dtype=X.dtype,
            ).clone()

            for stride_idx, current_stride in enumerate(self._stride):
                # indeces of points falling into filter range
                indeces = self._index[stride_idx][batch_idx]

                # number of points for each channel
                numb_pts_channel = tuple(indeces.sum(dim=-1))

                # extracting points for each channel
                point_stride = x[indeces]

                # if no points to upsample we just skip
                if point_stride.nelement() == 0:
                    continue

                # mapping points in filter domain
                map_points_(point_stride[..., :-1], current_stride)

                # input points for kernels
                # we split for extracting number of points for each channel
                nn_input_pts = point_stride[..., :-1].split(numb_pts_channel)

                # accumulate partial convolution results for each field
                res_tmp = []

                # for each channel field compute transpose convolution
                for idx_conv, idx_channel_out in enumerate(iterate_conv):

                    # index for input channels
                    idx_channel_in = idx_conv % self._input_numb_field

                    # extract filter
                    net = self._net[idx_conv]

                    # calculate filter value
                    staked_output = net(nn_input_pts[idx_channel_out])

                    # perform integral for all strides in one field
                    integral = (
                        staked_output
                        * integrals[batch_idx, idx_channel_in, stride_idx]
                    )
                    # append results
                    res_tmp.append(integral.flatten())

                # computing channel sum
                channel_sum = []
                start = 0
                for _ in range(self._output_numb_field):
                    tmp = res_tmp[start : start + self._input_numb_field]
                    tmp = torch.vstack(tmp).sum(dim=0)
                    channel_sum.append(tmp)
                    start += self._input_numb_field

                # accumulate the results
                accumulator_batch[indeces] += torch.hstack(channel_sum)

            # save results of accumulation for each batch
            conv_transposed[batch_idx, ..., -1] = accumulator_batch

        return conv_transposed
