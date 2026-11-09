"""Module to perform integration for continuous convolution."""

import torch


class Integral:
    """
    Class allowing integration for continous convolution.

    :Example:

        >>> import torch
        >>> from pina._src.model.block.integral import Integral
        >>> integral = Integral("discrete")
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> y = torch.tensor([4.0, 5.0, 6.0])
        >>> idx = [2, 1]
        >>> integral(x, y, idx)
        tensor([14., 18.])
    """

    def __init__(self, param):
        """
        Initialization of the :class:`Integral` class.

        :param str param: Integration mode, ``"discrete"`` or
            ``"continuous"``. The continuous routine is not implemented.
        :raises TypeError: If the parameter is neither ``discrete``
            nor ``continuous``.
        """
        if param == "discrete":
            self.make_integral = self.integral_param_disc
        elif param == "continuous":
            self.make_integral = self.integral_param_cont
        else:
            raise TypeError

    def __call__(self, *args, **kwds):
        """
        Call the integral function

        :param tuple[Any, ...] args: Positional ``x``, ``y``, and ``idx`` for
            selected integration routine.
        :param dict[str, Any] kwds: Named ``x``, ``y``, and ``idx`` arguments.
        :raises NotImplementedError: If the selected mode is continuous.
        :return: The integral of the input.
        :rtype: torch.Tensor
        """
        return self.make_integral(*args, **kwds)

    def _prepend_zero(self, x):
        """
        Create bins to perform integration.

        :param torch.Tensor x: The input tensor.
        :return: The bins for the integral.
        :rtype: torch.Tensor
        """
        return torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), x))

    def integral_param_disc(self, x, y, idx):
        """
        Perform discrete integration with discrete parameters.

        :param torch.Tensor x: The first input tensor.
        :param torch.Tensor y: The second input tensor.
        :param idx: Point counts for contiguous stride segments, not indices.
            Their sum must equal the number of flattened products.
        :type idx: list[int] | tuple[int, ...]
        :return: Sum of flattened ``x * y`` in each segment, with shape
            ``(len(idx),)``. No mesh spacing or quadrature weights are applied.
        :rtype: torch.Tensor
        """
        cs_idxes = self._prepend_zero(torch.cumsum(torch.tensor(idx), 0))
        cs = self._prepend_zero(torch.cumsum(x.flatten() * y.flatten(), 0))
        return cs[cs_idxes[1:]] - cs[cs_idxes[:-1]]

    def integral_param_cont(self, x, y, idx):
        """
        Perform continuous integration with continuous parameters.

        :param torch.Tensor x: The first input tensor.
        :param torch.Tensor y: The second input tensor.
        :param idx: Point counts for each stride segment.
        :type idx: list[int] | tuple[int, ...]
        :raises NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError
