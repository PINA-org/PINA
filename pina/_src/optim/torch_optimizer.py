"""Module for wrapping PyTorch optimizers."""

import torch
from pina._src.core.utils import check_consistency
from pina._src.optim.optimizer_interface import OptimizerInterface


class TorchOptimizer(OptimizerInterface):
    """
    The wrapper class for PyTorch optimizers.

    This class wraps a ``torch.optim.Optimizer`` class and defers its
    instantiation until runtime. It enables a consistent interface across
    different optimizer backends while leveraging PyTorch's optimization
    algorithms.

    :Example:

        >>> from pina.optim import TorchOptimizer
        >>> import torch
        >>> optimizer = TorchOptimizer(torch.optim.Adam, lr=0.001)
        >>> optimizer.optimizer_class
        <class 'torch.optim.adam.Adam'>
    """

    def __init__(self, optimizer_class, **kwargs):
        """
        Initialization of the :class:`TorchOptimizer` class.

        :param type[torch.optim.Optimizer] optimizer_class: The optimizer class,
            not an already initialized instance.
        :param dict[str, Any] kwargs: Options forwarded unchanged to that
            class's constructor during ``hook``, such as ``lr``,
            ``weight_decay``, or optimizer-specific options. Do not include
            ``params``; parameters are supplied by the hook. See more
            `here <https://pytorch.org/docs/stable/optim.html#algorithms>`_.
        :raises ValueError: If ``optimizer_class`` is not a subclass of
            ``torch.optim.Optimizer``.
        """
        # Check consistency
        check_consistency(optimizer_class, torch.optim.Optimizer, subclass=True)

        # Initialize attributes
        self.optimizer_class = optimizer_class
        self.kwargs = kwargs
        self._optimizer_instance = None

    def hook(self, parameters):
        """
        Construct and store the optimizer for the supplied parameters.

        Repeated calls replace the stored optimizer rather than preserving its
        optimization state. Parameters are passed directly to the constructor.

        :param parameters: The parameters to optimize, such as the iterator
            returned by :meth:`torch.nn.Module.parameters`, or PyTorch parameter
            groups containing a ``"params"`` entry and per-group options.
        :type parameters: Iterable[torch.Tensor] | Iterable[dict[str, Any]]
        :raises ValueError: If the optimizer rejects empty parameters or invalid
            option values.
        :raises TypeError: If parameter types or constructor options are
            invalid.
        """
        self._optimizer_instance = self.optimizer_class(
            parameters, **self.kwargs
        )

    @property
    def instance(self):
        """
        The underlying optimizer object.

        :return: The stored optimizer, or ``None`` until ``hook`` succeeds.
        :rtype: torch.optim.Optimizer | None
        """
        return self._optimizer_instance
