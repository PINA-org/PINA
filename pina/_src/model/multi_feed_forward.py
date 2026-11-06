"""Module for the Multi Feed Forward model class."""

from abc import ABC, abstractmethod
import torch
from pina._src.model.feed_forward import FeedForward


class MultiFeedForward(torch.nn.Module, ABC):
    """
    Multi Feed Forward neural network model class.

    This model allows to create a network with multiple Feed Forward neural
    networks combined together. The user is required to define the ``forward``
    method to choose how to combine the networks.

    :Example:

        >>> from pina.model import MultiFeedForward
        >>> class MyModel(MultiFeedForward):
        ...     def forward(self, x):
        ...         return self.ffn1(x) + self.ffn2(x)
        >>> model = MyModel({
        ...     "ffn1": {"input_dimensions": 2, "output_dimensions": 1},
        ...     "ffn2": {"input_dimensions": 2, "output_dimensions": 1},
        ... })
    """

    def __init__(self, ffn_dict):
        """
        Initialization of the :class:`MultiFeedForward` class.

        :param dict[str, dict[str, Any]] ffn_dict: Attribute names mapped to
            keyword arguments for :class:`pina.model.FeedForward`, including
            ``input_dimensions``, ``output_dimensions``, and optional hidden
            widths, activation classes, and bias settings. Each network is
            constructed and registered under its supplied name.
        :raises TypeError: If the input is not a dictionary.
        :raises ValueError: If an inner network rejects its dimensions.
        :raises RuntimeError: If inner layer/activation counts do not match.
        """
        super().__init__()

        if not isinstance(ffn_dict, dict):
            raise TypeError

        for name, constructor_args in ffn_dict.items():
            setattr(self, name, FeedForward(**constructor_args))

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for the :class:`MultiFeedForward` model.

        The user is required to define this method to choose how to combine the
        networks.

        :param tuple[Any, ...] args: Positional inputs defined by the subclass.
        :param dict[str, Any] kwargs: Keyword inputs defined by the subclass.
        """
