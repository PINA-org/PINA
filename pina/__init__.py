"""
A specialized framework for Scientific Machine Learning (SciML), providing
tools for Physics-Informed Neural Networks (PINNs), Neural Operators,
and data-driven physical modelling.
"""

import logging

__all__ = [
    "LabelTensor",
    "Trainer",
    "DataModule",
    "Graph",
]

logger = logging.getLogger("pina")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

from pina._src.core.label_tensor import LabelTensor
from pina._src.core.graph import Graph
from pina._src.core.trainer import Trainer
from pina._src.core.data_module import DataModule

# Back-compatibility with version 0.2, to be removed soon
import warnings

_DEPRECATED_IMPORTS = {"PinaDataModule": "DataModule"}


def __getattr__(name):
    if name == "Condition":
        warnings.warn(
            "Importing 'Condition' from 'pina' is deprecated; use the direct "
            "condition classes from 'pina.condition' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from pina._src.condition.condition import Condition

        return Condition

    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina' is deprecated; use "
            f"pina.{_DEPRECATED_IMPORTS[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return globals()[_DEPRECATED_IMPORTS[name]]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
