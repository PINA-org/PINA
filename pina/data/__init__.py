"""Module containing utilities for dataset and data loader management."""

__all__ = [
    "Batcher",
    "DataModule",
    "MultiLoader",
]

# Back-compatibility with version 0.2, to be removed soon
import warnings

from pina._src.data.batcher import Batcher
from pina._src.data.data_module import DataModule
from pina._src.data.loader import MultiLoader

_DEPRECATED_IMPORTS = {"PinaDataModule": "DataModule"}


def __getattr__(name):
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina.data' is deprecated; use "
            f"pina.data.{_DEPRECATED_IMPORTS[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return globals()[_DEPRECATED_IMPORTS[name]]
