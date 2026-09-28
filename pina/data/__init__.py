"""Module containing utilities for dataset and data loader management."""

__all__ = [
    "DataModule",
]

# Back-compatibility with version 0.2, to be removed soon
import warnings

from pina._src.data.data_module import DataModule

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
